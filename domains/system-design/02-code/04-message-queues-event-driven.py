"""
Jour 4 -- Message Queues & Event-Driven Architecture
Demonstrations interactives en Python.

Usage:
    python 04-message-queues-event-driven.py

Ce fichier implemente un mini-broker inspire de Kafka : topics, partitions,
consumer groups, offsets, rebalance, DLQ. Le but est de *montrer* comment
fonctionne en interne un systeme comme Kafka, sans les 200 Mo de Scala.

Chaque section est independante et peut etre executee via main().
"""

import time
import threading
import random
import hashlib
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

SEPARATOR = "=" * 70


# =============================================================================
# SECTION 1 : Message, Partition, Topic -- les primitives de Kafka
# =============================================================================


@dataclass
class Message:
    """Un message dans le broker.

    WHY key ? La cle sert a choisir la partition (hash(key) % n_partitions).
    Les messages avec la meme cle vont toujours dans la meme partition,
    ce qui garantit leur ordre. Ex : key = user_id pour garder l'ordre
    des evenements d'un user.

    WHY offset ? Position du message dans la partition. Le consumer stocke
    l'offset du dernier message traite pour pouvoir reprendre apres un crash.
    """

    key: Optional[str]
    value: Any
    offset: int = -1  # Assigne par la partition lors du append
    timestamp: float = field(default_factory=time.time)


class Partition:
    """Une partition = un log append-only ordonne.

    WHY append-only ? C'est l'idee centrale de Kafka : on n'edite jamais
    les messages, on ne fait qu'ajouter a la fin. Cela permet :
    1) des writes sequentielles tres rapides (disque mechanical friendly),
    2) une lecture concurrente par plusieurs consumers sans locks,
    3) la possibilite de rejouer l'historique (replay).
    """

    def __init__(self, partition_id: int):
        self.partition_id = partition_id
        self.log: list[Message] = []
        self._lock = threading.Lock()  # Protege les appends concurrents

    def append(self, msg: Message) -> int:
        """Append un message et retourne son offset."""
        with self._lock:
            msg.offset = len(self.log)
            self.log.append(msg)
            return msg.offset

    def read(self, from_offset: int, max_messages: int = 10) -> list[Message]:
        """Lit N messages a partir d'un offset donne.

        WHY from_offset ? Le consumer gere lui-meme son offset. Plusieurs
        consumers peuvent lire la meme partition a des offsets differents
        (ex : un consumer rapide en tete, un consumer de replay historique).
        """
        with self._lock:
            return self.log[from_offset:from_offset + max_messages]

    def high_watermark(self) -> int:
        """Le prochain offset a ecrire = nombre total de messages."""
        return len(self.log)


class Topic:
    """Un topic = un ensemble de partitions.

    WHY partitioning ? Un seul log serait limite par le CPU/disque d'une
    machine. Avec N partitions, on peut paralleliser sur N machines et
    avoir N consumers en parallele. Le prix : l'ordre global disparait,
    on a seulement l'ordre par partition.
    """

    def __init__(self, name: str, num_partitions: int):
        self.name = name
        self.num_partitions = num_partitions
        self.partitions = [Partition(i) for i in range(num_partitions)]

    def _partition_for(self, key: Optional[str]) -> int:
        """Choisit la partition pour une cle donnee.

        WHY hashing ? On veut une distribution uniforme ET deterministe :
        meme cle -> meme partition, meme si on redemarre le producer.
        """
        if key is None:
            # Pas de cle : round-robin aleatoire (simplification)
            return random.randint(0, self.num_partitions - 1)
        # hash stable via sha1 (pas python hash() qui est randomise)
        h = int(hashlib.sha1(key.encode()).hexdigest(), 16)
        return h % self.num_partitions

    def publish(self, key: Optional[str], value: Any) -> tuple[int, int]:
        """Publie un message et retourne (partition_id, offset)."""
        pid = self._partition_for(key)
        msg = Message(key=key, value=value)
        offset = self.partitions[pid].append(msg)
        return pid, offset


# =============================================================================
# SECTION 2 : Consumer Group -- le coeur du modele pub/sub + load balancing
# =============================================================================


class ConsumerGroup:
    """Un consumer group se partage les partitions d'un topic.

    WHY groups ? Ils permettent deux choses a la fois :
    1) Load balancing A L'INTERIEUR d'un groupe : les partitions sont
       reparties entre les consumers du groupe (chaque partition a UN seul
       consumer dans le groupe).
    2) Pub/sub ENTRE groupes : chaque groupe consomme independamment le
       meme flux. Le groupe "email" et le groupe "analytics" recoivent
       tous les deux chaque message.

    L'offset est stocke PAR (group, partition), pas par consumer individuel.
    Si un consumer tombe, un autre reprend a l'offset sauvegarde.
    """

    def __init__(self, group_id: str, topic: Topic):
        self.group_id = group_id
        self.topic = topic
        # offsets[partition_id] = prochain offset a lire
        self.offsets: dict[int, int] = {p: 0 for p in range(topic.num_partitions)}
        self.consumers: list["Consumer"] = []
        self.assignment: dict[str, list[int]] = {}  # consumer_id -> [partitions]
        self._lock = threading.Lock()

    def join(self, consumer: "Consumer"):
        """Ajoute un consumer et declenche un rebalance."""
        with self._lock:
            self.consumers.append(consumer)
            self._rebalance()

    def leave(self, consumer: "Consumer"):
        """Retire un consumer et declenche un rebalance."""
        with self._lock:
            self.consumers.remove(consumer)
            self._rebalance()

    def _rebalance(self):
        """Repartit les partitions entre les consumers actifs.

        WHY rebalance ? Chaque fois qu'un consumer arrive ou part, il faut
        redistribuer les partitions. Algorithme simple : round-robin.
        En vrai Kafka, il y a RangeAssignor, RoundRobinAssignor, StickyAssignor...
        Pendant le rebalance, la consommation est pausee : c'est un point
        douloureux a minimiser.
        """
        self.assignment = {c.consumer_id: [] for c in self.consumers}
        if not self.consumers:
            return
        for pid in range(self.topic.num_partitions):
            consumer = self.consumers[pid % len(self.consumers)]
            self.assignment[consumer.consumer_id].append(pid)
        print(f"  [REBALANCE group={self.group_id}] assignment = {self.assignment}")

    def commit(self, partition_id: int, offset: int):
        """Valide l'offset traite.

        WHY commit APRES traitement ? C'est le at-least-once pattern :
        si le consumer crash entre read et commit, un autre consumer
        reprendra au dernier offset committe et retraitera les messages.
        -> Le consumer DOIT etre idempotent.
        """
        with self._lock:
            # On ne regresse jamais un offset (evite les corner cases)
            if offset + 1 > self.offsets[partition_id]:
                self.offsets[partition_id] = offset + 1


# =============================================================================
# SECTION 3 : Consumer -- boucle de consommation avec DLQ et retries
# =============================================================================


class Consumer:
    """Un consumer qui poll les partitions qui lui sont assignees.

    WHY polling ? Kafka utilise un modele pull : le consumer demande
    au broker "donne-moi les messages depuis l'offset X". Cela permet
    au consumer d'aller a son rythme sans etre submerge (backpressure).
    Push serait plus simple mais impose au broker de gerer la vitesse
    de chaque consumer.
    """

    def __init__(
        self,
        consumer_id: str,
        group: ConsumerGroup,
        handler: Callable[[Message], None],
        max_retries: int = 3,
        dlq: Optional["DLQ"] = None,
    ):
        self.consumer_id = consumer_id
        self.group = group
        self.handler = handler
        self.max_retries = max_retries
        self.dlq = dlq
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def start(self):
        self.group.join(self)
        self.running = True
        self.thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        self.group.leave(self)

    def _poll_loop(self):
        while self.running:
            # Recupere les partitions assignees a ce consumer
            assigned = self.group.assignment.get(self.consumer_id, [])
            for pid in assigned:
                # Lire a partir de l'offset actuel du groupe pour cette partition
                offset = self.group.offsets[pid]
                msgs = self.group.topic.partitions[pid].read(offset, max_messages=5)
                for msg in msgs:
                    self._process(pid, msg)
            time.sleep(0.05)  # poll interval

    def _process(self, partition_id: int, msg: Message):
        """Traite un message avec retry et DLQ.

        WHY retry + DLQ ? Un bug transitoire (DB momentanement down) ne doit
        pas perdre le message. On retry N fois avec backoff, puis si ca
        continue, on envoie en DLQ pour inspection manuelle. Sans DLQ, un
        "poison message" bloquerait la partition indefiniment.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                self.handler(msg)
                self.group.commit(partition_id, msg.offset)
                return
            except Exception as e:
                print(f"    [{self.consumer_id}] p{partition_id}@{msg.offset} "
                      f"retry {attempt}/{self.max_retries} : {e}")
                time.sleep(0.02 * attempt)  # backoff lineaire (en prod : exponentiel)
        # Echec definitif : envoyer en DLQ et committer pour debloquer
        if self.dlq:
            self.dlq.push(msg, reason="max retries exceeded")
        print(f"    [{self.consumer_id}] p{partition_id}@{msg.offset} -> DLQ")
        self.group.commit(partition_id, msg.offset)


# =============================================================================
# SECTION 4 : Dead Letter Queue
# =============================================================================


class DLQ:
    """Dead Letter Queue : messages qui ont echoue malgre les retries."""

    def __init__(self):
        self.messages: list[tuple[Message, str]] = []
        self._lock = threading.Lock()

    def push(self, msg: Message, reason: str):
        with self._lock:
            self.messages.append((msg, reason))

    def size(self) -> int:
        return len(self.messages)


# =============================================================================
# SECTION 5 : Demos
# =============================================================================


def demo_ordering_by_key():
    """Montre que les messages avec la meme cle vont dans la meme partition
    et sont donc ordonnes."""
    print(f"\n{SEPARATOR}\n  DEMO 1 : Ordering par cle de partition\n{SEPARATOR}")
    topic = Topic("orders", num_partitions=3)
    # On envoie 5 events pour user-A et 5 pour user-B, entrelaces
    for i in range(5):
        topic.publish(key="user-A", value=f"A-event-{i}")
        topic.publish(key="user-B", value=f"B-event-{i}")
    for pid, p in enumerate(topic.partitions):
        vals = [m.value for m in p.log]
        print(f"  Partition {pid} : {vals}")
    print("  Observe : tous les events de user-A sont sur la meme partition,")
    print("  dans l'ordre. Pareil pour user-B. L'ordre global n'existe pas,")
    print("  mais l'ordre par cle est garanti.")


def demo_consumer_group_load_balancing():
    """Deux consumers dans un groupe se partagent 4 partitions."""
    print(f"\n{SEPARATOR}\n  DEMO 2 : Consumer group & load balancing\n{SEPARATOR}")
    topic = Topic("events", num_partitions=4)
    for i in range(20):
        topic.publish(key=f"k{i % 8}", value=f"msg-{i}")

    group = ConsumerGroup("workers", topic)
    processed = defaultdict(list)

    def handler_factory(cid):
        def h(msg: Message):
            processed[cid].append(msg.value)
        return h

    c1 = Consumer("C1", group, handler_factory("C1"))
    c2 = Consumer("C2", group, handler_factory("C2"))
    c1.start()
    c2.start()
    time.sleep(0.4)  # laisser le temps de consommer
    c1.stop()
    c2.stop()
    for cid, msgs in processed.items():
        print(f"  {cid} a traite {len(msgs)} messages : {msgs[:5]}...")
    total = sum(len(v) for v in processed.values())
    print(f"  Total : {total}/20 (exactly-once au sein du groupe)")


def demo_pubsub_multiple_groups():
    """Deux groupes lisent le meme topic independamment."""
    print(f"\n{SEPARATOR}\n  DEMO 3 : Pub/sub entre groupes\n{SEPARATOR}")
    topic = Topic("order-events", num_partitions=2)
    for i in range(6):
        topic.publish(key=f"order-{i}", value=f"OrderPlaced#{i}")

    email_group = ConsumerGroup("email-service", topic)
    analytics_group = ConsumerGroup("analytics-service", topic)
    email_received = []
    analytics_received = []

    c_email = Consumer("email-worker", email_group, lambda m: email_received.append(m.value))
    c_ana = Consumer("analytics-worker", analytics_group, lambda m: analytics_received.append(m.value))
    c_email.start()
    c_ana.start()
    time.sleep(0.3)
    c_email.stop()
    c_ana.stop()

    print(f"  email-service    : {len(email_received)} messages -> {email_received}")
    print(f"  analytics-service: {len(analytics_received)} messages -> {analytics_received}")
    print("  Les deux groupes ont recu tous les messages : c'est du pub/sub.")


def demo_dlq_with_poison_message():
    """Un message qui plante systematiquement finit en DLQ."""
    print(f"\n{SEPARATOR}\n  DEMO 4 : Dead Letter Queue\n{SEPARATOR}")
    topic = Topic("payments", num_partitions=1)
    topic.publish(key=None, value={"amount": 100, "ok": True})
    topic.publish(key=None, value={"amount": -1, "ok": False})  # poison
    topic.publish(key=None, value={"amount": 50, "ok": True})

    dlq = DLQ()
    group = ConsumerGroup("billing", topic)
    processed = []

    def handler(msg: Message):
        if not msg.value["ok"]:
            raise ValueError(f"invalid amount {msg.value['amount']}")
        processed.append(msg.value)

    c = Consumer("billing-1", group, handler, max_retries=2, dlq=dlq)
    c.start()
    time.sleep(0.5)
    c.stop()

    print(f"  Messages processes OK : {processed}")
    print(f"  DLQ size : {dlq.size()}")
    print(f"  DLQ content : {[m.value for m, _ in dlq.messages]}")


def demo_rebalance():
    """On ajoute un consumer en plein traitement : rebalance."""
    print(f"\n{SEPARATOR}\n  DEMO 5 : Rebalance au join\n{SEPARATOR}")
    topic = Topic("stream", num_partitions=4)
    for i in range(8):
        topic.publish(key=None, value=f"m{i}")
    group = ConsumerGroup("g", topic)

    c1 = Consumer("C1", group, lambda m: None)
    c1.start()
    print("  Etat apres C1 seul : tout sur C1")
    time.sleep(0.1)

    c2 = Consumer("C2", group, lambda m: None)
    c2.start()
    print("  Etat apres ajout de C2 :")
    time.sleep(0.1)

    c3 = Consumer("C3", group, lambda m: None)
    c3.start()
    print("  Etat apres ajout de C3 :")
    time.sleep(0.1)

    c1.stop()
    c2.stop()
    c3.stop()


def main():
    random.seed(42)
    demo_ordering_by_key()
    demo_consumer_group_load_balancing()
    demo_pubsub_multiple_groups()
    demo_dlq_with_poison_message()
    demo_rebalance()
    print(f"\n{SEPARATOR}\n  Fin des demos.\n{SEPARATOR}")


if __name__ == "__main__":
    main()
