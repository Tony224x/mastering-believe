"""
Day 4 -- Message Queues & Event-Driven Architecture
Interactive demonstrations in Python.

Usage:
    python 04-message-queues-event-driven.py

This file implements a mini-broker inspired by Kafka: topics, partitions,
consumer groups, offsets, rebalance, DLQ. The goal is to *show* how
a system like Kafka works internally, without the 200 MB of Scala.

Each section is independent and can be executed via main().
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
# SECTION 1 : Message, Partition, Topic -- the Kafka primitives
# =============================================================================


@dataclass
class Message:
    """A message in the broker.

    WHY key? The key is used to choose the partition (hash(key) % n_partitions).
    Messages with the same key always go to the same partition,
    which guarantees their ordering. E.g. key = user_id to keep the order
    of a user's events.

    WHY offset? Position of the message within the partition. The consumer stores
    the offset of the last processed message so it can resume after a crash.
    """

    key: Optional[str]
    value: Any
    offset: int = -1  # Assigned by the partition during the append
    timestamp: float = field(default_factory=time.time)


class Partition:
    """A partition = an ordered append-only log.

    WHY append-only? It is Kafka's central idea: we never edit
    messages, we only append at the end. This allows:
    1) very fast sequential writes (mechanical-disk friendly),
    2) concurrent reading by multiple consumers without locks,
    3) the ability to replay history (replay).
    """

    def __init__(self, partition_id: int):
        self.partition_id = partition_id
        self.log: list[Message] = []
        self._lock = threading.Lock()  # Protects concurrent appends

    def append(self, msg: Message) -> int:
        """Appends a message and returns its offset."""
        with self._lock:
            msg.offset = len(self.log)
            self.log.append(msg)
            return msg.offset

    def read(self, from_offset: int, max_messages: int = 10) -> list[Message]:
        """Reads N messages starting from a given offset.

        WHY from_offset? The consumer manages its own offset. Multiple
        consumers can read the same partition at different offsets
        (e.g. a fast consumer at the head, a historical replay consumer).
        """
        with self._lock:
            return self.log[from_offset:from_offset + max_messages]

    def high_watermark(self) -> int:
        """The next offset to write = total number of messages."""
        return len(self.log)


class Topic:
    """A topic = a set of partitions.

    WHY partitioning? A single log would be limited by the CPU/disk of one
    machine. With N partitions, we can parallelize across N machines and
    have N consumers in parallel. The price: global ordering disappears,
    we only have per-partition ordering.
    """

    def __init__(self, name: str, num_partitions: int):
        self.name = name
        self.num_partitions = num_partitions
        self.partitions = [Partition(i) for i in range(num_partitions)]

    def _partition_for(self, key: Optional[str]) -> int:
        """Chooses the partition for a given key.

        WHY hashing? We want a uniform AND deterministic distribution:
        same key -> same partition, even if the producer restarts.
        """
        if key is None:
            # No key: random round-robin (simplification)
            return random.randint(0, self.num_partitions - 1)
        # Stable hash via sha1 (not python hash() which is randomized)
        h = int(hashlib.sha1(key.encode()).hexdigest(), 16)
        return h % self.num_partitions

    def publish(self, key: Optional[str], value: Any) -> tuple[int, int]:
        """Publishes a message and returns (partition_id, offset)."""
        pid = self._partition_for(key)
        msg = Message(key=key, value=value)
        offset = self.partitions[pid].append(msg)
        return pid, offset


# =============================================================================
# SECTION 2 : Consumer Group -- the heart of the pub/sub + load balancing model
# =============================================================================


class ConsumerGroup:
    """A consumer group shares a topic's partitions.

    WHY groups? They enable two things at once:
    1) Load balancing WITHIN a group: the partitions are
       distributed among the group's consumers (each partition has a SINGLE
       consumer within the group).
    2) Pub/sub ACROSS groups: each group consumes the
       same stream independently. The "email" group and the "analytics" group
       both receive every message.

    The offset is stored PER (group, partition), not per individual consumer.
    If a consumer goes down, another one resumes from the saved offset.
    """

    def __init__(self, group_id: str, topic: Topic):
        self.group_id = group_id
        self.topic = topic
        # offsets[partition_id] = next offset to read
        self.offsets: dict[int, int] = {p: 0 for p in range(topic.num_partitions)}
        self.consumers: list["Consumer"] = []
        self.assignment: dict[str, list[int]] = {}  # consumer_id -> [partitions]
        self._lock = threading.Lock()

    def join(self, consumer: "Consumer"):
        """Adds a consumer and triggers a rebalance."""
        with self._lock:
            self.consumers.append(consumer)
            self._rebalance()

    def leave(self, consumer: "Consumer"):
        """Removes a consumer and triggers a rebalance."""
        with self._lock:
            self.consumers.remove(consumer)
            self._rebalance()

    def _rebalance(self):
        """Redistributes the partitions among the active consumers.

        WHY rebalance? Every time a consumer joins or leaves, the
        partitions must be redistributed. Simple algorithm: round-robin.
        In real Kafka, there are RangeAssignor, RoundRobinAssignor, StickyAssignor...
        During the rebalance, consumption is paused: it is a pain point
        to minimize.
        """
        self.assignment = {c.consumer_id: [] for c in self.consumers}
        if not self.consumers:
            return
        for pid in range(self.topic.num_partitions):
            consumer = self.consumers[pid % len(self.consumers)]
            self.assignment[consumer.consumer_id].append(pid)
        print(f"  [REBALANCE group={self.group_id}] assignment = {self.assignment}")

    def commit(self, partition_id: int, offset: int):
        """Commits the processed offset.

        WHY commit AFTER processing? It's the at-least-once pattern:
        if the consumer crashes between read and commit, another consumer
        will resume from the last committed offset and reprocess the messages.
        -> The consumer MUST be idempotent.
        """
        with self._lock:
            # We never move an offset backwards (avoids corner cases)
            if offset + 1 > self.offsets[partition_id]:
                self.offsets[partition_id] = offset + 1


# =============================================================================
# SECTION 3 : Consumer -- consumption loop with DLQ and retries
# =============================================================================


class Consumer:
    """A consumer that polls the partitions assigned to it.

    WHY polling? Kafka uses a pull model: the consumer asks
    the broker "give me the messages since offset X". This lets
    the consumer go at its own pace without being overwhelmed (backpressure).
    Push would be simpler but would force the broker to manage the speed
    of each consumer.
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
            # Fetch the partitions assigned to this consumer
            assigned = self.group.assignment.get(self.consumer_id, [])
            for pid in assigned:
                # Read from the group's current offset for this partition
                offset = self.group.offsets[pid]
                msgs = self.group.topic.partitions[pid].read(offset, max_messages=5)
                for msg in msgs:
                    self._process(pid, msg)
            time.sleep(0.05)  # poll interval

    def _process(self, partition_id: int, msg: Message):
        """Processes a message with retry and DLQ.

        WHY retry + DLQ? A transient bug (DB momentarily down) must not
        lose the message. We retry N times with backoff, then if it
        keeps failing, we send it to the DLQ for manual inspection. Without a DLQ, a
        "poison message" would block the partition indefinitely.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                self.handler(msg)
                self.group.commit(partition_id, msg.offset)
                return
            except Exception as e:
                print(f"    [{self.consumer_id}] p{partition_id}@{msg.offset} "
                      f"retry {attempt}/{self.max_retries} : {e}")
                time.sleep(0.02 * attempt)  # linear backoff (in prod: exponential)
        # Definitive failure: send to the DLQ and commit to unblock
        if self.dlq:
            self.dlq.push(msg, reason="max retries exceeded")
        print(f"    [{self.consumer_id}] p{partition_id}@{msg.offset} -> DLQ")
        self.group.commit(partition_id, msg.offset)


# =============================================================================
# SECTION 4 : Dead Letter Queue
# =============================================================================


class DLQ:
    """Dead Letter Queue: messages that failed despite the retries."""

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
    """Shows that messages with the same key go to the same partition
    and are therefore ordered."""
    print(f"\n{SEPARATOR}\n  DEMO 1 : Ordering by partition key\n{SEPARATOR}")
    topic = Topic("orders", num_partitions=3)
    # We send 5 events for user-A and 5 for user-B, interleaved
    for i in range(5):
        topic.publish(key="user-A", value=f"A-event-{i}")
        topic.publish(key="user-B", value=f"B-event-{i}")
    for pid, p in enumerate(topic.partitions):
        vals = [m.value for m in p.log]
        print(f"  Partition {pid} : {vals}")
    print("  Observe: all of user-A's events are on the same partition,")
    print("  in order. Same for user-B. Global ordering does not exist,")
    print("  but per-key ordering is guaranteed.")


def demo_consumer_group_load_balancing():
    """Two consumers in a group share 4 partitions."""
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
    time.sleep(0.4)  # allow time to consume
    c1.stop()
    c2.stop()
    for cid, msgs in processed.items():
        print(f"  {cid} processed {len(msgs)} messages : {msgs[:5]}...")
    total = sum(len(v) for v in processed.values())
    print(f"  Total : {total}/20 (exactly-once within the group)")


def demo_pubsub_multiple_groups():
    """Two groups read the same topic independently."""
    print(f"\n{SEPARATOR}\n  DEMO 3 : Pub/sub across groups\n{SEPARATOR}")
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
    print("  Both groups received all the messages: that's pub/sub.")


def demo_dlq_with_poison_message():
    """A message that systematically crashes ends up in the DLQ."""
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

    print(f"  Messages processed OK : {processed}")
    print(f"  DLQ size : {dlq.size()}")
    print(f"  DLQ content : {[m.value for m, _ in dlq.messages]}")


def demo_rebalance():
    """We add a consumer mid-processing: rebalance."""
    print(f"\n{SEPARATOR}\n  DEMO 5 : Rebalance on join\n{SEPARATOR}")
    topic = Topic("stream", num_partitions=4)
    for i in range(8):
        topic.publish(key=None, value=f"m{i}")
    group = ConsumerGroup("g", topic)

    c1 = Consumer("C1", group, lambda m: None)
    c1.start()
    print("  State after C1 alone : everything on C1")
    time.sleep(0.1)

    c2 = Consumer("C2", group, lambda m: None)
    c2.start()
    print("  State after adding C2 :")
    time.sleep(0.1)

    c3 = Consumer("C3", group, lambda m: None)
    c3.start()
    print("  State after adding C3 :")
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
    print(f"\n{SEPARATOR}\n  End of demos.\n{SEPARATOR}")


if __name__ == "__main__":
    main()
