"""
Jour 2 -- Stockage & Databases
Demonstrations interactives en Python.

Usage:
    python 02-stockage-databases.py

Chaque section est independante et peut etre executee via la fonction main().
"""

import time
import random
import hashlib
import math
import bisect
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional

# =============================================================================
# SECTION 1 : B-Tree Index vs Full Scan — Speedup demo
# =============================================================================


class SimpleBTreeNode:
    """Noeud d'un B-Tree simplifie (ordre 4).

    Un vrai B-Tree a des contraintes d'equilibrage complexes.
    On simplifie ici pour montrer le principe : recherche en O(log n)
    via une structure arborescente triee.
    """

    def __init__(self, order: int = 4):
        self.order = order          # Nombre max de cles par noeud
        self.keys: list = []        # Cles triees dans ce noeud
        self.values: list = []      # Valeurs associees (meme index que keys)
        self.children: list = []    # Enfants (len = len(keys) + 1 si noeud interne)
        self.is_leaf: bool = True

    def search(self, key) -> Optional[any]:
        """Recherche une cle dans le B-Tree. Retourne la valeur ou None.

        Complexite : O(log n) — a chaque noeud, on elimine une branche.
        Comme une recherche dichotomique, mais en arbre.
        """
        # Trouver la position ou la cle devrait etre
        i = bisect.bisect_left(self.keys, key)

        # Si la cle est dans ce noeud, la retourner
        if i < len(self.keys) and self.keys[i] == key:
            return self.values[i]

        # Si c'est une feuille, la cle n'existe pas
        if self.is_leaf:
            return None

        # Sinon, descendre dans le bon enfant
        return self.children[i].search(key)


class SimpleBTree:
    """B-Tree simplifie pour la demonstration.

    Supporte insert et search. Le split est simplifie
    (on ne fait pas un vrai split de noeud interne pour garder le code lisible).
    """

    def __init__(self, order: int = 100):
        self.root = SimpleBTreeNode(order)
        self.order = order
        self._comparisons = 0  # Compteur pour mesurer le travail

    def insert(self, key, value):
        """Insere une cle-valeur. Simplifie : on fait un insert trie dans la feuille."""
        node = self.root
        # Descendre jusqu'a la feuille appropriee
        while not node.is_leaf:
            i = bisect.bisect_left(node.keys, key)
            node = node.children[i]

        # Inserer dans la feuille (maintenir l'ordre)
        i = bisect.bisect_left(node.keys, key)
        node.keys.insert(i, key)
        node.values.insert(i, value)

        # Si le noeud deborde, le splitter (version simplifiee)
        if len(node.keys) > self.order:
            self._split_leaf(node)

    def _split_leaf(self, node: SimpleBTreeNode):
        """Split simplifie : cree un nouveau noeud racine avec deux enfants."""
        mid = len(node.keys) // 2

        # Nouveau noeud droit avec la moitie superieure
        right = SimpleBTreeNode(self.order)
        right.keys = node.keys[mid:]
        right.values = node.values[mid:]

        # Le noeud actuel garde la moitie inferieure
        left_keys = node.keys[:mid]
        left_values = node.values[:mid]

        # Si c'est la racine, creer une nouvelle racine
        if node is self.root:
            new_root = SimpleBTreeNode(self.order)
            new_root.is_leaf = False
            new_root.keys = [right.keys[0]]
            new_root.values = [right.values[0]]
            new_root.children = [SimpleBTreeNode(self.order), right]
            new_root.children[0].keys = left_keys
            new_root.children[0].values = left_values
            self.root = new_root

    def search(self, key) -> Optional[any]:
        """Recherche via le B-Tree."""
        return self.root.search(key)


def demo_btree_vs_fullscan():
    """Compare la vitesse d'un B-Tree index vs un full table scan.

    On insere N elements, puis on mesure le temps de recherche
    dans une liste non-triee (full scan) vs un B-Tree (index lookup).
    """
    print("\n" + "=" * 60)
    print("  SECTION 1 : B-Tree Index vs Full Scan")
    print("=" * 60)

    # Generer des donnees : N enregistrements avec des cles aleatoires
    N = 100_000
    print(f"\n  Dataset : {N:,} enregistrements")

    # Full scan : liste non-triee de tuples (key, value)
    data = [(random.randint(1, N * 10), f"record_{i}") for i in range(N)]

    # B-Tree : inserer les memes donnees dans l'index
    btree = SimpleBTree(order=100)
    for key, value in data:
        btree.insert(key, value)

    # Choisir des cles a rechercher (certaines existent, d'autres non)
    search_keys = [data[random.randint(0, N - 1)][0] for _ in range(1000)]

    # --- Full Scan ---
    start = time.perf_counter()
    found_scan = 0
    for key in search_keys:
        # Parcourir TOUTE la liste pour trouver la cle : O(n) par recherche
        for k, v in data:
            if k == key:
                found_scan += 1
                break
    elapsed_scan = (time.perf_counter() - start) * 1000

    # --- B-Tree Index Lookup ---
    start = time.perf_counter()
    found_btree = 0
    for key in search_keys:
        # Parcourir l'arbre : O(log n) par recherche
        result = btree.search(key)
        if result is not None:
            found_btree += 1
    elapsed_btree = (time.perf_counter() - start) * 1000

    speedup = elapsed_scan / elapsed_btree if elapsed_btree > 0 else float("inf")

    print(f"\n  {'Methode':<25} {'Temps (ms)':>12} {'Trouve':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Full Scan (O(n))':<25} {elapsed_scan:>12.1f} {found_scan:>10}")
    print(f"  {'B-Tree Index (O(log n))':<25} {elapsed_btree:>12.1f} {found_btree:>10}")
    print(f"\n  Speedup : {speedup:.1f}x")
    print(f"\n  >> Lecon : Avec {N:,} lignes, l'index est ~{speedup:.0f}x plus rapide.")
    print(f"  >> En production (10M+ lignes), le speedup est encore plus dramatique.")
    print(f"  >> Un full scan = O(n). Un index B-Tree = O(log n).")
    print(f"  >> log2({N:,}) = {math.log2(N):.0f} comparaisons vs {N:,} pour le scan.")


# =============================================================================
# SECTION 2 : Consistent Hashing — Implementation from scratch
# =============================================================================


class ConsistentHashRing:
    """Consistent hashing avec virtual nodes.

    Principe : les noeuds et les cles sont places sur un anneau
    de 0 a 2^32. Chaque cle est assignee au prochain noeud
    dans le sens horaire.

    Virtual nodes : chaque noeud physique a plusieurs positions
    sur l'anneau pour une meilleure distribution.
    """

    def __init__(self, num_virtual_nodes: int = 150):
        self.num_virtual_nodes = num_virtual_nodes
        # L'anneau : liste triee de (position, node_name)
        self._ring: list[tuple[int, str]] = []
        # Positions triees pour la recherche binaire
        self._sorted_positions: list[int] = []
        # Mapping node -> ses positions sur l'anneau
        self._node_positions: dict[str, list[int]] = {}

    def _hash(self, key: str) -> int:
        """Hash une cle sur l'espace [0, 2^32).

        On utilise MD5 pour sa distribution uniforme.
        (En prod, on utiliserait MurmurHash3 ou xxHash pour la performance.)
        """
        digest = hashlib.md5(key.encode()).hexdigest()
        return int(digest[:8], 16)  # Prendre les 8 premiers hex = 32 bits

    def add_node(self, node_name: str):
        """Ajoute un noeud physique avec ses virtual nodes sur l'anneau."""
        positions = []
        for i in range(self.num_virtual_nodes):
            # Chaque virtual node a une cle unique : "node_name#i"
            virtual_key = f"{node_name}#vn{i}"
            pos = self._hash(virtual_key)
            self._ring.append((pos, node_name))
            positions.append(pos)

        self._node_positions[node_name] = positions
        # Re-trier l'anneau apres ajout
        self._ring.sort(key=lambda x: x[0])
        self._sorted_positions = [pos for pos, _ in self._ring]

    def remove_node(self, node_name: str):
        """Supprime un noeud et tous ses virtual nodes de l'anneau."""
        self._ring = [(pos, name) for pos, name in self._ring if name != node_name]
        self._sorted_positions = [pos for pos, _ in self._ring]
        del self._node_positions[node_name]

    def get_node(self, key: str) -> str:
        """Trouve le noeud responsable d'une cle.

        Methode : hash la cle, trouver le prochain noeud dans le sens horaire
        sur l'anneau (via recherche binaire).
        """
        if not self._ring:
            raise ValueError("Anneau vide — aucun noeud disponible")

        key_hash = self._hash(key)
        # bisect_right trouve l'index du premier element > key_hash
        idx = bisect.bisect_right(self._sorted_positions, key_hash)

        # Si on depasse la fin de l'anneau, on revient au debut (c'est un anneau)
        if idx == len(self._ring):
            idx = 0

        return self._ring[idx][1]  # Retourner le nom du noeud

    def get_distribution(self, keys: list[str]) -> dict[str, int]:
        """Calcule la distribution des cles par noeud."""
        distribution = defaultdict(int)
        for key in keys:
            node = self.get_node(key)
            distribution[node] += 1
        return dict(distribution)


def demo_consistent_hashing():
    """Demontre le consistent hashing avec ajout/suppression de noeuds.

    Montre que seule une fraction des cles est redistribuee quand un noeud
    est ajoute ou supprime (contrairement au hash modulo classique).
    """
    print("\n" + "=" * 60)
    print("  SECTION 2 : Consistent Hashing")
    print("=" * 60)

    # --- Setup initial : 4 noeuds ---
    ring = ConsistentHashRing(num_virtual_nodes=150)
    initial_nodes = ["shard-A", "shard-B", "shard-C", "shard-D"]
    for node in initial_nodes:
        ring.add_node(node)

    # Generer 10 000 cles aleatoires
    keys = [f"user_{i}" for i in range(10_000)]

    # Distribution initiale
    dist_before = ring.get_distribution(keys)
    print(f"\n  Distribution initiale ({len(initial_nodes)} noeuds, {len(keys):,} cles) :")
    for node, count in sorted(dist_before.items()):
        pct = count / len(keys) * 100
        bar = "#" * int(pct / 2)
        print(f"    {node:<12} : {count:>5} ({pct:>5.1f}%) {bar}")

    # Calculer l'ecart-type pour mesurer l'equilibrage
    counts = list(dist_before.values())
    ideal = len(keys) / len(initial_nodes)
    std_dev = (sum((c - ideal) ** 2 for c in counts) / len(counts)) ** 0.5
    print(f"\n  Ideal par noeud : {ideal:,.0f}")
    print(f"  Ecart-type      : {std_dev:,.0f} ({std_dev / ideal * 100:.1f}% de l'ideal)")

    # --- Sauvegarder l'assignation avant ---
    assignment_before = {key: ring.get_node(key) for key in keys}

    # --- Ajouter un 5eme noeud ---
    print(f"\n  {'='*50}")
    print(f"  Ajout de shard-E...")
    ring.add_node("shard-E")

    dist_after_add = ring.get_distribution(keys)
    print(f"\n  Distribution apres ajout ({len(initial_nodes) + 1} noeuds) :")
    for node, count in sorted(dist_after_add.items()):
        pct = count / len(keys) * 100
        bar = "#" * int(pct / 2)
        print(f"    {node:<12} : {count:>5} ({pct:>5.1f}%) {bar}")

    # Compter les cles qui ont change de noeud
    assignment_after_add = {key: ring.get_node(key) for key in keys}
    moved = sum(1 for k in keys if assignment_before[k] != assignment_after_add[k])
    print(f"\n  Cles deplacees : {moved:,} / {len(keys):,} ({moved / len(keys) * 100:.1f}%)")
    print(f"  Ideal theorique : {1 / (len(initial_nodes) + 1) * 100:.1f}% (= 1/N)")
    print(f"\n  >> Avec hash modulo classique, ~80% des cles auraient ete deplacees.")
    print(f"  >> Consistent hashing deplace seulement ~{moved / len(keys) * 100:.0f}%.")

    # --- Supprimer un noeud ---
    print(f"\n  {'='*50}")
    print(f"  Suppression de shard-B...")
    assignment_before_remove = {key: ring.get_node(key) for key in keys}
    ring.remove_node("shard-B")

    dist_after_remove = ring.get_distribution(keys)
    print(f"\n  Distribution apres suppression ({len(initial_nodes)} noeuds) :")
    for node, count in sorted(dist_after_remove.items()):
        pct = count / len(keys) * 100
        bar = "#" * int(pct / 2)
        print(f"    {node:<12} : {count:>5} ({pct:>5.1f}%) {bar}")

    assignment_after_remove = {key: ring.get_node(key) for key in keys}
    moved_remove = sum(
        1 for k in keys
        if assignment_before_remove[k] != assignment_after_remove[k]
    )
    print(f"\n  Cles deplacees : {moved_remove:,} / {len(keys):,} ({moved_remove / len(keys) * 100:.1f}%)")

    # --- Comparaison avec hash modulo classique ---
    print(f"\n  {'='*50}")
    print(f"  COMPARAISON : Consistent Hashing vs Hash Modulo")
    print(f"  {'='*50}")

    def hash_modulo_distribution(keys_list, num_nodes):
        """Hash modulo classique : hash(key) % num_nodes."""
        dist = defaultdict(int)
        for key in keys_list:
            h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
            node_idx = h % num_nodes
            dist[f"shard-{node_idx}"] = dist.get(f"shard-{node_idx}", 0) + 1
        return dist

    # Avec 4 noeuds
    modulo_4 = hash_modulo_distribution(keys, 4)
    # Ajouter un 5eme noeud : combien de cles changent ?
    modulo_5 = hash_modulo_distribution(keys, 5)
    modulo_moved = sum(
        1 for key in keys
        if int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % 4
        != int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % 5
    )
    modulo_pct = modulo_moved / len(keys) * 100

    print(f"\n  Hash modulo : passage de 4 a 5 noeuds")
    print(f"  Cles deplacees : {modulo_moved:,} / {len(keys):,} ({modulo_pct:.1f}%)")
    print(f"\n  Consistent hashing : {moved / len(keys) * 100:.1f}% de cles deplacees")
    print(f"  Hash modulo        : {modulo_pct:.1f}% de cles deplacees")
    print(f"\n  >> Le consistent hashing deplace {modulo_pct / (moved / len(keys) * 100):.0f}x moins de cles.")


# =============================================================================
# SECTION 3 : Leader-Follower Replication avec Replication Lag
# =============================================================================


@dataclass
class ReplicationEvent:
    """Un event de replication : une ecriture a propager du leader aux followers."""
    key: str
    value: str
    timestamp: float       # Quand l'ecriture a ete faite sur le leader
    sequence_number: int   # Numero de sequence dans le WAL


class LeaderNode:
    """Simule un leader de replication qui accepte les ecritures."""

    def __init__(self, name: str):
        self.name = name
        self.data: dict[str, str] = {}  # Le stockage local
        self.wal: list[ReplicationEvent] = []  # Write-Ahead Log
        self._seq = 0

    def write(self, key: str, value: str) -> ReplicationEvent:
        """Ecriture sur le leader : ecrit dans le data store ET le WAL."""
        self._seq += 1
        self.data[key] = value
        event = ReplicationEvent(
            key=key, value=value,
            timestamp=time.time(), sequence_number=self._seq
        )
        self.wal.append(event)
        return event

    def read(self, key: str) -> Optional[str]:
        """Lecture directe sur le leader : toujours a jour."""
        return self.data.get(key)


class FollowerNode:
    """Simule un follower qui replique les ecritures du leader avec un delai."""

    def __init__(self, name: str, lag_ms: float):
        self.name = name
        self.lag_ms = lag_ms        # Delai de replication simule (en ms)
        self.data: dict[str, str] = {}
        self.last_applied_seq: int = 0  # Dernier event applique
        self._pending: list[ReplicationEvent] = []  # Events en attente

    def receive_event(self, event: ReplicationEvent):
        """Recoit un event de replication du leader.

        Le follower ne l'applique pas immediatement :
        il simule un delai de replication (reseau + application).
        """
        self._pending.append(event)

    def apply_pending(self):
        """Applique les events en attente qui ont depasse le delai de replication."""
        now = time.time()
        applied = 0
        remaining = []

        for event in self._pending:
            # L'event est applique seulement si assez de temps s'est ecoule
            age_ms = (now - event.timestamp) * 1000
            if age_ms >= self.lag_ms:
                self.data[event.key] = event.value
                self.last_applied_seq = event.sequence_number
                applied += 1
            else:
                remaining.append(event)

        self._pending = remaining
        return applied

    def read(self, key: str) -> Optional[str]:
        """Lecture sur le follower : peut retourner une valeur obsolete (stale read)."""
        self.apply_pending()  # Tenter d'appliquer les events en attente
        return self.data.get(key)


def demo_replication_lag():
    """Simule un systeme leader-follower et montre l'impact du replication lag.

    Scenario : Un leader recoit des ecritures. Deux followers repliquent
    avec des delais differents. On montre les stale reads.
    """
    print("\n" + "=" * 60)
    print("  SECTION 3 : Leader-Follower Replication")
    print("=" * 60)

    # Setup : 1 leader + 2 followers avec des lags differents
    leader = LeaderNode("leader")
    follower_fast = FollowerNode("follower-fast", lag_ms=50)   # 50ms de lag
    follower_slow = FollowerNode("follower-slow", lag_ms=500)  # 500ms de lag

    followers = [follower_fast, follower_slow]

    print(f"\n  Setup : 1 leader + 2 followers")
    print(f"    follower-fast : lag = {follower_fast.lag_ms}ms")
    print(f"    follower-slow : lag = {follower_slow.lag_ms}ms")

    # Phase 1 : Ecritures sur le leader
    print(f"\n  Phase 1 : 10 ecritures sur le leader...")
    events = []
    for i in range(10):
        event = leader.write(f"key_{i}", f"value_{i}_v1")
        events.append(event)
        # Propager a tous les followers
        for f in followers:
            f.receive_event(event)
        time.sleep(0.01)  # 10ms entre chaque ecriture

    # Phase 2 : Lectures immediates (avant que le lag soit ecoule)
    print(f"\n  Phase 2 : Lectures immediates apres les ecritures")
    print(f"\n  {'Cle':<10} {'Leader':>15} {'Fast Follower':>15} {'Slow Follower':>15}")
    print(f"  {'-'*58}")

    stale_fast = 0
    stale_slow = 0

    for i in range(10):
        key = f"key_{i}"
        leader_val = leader.read(key)
        fast_val = follower_fast.read(key) or "(stale/missing)"
        slow_val = follower_slow.read(key) or "(stale/missing)"

        if fast_val != leader_val:
            stale_fast += 1
        if slow_val != leader_val:
            stale_slow += 1

        print(f"  {key:<10} {str(leader_val):>15} {str(fast_val):>15} {str(slow_val):>15}")

    print(f"\n  Stale reads : fast={stale_fast}/10, slow={stale_slow}/10")

    # Phase 3 : Attendre que le lag passe, puis relire
    print(f"\n  Phase 3 : Attente de 600ms pour que tous les followers rattrapent...")
    time.sleep(0.6)

    print(f"\n  {'Cle':<10} {'Leader':>15} {'Fast Follower':>15} {'Slow Follower':>15}")
    print(f"  {'-'*58}")

    for i in range(10):
        key = f"key_{i}"
        leader_val = leader.read(key)
        fast_val = follower_fast.read(key) or "(missing)"
        slow_val = follower_slow.read(key) or "(missing)"
        print(f"  {key:<10} {str(leader_val):>15} {str(fast_val):>15} {str(slow_val):>15}")

    print(f"\n  >> Apres le lag, tous les followers sont a jour.")

    # Phase 4 : Read-your-writes pattern
    print(f"\n  Phase 4 : Probleme du 'read-your-writes'")
    print(f"  {'-'*50}")

    # L'utilisateur ecrit une mise a jour
    event = leader.write("profile_name", "Anthony_Updated")
    for f in followers:
        f.receive_event(event)

    # Lecture immediate depuis un follower (simule un load balancer)
    fast_read = follower_fast.read("profile_name") or "(pas encore replique)"
    print(f"\n  Ecriture : profile_name = 'Anthony_Updated' (sur leader)")
    print(f"  Lecture immediate depuis follower-fast : '{fast_read}'")
    print(f"  L'utilisateur voit son ancien profil alors qu'il vient de le modifier !")
    print(f"\n  Solution : 'read-your-writes consistency'")
    print(f"  -> Apres une ecriture, forcer les reads de CET utilisateur vers le leader")
    print(f"     pendant {follower_slow.lag_ms}ms (le temps du pire lag).")


# =============================================================================
# SECTION 4 : DB Selection Advisor
# =============================================================================


@dataclass
class DBRequirements:
    """Les requirements d'un systeme pour le choix de DB."""
    name: str
    needs_acid: bool = False               # Transactions ACID requises
    needs_joins: bool = False              # Jointures complexes
    schema_flexible: bool = False          # Schema variable
    write_heavy: bool = False              # > 80% writes
    read_heavy: bool = False               # > 80% reads
    needs_full_text_search: bool = False   # Recherche full-text
    needs_graph_traversal: bool = False    # Relations profondes
    needs_sub_ms_latency: bool = False     # Latence < 1ms
    data_is_time_series: bool = False      # Donnees temporelles
    expected_qps: int = 0                  # QPS attendu
    data_size_gb: int = 0                  # Volume de donnees en Go
    ttl_required: bool = False             # Expiration automatique


@dataclass
class DBRecommendation:
    """Resultat de l'advisor : une DB recommandee avec justification."""
    db_name: str
    category: str
    justification: str
    warning: str = ""


def recommend_db(req: DBRequirements) -> list[DBRecommendation]:
    """Recommande une ou plusieurs bases de donnees selon les requirements.

    Decision tree base sur les contraintes critiques :
    1. ACID + joins -> SQL
    2. Sub-ms latency -> Redis
    3. Graph traversal -> Neo4j
    4. Write-heavy + time-series -> Cassandra/TimescaleDB
    5. Full-text search -> Elasticsearch (+ DB primaire)
    6. Schema flexible -> MongoDB
    7. Defaut -> PostgreSQL
    """
    recommendations = []

    # Regle 1 : Besoin de transactions ACID et jointures -> SQL
    if req.needs_acid and req.needs_joins:
        rec = DBRecommendation(
            db_name="PostgreSQL",
            category="OLTP SQL",
            justification=(
                f"Transactions ACID + jointures complexes requises. "
                f"PostgreSQL supporte jusqu'a ~50K QPS par noeud. "
                f"Pour {req.expected_qps:,} QPS, "
                f"{'1 noeud suffit' if req.expected_qps <= 50_000 else 'il faudra du sharding (Citus)'}."
            ),
        )
        if req.data_size_gb > 5_000:
            rec.warning = (
                f"Attention : {req.data_size_gb:,} Go est large pour un seul PostgreSQL. "
                f"Envisager Citus (PostgreSQL distribue) ou le partitionnement natif."
            )
        recommendations.append(rec)

    # Regle 2 : Latence sub-ms -> Redis
    if req.needs_sub_ms_latency:
        recommendations.append(DBRecommendation(
            db_name="Redis",
            category="Key-Value (in-memory)",
            justification=(
                f"Latence < 1ms requise. Redis offre ~100K-1M ops/s par noeud. "
                f"Ideal pour {req.name} si les donnees tiennent en RAM "
                f"({req.data_size_gb} Go {'< 100 Go : OK' if req.data_size_gb < 100 else '> 100 Go : Redis Cluster necessaire'})."
            ),
            warning="Redis perd les donnees au crash sans persistence (RDB/AOF). Pas une DB primaire." if not req.ttl_required else "",
        ))

    # Regle 3 : Graph traversal -> Neo4j
    if req.needs_graph_traversal:
        recommendations.append(DBRecommendation(
            db_name="Neo4j",
            category="Graph",
            justification=(
                "Traversals de relations profondes (amis d'amis, recommandations, fraude). "
                "En SQL, une jointure recursive sur 4+ niveaux est impraticable. "
                "Neo4j traverse en ms ce qui prendrait des secondes en SQL."
            ),
        ))

    # Regle 4 : Write-heavy + time-series -> Cassandra ou TimescaleDB
    if req.write_heavy and req.data_is_time_series:
        recommendations.append(DBRecommendation(
            db_name="Cassandra (ou TimescaleDB si besoin SQL)",
            category="Column-family / Time-series",
            justification=(
                f"Write-heavy ({req.expected_qps:,} QPS) + time-series. "
                "Cassandra : LSM-Tree optimise pour les ecritures massives, "
                "scaling lineaire. Partition par sensor_id + clustering par timestamp. "
                "TimescaleDB si tu veux garder SQL (extension PostgreSQL)."
            ),
        ))
    elif req.write_heavy:
        recommendations.append(DBRecommendation(
            db_name="Cassandra / ScyllaDB",
            category="Column-family",
            justification=(
                f"Write-heavy ({req.expected_qps:,} QPS). "
                "Cassandra/ScyllaDB sont optimises pour les ecritures massives "
                "grace au LSM-Tree (append-only, pas de random writes)."
            ),
        ))

    # Regle 5 : Full-text search -> Elasticsearch
    if req.needs_full_text_search:
        recommendations.append(DBRecommendation(
            db_name="Elasticsearch (complement, pas DB primaire)",
            category="Search engine",
            justification=(
                "Recherche full-text avec scoring, fuzzy matching, facettes. "
                "Elasticsearch utilise un index inverse optimise pour ca. "
                "A coupler avec une DB primaire (PostgreSQL ou MongoDB)."
            ),
            warning="Elasticsearch n'est PAS une DB primaire. Pas de transactions, perte possible en crash.",
        ))

    # Regle 6 : Schema flexible -> MongoDB
    if req.schema_flexible and not req.needs_acid:
        recommendations.append(DBRecommendation(
            db_name="MongoDB",
            category="Document store",
            justification=(
                f"Schema variable pour {req.name}. Chaque document peut avoir "
                "des attributs differents. Pas de migration DDL. "
                "Iteration rapide. Bon pour les catalogues, CMS, profils enrichis."
            ),
        ))

    # Defaut : PostgreSQL si aucune recommandation forte
    if not recommendations:
        recommendations.append(DBRecommendation(
            db_name="PostgreSQL",
            category="OLTP SQL (defaut safe)",
            justification=(
                "Aucune contrainte forte identifiee. PostgreSQL est le choix par defaut "
                "le plus polyvalent : ACID, JSON(B), full-text search basique, "
                "extensions (PostGIS, TimescaleDB, pgvector). Tu peux specialiser plus tard."
            ),
        ))

    return recommendations


def demo_db_advisor():
    """Execute l'advisor sur plusieurs scenarios et affiche les recommandations."""
    print("\n" + "=" * 60)
    print("  SECTION 4 : DB Selection Advisor")
    print("=" * 60)

    scenarios = [
        DBRequirements(
            name="Systeme de commandes e-commerce",
            needs_acid=True, needs_joins=True,
            expected_qps=5_000, data_size_gb=200,
        ),
        DBRequirements(
            name="Cache de sessions utilisateur",
            needs_sub_ms_latency=True, ttl_required=True,
            expected_qps=100_000, data_size_gb=10,
        ),
        DBRequirements(
            name="Plateforme IoT (500K capteurs)",
            write_heavy=True, data_is_time_series=True,
            expected_qps=500_000, data_size_gb=50_000,
        ),
        DBRequirements(
            name="Reseau social — detection de fraude",
            needs_graph_traversal=True,
            expected_qps=10_000, data_size_gb=500,
        ),
        DBRequirements(
            name="Catalogue produit marketplace",
            schema_flexible=True, needs_full_text_search=True,
            read_heavy=True,
            expected_qps=50_000, data_size_gb=100,
        ),
        DBRequirements(
            name="App de notes simple (10K users)",
            expected_qps=100, data_size_gb=1,
        ),
    ]

    for req in scenarios:
        recs = recommend_db(req)
        print(f"\n  {'='*56}")
        print(f"  Scenario : {req.name}")
        print(f"  QPS: {req.expected_qps:,} | Data: {req.data_size_gb:,} Go")
        print(f"  {'='*56}")

        for i, rec in enumerate(recs, 1):
            print(f"\n  Recommandation {i} : {rec.db_name} ({rec.category})")
            print(f"  Justification : {rec.justification}")
            if rec.warning:
                print(f"  /!\\ Warning : {rec.warning}")


# =============================================================================
# SECTION 5 : Sharding Simulator — Distribution et Hot Spots
# =============================================================================


def demo_sharding_simulator():
    """Simule la distribution de cles sur des shards et montre les hot spots.

    Compare le range sharding (prone aux hot spots) et le hash sharding
    (distribution plus uniforme).
    """
    print("\n" + "=" * 60)
    print("  SECTION 5 : Sharding Simulator — Hot Spots")
    print("=" * 60)

    NUM_SHARDS = 4
    NUM_KEYS = 10_000

    # --- Scenario 1 : Range Sharding ---
    # Les users recents (IDs hauts) sont les plus actifs
    print(f"\n  Scenario 1 : Range Sharding")
    print(f"  {'-'*50}")

    # Simuler un trafic realiste : les users recents sont plus actifs
    # Distribution Zipf-like : user_id haut = plus de requetes
    requests_range = []
    for _ in range(NUM_KEYS):
        # 80% du trafic vient des 20% de users les plus recents (IDs hauts)
        if random.random() < 0.8:
            user_id = random.randint(7500, 10000)  # Users recents
        else:
            user_id = random.randint(1, 7500)       # Users anciens

        # Range sharding : chaque shard couvre une plage de IDs
        shard_size = 10000 // NUM_SHARDS
        shard = min(user_id // shard_size, NUM_SHARDS - 1)
        requests_range.append((user_id, f"shard-{shard}"))

    # Compter les requetes par shard
    range_dist = defaultdict(int)
    for _, shard in requests_range:
        range_dist[shard] += 1

    print(f"\n  Distribution du trafic (range sharding, {NUM_KEYS:,} requetes) :")
    max_load = max(range_dist.values())
    for shard in sorted(range_dist.keys()):
        count = range_dist[shard]
        pct = count / NUM_KEYS * 100
        bar = "#" * int(pct)
        hot = " << HOT SPOT!" if count > NUM_KEYS / NUM_SHARDS * 1.5 else ""
        print(f"    {shard:<10} : {count:>5} ({pct:>5.1f}%) {bar}{hot}")

    # --- Scenario 2 : Hash Sharding ---
    print(f"\n  Scenario 2 : Hash Sharding")
    print(f"  {'-'*50}")

    hash_dist = defaultdict(int)
    for user_id, _ in requests_range:
        # Hash du user_id pour distribuer uniformement
        h = int(hashlib.md5(str(user_id).encode()).hexdigest()[:8], 16)
        shard = f"shard-{h % NUM_SHARDS}"
        hash_dist[shard] += 1

    print(f"\n  Distribution du trafic (hash sharding, memes requetes) :")
    for shard in sorted(hash_dist.keys()):
        count = hash_dist[shard]
        pct = count / NUM_KEYS * 100
        bar = "#" * int(pct)
        print(f"    {shard:<10} : {count:>5} ({pct:>5.1f}%) {bar}")

    # --- Comparaison ---
    print(f"\n  Comparaison :")
    range_std = (sum((c - NUM_KEYS / NUM_SHARDS) ** 2 for c in range_dist.values()) / NUM_SHARDS) ** 0.5
    hash_std = (sum((c - NUM_KEYS / NUM_SHARDS) ** 2 for c in hash_dist.values()) / NUM_SHARDS) ** 0.5
    print(f"    Range sharding — ecart-type : {range_std:,.0f} ({range_std / (NUM_KEYS / NUM_SHARDS) * 100:.1f}%)")
    print(f"    Hash sharding  — ecart-type : {hash_std:,.0f} ({hash_std / (NUM_KEYS / NUM_SHARDS) * 100:.1f}%)")

    # --- Scenario 3 : Celebrity problem (hot key) ---
    print(f"\n  Scenario 3 : Celebrity Problem (hot key)")
    print(f"  {'-'*50}")

    celebrity_dist = defaultdict(int)
    # 10 000 requetes normales + 5 000 requetes pour UN seul user (viral)
    for _ in range(NUM_KEYS):
        user_id = random.randint(1, 10000)
        h = int(hashlib.md5(str(user_id).encode()).hexdigest()[:8], 16)
        shard = f"shard-{h % NUM_SHARDS}"
        celebrity_dist[shard] += 1

    # Le celebrity : user_id = 42 recoit 50% du trafic total
    celebrity_hash = int(hashlib.md5(b"42").hexdigest()[:8], 16)
    celebrity_shard = f"shard-{celebrity_hash % NUM_SHARDS}"
    celebrity_dist[celebrity_shard] += NUM_KEYS  # Doubler le trafic sur cette shard

    total_with_celeb = sum(celebrity_dist.values())
    print(f"\n  Distribution avec un 'celebrity user' (user_id=42, +{NUM_KEYS:,} requetes) :")
    for shard in sorted(celebrity_dist.keys()):
        count = celebrity_dist[shard]
        pct = count / total_with_celeb * 100
        bar = "#" * int(pct)
        hot = " << CELEBRITY SHARD!" if shard == celebrity_shard else ""
        print(f"    {shard:<10} : {count:>5} ({pct:>5.1f}%) {bar}{hot}")

    print(f"\n  >> Solution 'key splitting' pour le celebrity :")
    print(f"     Au lieu de hash('42'), utiliser hash('42_0'), hash('42_1'), ..., hash('42_9')")
    print(f"     Les reads sont distribues sur 10 cles au lieu d'une seule.")

    # Simuler le key splitting
    split_dist = defaultdict(int)
    # Re-distribuer les requetes du celebrity sur 10 cles
    for shard in sorted(celebrity_dist.keys()):
        if shard != celebrity_shard:
            split_dist[shard] = celebrity_dist[shard]
        else:
            # La shard du celebrity garde ses requetes normales
            split_dist[shard] = celebrity_dist[shard] - NUM_KEYS

    # Les requetes du celebrity sont distribuees sur 10 split keys
    for i in range(NUM_KEYS):
        split_key = f"42_{i % 10}"
        h = int(hashlib.md5(split_key.encode()).hexdigest()[:8], 16)
        shard = f"shard-{h % NUM_SHARDS}"
        split_dist[shard] += 1

    total_split = sum(split_dist.values())
    print(f"\n  Distribution apres key splitting :")
    for shard in sorted(split_dist.keys()):
        count = split_dist[shard]
        pct = count / total_split * 100
        bar = "#" * int(pct)
        print(f"    {shard:<10} : {count:>5} ({pct:>5.1f}%) {bar}")

    print(f"\n  >> Le hot spot est elimine. Le trafic du celebrity est reparti uniformement.")


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Execute toutes les demos sequentiellement."""
    print("\n" + "#" * 60)
    print("#  JOUR 2 -- STOCKAGE & DATABASES")
    print("#" * 60)

    demo_btree_vs_fullscan()
    demo_consistent_hashing()
    demo_replication_lag()
    demo_db_advisor()
    demo_sharding_simulator()

    print("\n" + "#" * 60)
    print("#  FIN -- Toutes les demos ont ete executees.")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
