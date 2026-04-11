"""
Jour 3 -- Caching & CDN
Demonstrations interactives en Python.

Usage:
    python 03-caching-cdn.py

Chaque section est independante et peut etre executee via la fonction main().
"""

import time
import random
import math
import threading
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Optional, Any

SEPARATOR = "=" * 70


# =============================================================================
# SECTION 1 : LRU Cache — Implementation avec OrderedDict
# =============================================================================


class LRUCacheOrderedDict:
    """LRU Cache utilisant OrderedDict de Python.

    OrderedDict maintient l'ordre d'insertion. On utilise move_to_end()
    pour deplacer un element accede vers la fin (= le plus recent).
    L'element au debut (= le moins recent) est evince quand le cache est plein.

    Complexite : O(1) pour get et put grace au dict + doubly-linked list interne.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()  # Ordre = anciennete (debut = LRU, fin = MRU)
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Recupere une valeur. Retourne None si absente."""
        if key in self.cache:
            # Deplacer vers la fin = marquer comme "recemment utilise"
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key: str, value: Any) -> Optional[str]:
        """Insere ou met a jour. Retourne la cle evincee si eviction."""
        evicted = None
        if key in self.cache:
            # Mise a jour : deplacer vers la fin
            self.cache.move_to_end(key)
            self.cache[key] = value
        else:
            if len(self.cache) >= self.capacity:
                # Evincer le plus ancien (debut de l'OrderedDict)
                evicted, _ = self.cache.popitem(last=False)  # last=False = FIFO order = LRU
            self.cache[key] = value
        return evicted

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def __repr__(self) -> str:
        return f"LRUCache(size={len(self.cache)}/{self.capacity}, hit_rate={self.hit_rate:.1%})"


# =============================================================================
# SECTION 2 : LRU Cache — Implementation from scratch (Doubly-Linked List + Dict)
# =============================================================================


class DLLNode:
    """Noeud d'une doubly-linked list.

    Chaque noeud stocke une paire cle-valeur et des pointeurs prev/next.
    La cle est stockee dans le noeud pour pouvoir retrouver la cle
    lors de l'eviction (on evince le noeud, mais il faut aussi supprimer du dict).
    """

    __slots__ = ("key", "value", "prev", "next")  # Economie memoire

    def __init__(self, key: str = "", value: Any = None):
        self.key = key
        self.value = value
        self.prev: Optional["DLLNode"] = None
        self.next: Optional["DLLNode"] = None


class LRUCacheFromScratch:
    """LRU Cache implemente sans OrderedDict.

    Structure interne :
    - Un dict (key -> DLLNode) pour l'acces O(1) par cle
    - Une doubly-linked list pour l'ordre d'acces O(1)
      - head.next = noeud le moins recent (LRU)
      - tail.prev = noeud le plus recent (MRU)
    - head et tail sont des noeuds sentinelles (dummy) pour simplifier les operations

    Pourquoi c'est demande en entretien :
    Cela teste la maitrise des pointeurs, des structures de donnees,
    et la capacite a combiner deux structures pour obtenir O(1) partout.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: dict[str, DLLNode] = {}
        self.hits = 0
        self.misses = 0

        # Noeuds sentinelles : evitent les cas limites (liste vide, un seul element)
        self.head = DLLNode()  # Dummy head — le noeud apres head est le LRU
        self.tail = DLLNode()  # Dummy tail — le noeud avant tail est le MRU
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: DLLNode) -> None:
        """Retire un noeud de la liste (O(1)).

        Avant : ... <-> prev <-> node <-> next <-> ...
        Apres : ... <-> prev <-> next <-> ...
        """
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def _add_to_tail(self, node: DLLNode) -> None:
        """Ajoute un noeud juste avant le tail (= position MRU) (O(1)).

        Avant : ... <-> last_real <-> tail
        Apres : ... <-> last_real <-> node <-> tail
        """
        prev_node = self.tail.prev  # L'ancien dernier noeud reel
        prev_node.next = node
        node.prev = prev_node
        node.next = self.tail
        self.tail.prev = node

    def get(self, key: str) -> Optional[Any]:
        """Recupere une valeur et deplace le noeud en position MRU."""
        if key in self.cache:
            node = self.cache[key]
            # Deplacer vers la position MRU (retirer puis re-ajouter en fin)
            self._remove(node)
            self._add_to_tail(node)
            self.hits += 1
            return node.value
        self.misses += 1
        return None

    def put(self, key: str, value: Any) -> Optional[str]:
        """Insere ou met a jour. Retourne la cle evincee si eviction."""
        evicted = None
        if key in self.cache:
            # Mise a jour : retirer et re-ajouter en fin
            node = self.cache[key]
            node.value = value
            self._remove(node)
            self._add_to_tail(node)
        else:
            if len(self.cache) >= self.capacity:
                # Evincer le LRU (le noeud juste apres head)
                lru_node = self.head.next
                self._remove(lru_node)
                del self.cache[lru_node.key]
                evicted = lru_node.key

            # Creer un nouveau noeud et l'ajouter en fin
            new_node = DLLNode(key, value)
            self.cache[key] = new_node
            self._add_to_tail(new_node)
        return evicted

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# =============================================================================
# SECTION 3 : LFU Cache — Implementation from scratch
# =============================================================================


class LFUCache:
    """Least Frequently Used Cache.

    Structure interne :
    - key_to_val: dict[key -> value]          — acces O(1) a la valeur
    - key_to_freq: dict[key -> frequency]     — compteur de frequence par cle
    - freq_to_keys: dict[freq -> OrderedDict] — ensemble ordonne de cles par frequence
    - min_freq: int                           — la frequence minimale actuelle

    L'OrderedDict par frequence permet de trouver le LRU parmi les cles
    de meme frequence (en cas d'egalite, on evince le moins recemment utilise).

    Complexite : O(1) pour get et put.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.key_to_val: dict[str, Any] = {}
        self.key_to_freq: dict[str, int] = {}
        self.freq_to_keys: dict[int, OrderedDict] = defaultdict(OrderedDict)
        self.min_freq = 0  # Frequence la plus basse actuellement dans le cache
        self.hits = 0
        self.misses = 0

    def _update_freq(self, key: str) -> None:
        """Incremente la frequence d'une cle et maintient les structures."""
        freq = self.key_to_freq[key]
        new_freq = freq + 1

        # Retirer de l'ancien bucket de frequence
        del self.freq_to_keys[freq][key]

        # Si l'ancien bucket est vide et c'etait la freq minimale, incrementer min_freq
        if not self.freq_to_keys[freq]:
            del self.freq_to_keys[freq]
            if self.min_freq == freq:
                self.min_freq = new_freq

        # Ajouter au nouveau bucket de frequence
        self.freq_to_keys[new_freq][key] = None  # OrderedDict comme ordered set
        self.key_to_freq[key] = new_freq

    def get(self, key: str) -> Optional[Any]:
        """Recupere une valeur et incremente sa frequence."""
        if key not in self.key_to_val:
            self.misses += 1
            return None
        self.hits += 1
        self._update_freq(key)
        return self.key_to_val[key]

    def put(self, key: str, value: Any) -> Optional[str]:
        """Insere ou met a jour. Retourne la cle evincee si eviction."""
        if self.capacity <= 0:
            return None

        evicted = None

        if key in self.key_to_val:
            # Mise a jour d'une cle existante
            self.key_to_val[key] = value
            self._update_freq(key)
        else:
            if len(self.key_to_val) >= self.capacity:
                # Evincer la cle avec la plus basse frequence (et la plus ancienne si egalite)
                # popitem(last=False) donne la plus ancienne dans l'OrderedDict
                evicted_key, _ = self.freq_to_keys[self.min_freq].popitem(last=False)
                if not self.freq_to_keys[self.min_freq]:
                    del self.freq_to_keys[self.min_freq]
                del self.key_to_val[evicted_key]
                del self.key_to_freq[evicted_key]
                evicted = evicted_key

            # Nouvelle cle : frequence = 1, min_freq = 1
            self.key_to_val[key] = value
            self.key_to_freq[key] = 1
            self.freq_to_keys[1][key] = None
            self.min_freq = 1

        return evicted

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# =============================================================================
# SECTION 4 : Cache-Aside Pattern — Simulation avec metriques
# =============================================================================


class SimulatedDB:
    """Simule une base de donnees avec latence configurable."""

    def __init__(self, latency_ms: float = 10.0):
        self.latency_ms = latency_ms
        self.data: dict[str, Any] = {}
        self.query_count = 0

    def get(self, key: str) -> Optional[Any]:
        """Simule un SELECT avec la latence."""
        time.sleep(self.latency_ms / 1000)  # Convertir ms en secondes
        self.query_count += 1
        return self.data.get(key)

    def set(self, key: str, value: Any) -> None:
        """Simule un INSERT/UPDATE."""
        time.sleep(self.latency_ms / 1000)
        self.data[key] = value


class SimulatedRedis:
    """Simule un cache Redis avec TTL et latence."""

    def __init__(self, latency_ms: float = 0.5):
        self.latency_ms = latency_ms
        self.data: dict[str, Any] = {}
        self.expiry: dict[str, float] = {}  # key -> timestamp d'expiration
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """GET avec verification du TTL."""
        time.sleep(self.latency_ms / 1000)

        if key in self.data:
            # Verifier si la cle a expire
            if key in self.expiry and time.time() > self.expiry[key]:
                # Cle expiree -> supprimer et retourner None
                del self.data[key]
                del self.expiry[key]
                self.misses += 1
                return None
            self.hits += 1
            return self.data[key]
        self.misses += 1
        return None

    def setex(self, key: str, ttl_seconds: int, value: Any) -> None:
        """SET avec TTL en secondes."""
        time.sleep(self.latency_ms / 1000)
        self.data[key] = value
        self.expiry[key] = time.time() + ttl_seconds

    def delete(self, key: str) -> None:
        """DELETE une cle."""
        time.sleep(self.latency_ms / 1000)
        self.data.pop(key, None)
        self.expiry.pop(key, None)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class CacheAsideService:
    """Service avec pattern Cache-Aside.

    Flow READ :
    1. Verifier le cache
    2. Si HIT -> retourner
    3. Si MISS -> lire la DB, ecrire dans le cache, retourner

    Flow WRITE :
    1. Ecrire dans la DB
    2. Invalider le cache (DELETE, pas SET pour eviter les race conditions)
    """

    def __init__(self, db: SimulatedDB, cache: SimulatedRedis, ttl: int = 300):
        self.db = db
        self.cache = cache
        self.ttl = ttl

    def get(self, key: str) -> Optional[Any]:
        """Lecture cache-aside."""
        # 1. Verifier le cache
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        # 2. Cache miss -> lire la DB
        value = self.db.get(key)
        if value is not None:
            # 3. Ecrire dans le cache
            self.cache.setex(key, self.ttl, value)
        return value

    def update(self, key: str, value: Any) -> None:
        """Ecriture cache-aside : DB first, puis invalidation cache."""
        # 1. Ecrire dans la DB
        self.db.set(key, value)
        # 2. Invalider le cache (DELETE, pas SET !)
        self.cache.delete(key)


# =============================================================================
# SECTION 5 : Write-Through vs Write-Behind — Comparaison
# =============================================================================


class WriteThroughService:
    """Write-Through : chaque write passe par le cache ET la DB de maniere synchrone.

    Le cache est toujours a jour, mais les writes sont plus lents
    car on attend la confirmation des deux (cache + DB).
    """

    def __init__(self, db: SimulatedDB, cache: SimulatedRedis, ttl: int = 300):
        self.db = db
        self.cache = cache
        self.ttl = ttl
        self.write_count = 0

    def write(self, key: str, value: Any) -> float:
        """Ecriture synchrone dans cache + DB. Retourne le temps total."""
        start = time.time()
        # Ecriture synchrone : cache d'abord, puis DB
        self.cache.setex(key, self.ttl, value)
        self.db.set(key, value)
        self.write_count += 1
        return (time.time() - start) * 1000  # ms

    def read(self, key: str) -> Optional[Any]:
        """Lecture directe depuis le cache (toujours a jour)."""
        return self.cache.get(key)


class WriteBehindService:
    """Write-Behind : les writes vont dans le cache, puis flush en batch vers la DB.

    Les writes sont tres rapides (seulement le cache), mais il y a un risque
    de perte de donnees si le cache crash avant le flush.
    """

    def __init__(self, db: SimulatedDB, cache: SimulatedRedis,
                 ttl: int = 300, flush_interval: float = 1.0):
        self.db = db
        self.cache = cache
        self.ttl = ttl
        self.flush_interval = flush_interval  # Secondes entre chaque flush
        self.write_buffer: list[tuple[str, Any]] = []  # Buffer des writes en attente
        self.write_count = 0
        self.flushed_count = 0

    def write(self, key: str, value: Any) -> float:
        """Ecriture dans le cache seulement. Le buffer sera flush plus tard."""
        start = time.time()
        self.cache.setex(key, self.ttl, value)
        self.write_buffer.append((key, value))  # Ajouter au buffer
        self.write_count += 1
        return (time.time() - start) * 1000  # ms

    def flush(self) -> int:
        """Flush le buffer vers la DB. Retourne le nombre d'ecritures."""
        count = 0
        for key, value in self.write_buffer:
            self.db.set(key, value)
            count += 1
        self.flushed_count += count
        self.write_buffer.clear()
        return count

    def read(self, key: str) -> Optional[Any]:
        """Lecture depuis le cache (qui est en avance sur la DB)."""
        return self.cache.get(key)


# =============================================================================
# SECTION 6 : Cache Stampede — Simulation et Fix (Locking)
# =============================================================================


class StampedeDemo:
    """Demonstration du cache stampede et de la solution par locking.

    Sans lock : N threads font tous un cache miss simultane et requetent la DB.
    Avec lock : un seul thread reconstruit le cache, les autres attendent.
    """

    def __init__(self):
        self.db_query_count = 0
        self.lock = threading.Lock()
        self.cache: dict[str, Any] = {}
        self.db_latency_ms = 50  # La DB est lente

    def _db_query(self, key: str) -> str:
        """Simule une requete DB couteuse."""
        time.sleep(self.db_latency_ms / 1000)
        self.db_query_count += 1
        return f"value_for_{key}"

    def get_without_lock(self, key: str) -> str:
        """GET sans protection contre le stampede.

        Tous les threads qui arrivent pendant un cache miss
        vont requeter la DB en parallele -> surcharge.
        """
        if key in self.cache:
            return self.cache[key]

        # Cache miss -> requeter la DB (TOUS les threads font ca)
        value = self._db_query(key)
        self.cache[key] = value
        return value

    def get_with_lock(self, key: str) -> str:
        """GET avec mutex pour eviter le stampede.

        Un seul thread reconstruit le cache.
        Les autres attendent que le lock soit relache puis lisent le cache.
        """
        if key in self.cache:
            return self.cache[key]

        # Tenter d'acquerir le lock
        with self.lock:
            # Double-check : un autre thread a peut-etre deja reconstruit
            if key in self.cache:
                return self.cache[key]

            # Ce thread (et seulement celui-ci) reconstruit le cache
            value = self._db_query(key)
            self.cache[key] = value
            return value

    def reset(self):
        """Reset les compteurs et le cache pour un nouveau test."""
        self.db_query_count = 0
        self.cache.clear()


# =============================================================================
# SECTION 7 : TTL-based Cache avec expiration automatique
# =============================================================================


class TTLCache:
    """Cache avec TTL (Time-To-Live) par entree.

    Chaque entree a un timestamp d'expiration. Les entrees expirees
    sont supprimees paresseusement (lazy deletion) lors des acces,
    et periodiquement (active deletion) via cleanup().

    Lazy deletion = on verifie a chaque GET si l'entree est expiree.
    Active deletion = on scanne periodiquement pour supprimer les expirees.
    Redis utilise les deux strategies combinees.
    """

    def __init__(self, default_ttl: int = 60):
        self.default_ttl = default_ttl
        self.data: dict[str, Any] = {}
        self.expiry: dict[str, float] = {}
        self.hits = 0
        self.misses = 0
        self.expired_count = 0

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set avec TTL optionnel (defaut = default_ttl)."""
        self.data[key] = value
        effective_ttl = ttl if ttl is not None else self.default_ttl
        self.expiry[key] = time.time() + effective_ttl

    def get(self, key: str) -> Optional[Any]:
        """Get avec lazy deletion des entrees expirees."""
        if key in self.data:
            if time.time() > self.expiry.get(key, float("inf")):
                # Lazy deletion : supprimer l'entree expiree
                del self.data[key]
                del self.expiry[key]
                self.expired_count += 1
                self.misses += 1
                return None
            self.hits += 1
            return self.data[key]
        self.misses += 1
        return None

    def cleanup(self) -> int:
        """Active deletion : scanner et supprimer les entrees expirees.

        Redis fait ca periodiquement (10 fois/sec) en echantillonnant
        20 cles aleatoires parmi celles avec un TTL.

        Retourne le nombre d'entrees supprimees.
        """
        now = time.time()
        expired_keys = [k for k, exp in self.expiry.items() if now > exp]
        for key in expired_keys:
            del self.data[key]
            del self.expiry[key]
            self.expired_count += 1
        return len(expired_keys)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def __len__(self) -> int:
        return len(self.data)


# =============================================================================
# SECTION 8 : CDN Simulator — Edge -> Regional -> Origin
# =============================================================================


@dataclass
class CDNNode:
    """Un noeud du CDN (edge, regional, ou origin).

    Chaque noeud a son propre cache, sa latence d'acces,
    et un noeud parent (upstream) pour les cache miss.
    """
    name: str
    latency_ms: float           # Latence pour atteindre ce noeud
    cache: dict = field(default_factory=dict)
    cache_ttl: dict = field(default_factory=dict)
    upstream: Optional["CDNNode"] = None  # Noeud parent (pour les miss)
    hits: int = 0
    misses: int = 0

    def get(self, key: str) -> tuple[Any, float, list[str]]:
        """Recupere une ressource. Remonte les niveaux en cas de miss.

        Retourne (valeur, latence_totale_ms, chemin_parcouru).
        Le chemin montre quels noeuds ont ete traverses (utile pour le debug).
        """
        total_latency = self.latency_ms  # Latence pour atteindre ce noeud
        path = [self.name]

        # Verifier le cache local
        if key in self.cache:
            # Verifier le TTL
            if key in self.cache_ttl and time.time() > self.cache_ttl[key]:
                # Expire -> supprimer
                del self.cache[key]
                del self.cache_ttl[key]
            else:
                self.hits += 1
                return self.cache[key], total_latency, path

        self.misses += 1

        # Cache miss -> remonter vers l'upstream
        if self.upstream:
            value, upstream_latency, upstream_path = self.upstream.get(key)
            total_latency += upstream_latency
            path.extend(upstream_path)

            # Stocker dans le cache local pour les futurs acces
            if value is not None:
                self.cache[key] = value
                self.cache_ttl[key] = time.time() + 60  # TTL par defaut 60s

            return value, total_latency, path

        # Pas d'upstream = on est l'origin, la donnee n'existe pas
        return None, total_latency, path

    def put_origin(self, key: str, value: Any) -> None:
        """Ajoute une ressource a l'origin (pas de TTL pour l'origin)."""
        self.cache[key] = value


class CDNSimulator:
    """Simule un CDN a 3 niveaux : Edge -> Regional -> Origin.

    Architecture :
        [Client] -> [Edge Paris/Tokyo] -> [Regional EU/APAC] -> [Origin US]

    Latences typiques :
        - Edge (meme ville) : 2-5 ms
        - Regional (meme continent) : 15-30 ms
        - Origin (intercontinental) : 80-150 ms
    """

    def __init__(self):
        # Origin (source de verite)
        self.origin = CDNNode(name="Origin-US", latency_ms=100.0)

        # Regional caches (un par continent)
        self.regional_eu = CDNNode(name="Regional-EU", latency_ms=25.0, upstream=self.origin)
        self.regional_apac = CDNNode(name="Regional-APAC", latency_ms=30.0, upstream=self.origin)

        # Edge caches (un par ville)
        self.edge_paris = CDNNode(name="Edge-Paris", latency_ms=3.0, upstream=self.regional_eu)
        self.edge_tokyo = CDNNode(name="Edge-Tokyo", latency_ms=4.0, upstream=self.regional_apac)
        self.edge_nyc = CDNNode(name="Edge-NYC", latency_ms=2.0, upstream=self.origin)

    def populate_origin(self, resources: dict[str, Any]) -> None:
        """Charge des ressources dans l'origin."""
        for key, value in resources.items():
            self.origin.put_origin(key, value)


# =============================================================================
# DEMONSTRATIONS
# =============================================================================


def demo_lru_caches():
    """Compare les deux implementations LRU (OrderedDict vs from scratch)."""
    print(f"\n{SEPARATOR}")
    print("  SECTION 1 & 2 : LRU Cache — OrderedDict vs Doubly-Linked List")
    print(SEPARATOR)

    # Test avec les memes operations sur les deux implementations
    for name, cache_class in [("OrderedDict", LRUCacheOrderedDict),
                               ("From Scratch", LRUCacheFromScratch)]:
        cache = cache_class(capacity=3)
        print(f"\n  --- {name} (capacity=3) ---")

        # Inserer 3 elements
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        print(f"  Apres put(a,b,c) : taille = {len(cache.cache)}")

        # Acceder a 'a' pour le marquer comme recent
        cache.get("a")
        print(f"  get('a') -> 'a' est maintenant le MRU")

        # Inserer 'd' -> 'b' devrait etre evince (LRU)
        evicted = cache.put("d", 4)
        print(f"  put('d') -> evicted = '{evicted}' (attendu: 'b')")

        # Verifier que 'b' n'est plus la
        result = cache.get("b")
        print(f"  get('b') -> {result} (attendu: None)")

        # Verifier que 'a' est toujours la
        result = cache.get("a")
        print(f"  get('a') -> {result} (attendu: 1)")

    # Benchmark avec des acces Zipf (realiste)
    print(f"\n  --- Benchmark : distribution Zipf (1000 cles, cache=100) ---")
    random.seed(42)

    # Generer des acces selon une loi Zipf (les premieres cles sont plus populaires)
    num_keys = 1000
    num_accesses = 10000
    cache_size = 100

    # Zipf : P(key_i) proportionnel a 1/i
    weights = [1.0 / (i + 1) for i in range(num_keys)]
    total_weight = sum(weights)
    probabilities = [w / total_weight for w in weights]

    # Generer la sequence d'acces
    access_sequence = random.choices(range(num_keys), weights=probabilities, k=num_accesses)

    for name, cache_class in [("OrderedDict", LRUCacheOrderedDict),
                               ("From Scratch", LRUCacheFromScratch)]:
        cache = cache_class(capacity=cache_size)

        start = time.time()
        for key_idx in access_sequence:
            key = f"key_{key_idx}"
            result = cache.get(key)
            if result is None:
                cache.put(key, f"value_{key_idx}")
        elapsed = (time.time() - start) * 1000

        print(f"  {name:15s} : hit_rate={cache.hit_rate:.1%}, "
              f"temps={elapsed:.1f}ms pour {num_accesses} ops")


def demo_lfu_cache():
    """Demonstation du LFU cache et comparaison avec LRU."""
    print(f"\n{SEPARATOR}")
    print("  SECTION 3 : LFU Cache")
    print(SEPARATOR)

    lfu = LFUCache(capacity=3)

    # Inserer 3 elements
    lfu.put("a", 1)  # freq(a) = 1
    lfu.put("b", 2)  # freq(b) = 1
    lfu.put("c", 3)  # freq(c) = 1

    # Acceder a 'a' 3 fois et 'b' 2 fois
    lfu.get("a")  # freq(a) = 2
    lfu.get("a")  # freq(a) = 3
    lfu.get("b")  # freq(b) = 2

    # Inserer 'd' -> 'c' est evince (freq=1, la plus basse)
    evicted = lfu.put("d", 4)
    print(f"\n  Apres acces a=3x, b=2x, c=1x : put('d') evince '{evicted}' (attendu: 'c')")

    # Comparer LFU vs LRU sur un workload avec des hot keys
    print(f"\n  --- Comparaison LFU vs LRU avec hot keys ---")
    random.seed(42)

    # Scenario : 10 hot keys accedees 80% du temps, 990 cold keys accedees 20% du temps
    hot_keys = [f"hot_{i}" for i in range(10)]
    cold_keys = [f"cold_{i}" for i in range(990)]
    num_ops = 10000
    cache_size = 50  # Le cache peut contenir 50 entrees

    for name, cache_class in [("LRU", LRUCacheOrderedDict), ("LFU", LFUCache)]:
        if name == "LRU":
            cache = cache_class(capacity=cache_size)
        else:
            cache = cache_class(capacity=cache_size)

        for _ in range(num_ops):
            if random.random() < 0.8:
                # 80% du temps : acceder a une hot key
                key = random.choice(hot_keys)
            else:
                # 20% du temps : acceder a une cold key
                key = random.choice(cold_keys)

            result = cache.get(key)
            if result is None:
                cache.put(key, f"value_for_{key}")

        print(f"  {name:5s} : hit_rate = {cache.hit_rate:.1%}")

    print(f"\n  LFU devrait avoir un meilleur hit rate car il protege les hot keys")
    print(f"  LRU peut evincer une hot key si un burst de cold keys remplit le cache")


def demo_cache_aside():
    """Demonstration du pattern Cache-Aside avec metriques."""
    print(f"\n{SEPARATOR}")
    print("  SECTION 4 : Cache-Aside Pattern")
    print(SEPARATOR)

    db = SimulatedDB(latency_ms=10.0)
    cache = SimulatedRedis(latency_ms=0.5)
    service = CacheAsideService(db, cache, ttl=300)

    # Prepopuler la DB
    for i in range(100):
        db.data[f"user:{i}"] = {"id": i, "name": f"User_{i}"}

    # Simuler des acces (memes users accedes plusieurs fois)
    print(f"\n  Simulation : 200 acces sur 100 users (distribution Zipf)")
    random.seed(42)
    weights = [1.0 / (i + 1) for i in range(100)]
    access_keys = random.choices([f"user:{i}" for i in range(100)],
                                  weights=weights, k=200)

    start = time.time()
    for key in access_keys:
        service.get(key)
    elapsed = (time.time() - start) * 1000

    print(f"  Cache hit rate : {cache.hit_rate:.1%}")
    print(f"  DB queries     : {db.query_count} (sur 200 acces)")
    print(f"  Temps total    : {elapsed:.0f} ms")

    # Demonstration de l'invalidation
    print(f"\n  --- Invalidation apres update ---")
    service.get("user:0")  # S'assurer que c'est en cache
    print(f"  get('user:0') -> en cache (hit)")

    service.update("user:0", {"id": 0, "name": "User_0_updated"})
    print(f"  update('user:0') -> DB mise a jour, cache invalide")

    result = service.get("user:0")
    print(f"  get('user:0') -> {result['name']} (cache miss -> DB -> re-cache)")


def demo_write_through_vs_behind():
    """Compare Write-Through (synchrone) et Write-Behind (asynchrone)."""
    print(f"\n{SEPARATOR}")
    print("  SECTION 5 : Write-Through vs Write-Behind")
    print(SEPARATOR)

    num_writes = 50

    # Write-Through
    db_wt = SimulatedDB(latency_ms=10.0)
    cache_wt = SimulatedRedis(latency_ms=0.5)
    wt_service = WriteThroughService(db_wt, cache_wt)

    start = time.time()
    for i in range(num_writes):
        wt_service.write(f"key:{i}", f"value_{i}")
    wt_time = (time.time() - start) * 1000

    # Write-Behind
    db_wb = SimulatedDB(latency_ms=10.0)
    cache_wb = SimulatedRedis(latency_ms=0.5)
    wb_service = WriteBehindService(db_wb, cache_wb)

    start = time.time()
    for i in range(num_writes):
        wb_service.write(f"key:{i}", f"value_{i}")
    wb_write_time = (time.time() - start) * 1000

    # Flush le buffer de Write-Behind
    start = time.time()
    flushed = wb_service.flush()
    wb_flush_time = (time.time() - start) * 1000

    print(f"\n  {num_writes} ecritures :")
    print(f"  Write-Through : {wt_time:.0f} ms (synchrone cache + DB)")
    print(f"  Write-Behind  : {wb_write_time:.0f} ms (cache seulement)")
    print(f"  Write-Behind flush : {wb_flush_time:.0f} ms ({flushed} ecritures en batch)")
    print(f"\n  Speedup write : {wt_time / wb_write_time:.1f}x")
    print(f"  Risque Write-Behind : si crash avant flush, {len(wb_service.write_buffer)} ecritures perdues")

    # Verification consistance
    print(f"\n  --- Verification consistance ---")
    print(f"  Write-Through DB queries : {db_wt.query_count} (= {num_writes} writes)")
    print(f"  Write-Behind DB queries  : {db_wb.query_count} (= {flushed} apres flush)")


def demo_cache_stampede():
    """Simule un cache stampede et montre l'effet du locking."""
    print(f"\n{SEPARATOR}")
    print("  SECTION 6 : Cache Stampede — Sans lock vs Avec lock")
    print(SEPARATOR)

    num_threads = 20
    key = "popular_item"

    # Test SANS lock
    demo = StampedeDemo()
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=demo.get_without_lock, args=(key,))
        threads.append(t)

    start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    no_lock_time = (time.time() - start) * 1000

    print(f"\n  Sans lock : {num_threads} threads simultanes")
    print(f"  DB queries : {demo.db_query_count} (idealement 1, en realite {demo.db_query_count})")
    print(f"  Temps total : {no_lock_time:.0f} ms")

    # Test AVEC lock
    demo.reset()
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=demo.get_with_lock, args=(key,))
        threads.append(t)

    start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    lock_time = (time.time() - start) * 1000

    print(f"\n  Avec lock : {num_threads} threads simultanes")
    print(f"  DB queries : {demo.db_query_count} (exactement 1)")
    print(f"  Temps total : {lock_time:.0f} ms")

    print(f"\n  Resultat : le lock reduit les DB queries de {num_threads}x a 1")
    print(f"  (en production, {num_threads} queries simultanes pourraient faire tomber la DB)")


def demo_ttl_cache():
    """Demonstration du TTL cache avec expiration."""
    print(f"\n{SEPARATOR}")
    print("  SECTION 7 : TTL Cache avec expiration automatique")
    print(SEPARATOR)

    cache = TTLCache(default_ttl=2)  # TTL de 2 secondes pour la demo

    # Inserer des donnees
    cache.set("fast", "expire vite", ttl=1)
    cache.set("slow", "expire lentement", ttl=5)
    cache.set("default", "ttl par defaut")  # 2 secondes

    print(f"\n  Donnees inserees : fast(1s), slow(5s), default(2s)")
    print(f"  Taille cache : {len(cache)}")

    # Lire immediatement
    print(f"\n  Lecture immediate :")
    print(f"    fast    = {cache.get('fast')}")
    print(f"    slow    = {cache.get('slow')}")
    print(f"    default = {cache.get('default')}")
    print(f"    hit_rate = {cache.hit_rate:.1%}")

    # Attendre 1.5 secondes
    time.sleep(1.5)
    print(f"\n  Apres 1.5 secondes :")
    print(f"    fast    = {cache.get('fast')} (expire apres 1s)")
    print(f"    slow    = {cache.get('slow')} (encore valide)")
    print(f"    default = {cache.get('default')} (encore valide)")

    # Attendre encore 1 seconde (total 2.5s)
    time.sleep(1.0)
    print(f"\n  Apres 2.5 secondes :")
    print(f"    fast    = {cache.get('fast')} (expire)")
    print(f"    slow    = {cache.get('slow')} (encore valide)")
    print(f"    default = {cache.get('default')} (expire apres 2s)")

    # Active cleanup
    time.sleep(3.0)  # Attendre que 'slow' expire aussi
    cleaned = cache.cleanup()
    print(f"\n  Apres 5.5 secondes + cleanup : {cleaned} entree(s) supprimee(s)")
    print(f"  Taille cache : {len(cache)}")
    print(f"  Total expired : {cache.expired_count}")


def demo_cdn_simulator():
    """Simule un CDN multi-niveaux et mesure les latences."""
    print(f"\n{SEPARATOR}")
    print("  SECTION 8 : CDN Simulator — Edge -> Regional -> Origin")
    print(SEPARATOR)

    cdn = CDNSimulator()

    # Charger des ressources dans l'origin
    cdn.populate_origin({
        "/index.html": "<html>Homepage</html>",
        "/app.js": "console.log('app')",
        "/style.css": "body { margin: 0 }",
        "/api/feed": '{"posts": [...]}',
        "/images/logo.png": "[binary data]",
    })

    print(f"\n  5 ressources chargees dans l'origin")

    # Simuler des requetes depuis differentes villes
    scenarios = [
        ("Paris  (1er acces)", cdn.edge_paris, "/index.html"),
        ("Paris  (2e acces) ", cdn.edge_paris, "/index.html"),
        ("Tokyo  (1er acces)", cdn.edge_tokyo, "/index.html"),
        ("Paris  (app.js)   ", cdn.edge_paris, "/app.js"),
        ("Paris  (app.js 2e)", cdn.edge_paris, "/app.js"),
        ("NYC    (index)    ", cdn.edge_nyc, "/index.html"),
        ("NYC    (index 2e) ", cdn.edge_nyc, "/index.html"),
    ]

    print(f"\n  {'Scenario':<25} {'Latence':>10} {'Chemin'}")
    print(f"  {'-'*25} {'-'*10} {'-'*40}")

    for label, edge, resource in scenarios:
        value, latency, path = edge.get(resource)
        path_str = " -> ".join(path)
        hit_or_miss = "HIT" if len(path) == 1 else "MISS->fill"
        print(f"  {label:<25} {latency:>7.0f} ms  {path_str} ({hit_or_miss})")

    # Stats par noeud
    print(f"\n  --- Stats par noeud ---")
    nodes = [cdn.edge_paris, cdn.edge_tokyo, cdn.edge_nyc,
             cdn.regional_eu, cdn.regional_apac, cdn.origin]
    for node in nodes:
        total = node.hits + node.misses
        hit_rate = node.hits / total if total > 0 else 0
        if total > 0:
            print(f"  {node.name:<20} hits={node.hits:>2}, misses={node.misses:>2}, "
                  f"hit_rate={hit_rate:.0%}")

    # Calculer les economies
    print(f"\n  --- Impact ---")
    total_requests = sum(n.hits + n.misses for n in [cdn.edge_paris, cdn.edge_tokyo, cdn.edge_nyc])
    origin_requests = cdn.origin.hits + cdn.origin.misses
    print(f"  Requetes totales aux edges : {total_requests}")
    print(f"  Requetes arrivees a l'origin : {origin_requests}")
    print(f"  Trafic economise sur l'origin : {(1 - origin_requests/total_requests)*100:.0f}%")


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Execute toutes les demonstrations."""
    print("\n" + "=" * 70)
    print("  JOUR 3 — CACHING & CDN : DEMONSTRATIONS INTERACTIVES")
    print("=" * 70)

    demo_lru_caches()
    demo_lfu_cache()
    demo_cache_aside()
    demo_write_through_vs_behind()
    demo_cache_stampede()
    demo_ttl_cache()
    demo_cdn_simulator()

    print(f"\n{'=' * 70}")
    print("  FIN DES DEMONSTRATIONS")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
