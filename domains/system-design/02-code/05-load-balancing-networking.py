"""
Jour 5 -- Load Balancing & Networking
Demonstrations interactives en Python.

Usage:
    python 05-load-balancing-networking.py

Simulations illustrant :
- Differents algorithmes de load balancing (RR, least-conn, consistent hash)
- Token bucket rate limiter
- Circuit breaker avec ses 3 etats
- Retry avec exponential backoff + jitter
"""

import time
import random
import bisect
import hashlib
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

SEPARATOR = "=" * 70


# =============================================================================
# SECTION 1 : Backend -- representation d'un serveur cible
# =============================================================================


@dataclass
class Backend:
    """Un backend (serveur web) avec ses metriques.

    WHY tracker les connexions ? Least-connections a besoin de savoir
    combien de requetes sont EN COURS sur chaque backend pour choisir
    le moins charge. En production, le LB compte les connexions TCP ouvertes.
    """

    name: str
    weight: int = 1
    active_connections: int = 0
    healthy: bool = True
    total_handled: int = 0

    def handle(self):
        """Simule le traitement d'une requete."""
        self.active_connections += 1
        self.total_handled += 1
        # En vrai, le temps depend de la requete. On simule avec un sleep aleatoire.
        time.sleep(random.uniform(0.001, 0.005))
        self.active_connections -= 1


# =============================================================================
# SECTION 2 : Round Robin et Weighted Round Robin
# =============================================================================


class RoundRobinLB:
    """Round robin : requete i va au backend i % N.

    WHY si simple ? Aucun etat a tracker par backend (juste un compteur
    global). Parfait si toutes les requetes ont la meme duree, mauvais
    si certaines sont 100x plus longues.
    """

    def __init__(self, backends: list[Backend]):
        self.backends = backends
        self.index = 0
        self._lock = threading.Lock()

    def pick(self) -> Backend:
        with self._lock:
            # On cherche le prochain backend healthy (skip les morts)
            for _ in range(len(self.backends)):
                b = self.backends[self.index % len(self.backends)]
                self.index += 1
                if b.healthy:
                    return b
            raise RuntimeError("No healthy backend available")


class WeightedRoundRobinLB:
    """Weighted RR : chaque backend a un poids.

    WHY weighted ? Pour les serveurs heterogenes (16-core vs 64-core)
    ou pour le canary deploy (10% du trafic sur la V2). Implementation
    naive : on duplique le backend dans la liste selon son poids.
    """

    def __init__(self, backends: list[Backend]):
        self.backends = backends
        # Liste etendue : [A, A, A, B, C] pour A(3), B(1), C(1)
        self.expanded: list[Backend] = []
        for b in backends:
            self.expanded.extend([b] * b.weight)
        self.index = 0
        self._lock = threading.Lock()

    def pick(self) -> Backend:
        with self._lock:
            for _ in range(len(self.expanded)):
                b = self.expanded[self.index % len(self.expanded)]
                self.index += 1
                if b.healthy:
                    return b
            raise RuntimeError("No healthy backend available")


# =============================================================================
# SECTION 3 : Least Connections
# =============================================================================


class LeastConnectionsLB:
    """Least Connections : envoyer au backend avec le moins de conns actives.

    WHY ca marche ? Si un backend est lent (gros request, thread bloque),
    ses connexions s'accumulent. Le LB detecte et envoie ailleurs. Auto-
    adaptation aux requetes de duree variable.
    """

    def __init__(self, backends: list[Backend]):
        self.backends = backends

    def pick(self) -> Backend:
        # On filtre les healthy puis on prend celui avec le min de conns
        healthy = [b for b in self.backends if b.healthy]
        if not healthy:
            raise RuntimeError("No healthy backend available")
        return min(healthy, key=lambda b: b.active_connections)


# =============================================================================
# SECTION 4 : Consistent Hashing avec virtual nodes
# =============================================================================


class ConsistentHashLB:
    """Consistent Hashing : l'anneau avec vnodes.

    WHY pas hash(key) % N ? Si on change N (ajout/suppression d'un
    backend), presque toutes les cles sont reaffectees. Un cache perdrait
    99% de ses entrees. Avec l'anneau, seuls ~1/N des cles bougent.

    WHY virtual nodes ? Si on a juste N points sur l'anneau, la
    distribution est mauvaise (un backend peut choper une grande zone,
    un autre une petite). En creant K vnodes par backend (K=100-200),
    la distribution est quasi uniforme.
    """

    def __init__(self, backends: list[Backend], vnodes_per_backend: int = 100):
        self.vnodes_per_backend = vnodes_per_backend
        self.ring: list[tuple[int, Backend]] = []  # trie par hash
        self.hashes: list[int] = []
        for b in backends:
            self._add(b)

    @staticmethod
    def _hash(key: str) -> int:
        # sha1 pour une distribution uniforme (pas python hash() qui est randomise)
        return int(hashlib.sha1(key.encode()).hexdigest(), 16)

    def _add(self, backend: Backend):
        for v in range(self.vnodes_per_backend):
            h = self._hash(f"{backend.name}-vnode-{v}")
            # bisect pour garder le ring trie (O(log N) lookups)
            idx = bisect.bisect(self.hashes, h)
            self.hashes.insert(idx, h)
            self.ring.insert(idx, (h, backend))

    def remove(self, backend: Backend):
        """WHY remove ? Quand un backend tombe en healthcheck, on retire
        ses vnodes. Seules les cles qui visaient ces vnodes sont
        redistribuees (les voisines horaires)."""
        new_ring = [(h, b) for h, b in self.ring if b is not backend]
        self.ring = new_ring
        self.hashes = [h for h, _ in new_ring]

    def pick(self, key: str) -> Backend:
        if not self.ring:
            raise RuntimeError("No backend in ring")
        h = self._hash(key)
        # Cherche le premier vnode apres h (wrap around si au bout)
        idx = bisect.bisect(self.hashes, h) % len(self.hashes)
        return self.ring[idx][1]


# =============================================================================
# SECTION 5 : Token Bucket Rate Limiter
# =============================================================================


class TokenBucket:
    """Token bucket rate limiter.

    WHY token bucket ? Permet les bursts (seau plein = tu peux envoyer
    'capacity' requetes d'un coup) tout en limitant la moyenne au taux
    de refill. C'est le choix par defaut de la plupart des APIs (Stripe,
    AWS, GitHub).

    Exemple : capacity=100, refill=10/s
    -> Burst : 100 req d'un coup OK, puis 10 req/s en moyenne.
    """

    def __init__(self, capacity: int, refill_per_sec: float):
        self.capacity = capacity
        self.refill_per_sec = refill_per_sec
        self.tokens = float(capacity)
        self.last_refill = time.time()
        self._lock = threading.Lock()

    def allow(self, cost: int = 1) -> bool:
        """Tente de consommer 'cost' tokens. Retourne True si OK."""
        with self._lock:
            now = time.time()
            elapsed = now - self.last_refill
            # Refill continu : chaque milliseconde ajoute refill_per_sec/1000 tokens
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_sec)
            self.last_refill = now
            if self.tokens >= cost:
                self.tokens -= cost
                return True
            return False


# =============================================================================
# SECTION 6 : Circuit Breaker
# =============================================================================


class CircuitBreaker:
    """Circuit breaker avec les 3 etats classiques.

    WHY ca sauve des vies ? Sans breaker, quand un service aval tombe,
    l'appelant continue d'essayer et sature ses threads sur des timeouts
    de 30s. Les threads meurent, l'appelant tombe aussi -> cascade.

    Le breaker detecte les echecs et OUVRE : toutes les requetes suivantes
    echouent IMMEDIATEMENT (pas d'appel reseau) avec une erreur ou un
    fallback. Apres un timeout, il tente prudemment 1 requete (HALF_OPEN).
    """

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_sec: float = 5.0,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout_sec = recovery_timeout_sec
        self.state = self.CLOSED
        self.failures = 0
        self.opened_at: Optional[float] = None
        self._lock = threading.Lock()

    def call(self, func: Callable[[], Any], fallback: Callable[[], Any] = None) -> Any:
        """Appelle func via le breaker. Si OPEN, retourne fallback() ou leve."""
        with self._lock:
            # Transition automatique OPEN -> HALF_OPEN apres timeout
            if self.state == self.OPEN:
                if time.time() - self.opened_at >= self.recovery_timeout_sec:
                    self.state = self.HALF_OPEN
                    print(f"    [breaker] OPEN -> HALF_OPEN (tentative de reprise)")
                else:
                    if fallback:
                        return fallback()
                    raise RuntimeError("Circuit breaker is OPEN")

        try:
            result = func()
        except Exception as e:
            self._on_failure()
            if fallback:
                return fallback()
            raise
        else:
            self._on_success()
            return result

    def _on_success(self):
        with self._lock:
            if self.state == self.HALF_OPEN:
                print(f"    [breaker] HALF_OPEN -> CLOSED (reprise OK)")
            self.state = self.CLOSED
            self.failures = 0
            self.opened_at = None

    def _on_failure(self):
        with self._lock:
            self.failures += 1
            if self.state == self.HALF_OPEN:
                # La requete test a echoue : on repart en OPEN
                self.state = self.OPEN
                self.opened_at = time.time()
                print(f"    [breaker] HALF_OPEN -> OPEN (echec de la requete test)")
            elif self.failures >= self.failure_threshold:
                self.state = self.OPEN
                self.opened_at = time.time()
                print(f"    [breaker] CLOSED -> OPEN (threshold {self.failure_threshold} atteint)")


# =============================================================================
# SECTION 7 : Retry avec exponential backoff + jitter
# =============================================================================


def retry_with_backoff(
    func: Callable[[], Any],
    max_attempts: int = 5,
    base_delay: float = 0.1,
    max_delay: float = 2.0,
) -> Any:
    """Retry avec exponential backoff et jitter.

    WHY exponentiel ? Laisse au service le temps de recuperer entre les
    tentatives (1s, 2s, 4s, 8s...).
    WHY jitter ? Sans randomisation, tous les clients retryent au meme
    moment -> thundering herd qui surcharge le service au moment ou il
    tente de remonter. Le jitter etale dans le temps.
    """
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            if attempt == max_attempts - 1:
                raise
            # delay = min(max, base * 2^attempt)
            delay = min(max_delay, base_delay * (2 ** attempt))
            # Jitter : multiplier par [0.5, 1.5] pour etaler
            delay = delay * random.uniform(0.5, 1.5)
            print(f"    attempt {attempt+1}/{max_attempts} failed: {e}, waiting {delay:.3f}s")
            time.sleep(delay)


# =============================================================================
# SECTION 8 : Demos
# =============================================================================


def demo_algorithms_comparison():
    """Compare la distribution de 1000 requetes sur 3 backends avec 3 algos."""
    print(f"\n{SEPARATOR}\n  DEMO 1 : Comparaison des algorithmes de LB\n{SEPARATOR}")

    def make_backends():
        return [Backend("srv-A", weight=3), Backend("srv-B", weight=1), Backend("srv-C", weight=1)]

    # Round Robin
    backends = make_backends()
    lb = RoundRobinLB(backends)
    for _ in range(100):
        lb.pick().total_handled += 1
    print(f"  Round Robin : {[(b.name, b.total_handled) for b in backends]}")

    # Weighted Round Robin
    backends = make_backends()
    lb = WeightedRoundRobinLB(backends)
    for _ in range(100):
        lb.pick().total_handled += 1
    print(f"  Weighted RR : {[(b.name, b.total_handled) for b in backends]} (poids 3:1:1)")

    # Least Connections (simule avec des durees variables)
    backends = make_backends()
    lb = LeastConnectionsLB(backends)
    for i in range(30):
        b = lb.pick()
        b.active_connections += 1
        # Simuler que srv-A est plus lent (conns ne se liberent pas vite)
        if b.name == "srv-A" and i % 3 != 0:
            continue
        b.active_connections = max(0, b.active_connections - 1)
    print(f"  Least Conn  : {[(b.name, b.active_connections) for b in backends]}")


def demo_consistent_hashing_stability():
    """Montre qu'ajouter/retirer un backend ne reshuffle qu'une fraction des cles."""
    print(f"\n{SEPARATOR}\n  DEMO 2 : Consistent hashing stability\n{SEPARATOR}")

    backends = [Backend(f"cache-{i}") for i in range(4)]
    lb = ConsistentHashLB(backends, vnodes_per_backend=100)

    # Assign 1000 keys -> record current owner
    keys = [f"user-{i}" for i in range(1000)]
    before = {k: lb.pick(k).name for k in keys}

    # Remove one backend (simulates failure)
    lb.remove(backends[2])
    after = {k: lb.pick(k).name for k in keys}

    moved = sum(1 for k in keys if before[k] != after[k])
    print(f"  Keys before: assigned over 4 backends")
    print(f"  Remove cache-2 : {moved}/1000 keys moved ({moved/10:.1f}%)")
    print(f"  Avec hash % N : 100% des keys auraient bouge. Avec consistent,")
    print(f"  seul ~25% (cles qui pointaient vers cache-2).")


def demo_token_bucket():
    """Test un token bucket avec un burst puis trafic soutenu."""
    print(f"\n{SEPARATOR}\n  DEMO 3 : Token bucket rate limiter\n{SEPARATOR}")
    bucket = TokenBucket(capacity=10, refill_per_sec=5)

    # Burst : 15 requetes d'un coup
    allowed = sum(1 for _ in range(15) if bucket.allow())
    print(f"  Burst 15 req : {allowed} allowed, {15-allowed} rejected")
    print(f"  (capacity=10, donc on peut burst 10 d'un coup)")

    # Attendre 1 seconde : refill de 5 tokens
    time.sleep(1.0)
    allowed = sum(1 for _ in range(10) if bucket.allow())
    print(f"  Apres 1s (refill ~5 tokens) : {allowed}/10 allowed")


def demo_circuit_breaker():
    """Simulate un service qui tombe et se releve."""
    print(f"\n{SEPARATOR}\n  DEMO 4 : Circuit breaker\n{SEPARATOR}")
    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout_sec=0.3)

    service_alive = [True]

    def flaky_service():
        if not service_alive[0]:
            raise ConnectionError("service down")
        return "OK"

    def fallback():
        return "CACHED_FALLBACK"

    # Service alive : tout passe
    print("  Phase 1 : service alive")
    for i in range(3):
        r = breaker.call(flaky_service, fallback)
        print(f"    req{i+1} : {r}, state={breaker.state}")

    # Service down : le breaker va s'ouvrir apres 3 echecs
    print("  Phase 2 : service down")
    service_alive[0] = False
    for i in range(5):
        r = breaker.call(flaky_service, fallback)
        print(f"    req{i+1} : {r}, state={breaker.state}")

    # Service revient. On attend le recovery timeout.
    print("  Phase 3 : service back, wait for HALF_OPEN")
    service_alive[0] = True
    time.sleep(0.35)
    r = breaker.call(flaky_service, fallback)
    print(f"    req : {r}, state={breaker.state}")


def demo_retry_with_jitter():
    """Simule un service qui repond apres 3 tentatives."""
    print(f"\n{SEPARATOR}\n  DEMO 5 : Retry exponential backoff + jitter\n{SEPARATOR}")
    attempts = [0]

    def flaky():
        attempts[0] += 1
        if attempts[0] < 3:
            raise ConnectionError(f"transient error")
        return "success"

    result = retry_with_backoff(flaky, max_attempts=5, base_delay=0.05)
    print(f"  Result : {result} after {attempts[0]} attempts")


def main():
    random.seed(42)
    demo_algorithms_comparison()
    demo_consistent_hashing_stability()
    demo_token_bucket()
    demo_circuit_breaker()
    demo_retry_with_jitter()
    print(f"\n{SEPARATOR}\n  Fin des demos.\n{SEPARATOR}")


if __name__ == "__main__":
    main()
