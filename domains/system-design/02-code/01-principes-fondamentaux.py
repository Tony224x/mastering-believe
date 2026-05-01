"""
Jour 1 -- Principes fondamentaux du System Design
Demonstrations interactives en Python.

Usage:
    python 01-principes-fondamentaux.py

Chaque section est independante et peut etre executee via la fonction main().
"""

import time
import threading
import random
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

# =============================================================================
# SECTION 1 : Back-of-the-envelope Calculator
# =============================================================================


@dataclass
class SystemEstimation:
    """Estimation rapide des besoins d'un systeme."""

    name: str
    dau: int                    # Daily Active Users
    actions_per_user_per_day: int
    avg_request_size_bytes: int
    avg_response_size_bytes: int
    storage_per_object_bytes: int
    retention_days: int

    @property
    def qps_average(self) -> float:
        """QPS moyen = DAU * actions/jour / 86400 secondes."""
        return self.dau * self.actions_per_user_per_day / 86_400

    @property
    def qps_peak(self) -> float:
        """QPS pic = QPS moyen * 3 (regle empirique : pic ~ 2-5x moyenne)."""
        return self.qps_average * 3

    @property
    def bandwidth_in_mbps(self) -> float:
        """Bande passante entrante = QPS pic * taille requete, convertie en Mbps."""
        bytes_per_sec = self.qps_peak * self.avg_request_size_bytes
        return bytes_per_sec * 8 / 1_000_000  # bits -> Mbps

    @property
    def bandwidth_out_mbps(self) -> float:
        """Bande passante sortante = QPS pic * taille reponse."""
        bytes_per_sec = self.qps_peak * self.avg_response_size_bytes
        return bytes_per_sec * 8 / 1_000_000

    @property
    def storage_per_day_gb(self) -> float:
        """Stockage/jour = QPS moyen * 86400 * taille objet."""
        # On utilise QPS moyen (pas pic) car le stockage est cumule sur 24h
        total_objects_per_day = self.dau * self.actions_per_user_per_day
        return total_objects_per_day * self.storage_per_object_bytes / (1024**3)

    @property
    def storage_total_tb(self) -> float:
        """Stockage total = stockage/jour * retention, en To."""
        return self.storage_per_day_gb * self.retention_days / 1024

    def report(self) -> str:
        """Genere un rapport lisible des estimations."""
        lines = [
            f"\n{'='*60}",
            f"  ESTIMATION : {self.name}",
            f"{'='*60}",
            f"  DAU                    : {self.dau:>15,}",
            f"  Actions/user/day       : {self.actions_per_user_per_day:>15,}",
            f"  {'-'*56}",
            f"  QPS moyen              : {self.qps_average:>15,.0f} req/s",
            f"  QPS pic (x3)           : {self.qps_peak:>15,.0f} req/s",
            f"  {'-'*56}",
            f"  Bande passante IN      : {self.bandwidth_in_mbps:>15,.1f} Mbps",
            f"  Bande passante OUT     : {self.bandwidth_out_mbps:>15,.1f} Mbps",
            f"  {'-'*56}",
            f"  Stockage/jour          : {self.storage_per_day_gb:>15,.2f} Go",
            f"  Stockage total ({self.retention_days}j)   : {self.storage_total_tb:>12,.2f} To",
            f"{'='*60}",
        ]
        return "\n".join(lines)


def demo_estimation():
    """Estime les besoins pour un clone Twitter et un service de chat."""
    print("\n" + "=" * 60)
    print("  SECTION 1 : Back-of-the-envelope Estimation")
    print("=" * 60)

    # Scenario 1 : Twitter-like
    twitter = SystemEstimation(
        name="Twitter-like (posts + timeline reads)",
        dau=200_000_000,
        actions_per_user_per_day=50,        # 50 lectures de timeline / tweets par jour
        avg_request_size_bytes=500,          # Requete legere (GET timeline)
        avg_response_size_bytes=5_000,       # 5 Ko de JSON (20 tweets avec metadata)
        storage_per_object_bytes=1_000,      # 1 Ko par tweet (texte + metadata)
        retention_days=365 * 5,              # 5 ans de retention
    )

    # Scenario 2 : Service de chat (WhatsApp-like)
    chat = SystemEstimation(
        name="Chat service (WhatsApp-like)",
        dau=500_000_000,
        actions_per_user_per_day=100,        # 100 messages envoyes/recus par jour
        avg_request_size_bytes=200,          # Message court
        avg_response_size_bytes=200,         # ACK + message
        storage_per_object_bytes=500,        # Message + metadata + index
        retention_days=365,                  # 1 an
    )

    print(twitter.report())
    print(chat.report())


# =============================================================================
# SECTION 2 : Latence -- Sequentiel vs Parallele
# =============================================================================


def simulate_api_call(service_name: str, latency_ms: int) -> dict:
    """Simule un appel API avec une latence fixe.

    En vrai, on ferait un HTTP call. Ici on simule avec sleep
    pour montrer l'impact de la parallelisation.
    """
    time.sleep(latency_ms / 1000)
    return {"service": service_name, "latency_ms": latency_ms, "status": "ok"}


def demo_latency():
    """Compare appels sequentiels vs paralleles -- meme travail, temps tres different."""
    print("\n" + "=" * 60)
    print("  SECTION 2 : Latence -- Sequentiel vs Parallele")
    print("=" * 60)

    # Scenario : un endpoint doit appeler 5 microservices pour construire une reponse
    services = [
        ("user-service", 50),
        ("product-service", 80),
        ("recommendation-service", 120),
        ("pricing-service", 60),
        ("inventory-service", 40),
    ]

    total_theoretical = sum(lat for _, lat in services)

    # --- Sequentiel ---
    start = time.perf_counter()
    results_seq = []
    for name, latency in services:
        results_seq.append(simulate_api_call(name, latency))
    elapsed_seq = (time.perf_counter() - start) * 1000

    print(f"\n  Sequentiel ({len(services)} appels)")
    print(f"  Somme des latences individuelles : {total_theoretical} ms")
    print(f"  Temps reel mesure                : {elapsed_seq:.0f} ms")

    # --- Parallele ---
    # ThreadPoolExecutor lance tous les calls en meme temps
    # Le temps total = max(latences), pas la somme
    start = time.perf_counter()
    results_par = []
    with ThreadPoolExecutor(max_workers=len(services)) as executor:
        futures = {
            executor.submit(simulate_api_call, name, latency): name
            for name, latency in services
        }
        for future in as_completed(futures):
            results_par.append(future.result())
    elapsed_par = (time.perf_counter() - start) * 1000

    max_latency = max(lat for _, lat in services)
    print(f"\n  Parallele ({len(services)} appels)")
    print(f"  Latence du plus lent             : {max_latency} ms")
    print(f"  Temps reel mesure                : {elapsed_par:.0f} ms")
    print(f"\n  Speedup                          : {elapsed_seq / elapsed_par:.1f}x")
    print(f"  Temps economise                  : {elapsed_seq - elapsed_par:.0f} ms")

    # Lecon cle
    print("\n  >> Lecon : En parallele, la latence totale = max(latences individuelles)")
    print("  >> C'est pourquoi les architectures modernes fan-out puis aggregent.")


# =============================================================================
# SECTION 3 : Consistency Models -- Strong vs Eventual
# =============================================================================


class StrongConsistencyStore:
    """Simule un store avec strong consistency via un lock global.

    Chaque ecriture bloque toutes les lectures et ecritures
    jusqu'a ce que TOUS les replicas soient mis a jour.
    """

    def __init__(self, num_replicas: int = 3):
        self._lock = threading.Lock()  # Le lock simule la synchronisation entre replicas
        self._replicas = [None] * num_replicas
        self._write_count = 0
        self._stale_reads = 0
        self._total_reads = 0

    def write(self, value):
        """Ecriture : acquiert le lock, met a jour TOUS les replicas, puis libere."""
        with self._lock:
            # Simule le temps de replication synchrone (tous les replicas)
            time.sleep(0.01 * len(self._replicas))
            for i in range(len(self._replicas)):
                self._replicas[i] = value
            self._write_count += 1

    def read(self) -> tuple:
        """Lecture : attend le lock (garantit de lire la derniere ecriture)."""
        with self._lock:
            self._total_reads += 1
            # Tous les replicas sont identiques grace au lock
            replica_idx = random.randint(0, len(self._replicas) - 1)
            return self._replicas[replica_idx], replica_idx


class EventualConsistencyStore:
    """Simule un store avec eventual consistency.

    L'ecriture met a jour UN replica immediatement.
    La propagation aux autres replicas se fait en arriere-plan (async).
    Les lectures peuvent toucher un replica pas encore mis a jour = stale read.
    """

    def __init__(self, num_replicas: int = 3):
        self._replicas = [None] * num_replicas
        self._write_count = 0
        self._stale_reads = 0
        self._total_reads = 0
        self._latest_value = None

    def write(self, value):
        """Ecriture : met a jour UN seul replica, lance la propagation en background."""
        self._latest_value = value
        self._replicas[0] = value  # Seul le replica primaire est a jour immediatement
        self._write_count += 1

        # Propagation asynchrone aux autres replicas (simule un delai reseau)
        def propagate():
            for i in range(1, len(self._replicas)):
                time.sleep(random.uniform(0.005, 0.03))  # Delai variable par replica
                self._replicas[i] = value

        threading.Thread(target=propagate, daemon=True).start()

    def read(self) -> tuple:
        """Lecture : choisit un replica aleatoire (peut etre stale)."""
        self._total_reads += 1
        replica_idx = random.randint(0, len(self._replicas) - 1)
        value = self._replicas[replica_idx]
        if value != self._latest_value:
            self._stale_reads += 1
        return value, replica_idx


def demo_consistency():
    """Montre la difference entre strong et eventual consistency avec des threads."""
    print("\n" + "=" * 60)
    print("  SECTION 3 : Consistency Models")
    print("=" * 60)

    num_operations = 50
    num_replicas = 3

    # --- Strong Consistency ---
    strong = StrongConsistencyStore(num_replicas)
    start = time.perf_counter()

    def strong_writer():
        for i in range(num_operations):
            strong.write(f"v{i}")
            time.sleep(0.001)

    def strong_reader():
        for _ in range(num_operations * 2):
            strong.read()
            time.sleep(0.001)

    t_write = threading.Thread(target=strong_writer)
    t_read = threading.Thread(target=strong_reader)
    t_write.start()
    t_read.start()
    t_write.join()
    t_read.join()

    elapsed_strong = (time.perf_counter() - start) * 1000

    # --- Eventual Consistency ---
    eventual = EventualConsistencyStore(num_replicas)
    start = time.perf_counter()

    def eventual_writer():
        for i in range(num_operations):
            eventual.write(f"v{i}")
            time.sleep(0.001)

    def eventual_reader():
        for _ in range(num_operations * 2):
            eventual.read()
            time.sleep(0.001)

    t_write = threading.Thread(target=eventual_writer)
    t_read = threading.Thread(target=eventual_reader)
    t_write.start()
    t_read.start()
    t_write.join()
    t_read.join()

    elapsed_eventual = (time.perf_counter() - start) * 1000

    # -- Wait a bit for propagation to complete --
    time.sleep(0.1)

    print(f"\n  {'Metrique':<30} {'Strong':>12} {'Eventual':>12}")
    print(f"  {'-'*54}")
    print(f"  {'Ecritures':<30} {strong._write_count:>12} {eventual._write_count:>12}")
    print(f"  {'Lectures totales':<30} {strong._total_reads:>12} {eventual._total_reads:>12}")
    print(f"  {'Stale reads':<30} {strong._stale_reads:>12} {eventual._stale_reads:>12}")
    print(f"  {'Temps total (ms)':<30} {elapsed_strong:>12.0f} {elapsed_eventual:>12.0f}")

    stale_pct = (
        (eventual._stale_reads / eventual._total_reads * 100)
        if eventual._total_reads > 0
        else 0
    )

    print(f"\n  >> Strong : 0 stale reads, mais {elapsed_strong:.0f}ms (lock bloque les readers)")
    print(f"  >> Eventual : {eventual._stale_reads} stale reads ({stale_pct:.1f}%), mais {elapsed_eventual:.0f}ms")
    print(f"  >> Tradeoff : consistance vs performance. Choisis selon le domaine metier.")


# =============================================================================
# SECTION 4 : SLA Calculator
# =============================================================================


def sla_downtime(uptime_pct: float) -> dict:
    """Calcule le downtime autorise pour un SLA donne.

    Args:
        uptime_pct: Pourcentage d'uptime (ex: 99.99)

    Returns:
        Dict avec downtime par an, mois, semaine, jour en format lisible.
    """
    # Fraction de downtime = 1 - uptime/100
    downtime_fraction = 1 - (uptime_pct / 100)

    # Secondes dans chaque periode
    seconds_per_year = 365.25 * 24 * 3600
    seconds_per_month = seconds_per_year / 12
    seconds_per_week = 7 * 24 * 3600
    seconds_per_day = 24 * 3600

    def format_duration(seconds: float) -> str:
        """Formate une duree en secondes en format humain."""
        if seconds >= 86400:
            return f"{seconds / 86400:.2f} jours"
        elif seconds >= 3600:
            return f"{seconds / 3600:.2f} heures"
        elif seconds >= 60:
            return f"{seconds / 60:.2f} minutes"
        else:
            return f"{seconds:.2f} secondes"

    return {
        "uptime": f"{uptime_pct}%",
        "nines": _count_nines(uptime_pct),
        "downtime_per_year": format_duration(downtime_fraction * seconds_per_year),
        "downtime_per_month": format_duration(downtime_fraction * seconds_per_month),
        "downtime_per_week": format_duration(downtime_fraction * seconds_per_week),
        "downtime_per_day": format_duration(downtime_fraction * seconds_per_day),
    }


def _count_nines(uptime_pct: float) -> str:
    """Compte le nombre de '9' dans un SLA (ex: 99.99% = 'four nines').

    Methode : -log10(1 - uptime/100) donne le nombre de nines.
    99.9% -> -log10(0.001) = 3 -> 'three nines'
    """
    if uptime_pct >= 100:
        return "infinite (impossible)"
    nines_count = -math.log10(1 - uptime_pct / 100)
    names = {1: "one nine", 2: "two nines", 3: "three nines",
             4: "four nines", 5: "five nines", 6: "six nines"}
    rounded = round(nines_count, 1)
    # Si c'est un entier, utilise le nom
    if rounded == int(rounded) and int(rounded) in names:
        return f"{names[int(rounded)]} ({nines_count:.2f})"
    return f"{nines_count:.2f} nines"


def demo_sla():
    """Affiche un tableau comparatif des SLA courants."""
    print("\n" + "=" * 60)
    print("  SECTION 4 : SLA Calculator")
    print("=" * 60)

    sla_levels = [99.0, 99.5, 99.9, 99.95, 99.99, 99.999]

    print(f"\n  {'SLA':>8}  {'Nines':<22}  {'Downtime/an':<18}  {'Downtime/mois':<18}  {'Downtime/jour':<18}")
    print(f"  {'-'*90}")

    for level in sla_levels:
        info = sla_downtime(level)
        print(
            f"  {info['uptime']:>8}  "
            f"{info['nines']:<22}  "
            f"{info['downtime_per_year']:<18}  "
            f"{info['downtime_per_month']:<18}  "
            f"{info['downtime_per_day']:<18}"
        )

    # Cas pratique : combiner des SLAs
    print(f"\n  {'-'*60}")
    print("  COMBINAISON DE SLAs")
    print(f"  {'-'*60}")
    print("  Si ton systeme depend de 3 services chacun a 99.9% :")
    combined = 0.999 ** 3
    print(f"  SLA combine = 99.9% * 99.9% * 99.9% = {combined * 100:.4f}%")
    combined_info = sla_downtime(combined * 100)
    print(f"  Downtime/an = {combined_info['downtime_per_year']}")
    print(f"\n  >> Lecon : Les SLAs se multiplient. 3 services a 'three nines'")
    print(f"     donnent un systeme a ~{combined * 100:.2f}% -- presque 'two nines'.")
    print(f"     C'est pourquoi la redondance et les fallbacks sont essentiels.")


# =============================================================================
# SECTION 5 : Latency Numbers Every Programmer Should Know
# =============================================================================


def demo_latency_numbers():
    """Affiche les ordres de grandeur de latence avec echelle visuelle."""
    print("\n" + "=" * 60)
    print("  SECTION 5 : Latency Numbers")
    print("=" * 60)

    # (operation, latence en nanosecondes)
    latencies = [
        ("L1 cache reference", 1),
        ("Branch mispredict", 3),
        ("L2 cache reference", 4),
        ("Mutex lock/unlock", 100),
        ("Main memory reference", 100),
        ("Compress 1KB (Snappy)", 3_000),
        ("Read 1MB from RAM", 3_000),
        ("SSD random read", 16_000),
        ("Read 1MB from SSD", 49_000),
        ("Datacenter round trip", 500_000),
        ("Read 1MB from HDD", 825_000),
        ("HDD disk seek", 2_000_000),
        ("Read 1MB from network (1Gbps)", 10_000_000),
        ("US coast-to-coast round trip", 40_000_000),
        ("Europe-US round trip", 80_000_000),
        ("Europe-Asia round trip", 150_000_000),
    ]

    # Echelle logarithmique visuelle : chaque '#' = un ordre de grandeur
    print(f"\n  {'Operation':<38} {'Latence':>12}  Echelle (log)")
    print(f"  {'-'*75}")

    for op, ns in latencies:
        # Formater la latence en unite lisible
        if ns < 1_000:
            lat_str = f"{ns} ns"
        elif ns < 1_000_000:
            lat_str = f"{ns / 1_000:.0f} us"
        else:
            lat_str = f"{ns / 1_000_000:.0f} ms"

        # Barre logarithmique : log10(ns) donne l'echelle
        bar_len = int(math.log10(max(ns, 1)) * 3)  # *3 pour lisibilite
        bar = "#" * bar_len

        print(f"  {op:<38} {lat_str:>12}  {bar}")

    print(f"\n  >> Facteur RAM vs SSD       : ~{49_000 / 3_000:.0f}x")
    print(f"  >> Facteur SSD vs Network   : ~{10_000_000 / 49_000:.0f}x")
    print(f"  >> Facteur RAM vs Network   : ~{10_000_000 / 3_000:.0f}x")
    print(f"\n  >> Conclusion : un cache memoire evite un round-trip reseau = gain de 3000x")


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Execute toutes les demos sequentiellement."""
    print("\n" + "#" * 60)
    print("#  JOUR 1 -- PRINCIPES FONDAMENTAUX DU SYSTEM DESIGN")
    print("#" * 60)

    demo_estimation()
    demo_latency_numbers()
    demo_latency()
    demo_consistency()
    demo_sla()

    print("\n" + "#" * 60)
    print("#  FIN -- Toutes les demos ont ete executees.")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
