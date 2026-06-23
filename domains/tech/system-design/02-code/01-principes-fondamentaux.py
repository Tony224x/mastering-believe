"""
Day 1 -- System Design Fundamentals
Interactive demonstrations in Python.

Usage:
    python 01-principes-fondamentaux.py

Each section is independent and can be executed via the main() function.
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
    """Quick estimation of a system's resource needs."""

    name: str
    dau: int                    # Daily Active Users
    actions_per_user_per_day: int
    avg_request_size_bytes: int
    avg_response_size_bytes: int
    storage_per_object_bytes: int
    retention_days: int

    @property
    def qps_average(self) -> float:
        """Average QPS = DAU * actions/day / 86400 seconds."""
        return self.dau * self.actions_per_user_per_day / 86_400

    @property
    def qps_peak(self) -> float:
        """Peak QPS = average QPS * 3 (rule of thumb: peak ~ 2-5x average)."""
        return self.qps_average * 3

    @property
    def bandwidth_in_mbps(self) -> float:
        """Inbound bandwidth = peak QPS * request size, converted to Mbps."""
        bytes_per_sec = self.qps_peak * self.avg_request_size_bytes
        return bytes_per_sec * 8 / 1_000_000  # bits -> Mbps

    @property
    def bandwidth_out_mbps(self) -> float:
        """Outbound bandwidth = peak QPS * response size."""
        bytes_per_sec = self.qps_peak * self.avg_response_size_bytes
        return bytes_per_sec * 8 / 1_000_000

    @property
    def storage_per_day_gb(self) -> float:
        """Storage/day = average QPS * 86400 * object size."""
        # We use average QPS (not peak) because storage accumulates over 24h
        total_objects_per_day = self.dau * self.actions_per_user_per_day
        return total_objects_per_day * self.storage_per_object_bytes / (1024**3)

    @property
    def storage_total_tb(self) -> float:
        """Total storage = storage/day * retention, in TB."""
        return self.storage_per_day_gb * self.retention_days / 1024

    def report(self) -> str:
        """Generates a readable report of the estimates."""
        lines = [
            f"\n{'='*60}",
            f"  ESTIMATION : {self.name}",
            f"{'='*60}",
            f"  DAU                    : {self.dau:>15,}",
            f"  Actions/user/day       : {self.actions_per_user_per_day:>15,}",
            f"  {'-'*56}",
            f"  Average QPS            : {self.qps_average:>15,.0f} req/s",
            f"  Peak QPS (x3)          : {self.qps_peak:>15,.0f} req/s",
            f"  {'-'*56}",
            f"  Bandwidth IN           : {self.bandwidth_in_mbps:>15,.1f} Mbps",
            f"  Bandwidth OUT          : {self.bandwidth_out_mbps:>15,.1f} Mbps",
            f"  {'-'*56}",
            f"  Storage/day            : {self.storage_per_day_gb:>15,.2f} GB",
            f"  Total storage ({self.retention_days}d) : {self.storage_total_tb:>12,.2f} TB",
            f"{'='*60}",
        ]
        return "\n".join(lines)


def demo_estimation():
    """Estimates the resource needs for a Twitter clone and a chat service."""
    print("\n" + "=" * 60)
    print("  SECTION 1 : Back-of-the-envelope Estimation")
    print("=" * 60)

    # Scenario 1 : Twitter-like
    twitter = SystemEstimation(
        name="Twitter-like (posts + timeline reads)",
        dau=200_000_000,
        actions_per_user_per_day=50,        # 50 timeline reads / tweets per day
        avg_request_size_bytes=500,          # Lightweight request (GET timeline)
        avg_response_size_bytes=5_000,       # 5 KB of JSON (20 tweets with metadata)
        storage_per_object_bytes=1_000,      # 1 KB per tweet (text + metadata)
        retention_days=365 * 5,              # 5 years of retention
    )

    # Scenario 2 : Chat service (WhatsApp-like)
    chat = SystemEstimation(
        name="Chat service (WhatsApp-like)",
        dau=500_000_000,
        actions_per_user_per_day=100,        # 100 messages sent/received per day
        avg_request_size_bytes=200,          # Short message
        avg_response_size_bytes=200,         # ACK + message
        storage_per_object_bytes=500,        # Message + metadata + index
        retention_days=365,                  # 1 year
    )

    print(twitter.report())
    print(chat.report())


# =============================================================================
# SECTION 2 : Latency -- Sequential vs Parallel
# =============================================================================


def simulate_api_call(service_name: str, latency_ms: int) -> dict:
    """Simulates an API call with a fixed latency.

    In real life, we would make an HTTP call. Here we simulate with sleep
    to show the impact of parallelization.
    """
    time.sleep(latency_ms / 1000)
    return {"service": service_name, "latency_ms": latency_ms, "status": "ok"}


def demo_latency():
    """Compares sequential vs parallel calls -- same work, very different time."""
    print("\n" + "=" * 60)
    print("  SECTION 2 : Latency -- Sequential vs Parallel")
    print("=" * 60)

    # Scenario: an endpoint must call 5 microservices to build a response
    services = [
        ("user-service", 50),
        ("product-service", 80),
        ("recommendation-service", 120),
        ("pricing-service", 60),
        ("inventory-service", 40),
    ]

    total_theoretical = sum(lat for _, lat in services)

    # --- Sequential ---
    start = time.perf_counter()
    results_seq = []
    for name, latency in services:
        results_seq.append(simulate_api_call(name, latency))
    elapsed_seq = (time.perf_counter() - start) * 1000

    print(f"\n  Sequential ({len(services)} calls)")
    print(f"  Sum of individual latencies      : {total_theoretical} ms")
    print(f"  Actual measured time             : {elapsed_seq:.0f} ms")

    # --- Parallel ---
    # ThreadPoolExecutor launches all the calls at the same time
    # Total time = max(latencies), not the sum
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
    print(f"\n  Parallel ({len(services)} calls)")
    print(f"  Latency of the slowest one       : {max_latency} ms")
    print(f"  Actual measured time             : {elapsed_par:.0f} ms")
    print(f"\n  Speedup                          : {elapsed_seq / elapsed_par:.1f}x")
    print(f"  Time saved                       : {elapsed_seq - elapsed_par:.0f} ms")

    # Key lesson
    print("\n  >> Lesson: In parallel, total latency = max(individual latencies)")
    print("  >> This is why modern architectures fan out then aggregate.")


# =============================================================================
# SECTION 3 : Consistency Models -- Strong vs Eventual
# =============================================================================


class StrongConsistencyStore:
    """Simulates a store with strong consistency via a global lock.

    Every write blocks all reads and writes
    until ALL replicas have been updated.
    """

    def __init__(self, num_replicas: int = 3):
        self._lock = threading.Lock()  # The lock simulates synchronization between replicas
        self._replicas = [None] * num_replicas
        self._write_count = 0
        self._stale_reads = 0
        self._total_reads = 0

    def write(self, value):
        """Write: acquires the lock, updates ALL replicas, then releases."""
        with self._lock:
            # Simulates synchronous replication time (all replicas)
            time.sleep(0.01 * len(self._replicas))
            for i in range(len(self._replicas)):
                self._replicas[i] = value
            self._write_count += 1

    def read(self) -> tuple:
        """Read: waits for the lock (guarantees reading the latest write)."""
        with self._lock:
            self._total_reads += 1
            # All replicas are identical thanks to the lock
            replica_idx = random.randint(0, len(self._replicas) - 1)
            return self._replicas[replica_idx], replica_idx


class EventualConsistencyStore:
    """Simulates a store with eventual consistency.

    A write updates ONE replica immediately.
    Propagation to the other replicas happens in the background (async).
    Reads may hit a replica that has not been updated yet = stale read.
    """

    def __init__(self, num_replicas: int = 3):
        self._replicas = [None] * num_replicas
        self._write_count = 0
        self._stale_reads = 0
        self._total_reads = 0
        self._latest_value = None

    def write(self, value):
        """Write: updates a SINGLE replica, kicks off propagation in the background."""
        self._latest_value = value
        self._replicas[0] = value  # Only the primary replica is up to date immediately
        self._write_count += 1

        # Asynchronous propagation to the other replicas (simulates network delay)
        def propagate():
            for i in range(1, len(self._replicas)):
                time.sleep(random.uniform(0.005, 0.03))  # Variable delay per replica
                self._replicas[i] = value

        threading.Thread(target=propagate, daemon=True).start()

    def read(self) -> tuple:
        """Read: picks a random replica (may be stale)."""
        self._total_reads += 1
        replica_idx = random.randint(0, len(self._replicas) - 1)
        value = self._replicas[replica_idx]
        if value != self._latest_value:
            self._stale_reads += 1
        return value, replica_idx


def demo_consistency():
    """Shows the difference between strong and eventual consistency using threads."""
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

    print(f"\n  {'Metric':<30} {'Strong':>12} {'Eventual':>12}")
    print(f"  {'-'*54}")
    print(f"  {'Writes':<30} {strong._write_count:>12} {eventual._write_count:>12}")
    print(f"  {'Total reads':<30} {strong._total_reads:>12} {eventual._total_reads:>12}")
    print(f"  {'Stale reads':<30} {strong._stale_reads:>12} {eventual._stale_reads:>12}")
    print(f"  {'Total time (ms)':<30} {elapsed_strong:>12.0f} {elapsed_eventual:>12.0f}")

    stale_pct = (
        (eventual._stale_reads / eventual._total_reads * 100)
        if eventual._total_reads > 0
        else 0
    )

    print(f"\n  >> Strong : 0 stale reads, but {elapsed_strong:.0f}ms (lock blocks the readers)")
    print(f"  >> Eventual : {eventual._stale_reads} stale reads ({stale_pct:.1f}%), but {elapsed_eventual:.0f}ms")
    print(f"  >> Tradeoff: consistency vs performance. Choose based on the business domain.")


# =============================================================================
# SECTION 4 : SLA Calculator
# =============================================================================


def sla_downtime(uptime_pct: float) -> dict:
    """Computes the allowed downtime for a given SLA.

    Args:
        uptime_pct: Uptime percentage (e.g. 99.99)

    Returns:
        Dict with downtime per year, month, week, day in human-readable format.
    """
    # Downtime fraction = 1 - uptime/100
    downtime_fraction = 1 - (uptime_pct / 100)

    # Seconds in each period
    seconds_per_year = 365.25 * 24 * 3600
    seconds_per_month = seconds_per_year / 12
    seconds_per_week = 7 * 24 * 3600
    seconds_per_day = 24 * 3600

    def format_duration(seconds: float) -> str:
        """Formats a duration in seconds into a human-readable string."""
        if seconds >= 86400:
            return f"{seconds / 86400:.2f} days"
        elif seconds >= 3600:
            return f"{seconds / 3600:.2f} hours"
        elif seconds >= 60:
            return f"{seconds / 60:.2f} minutes"
        else:
            return f"{seconds:.2f} seconds"

    return {
        "uptime": f"{uptime_pct}%",
        "nines": _count_nines(uptime_pct),
        "downtime_per_year": format_duration(downtime_fraction * seconds_per_year),
        "downtime_per_month": format_duration(downtime_fraction * seconds_per_month),
        "downtime_per_week": format_duration(downtime_fraction * seconds_per_week),
        "downtime_per_day": format_duration(downtime_fraction * seconds_per_day),
    }


def _count_nines(uptime_pct: float) -> str:
    """Counts the number of '9's in an SLA (e.g. 99.99% = 'four nines').

    Method: -log10(1 - uptime/100) gives the number of nines.
    99.9% -> -log10(0.001) = 3 -> 'three nines'
    """
    if uptime_pct >= 100:
        return "infinite (impossible)"
    nines_count = -math.log10(1 - uptime_pct / 100)
    names = {1: "one nine", 2: "two nines", 3: "three nines",
             4: "four nines", 5: "five nines", 6: "six nines"}
    rounded = round(nines_count, 1)
    # If it is an integer, use the name
    if rounded == int(rounded) and int(rounded) in names:
        return f"{names[int(rounded)]} ({nines_count:.2f})"
    return f"{nines_count:.2f} nines"


def demo_sla():
    """Displays a comparison table of common SLAs."""
    print("\n" + "=" * 60)
    print("  SECTION 4 : SLA Calculator")
    print("=" * 60)

    sla_levels = [99.0, 99.5, 99.9, 99.95, 99.99, 99.999]

    print(f"\n  {'SLA':>8}  {'Nines':<22}  {'Downtime/year':<18}  {'Downtime/month':<18}  {'Downtime/day':<18}")
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

    # Practical case: combining SLAs
    print(f"\n  {'-'*60}")
    print("  COMBINING SLAs")
    print(f"  {'-'*60}")
    print("  If your system depends on 3 services each at 99.9% :")
    combined = 0.999 ** 3
    print(f"  Combined SLA = 99.9% * 99.9% * 99.9% = {combined * 100:.4f}%")
    combined_info = sla_downtime(combined * 100)
    print(f"  Downtime/year = {combined_info['downtime_per_year']}")
    print(f"\n  >> Lesson: SLAs multiply. 3 services at 'three nines'")
    print(f"     yield a system at ~{combined * 100:.2f}% -- almost 'two nines'.")
    print(f"     This is why redundancy and fallbacks are essential.")


# =============================================================================
# SECTION 5 : Latency Numbers Every Programmer Should Know
# =============================================================================


def demo_latency_numbers():
    """Displays latency orders of magnitude with a visual scale."""
    print("\n" + "=" * 60)
    print("  SECTION 5 : Latency Numbers")
    print("=" * 60)

    # (operation, latency in nanoseconds)
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

    # Visual logarithmic scale: each '#' = one order of magnitude
    print(f"\n  {'Operation':<38} {'Latency':>12}  Scale (log)")
    print(f"  {'-'*75}")

    for op, ns in latencies:
        # Format the latency into a readable unit
        if ns < 1_000:
            lat_str = f"{ns} ns"
        elif ns < 1_000_000:
            lat_str = f"{ns / 1_000:.0f} us"
        else:
            lat_str = f"{ns / 1_000_000:.0f} ms"

        # Logarithmic bar: log10(ns) gives the scale
        bar_len = int(math.log10(max(ns, 1)) * 3)  # *3 for readability
        bar = "#" * bar_len

        print(f"  {op:<38} {lat_str:>12}  {bar}")

    print(f"\n  >> RAM vs SSD factor        : ~{49_000 / 3_000:.0f}x")
    print(f"  >> SSD vs Network factor    : ~{10_000_000 / 49_000:.0f}x")
    print(f"  >> RAM vs Network factor    : ~{10_000_000 / 3_000:.0f}x")
    print(f"\n  >> Conclusion: an in-memory cache avoids a network round trip = 3000x gain")


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Runs all the demos sequentially."""
    print("\n" + "#" * 60)
    print("#  DAY 1 -- SYSTEM DESIGN FUNDAMENTALS")
    print("#" * 60)

    demo_estimation()
    demo_latency_numbers()
    demo_latency()
    demo_consistency()
    demo_sla()

    print("\n" + "#" * 60)
    print("#  DONE -- All demos have been executed.")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
