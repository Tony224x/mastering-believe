"""
Jour 9 -- Inference at scale : simulateur de dynamic batching.

Usage:
    python 09-inference-at-scale.py

On simule un serveur d'inference avec :
  - une queue d'entree
  - un batcher qui flush sur max_batch_size OU max_wait
  - un "GPU" qui traite un batch avec une latence depend du batch size
  - des metriques par requete et agregees (throughput, p50, p99, GPU util)

On compare 3 configurations : no batching, naive static batching, dynamic batching.
Le but est de montrer le tradeoff throughput vs latency.
"""

import time
import random
import heapq
from dataclasses import dataclass, field
from collections import deque
from typing import Optional

SEPARATOR = "=" * 70


# =============================================================================
# SECTION 1 : Modele de latence du "GPU"
# =============================================================================


def gpu_process_time(batch_size: int, base_ms: float = 40.0, per_item_ms: float = 2.0) -> float:
    """Simule le temps que met un GPU a traiter un batch.

    Modele simple : latence base (setup, attention matmul) + cost per item.
    En realite la fonction est sous-lineaire : doubler la batch ne double pas
    le temps, on amortit les couts fixes.
    """
    return base_ms + per_item_ms * batch_size


# =============================================================================
# SECTION 2 : Request / Response data classes
# =============================================================================


@dataclass(order=True)
class Request:
    """Une requete entrante. Ordered by arrival_time for heap usage."""

    arrival_time: float
    req_id: int = field(compare=False)
    payload: str = field(compare=False, default="")
    dispatched_time: Optional[float] = field(compare=False, default=None)
    completed_time: Optional[float] = field(compare=False, default=None)

    @property
    def wait_time_ms(self) -> float:
        if self.dispatched_time is None:
            return 0.0
        return (self.dispatched_time - self.arrival_time) * 1000

    @property
    def total_latency_ms(self) -> float:
        if self.completed_time is None:
            return 0.0
        return (self.completed_time - self.arrival_time) * 1000


# =============================================================================
# SECTION 3 : Traffic generator (Poisson arrivals)
# =============================================================================


def generate_traffic(n_requests: int, mean_iat_ms: float, seed: int = 7) -> list[Request]:
    """Genere n requetes avec des inter-arrival times exponentiels.

    mean_iat_ms petit -> forte charge. Inverse = rate (req/s).
    """
    rng = random.Random(seed)
    requests = []
    t = 0.0
    for i in range(n_requests):
        iat = rng.expovariate(1 / (mean_iat_ms / 1000))  # seconds
        t += iat
        requests.append(Request(arrival_time=t, req_id=i))
    return requests


# =============================================================================
# SECTION 4 : Trois strategies de serving
# =============================================================================


def serve_no_batching(requests: list[Request]) -> dict:
    """Strategy 1 : Une requete a la fois (batch de 1).

    Le GPU est tres sous-utilise : cout fixe paye pour chaque requete.
    """
    gpu_free_at = 0.0
    total_gpu_busy = 0.0
    for r in sorted(requests, key=lambda x: x.arrival_time):
        start = max(gpu_free_at, r.arrival_time)
        r.dispatched_time = start
        proc_s = gpu_process_time(1) / 1000
        r.completed_time = start + proc_s
        gpu_free_at = r.completed_time
        total_gpu_busy += proc_s
    return summarize(requests, gpu_busy_s=total_gpu_busy, label="no_batching")


def serve_static_batching(requests: list[Request], batch_size: int = 8) -> dict:
    """Strategy 2 : static batching.

    On attend d'avoir exactement batch_size requetes avant de flusher.
    Si moins, on attend indefiniment -> les dernieres requetes du run
    risquent de ne jamais etre traitees (on les force a la fin ici).
    """
    queue = sorted(requests, key=lambda x: x.arrival_time)
    idx = 0
    gpu_free_at = 0.0
    total_gpu_busy = 0.0
    while idx < len(queue):
        # Peek: arrival time of the batch_size-th request (full batch)
        end_idx = min(idx + batch_size, len(queue))
        batch = queue[idx:end_idx]
        # The batch can only start once the last request has arrived AND the gpu is free
        ready_at = max(batch[-1].arrival_time, gpu_free_at)
        bs = len(batch)
        proc_s = gpu_process_time(bs) / 1000
        for r in batch:
            r.dispatched_time = ready_at
            r.completed_time = ready_at + proc_s
        gpu_free_at = ready_at + proc_s
        total_gpu_busy += proc_s
        idx = end_idx
    return summarize(requests, gpu_busy_s=total_gpu_busy, label=f"static_bs={batch_size}")


def serve_dynamic_batching(
    requests: list[Request],
    max_batch_size: int = 16,
    max_wait_ms: float = 20.0,
) -> dict:
    """Strategy 3 : dynamic batching.

    Flush le batch quand :
      - max_batch_size atteint
      - OU max_wait depasse depuis la premiere requete du batch

    Algorithme simple et correct : on simule un scheduler qui, a chaque
    iteration, decide QUAND ouvrir le prochain batch et QUOI mettre dedans.
    Une iteration = un batch. On termine quand toutes les requetes sont
    dispatchees.
    """
    sorted_reqs = sorted(requests, key=lambda x: x.arrival_time)
    n = len(sorted_reqs)
    idx = 0  # index of next request to consider
    gpu_free_at = 0.0
    total_gpu_busy = 0.0
    max_wait_s = max_wait_ms / 1000

    while idx < n:
        # The GPU is free at gpu_free_at. The first request in the next batch
        # is sorted_reqs[idx]. Its effective "start of batching" is
        # max(its arrival, gpu_free_at).
        first = sorted_reqs[idx]
        batch_open_at = max(first.arrival_time, gpu_free_at)
        # Deadline : we must flush at batch_open_at + max_wait_s, or when the
        # batch hits max_batch_size, whichever comes first.
        deadline = batch_open_at + max_wait_s

        # Collect as many requests as possible up to max_batch_size that
        # arrive <= deadline.
        batch: list[Request] = [first]
        j = idx + 1
        while j < n and len(batch) < max_batch_size and sorted_reqs[j].arrival_time <= deadline:
            batch.append(sorted_reqs[j])
            j += 1

        # Determine actual start: the moment we have all batch members AND GPU is free.
        # That is max(last_batch_arrival, gpu_free_at).
        start = max(batch[-1].arrival_time, gpu_free_at)
        # If we hit max_batch_size before deadline, we could flush even earlier,
        # which is what 'start' already reflects.
        bs = len(batch)
        proc_s = gpu_process_time(bs) / 1000
        for r in batch:
            r.dispatched_time = start
            r.completed_time = start + proc_s
        gpu_free_at = start + proc_s
        total_gpu_busy += proc_s
        idx = j

    return summarize(
        requests,
        gpu_busy_s=total_gpu_busy,
        label=f"dynamic_bs<={max_batch_size}_wait={max_wait_ms}ms",
    )


# =============================================================================
# SECTION 5 : Metrics summary
# =============================================================================


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = min(int(len(s) * p), len(s) - 1)
    return s[k]


def summarize(requests: list[Request], gpu_busy_s: float, label: str) -> dict:
    latencies = [r.total_latency_ms for r in requests if r.completed_time is not None]
    completed = [r for r in requests if r.completed_time is not None]
    start = min(r.arrival_time for r in requests)
    end = max((r.completed_time or 0) for r in completed)
    wallclock = max(end - start, 1e-6)
    return {
        "label": label,
        "n": len(completed),
        "p50_ms": round(percentile(latencies, 0.50), 1),
        "p95_ms": round(percentile(latencies, 0.95), 1),
        "p99_ms": round(percentile(latencies, 0.99), 1),
        "max_ms": round(max(latencies) if latencies else 0, 1),
        "throughput_rps": round(len(completed) / wallclock, 1),
        "gpu_utilization": round(gpu_busy_s / wallclock, 3),
    }


# =============================================================================
# SECTION 6 : Demo -- compare strategies
# =============================================================================


def demo() -> None:
    print(SEPARATOR)
    print("DYNAMIC BATCHING SIMULATOR -- throughput vs latency tradeoff")
    print(SEPARATOR)
    print("Traffic: 400 requests, Poisson arrivals, 10 ms mean inter-arrival (~100 rps).")
    n = 400
    mean_iat_ms = 10
    reqs1 = generate_traffic(n, mean_iat_ms, seed=1)
    reqs2 = generate_traffic(n, mean_iat_ms, seed=1)
    reqs3 = generate_traffic(n, mean_iat_ms, seed=1)

    s1 = serve_no_batching(reqs1)
    s2 = serve_static_batching(reqs2, batch_size=8)
    s3 = serve_dynamic_batching(reqs3, max_batch_size=16, max_wait_ms=20)

    header = f"{'strategy':<32} {'p50':>8} {'p95':>8} {'p99':>8} {'rps':>8} {'gpu':>8}"
    print("\n" + header)
    print("-" * len(header))
    for s in (s1, s2, s3):
        print(
            f"{s['label']:<32} {s['p50_ms']:>8} {s['p95_ms']:>8} {s['p99_ms']:>8} "
            f"{s['throughput_rps']:>8} {s['gpu_utilization']:>8}"
        )

    print("\nLessons:")
    print("  - no_batching : lowest latency but GPU underutilized, throughput capped")
    print("  - static_bs=8 : better throughput but tail latency spikes (wait for full batch)")
    print("  - dynamic     : balances both -- flush on batch full OR wait deadline")
    print("  - Tune max_batch_size and max_wait to match your SLA.")


if __name__ == "__main__":
    demo()
