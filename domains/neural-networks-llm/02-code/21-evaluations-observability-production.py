"""
Jour 21 — Evaluations & observability LLM en production
========================================================
Pure Python. On construit :
  1. Un eval set golden + metriques exact/semantic/rule-based
  2. Un LLM-as-judge simule avec biais (position, length, self-pref)
  3. Un pairwise comparator avec bootstrap CI
  4. Un tracker de prod metrics (cache, cost, latency, errors)
  5. Une detection de drift via sliding window

Run : python 02-code/21-evaluations-observability-production.py
"""

from __future__ import annotations
import sys, io, random, math, statistics, hashlib
from dataclasses import dataclass, field
from collections import deque

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

random.seed(21)

# ============================================================================
# 1) Eval set golden + metriques basiques
# ============================================================================

EVAL_SET = [
    ("Capitale de la France ?", "Paris"),
    ("Racine carree de 144 ?", "12"),
    ("Auteur de 1984 ?", "George Orwell"),
    ("Annee de la Revolution francaise ?", "1789"),
    ("Plus grand ocean ?", "Pacifique"),
]

def exact_match(pred, gold):
    return pred.strip().lower() == gold.strip().lower()


def contains_match(pred, gold):
    return gold.strip().lower() in pred.strip().lower()


def char_overlap(a, b):
    set_a, set_b = set(a.lower()), set(b.lower())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


# Deux "modeles" fictifs : model_a repond concis, model_b repond verbeux.
def model_a(q):
    return {
        "Capitale de la France ?": "Paris",
        "Racine carree de 144 ?": "12",
        "Auteur de 1984 ?": "Orwell",
        "Annee de la Revolution francaise ?": "1789",
        "Plus grand ocean ?": "Pacifique",
    }.get(q, "je ne sais pas")


def model_b(q):
    return {
        "Capitale de la France ?": "La capitale de la France est Paris, situee au bord de la Seine.",
        "Racine carree de 144 ?": "La racine carree de 144 est 12.",
        "Auteur de 1984 ?": "Le roman 1984 a ete ecrit par George Orwell.",
        "Annee de la Revolution francaise ?": "La Revolution francaise a commence en 1789.",
        "Plus grand ocean ?": "L'ocean Pacifique est le plus grand ocean du monde.",
    }.get(q, "Je ne connais pas la reponse.")


print("=" * 70)
print("1) Eval set golden — 3 metriques differentes")
print("=" * 70)

for name, model in [("model_a (concis)", model_a), ("model_b (verbeux)", model_b)]:
    em = sum(exact_match(model(q), g) for q, g in EVAL_SET) / len(EVAL_SET)
    cm = sum(contains_match(model(q), g) for q, g in EVAL_SET) / len(EVAL_SET)
    ov = sum(char_overlap(model(q), g) for q, g in EVAL_SET) / len(EVAL_SET)
    print(f"  {name:<22}  exact={em:.2f}  contains={cm:.2f}  char_overlap={ov:.2f}")
print("  → 'exact match' est trop strict pour model_b qui donne la bonne")
print("    reponse mais habillee. 'contains' est plus juste pour cette tache.\n")


# ============================================================================
# 2) LLM-as-judge simule avec biais
# ============================================================================

@dataclass
class Judgement:
    score_groundedness: int   # 0-5
    score_relevance: int
    score_fluency: int


def llm_judge(query, answer, context=None, position=0, is_self=False,
              calibrated=True):
    """
    Simule un juge LLM. Par defaut, on simule un modele calibre qui note
    en moyenne correctement mais :
      - avec un position bias (reponse en 1ere position +0.3)
      - un length bias (plus c'est long, +0.5)
      - un self-preference bias (+0.4 si on lui demande de juger sa propre sortie)
    """
    base = 3 + (1 if len(answer) > 30 else 0)  # plausible baseline
    if not calibrated:
        base += random.gauss(0, 1)
    bias = 0
    if position == 0:
        bias += 0.3
    bias += min(len(answer) / 100, 0.5)
    if is_self:
        bias += 0.4
    s = max(0, min(5, round(base + bias)))
    return Judgement(
        score_groundedness=max(0, min(5, s + random.randint(-1, 1))),
        score_relevance=max(0, min(5, s + random.randint(-1, 1))),
        score_fluency=max(0, min(5, s + random.randint(-1, 1))),
    )


print("=" * 70)
print("2) LLM-as-judge : demonstration des biais")
print("=" * 70)

# Meme reponse, positions differentes : observe le position bias
ans = "Paris est la capitale de la France."
j1 = llm_judge("Capitale ?", ans, position=0)
j2 = llm_judge("Capitale ?", ans, position=1)
print(f"  Meme reponse, position 0: relevance={j1.score_relevance}  "
      f"position 1: relevance={j2.score_relevance}  (position bias)")

# Length bias : reponse courte vs longue, meme qualite factuelle
j_short = llm_judge("Capitale ?", "Paris", position=0)
j_long = llm_judge("Capitale ?",
                   "La capitale de la France est Paris, ville lumiere reputee "
                   "mondialement pour son architecture et sa culture riche.",
                   position=0)
print(f"  Court: {j_short.score_fluency}  Long: {j_long.score_fluency}  (length bias)")

# Self-preference : juger ta propre sortie vs une autre
j_self = llm_judge("Capitale ?", ans, position=0, is_self=True)
j_other = llm_judge("Capitale ?", ans, position=0, is_self=False)
print(f"  Self: {j_self.score_groundedness}  Other: {j_other.score_groundedness}  (self-pref bias)\n")


# ============================================================================
# 3) Pairwise + bootstrap CI
# ============================================================================


def pairwise_winner(q, a, b, judge):
    """Le juge choisit A, B ou Tie. On evalue avec position-swap pour enlever
    le position bias : run judge(A,B) et judge(B,A), moyenne."""
    j1 = judge(q, a, position=0)
    j2 = judge(q, b, position=1)
    s_a = j1.score_relevance + j1.score_fluency
    s_b = j2.score_relevance + j2.score_fluency
    j3 = judge(q, b, position=0)
    j4 = judge(q, a, position=1)
    s_b += j3.score_relevance + j3.score_fluency
    s_a += j4.score_relevance + j4.score_fluency
    if abs(s_a - s_b) < 1:
        return "tie"
    return "a" if s_a > s_b else "b"


# Comparer model_a vs model_b sur l'eval set avec position-swap
queries = [q for q, _ in EVAL_SET]
results = []
for q in queries:
    r = pairwise_winner(q, model_a(q), model_b(q), llm_judge)
    results.append(r)


def bootstrap_winrate(results, n_iter=1000):
    wins = []
    for _ in range(n_iter):
        sample = [random.choice(results) for _ in results]
        wr = sum(1 for r in sample if r == "b") / len(sample)
        wins.append(wr)
    wins.sort()
    return wins[int(0.025 * n_iter)], wins[int(0.975 * n_iter)]


lo, hi = bootstrap_winrate(results)
print("=" * 70)
print("3) Pairwise avec position-swap + bootstrap CI")
print("=" * 70)
print(f"  Results: {results}")
print(f"  Win rate model_b: {sum(1 for r in results if r == 'b') / len(results):.2f}")
print(f"  95% CI bootstrap: [{lo:.2f}, {hi:.2f}]")
print("  → avec seulement 5 queries, l'IC est TRES large — on ne peut pas")
print("    conclure qu'un modele est meilleur. Il faut >100 exemples.\n")


# ============================================================================
# 4) Production tracker : cache, cost, latency, errors
# ============================================================================


@dataclass
class CallLog:
    timestamp_s: float
    endpoint: str
    input_tokens: int
    cache_read_tokens: int
    output_tokens: int
    latency_ms: float
    error: bool
    judge_score: float | None = None


class ProdMonitor:
    def __init__(self, window: int = 100):
        self.logs: deque[CallLog] = deque(maxlen=window)

    def log(self, l: CallLog):
        self.logs.append(l)

    def metrics(self):
        if not self.logs:
            return {}
        total = len(self.logs)
        errors = sum(1 for l in self.logs if l.error)
        lat = sorted(l.latency_ms for l in self.logs if not l.error)
        cache_total = sum(l.cache_read_tokens for l in self.logs)
        input_total = sum(l.input_tokens for l in self.logs)
        hit_rate = cache_total / max(cache_total + input_total, 1)
        scores = [l.judge_score for l in self.logs if l.judge_score is not None]
        return {
            "n": total,
            "error_rate": errors / total,
            "p50_latency_ms": lat[len(lat) // 2] if lat else 0,
            "p95_latency_ms": lat[int(0.95 * len(lat))] if lat else 0,
            "cache_hit_rate": hit_rate,
            "judge_mean": (sum(scores) / len(scores)) if scores else None,
        }

    def alerts(self):
        m = self.metrics()
        alerts = []
        if m.get("error_rate", 0) > 0.02:
            alerts.append(f"error_rate high: {m['error_rate']:.2%}")
        if m.get("p95_latency_ms", 0) > 5000:
            alerts.append(f"p95 latency > 5s: {m['p95_latency_ms']:.0f}ms")
        if m.get("cache_hit_rate", 1) < 0.4:
            alerts.append(f"cache hit rate low: {m['cache_hit_rate']:.1%}")
        if m.get("judge_mean") is not None and m["judge_mean"] < 3.0:
            alerts.append(f"judge_mean drift: {m['judge_mean']:.2f}")
        return alerts


# Simuler 300 appels avec un incident au milieu (cache drop + judge drop)
mon = ProdMonitor(window=200)
for i in range(300):
    incident = 120 <= i < 180  # fenetre d'incident
    cache_ratio = 0.15 if incident else 0.75
    input_tok = 2000
    log = CallLog(
        timestamp_s=i * 1.0,
        endpoint="/chat",
        input_tokens=int(input_tok * (1 - cache_ratio)),
        cache_read_tokens=int(input_tok * cache_ratio),
        output_tokens=400,
        latency_ms=random.gauss(1000 if not incident else 3500, 200),
        error=random.random() < (0.05 if incident else 0.005),
        judge_score=random.gauss(2.5 if incident else 4.2, 0.3),
    )
    mon.log(log)

print("=" * 70)
print("4) Production monitor — detection d'incident via sliding window")
print("=" * 70)
m = mon.metrics()
for k, v in m.items():
    print(f"  {k:<20} : {v}")
print(f"  ALERTS: {mon.alerts()}")
print("""
  → la fenetre finale montre un mix avant/apres incident. En prod,
    on garde plusieurs fenetres (5min/1h/24h) et on alerte des que les
    seuils sont depasses sur la courte.

Stack reelle 2026 pour observability LLM :
  - Langfuse / Braintrust / LangSmith / Phoenix / Helicone
  - OpenTelemetry pour lier aux traces services
  - Grafana dashboard custom sur le store
  - Offline eval CI : run eval set complet a chaque PR qui change prompt
  - Shadow eval online : 1-5% du traffic re-judge en background
""")
