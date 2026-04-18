"""
Jour 20 — Distillation & donnees synthetiques : pipeline minimal
==================================================================
Pure Python + simulation. On construit un pipeline complet de distillation
sur une tache jouet de classification (sentiment) pour voir chaque etape
exister :
  1. Seed prompts humains
  2. Generation synthetique par un "teacher" simule
  3. Filtrage rule-based + LLM-judge simule + dedup
  4. Mesure de data contamination (presence d'eval dans synth)
  5. Training d'un "student" (regression logistique a la main) sur les
     features bag-of-words du dataset filtre
  6. Evaluation sur un eval set hold-out vs. baseline non-distillee

Run : python 02-code/20-distillation-synthetic-data-slms.py
"""

from __future__ import annotations
import sys, io, random, math, re, hashlib
from collections import Counter, defaultdict

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

random.seed(20)

# ============================================================================
# 1) Seed prompts humains (varies) + eval set hold-out
# ============================================================================

SEEDS = [
    "Ce film etait incroyable, j'ai adore",
    "Produit defectueux, tres decu",
    "Service client excellent, je recommande",
    "Totalement inutile et cher",
    "Bonne experience dans l'ensemble",
    "Mauvais rapport qualite prix",
]

# Eval set : ne doit JAMAIS etre dans le train ni genere par le teacher.
EVAL_SET = [
    ("Commande livree rapidement, tres satisfaite", 1),
    ("Interface lente et buggee, frustrant", 0),
    ("Cette recette est delicieuse, a refaire", 1),
    ("Installation compliquee, notice incomprehensible", 0),
    ("L'hotel etait parfait, je reviendrai", 1),
    ("Accueil glacial, jamais plus", 0),
    ("Materiel solide, bonne finition", 1),
    ("Produit casse a l'arrivee", 0),
    ("J'ai passe un moment exceptionnel", 1),
    ("Decu par la qualite apres 2 semaines", 0),
]


# ============================================================================
# 2) Teacher simule : genere des exemples etiquetes
# ============================================================================

POS_SNIPPETS = ["excellent", "parfait", "incroyable", "super", "adore",
                "tres bon", "recommande", "ravi", "magnifique", "genial"]
NEG_SNIPPETS = ["nul", "decu", "mauvais", "catastrophique", "horrible",
                "deteste", "ne recommande pas", "fuir", "mediocre", "terrible"]


def teacher_generate(seed: str, n: int = 10) -> list[tuple[str, int]]:
    """
    Simule : un gros modele frontier reformule le seed en n variantes.
    On genere des phrases plausibles avec un label coherent. 5% de bruit
    (erreur teacher) : le teacher n'est pas parfait.
    """
    out = []
    is_positive = any(w in seed.lower() for w in ["adore", "excellent", "bon", "recommande"])
    for _ in range(n):
        if is_positive:
            words = random.sample(POS_SNIPPETS, k=random.randint(1, 3))
            text = f"Experience avec le produit : {' '.join(words)}."
            label = 1
        else:
            words = random.sample(NEG_SNIPPETS, k=random.randint(1, 3))
            text = f"Mon retour : {' '.join(words)}."
            label = 0
        # Bruit teacher : parfois label retourne
        if random.random() < 0.05:
            label = 1 - label
        out.append((text, label))
    return out


# ============================================================================
# 3) Filtrage : rule-based, LLM-judge simule, dedup
# ============================================================================

def rule_based_ok(text: str) -> bool:
    return 10 < len(text) < 500 and text.count(".") >= 1


def llm_judge(text: str, label: int) -> bool:
    """
    Simule un juge LLM qui verifie la coherence texte/label. On detecte
    un conflit evident (mots positifs + label 0, ou inverse). C'est ce
    qui filtre les erreurs du teacher.
    """
    has_pos = any(w in text.lower() for w in POS_SNIPPETS)
    has_neg = any(w in text.lower() for w in NEG_SNIPPETS)
    if label == 1 and has_neg and not has_pos:
        return False
    if label == 0 and has_pos and not has_neg:
        return False
    return True


def minhash_dedup(examples: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Dedup simple par hash de set de mots (approx minhash)."""
    seen = set()
    out = []
    for text, label in examples:
        key = tuple(sorted(set(re.findall(r"[a-z]+", text.lower()))))
        h = hashlib.md5(str(key).encode()).hexdigest()[:12]
        if h not in seen:
            seen.add(h)
            out.append((text, label))
    return out


def contamination_check(synth: list[tuple[str, int]],
                        eval_set: list[tuple[str, int]]) -> int:
    """Combien d'exemples synth sont quasi-identiques a des eval ?"""
    eval_words = [set(re.findall(r"[a-z]+", t.lower())) for t, _ in eval_set]
    count = 0
    for text, _ in synth:
        s = set(re.findall(r"[a-z]+", text.lower()))
        for e in eval_words:
            if s and len(s & e) / len(s | e) > 0.6:
                count += 1
                break
    return count


# ============================================================================
# 4) Student simple : regression logistique bag-of-words from scratch
# ============================================================================


class BagOfWordsLogReg:
    def __init__(self, lr=0.1, epochs=50):
        self.lr, self.epochs = lr, epochs
        self.w: dict[str, float] = defaultdict(float)
        self.b = 0.0

    @staticmethod
    def _feat(text: str) -> Counter:
        return Counter(re.findall(r"[a-z]+", text.lower()))

    def _predict_proba(self, x: Counter) -> float:
        z = self.b + sum(self.w[k] * v for k, v in x.items())
        return 1 / (1 + math.exp(-max(min(z, 30), -30)))

    def fit(self, data: list[tuple[str, int]]):
        for _ in range(self.epochs):
            random.shuffle(data)
            for text, y in data:
                x = self._feat(text)
                p = self._predict_proba(x)
                grad = (p - y)
                for k, v in x.items():
                    self.w[k] -= self.lr * grad * v
                self.b -= self.lr * grad

    def evaluate(self, data: list[tuple[str, int]]) -> float:
        hits = sum(1 for t, y in data if (self._predict_proba(self._feat(t)) > 0.5) == bool(y))
        return hits / len(data)


# ============================================================================
# 5) Pipeline complet
# ============================================================================

print("=" * 70)
print("PIPELINE DISTILLATION — dataset synthetique + SFT student")
print("=" * 70)

# Baseline : un student entraine UNIQUEMENT sur les seeds (6 exemples)
seeds_labeled = [
    (s, 1 if any(w in s.lower() for w in ["adore", "excellent", "bon", "recommande"]) else 0)
    for s in SEEDS
]
baseline = BagOfWordsLogReg()
baseline.fit(list(seeds_labeled))
print(f"\n  Baseline student (6 seeds only)    eval acc = {baseline.evaluate(EVAL_SET):.2f}")

# Distillation : generer 60 synth examples (10 par seed), filtrer, entrainer
print("\n  Generating synthetic data from teacher...")
synth = []
for s in SEEDS:
    synth.extend(teacher_generate(s, n=10))
print(f"    generated         : {len(synth)} examples")

# Filtrage en cascade
before = len(synth)
synth = [(t, l) for t, l in synth if rule_based_ok(t)]
print(f"    after rule-based  : {len(synth)}  (rejected {before - len(synth)})")
before = len(synth)
synth = [(t, l) for t, l in synth if llm_judge(t, l)]
print(f"    after LLM-judge   : {len(synth)}  (rejected {before - len(synth)})")
before = len(synth)
synth = minhash_dedup(synth)
print(f"    after dedup       : {len(synth)}  (rejected {before - len(synth)})")

# Verification contamination avec l'eval set
contam = contamination_check(synth, EVAL_SET)
print(f"    contamination     : {contam} synth examples matching eval")

# Entrainement student sur seeds + synth
student = BagOfWordsLogReg()
student.fit(list(seeds_labeled) + synth)
acc = student.evaluate(EVAL_SET)
print(f"\n  Distilled student                  eval acc = {acc:.2f}")

# Ablation : et sans filtrage ?
all_gen = []
for s in SEEDS:
    all_gen.extend(teacher_generate(s, n=10))
student_no_filter = BagOfWordsLogReg()
student_no_filter.fit(list(seeds_labeled) + all_gen)
acc_nf = student_no_filter.evaluate(EVAL_SET)
print(f"  Distilled WITHOUT filtering        eval acc = {acc_nf:.2f}")

print("""
Lecons :
  - Baseline 6-shot ≈ mauvais, le student n'a pas vu assez de vocab.
  - Distillation avec filtrage : accuracy monte vs baseline.
  - Sans filtrage : plus de donnees brutes peut parfois aider a court terme,
    mais a plus grande echelle le bruit teacher (5%) s'accumule et plombe
    l'accuracy. Ce qui compte : la qualite et la diversite, pas le volume.
  - Les filtres rule-based + LLM judge + dedup + contamination check sont
    la vraie valeur d'un pipeline de distillation.

Stack reelle 2026 pour distillation :
  - Seeds : 500-5000 humains bien divers
  - Teacher : Claude 4.5 Opus ou DeepSeek R1
  - Judge  : Claude Haiku ou Qwen 3 32B as-a-judge
  - SFT    : TRL / Axolotl sur Gemma 3 / Qwen 3 / Llama 3.3
  - Dedup  : MinHash LSH sur les prompts generes
""")
