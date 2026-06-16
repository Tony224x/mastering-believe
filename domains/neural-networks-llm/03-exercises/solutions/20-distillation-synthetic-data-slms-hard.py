"""
Solutions HARD — Jour 20 : Distillation, donnees synthetiques & SLMs
====================================================================
Exercices 7, 8, 9 (hard). PUR PYTHON STDLIB (random / math / re / hashlib /
collections), comme 02-code/20-distillation-synthetic-data-slms.py — PAS de
numpy, pour rester fidele au module et tourner sans dependance.

Chaque etape non triviale est commentee avec le POURQUOI. Le fichier est
auto-verifiant : il se termine par des assertions qui echouent si une
propriete pedagogique attendue n'est plus vraie.

Run: python3 03-exercises/solutions/20-distillation-synthetic-data-slms-hard.py
"""

from __future__ import annotations
import sys
import io
import random
import math
import re
import hashlib
from collections import Counter, defaultdict

# Garde UTF-8 : certains terminaux Windows/CI sortent en cp1252 et plantent
# sur les accents francais. On force un wrapper UTF-8 tolerant.
if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

random.seed(20)  # reproductibilite : memes tirages que 02-code


# ============================================================================
# Petites primitives numeriques A LA MAIN (pas de numpy)
# ============================================================================

def softmax(logits: list[float], T: float = 1.0) -> list[float]:
    """Softmax numeriquement stable a temperature T.

    POURQUOI le -max : exp(grand) deborde en float ; soustraire le max ne
    change pas le resultat (invariance par translation du softmax) mais
    garde les exposants <= 0.
    POURQUOI /T : la temperature lisse (T>1) ou pique (T<1) la distribution.
    """
    scaled = [z / T for z in logits]
    m = max(scaled)
    exps = [math.exp(s - m) for s in scaled]
    Z = sum(exps)
    return [e / Z for e in exps]


def cross_entropy(probs: list[float], target_idx: int) -> float:
    """-log p[target] : la loss hard-label classique."""
    return -math.log(probs[target_idx] + 1e-12)


def kl_divergence(p: list[float], q: list[float]) -> float:
    """KL(p || q) = sum_i p_i log(p_i / q_i). >= 0 (Gibbs), = 0 ssi p == q.

    En distillation p = teacher (cible fixe), q = student (on l'entraine
    a copier p). On l'utilise comme metrique ET comme loss.
    """
    s = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            s += pi * math.log(pi / (qi + 1e-12))
    return max(s, 0.0)  # clamp : le bruit float peut donner -1e-17


def argsort_desc(values: list[float]) -> list[int]:
    """Indices tries par valeur decroissante (pour raisonner sur les rangs)."""
    return sorted(range(len(values)), key=lambda i: values[i], reverse=True)


# ============================================================================
# EXERCISE 7 — Distillation de logits (Hinton 2015) : dark knowledge & T
# ============================================================================
# But : prouver que distiller les SOFT targets du teacher (T>1) transmet la
# distribution complete (dark knowledge : l'ordre relatif des perdantes), ce
# que le hard label one-hot efface. On entraine un student lineaire a la main
# et on mesure, sur un hold-out, (a) la fidelite a la distribution teacher
# (KL) et (b) la "dark-knowledge recovery" (le student range-t-il la SOEUR de
# la bonne classe devant une classe lointaine ?).
#
# Cle du design : (1) le teacher est DELIBEREMENT peu confiant (petits gaps de
# logits) -> sa distribution est riche en info ; (2) le hard student n'est
# entraine que sur l'argmax BRUITE -> sur les exemples ambigus son label est
# faux, donc il n'a aucun signal fiable sur l'ordre des perdantes.

print("=" * 70)
print("EXERCISE 7 : distillation de logits — dark knowledge & temperature")
print("=" * 70)

K = 4          # classes : 0<->1 soeurs (chat/tigre), 2<->3 soeurs (camion/voiture)
D = 8          # dimension des features
H = 12         # taille cachee du teacher (NON-lineaire)
N_TRAIN = 600
N_TEST = 300
SIBLING = {0: 1, 1: 0, 2: 3, 3: 2}  # paires de classes proches

# --- Teacher NON-LINEAIRE (MLP a 1 couche cachee) -------------------------
# POURQUOI non-lineaire alors que le student est lineaire : il FAUT un ecart
# de capacite. Si teacher et student etaient tous deux lineaires, le student
# copierait le teacher exactement (KL -> 0 a toute temperature) et la
# temperature n'aurait aucun effet -> pas de courbe en cloche, demonstration
# vide. Avec un teacher non-lineaire, le student lineaire ne peut PAS coller
# parfaitement : la temperature devient un vrai compromis (T=1 sur-pondere
# l'argmax inaccessible ; T tres grand aplatit tout), d'ou une cloche.
def make_teacher():
    rng = random.Random(7)
    W1 = [[rng.gauss(0, 1.0) for _ in range(D)] for _ in range(H)]   # (H, D)
    b1 = [rng.gauss(0, 0.3) for _ in range(H)]
    base = [[rng.gauss(0, 1.0) for _ in range(H)] for _ in range(K)]  # (K, H)
    W2 = [[0.0] * H for _ in range(K)]
    for c in range(K):                 # structure de similarite : 70% propre +
        s = SIBLING[c]                 # 30% de la soeur -> la soeur sort 2e.
        for j in range(H):
            W2[c][j] = 0.7 * base[c][j] + 0.3 * base[s][j]
    return W1, b1, W2

T_W1, T_B1, T_W2 = make_teacher()
# Facteur < 1 : ECRASE les logits -> teacher peu confiant -> distribution
# douce et informative (dark knowledge). Logits enormes => softmax quasi
# one-hot => rien a distiller.
TEACHER_CONFIDENCE = 0.55

def teacher_logits(x: list[float]) -> list[float]:
    h = [max(0.0, T_B1[i] + sum(T_W1[i][j] * x[j] for j in range(D))) for i in range(H)]  # ReLU
    return [TEACHER_CONFIDENCE * sum(T_W2[c][i] * h[i] for i in range(H)) for c in range(K)]

def sample_x(rng: random.Random, true_class: int) -> list[float]:
    """Feature gaussienne centree sur un prototype FAIBLE (signal/bruit bas)
    pour creer des exemples AMBIGUS ou le teacher hesite -> dark knowledge."""
    proto = [(1.1 if (j % K) == true_class else 0.0) for j in range(D)]
    return [proto[j] + rng.gauss(0, 1.0) for j in range(D)]

def build_dataset(n: int, seed: int):
    rng = random.Random(seed)
    data = []
    for _ in range(n):
        y_true = rng.randrange(K)
        x = sample_x(rng, y_true)
        zt = teacher_logits(x)
        hard = max(range(K), key=lambda c: zt[c])  # argmax teacher = pseudo-label
        # Bruit d'etiquetage : sur les exemples ambigus l'argmax est peu fiable
        # (le teacher n'est pas parfait — cf. bruit du 02-code). 15% ici pour
        # que le hard label perde vraiment de l'info que le soft conserve.
        if rng.random() < 0.15:
            hard = rng.randrange(K)
        data.append((x, zt, hard))
    return data

train = build_dataset(N_TRAIN, seed=1)
test = build_dataset(N_TEST, seed=2)


class LinearStudent:
    """logits = W x + b, softmax, SGD. Pas de numpy : listes + boucles."""

    def __init__(self, K: int, D: int, seed: int = 0):
        rng = random.Random(seed)
        # Memes inits pour hard et soft (meme seed) : seul le signal differe,
        # donc la comparaison est honnete.
        self.W = [[rng.gauss(0, 0.01) for _ in range(D)] for _ in range(K)]
        self.b = [0.0 for _ in range(K)]
        self.K, self.D = K, D

    def logits(self, x):
        return [self.b[c] + sum(self.W[c][j] * x[j] for j in range(self.D))
                for c in range(self.K)]

    def proba(self, x, T=1.0):
        return softmax(self.logits(x), T=T)

    def _apply_grad(self, x, grad_logits, lr):
        # dL/dW[c][j] = grad_logits[c] * x[j] (regle de la chaine softmax).
        for c in range(self.K):
            g = grad_logits[c]
            self.b[c] -= lr * g
            row = self.W[c]
            for j in range(self.D):
                row[j] -= lr * g * x[j]

    def step_hard(self, x, y, lr):
        """Gradient de la cross-entropy hard : (p - onehot(y))."""
        p = self.proba(x, T=1.0)
        grad = [p[c] - (1.0 if c == y else 0.0) for c in range(self.K)]
        self._apply_grad(x, grad, lr)

    def step_soft(self, x, teacher_soft, T, lr):
        """Gradient de T^2 * KL(teacher_soft || student_soft).

        d/dz_c de KL(t || softmax(z/T)) = (1/T)(q_c - t_c), q = softmax(z/T).
        On multiplie la loss par T^2 (Hinton) pour compenser le 1/T^2 du
        gradient -> facteur NET T sur le gradient. Sans ca, a T grand le
        signal s'evanouirait.
        """
        q = self.proba(x, T=T)
        grad = [T * (q[c] - teacher_soft[c]) for c in range(self.K)]
        self._apply_grad(x, grad, lr)


def train_hard(seed_init, epochs=80, lr=0.15):
    st = LinearStudent(K, D, seed=seed_init)
    idx = list(range(len(train)))
    for _ in range(epochs):
        random.shuffle(idx)
        for i in idx:
            x, _zt, hard = train[i]
            st.step_hard(x, hard, lr)
    return st


def mean_teacher_student_kl(st, T, data):
    tot = sum(kl_divergence(softmax(zt, T=T), st.proba(x, T=T)) for x, zt, _ in data)
    return tot / len(data)


def train_soft(seed_init, T, epochs=80, lr=0.15):
    st = LinearStudent(K, D, seed=seed_init)
    idx = list(range(len(train)))
    kl_curve = []
    for _ in range(epochs):
        kl_curve.append(mean_teacher_student_kl(st, T, train))  # avant l'epoch
        random.shuffle(idx)
        for i in idx:
            x, zt, _hard = train[i]
            st.step_soft(x, softmax(zt, T=T), T, lr)
    kl_curve.append(mean_teacher_student_kl(st, T, train))
    return st, kl_curve


def accuracy(st, data):
    """Accuracy vs argmax teacher (la vraie cible, sans le bruit d'etiquetage)."""
    hits = sum(
        max(range(K), key=lambda c: st.logits(x)[c])
        == max(range(K), key=lambda c: zt[c])
        for x, zt, _ in data)
    return hits / len(data)


def dark_knowledge_recovery(st, data):
    """Fraction d'exemples ou le student met la SOEUR (2e du teacher) devant
    une classe lointaine dans son PROPRE classement (a T=1).

    POURQUOI : c'est la signature de la dark knowledge. Le teacher dit
    "apres la bonne reponse vient la soeur". Le soft target transmet cet
    ordre ; le hard label one-hot (et a fortiori s'il est bruite) ne dit
    rien sur l'ordre des perdantes.
    """
    ok = used = 0
    for x, zt, _ in data:
        rank_t = argsort_desc(zt)
        winner, second = rank_t[0], rank_t[1]
        if SIBLING.get(winner) != second:
            continue  # on ne juge que quand la 2e place teacher EST la soeur
        far = next(c for c in rank_t if c not in (winner, second))
        zs = st.logits(x)
        ok += (zs[second] > zs[far])
        used += 1
    return ok / used if used else 0.0


SEED_INIT = 123
T_MAIN = 4.0
st_hard = train_hard(SEED_INIT)
st_soft, kl_curve = train_soft(SEED_INIT, T=T_MAIN)

acc_hard, acc_soft = accuracy(st_hard, test), accuracy(st_soft, test)
dk_hard, dk_soft = dark_knowledge_recovery(st_hard, test), dark_knowledge_recovery(st_soft, test)
# Fidelite a la distribution teacher (a T=1) sur le hold-out : la mesure la
# plus directe de dark knowledge. Le soft minimise exactement ca ; le hard non.
kl_hard = mean_teacher_student_kl(st_hard, 1.0, test)
kl_soft = mean_teacher_student_kl(st_soft, 1.0, test)

print(f"\n  Student HARD : acc={acc_hard:.3f}  dark-knowledge recovery={dk_hard:.3f}"
      f"  KL(teacher||student,hold-out)={kl_hard:.4f}")
print(f"  Student SOFT : acc={acc_soft:.3f}  dark-knowledge recovery={dk_soft:.3f}"
      f"  KL(teacher||student,hold-out)={kl_soft:.4f}")
print(f"\n  KL(teacher||student) au fil des epochs (soft, train) :")
print(f"    initiale = {kl_curve[0]:.4f}  -> finale = {kl_curve[-1]:.4f}")
descents = sum(1 for a, b in zip(kl_curve, kl_curve[1:]) if b <= a + 1e-9)
print(f"    pas decroissants : {descents}/{len(kl_curve) - 1}")

# Ablation temperature : on attend une cloche sur la fidelite (KL hold-out).
# T=1 : soft targets quasi one-hot -> peu de dark knowledge transmise.
# T tres grand : targets quasi uniformes -> le signal de classe se noie.
print(f"\n  Ablation T (KL hold-out teacher||student a T=1, plus bas = mieux) :")
kl_by_T = {}
for T in (1.0, 2.0, 4.0, 8.0, 20.0):
    st_T, _ = train_soft(SEED_INIT, T=T)
    kl_by_T[T] = mean_teacher_student_kl(st_T, 1.0, test)
    print(f"    T={T:<5} -> KL hold-out = {kl_by_T[T]:.4f}")
best_T = min(kl_by_T, key=kl_by_T.get)
print(f"  -> sweet spot ~ T={best_T} (T=1 trop pique, T tres grand trop plat).")


# ============================================================================
# EXERCISE 8 — Pipeline synthetique : ablation de filtres & qualite/quantite
# ============================================================================
# But : etendre le pipeline du 02-code en ablation LEAVE-ONE-FILTER-OUT,
# moyennee sur plusieurs seeds, + tradeoff volume/qualite honnete + demo de
# contamination. On reutilise les briques sentiment du 02-code.
#
# Cle du design : le vocabulaire des snippets generes RECOUVRE celui de l'eval
# (sinon le synth n'apprend rien d'utile et le brut "gagne" par hasard). Et le
# teacher est franchement bruite -> le judge a un vrai travail.

print("\n" + "=" * 70)
print("EXERCISE 8 : pipeline synthetique — ablation de filtres (leave-one-out)")
print("=" * 70)

SEEDS = [
    "Ce film etait incroyable, j'ai adore",
    "Produit defectueux, tres decu",
    "Service client excellent, je recommande",
    "Totalement inutile et cher",
    "Bonne experience dans l'ensemble",
    "Mauvais rapport qualite prix",
]

# Eval hold-out : vocabulaire choisi pour CHEVAUCHER les snippets (sinon le
# student ne peut rien transferer). C'est realiste : le synth doit couvrir le
# vocabulaire de la tache cible.
EVAL_SET = [
    ("Produit parfait, je recommande vivement", 1),
    ("Service horrible, je deteste", 0),
    ("Experience incroyable, tout etait super", 1),
    ("Achat catastrophique, vraiment mauvais", 0),
    ("Magnifique, ravi de mon achat", 1),
    ("Mediocre et decu, a fuir", 0),
    ("Excellent rapport qualite, genial", 1),
    ("Terrible, ne recommande pas du tout", 0),
    ("Tres bon produit, adore", 1),
    ("Nul et decevant, jamais plus", 0),
]

POS_SNIPPETS = ["excellent", "parfait", "incroyable", "super", "adore",
                "tres bon", "recommande", "ravi", "magnifique", "genial"]
NEG_SNIPPETS = ["nul", "decu", "mauvais", "catastrophique", "horrible",
                "deteste", "ne recommande pas", "fuir", "mediocre", "terrible"]


def teacher_generate(seed, n, p_noise, rng):
    """Teacher bruite : reformule le seed, p_noise = proba de retourner le label.
    p_noise eleve simule un teacher mediocre -> les filtres ont du travail."""
    out = []
    is_pos = any(w in seed.lower() for w in ["adore", "excellent", "bon", "recommande"])
    for _ in range(n):
        if is_pos:
            words = rng.sample(POS_SNIPPETS, k=rng.randint(1, 3))
            text, label = f"Avis client : {' '.join(words)}.", 1
        else:
            words = rng.sample(NEG_SNIPPETS, k=rng.randint(1, 3))
            text, label = f"Avis client : {' '.join(words)}.", 0
        if rng.random() < p_noise:   # bruit teacher : label retourne
            label = 1 - label
        out.append((text, label))
    return out


def rule_based_ok(text):
    return 10 < len(text) < 500 and text.count(".") >= 1


def llm_judge(text, label):
    """Juge simule : detecte un conflit texte/label (les retournements du
    teacher). C'est le filtre qui neutralise le p_noise."""
    has_pos = any(w in text.lower() for w in POS_SNIPPETS)
    has_neg = any(w in text.lower() for w in NEG_SNIPPETS)
    if label == 1 and has_neg and not has_pos:
        return False
    if label == 0 and has_pos and not has_neg:
        return False
    return True


def minhash_dedup(examples):
    seen, out = set(), []
    for text, label in examples:
        key = tuple(sorted(set(re.findall(r"[a-z]+", text.lower()))))
        h = hashlib.md5(str(key).encode()).hexdigest()[:12]
        if h not in seen:
            seen.add(h)
            out.append((text, label))
    return out


def contamination_check(synth, eval_set):
    """Nombre d'exemples synth quasi-identiques a un eval (Jaccard > 0.6)."""
    eval_words = [set(re.findall(r"[a-z]+", t.lower())) for t, _ in eval_set]
    count = 0
    for text, _ in synth:
        s = set(re.findall(r"[a-z]+", text.lower()))
        if not s:
            continue
        if any(len(s & e) / len(s | e) > 0.6 for e in eval_words):
            count += 1
    return count


class BagOfWordsLogReg:
    """Bag-of-words logreg from scratch (identique au 02-code)."""

    def __init__(self, lr=0.1, epochs=60):
        self.lr, self.epochs = lr, epochs
        self.w = defaultdict(float)
        self.b = 0.0

    @staticmethod
    def _feat(text):
        return Counter(re.findall(r"[a-z]+", text.lower()))

    def _proba(self, x):
        z = self.b + sum(self.w[k] * v for k, v in x.items())
        return 1 / (1 + math.exp(-max(min(z, 30), -30)))

    def fit(self, data):
        data = list(data)
        for _ in range(self.epochs):
            random.shuffle(data)
            for text, y in data:
                x = self._feat(text)
                grad = self._proba(x) - y
                for k, v in x.items():
                    self.w[k] -= self.lr * grad * v
                self.b -= self.lr * grad

    def evaluate(self, data):
        hits = sum(1 for t, y in data if (self._proba(self._feat(t)) > 0.5) == bool(y))
        return hits / len(data)


SEEDS_LABELED = [
    (s, 1 if any(w in s.lower() for w in ["adore", "excellent", "bon", "recommande"]) else 0)
    for s in SEEDS
]

P_NOISE = 0.35  # teacher franchement bruite : sans judge, ~35% de labels faux


def apply_filters(raw, use_rule=True, use_judge=True, use_dedup=True):
    """Cascade configurable -> permet l'ablation leave-one-filter-out."""
    data = list(raw)
    if use_rule:
        data = [(t, l) for t, l in data if rule_based_ok(t)]
    if use_judge:
        data = [(t, l) for t, l in data if llm_judge(t, l)]
    if use_dedup:
        data = minhash_dedup(data)
    return data


def run_pipeline(seed, n_per_seed=20, filters=None):
    """Genere du synth, applique une config de filtres, entraine, eval."""
    rng = random.Random(seed)
    raw = []
    for s in SEEDS:
        raw.extend(teacher_generate(s, n=n_per_seed, p_noise=P_NOISE, rng=rng))
    synth = apply_filters(raw, **(filters or {}))
    student = BagOfWordsLogReg()
    student.fit(list(SEEDS_LABELED) + synth)
    return student.evaluate(EVAL_SET), len(synth)


# Ablation leave-one-filter-out, MOYENNEE sur plusieurs seeds (robustesse :
# le SGD et le tirage des donnees sont stochastiques, un run unique = bruit).
N_SEEDS = 8
configs = {
    "full (rule+judge+dedup)": dict(use_rule=True, use_judge=True, use_dedup=True),
    "no rule":                 dict(use_rule=False, use_judge=True, use_dedup=True),
    "no judge":                dict(use_rule=True, use_judge=False, use_dedup=True),
    "no dedup":                dict(use_rule=True, use_judge=True, use_dedup=False),
    "raw (no filter)":         dict(use_rule=False, use_judge=False, use_dedup=False),
}

print(f"\n  p_noise={P_NOISE}, {N_SEEDS} seeds, moyenne accuracy hold-out :")
mean_acc = {}
for name, cfg in configs.items():
    accs = [run_pipeline(1000 + s, filters=cfg)[0] for s in range(N_SEEDS)]
    mean_acc[name] = sum(accs) / len(accs)
    print(f"    {name:<26} acc = {mean_acc[name]:.3f}")

full = mean_acc["full (rule+judge+dedup)"]
print(f"\n  Contribution de chaque filtre (chute d'acc si on le retire) :")
for name in ("no rule", "no judge", "no dedup"):
    print(f"    retirer {name[3:]:<8} -> delta = {mean_acc[name] - full:+.3f}")
most_valuable = min(("no rule", "no judge", "no dedup"), key=lambda n: mean_acc[n])
print(f"  -> filtre le plus precieux : {most_valuable[3:]} "
      f"(neutralise le bruit teacher p_noise={P_NOISE}).")

# Tradeoff volume vs qualite : a volume egal, filtre vs brut, + saturation.
print(f"\n  Volume vs qualite (moyenne {N_SEEDS} seeds) :")
print(f"    {'n/seed':<8} {'filtre':<10} {'brut':<10}")
for n in (5, 10, 20, 40):
    af = [run_pipeline(2000 + s, n_per_seed=n, filters=configs["full (rule+judge+dedup)"])[0]
          for s in range(N_SEEDS)]
    ar = [run_pipeline(2000 + s, n_per_seed=n, filters=configs["raw (no filter)"])[0]
          for s in range(N_SEEDS)]
    print(f"    {n:<8} {sum(af)/len(af):<10.3f} {sum(ar)/len(ar):<10.3f}")
print("  -> a teacher bruite, le filtre gagne ; le gain de volume sature vite")
print("     (le bruit s'accumule : doubler le brut n'achete presque rien).")

# Contamination injectee : detection + accuracy gonflee.
print(f"\n  Contamination injectee :")
rng_c = random.Random(999)
raw_pool = []
for s in SEEDS:
    raw_pool.extend(teacher_generate(s, n=20, p_noise=P_NOISE, rng=rng_c))
N_INJECT = 3
injected = EVAL_SET[:N_INJECT]               # on triche : on met de l'eval dans le train
poisoned = raw_pool + [(t, l) for t, l in injected]
detected = contamination_check(poisoned, EVAL_SET)
print(f"    injectes = {N_INJECT}, detectes par contamination_check = {detected}")
print("    -> sans eval prive + sans ce check, ces exemples gonfleraient l'eval.")
print("    (piege #1 du cours : l'eval ne doit JAMAIS etre dans le pipeline.)")


# ============================================================================
# EXERCISE 9 — Sequence distillation : teacher-forcing vs on-policy
# ============================================================================
# But : montrer l'exposure bias. Le student TF n'est entraine que sur les
# etats de la trajectoire teacher ; a l'inference auto-regressive sa 1ere
# erreur l'envoie dans un etat jamais vu -> erreur composee. L'etape on-policy
# lui fait visiter ses propres etats (re-scores par le teacher) -> il apprend
# a se recuperer. On asserte derailment(on-policy) < derailment(TF).
#
# Cle du design : pour qu'une erreur SE PRODUISE, la regle teacher est
# stochastique (branche minoritaire occasionnelle) ET un etat est volontaire-
# ment ABSENT de l'entrainement TF -> sur cet etat le student TF devine.

print("\n" + "=" * 70)
print("EXERCISE 9 : sequence distillation — teacher-forcing vs on-policy")
print("=" * 70)

V = 6  # alphabet de tokens (etats)

def teacher_rule(state, rng=None):
    """Transition teacher MAJORITAIRE (deterministe a l'eval, la 'verite').
    On garde une version deterministe pour definir la trajectoire de reference
    et scorer le student."""
    return (state + 1) % V

# La trajectoire teacher est un CYCLE complet 0->1->...->V-1->0. En TF on
# entraine le student sur toutes les transitions SAUF une (on retire l'etat
# 'HOLE') : c'est le trou d'exposition. Si le student atterrit sur HOLE a
# l'inference (apres une perturbation), il n'a aucune transition apprise.
HOLE = 3
full_transitions = [(s, teacher_rule(s)) for s in range(V)]
tf_pairs = [(s, nxt) for (s, nxt) in full_transitions if s != HOLE]


class TabularStudent:
    """Modele de transition tabulaire : next(state) = transition la plus vue.
    Auto-regressif au rollout (il consomme ses propres sorties)."""

    def __init__(self):
        self.counts = defaultdict(Counter)

    def train_on(self, pairs):
        for state, nxt in pairs:
            self.counts[state][nxt] += 1

    def predict(self, state):
        if state in self.counts and self.counts[state]:
            return self.counts[state].most_common(1)[0][0]
        # Etat jamais vu : trou de connaissance -> devine (fallback fixe et
        # FAUX par construction). C'est ici que l'exposure bias mord.
        return state  # boucle sur soi-meme : il reste coince hors-trajectoire

    def rollout(self, start, length, perturb_at=None, rng=None):
        """Rollout AUTO-REGRESSIF : state_{t+1} = predict(state_t).
        perturb_at : on force une erreur a ce pas (simule une faute initiale
        du student) pour observer l'error compounding qui suit."""
        seq = [start]
        for t in range(length):
            nxt = self.predict(seq[-1])
            if perturb_at is not None and t == perturb_at:
                # Perturbation : on envoie le student SUR le trou (HOLE).
                nxt = HOLE
            seq.append(nxt)
        return seq


def collect_on_policy(base_student, n_rollouts=60, length=8, rng=None):
    """Phase on-policy : le student deroule, le teacher RE-SCORE les etats
    VISITES par le student (y compris le trou). On ajoute (state_visite,
    teacher_rule(state_visite)) -> le student apprend a se recuperer du trou."""
    rng = rng or random.Random(7)
    extra = []
    for _ in range(n_rollouts):
        start = rng.randrange(V)
        # On force le passage par le trou de temps en temps pour le couvrir.
        roll = base_student.rollout(start, length, perturb_at=rng.randrange(length))
        for s in roll:
            extra.append((s, teacher_rule(s)))  # le teacher corrige l'etat visite
    return extra


def derailment_rate(student, n_rollouts=300, length=8):
    """Error compounding sur rollouts auto-regressifs AVEC une perturbation
    precoce (pas 1 : on force le passage par le trou).

    POURQUOI on ne compare PAS a la trajectoire teacher 'propre' : apres une
    perturbation forcee, AUCUN student ne peut re-coller a la trajectoire
    ideale. Ce qui distingue un bon student d'un mauvais, c'est sa capacite a
    refaire des transitions VALIDES (= se remettre sur le cycle), pas a
    rattraper le passe. On mesure donc, APRES le trou :
      - erreur/pas = fraction de transitions INVALIDES (roll[t] != rule(roll[t-1]))
      - derailment = le student est-il TOUJOURS sur une transition invalide a la fin ?
    Le TF reste coince sur le trou (HOLE->HOLE, invalide) ; l'on-policy a appris
    HOLE->4 (valide) et rejoint le cycle.
    """
    rng = random.Random(123)
    derailed = step_err = total = 0
    for _ in range(n_rollouts):
        start = rng.randrange(V)
        roll = student.rollout(start, length, perturb_at=1)  # 1 erreur au pas 1
        for t in range(2, length + 1):  # apres la perturbation : transitions valides ?
            total += 1
            step_err += (roll[t] != teacher_rule(roll[t - 1]))  # transition invalide ?
        # derailment final : la derniere transition est-elle encore invalide ?
        derailed += (roll[length] != teacher_rule(roll[length - 1]))
    return derailed / n_rollouts, step_err / total


# Student TF pur : n'a jamais vu la transition du trou.
st_tf = TabularStudent()
st_tf.train_on(tf_pairs)

# Student on-policy : TF + paires on-policy (inclut la transition du trou).
st_op = TabularStudent()
st_op.train_on(tf_pairs)
st_op.train_on(collect_on_policy(st_tf))

L = 8
der_tf, step_tf = derailment_rate(st_tf, length=L)
der_op, step_op = derailment_rate(st_op, length=L)

print(f"\n  Cycle teacher 0->...->{V-1}->0 ; etat-trou non vu en TF : HOLE={HOLE}")
print(f"  Rollouts auto-regressifs L={L} avec 1 perturbation au pas 1 :")
print(f"    teacher-forcing pur : derailment={der_tf:.3f}  erreur/pas(post)={step_tf:.3f}")
print(f"    + on-policy         : derailment={der_op:.3f}  erreur/pas(post)={step_op:.3f}")
print("  -> apres la 1ere erreur, le TF reste coince sur le trou (jamais vu) ;")
print("     l'on-policy a appris la transition du trou -> il se recupere.")

# --- Borne analytique (caveat honnete) ------------------------------------
# Ross et al. (DAgger, 2011) : en teacher-forcing l'erreur composee croit en
# O(eps * L^2) (eps se propage et s'amplifie quadratiquement avec L) ; une
# approche on-policy ramene a O(eps * L). On modelise explicitement
# l'inegalite (toy fidele : pas une vraie boucle RL, comme le 02-code
# documente honnetement son bruit teacher).
EPS = 0.1
bound_tf = EPS * L * L      # O(eps L^2)
bound_op = EPS * L          # O(eps L)
print(f"\n  Borne analytique (Ross/DAgger), eps={EPS}, L={L} :")
print(f"    teacher-forcing ~ eps*L^2 = {bound_tf:.2f}")
print(f"    on-policy       ~ eps*L   = {bound_op:.2f}")
print("  Caveat : toy fidele (modele tabulaire + borne), PAS une boucle RL")
print("  complete. Lien cours section 5 : le SFT pur (Type 2) souffre quand")
print("  meme d'exposure bias sur les taches longues -> Apple/DeepSeek")
print("  ajoutent une phase on-policy/RL (Type 3) pour le reasoning multi-pas.")


# ============================================================================
# ASSERTIONS — le fichier est auto-verifiant
# ============================================================================
print("\n" + "=" * 70)
print("ASSERTIONS (self-check)")
print("=" * 70)

# --- Exercise 7 ---
# (a) KL(teacher||student) decroit du debut a la fin de l'entrainement soft.
assert kl_curve[-1] < kl_curve[0], f"KL non decrue : {kl_curve[0]:.4f}->{kl_curve[-1]:.4f}"
# (b) dynamique globalement decroissante (>=80% des pas vont vers le bas).
assert descents >= 0.8 * (len(kl_curve) - 1), "KL non globalement decroissante"
# (c) le student SOFT colle mieux a la distribution teacher (dark knowledge).
assert kl_soft < kl_hard, f"soft ne colle pas mieux : KL {kl_soft:.4f} vs {kl_hard:.4f}"
# (d) le student SOFT recupere mieux l'ordre des perdantes (soeur > lointaine).
assert dk_soft > dk_hard, f"soft ne bat pas hard en dark-knowledge : {dk_soft:.3f} vs {dk_hard:.3f}"
# (e) la dark knowledge n'est pas payee par l'accuracy (soft ne s'effondre pas).
assert acc_soft >= acc_hard - 0.05, "le student soft s'effondre en accuracy"
# (f) ablation T : le sweet spot n'est PAS a T=1 (sinon pas de dark knowledge).
assert best_T > 1.0, f"sweet spot a T={best_T} : la dark knowledge n'apparait pas"
print("  [Ex7] KL decroit ; soft < hard en KL hold-out ; soft > hard en")
print("        dark-knowledge recovery ; sweet spot T>1  ->  OK")

# --- Exercise 8 ---
# (a) le pipeline filtre bat le brut (moyennes multi-seed).
assert mean_acc["full (rule+judge+dedup)"] > mean_acc["raw (no filter)"], \
    f"filtre ({full:.3f}) ne bat pas brut ({mean_acc['raw (no filter)']:.3f})"
# (b) le judge est le filtre le plus precieux avec un teacher bruite.
assert most_valuable == "no judge", f"filtre le plus precieux inattendu : {most_valuable}"
# (c) retirer le judge fait chuter l'accuracy.
assert mean_acc["no judge"] < full, "le judge n'a pas d'effet (devrait aider)"
# (d) la contamination injectee est detectee (au moins autant qu'injecte).
assert detected >= N_INJECT, f"contamination ratee : {detected} < {N_INJECT}"
print("  [Ex8] filtre > brut ; judge = filtre cle ; contamination detectee  ->  OK")

# --- Exercise 9 ---
# (a) empirique : l'on-policy derive STRICTEMENT moins que le TF pur.
assert der_op < der_tf, f"on-policy ne reduit pas le derailment : {der_op:.3f} >= {der_tf:.3f}"
# (b) on-policy reduit l'erreur par pas post-perturbation (recuperation).
assert step_op < step_tf, f"on-policy n'ameliore pas l'erreur/pas : {step_op:.3f} >= {step_tf:.3f}"
# (c) borne analytique : O(eps L) < O(eps L^2) pour L > 1.
assert bound_op < bound_tf, "borne on-policy >= borne TF (devrait etre <)"
print("  [Ex9] on-policy < TF (empirique) et borne eps*L < eps*L^2  ->  OK")

print("\nDone (HARD).  Tous les asserts passent.")
