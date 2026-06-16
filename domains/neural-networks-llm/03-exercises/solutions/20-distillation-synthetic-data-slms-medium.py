"""
Solutions MEDIUM — Jour 20 : Distillation, donnees synthetiques & SLMs
=====================================================================
Exercices 4, 5, 6 (medium). PUR PYTHON STDLIB (random / math / re / hashlib /
collections), comme 02-code/20-distillation-synthetic-data-slms.py — PAS de
numpy, pour rester fidele au module et tourner sans dependance.

Chaque etape non triviale est commentee avec le POURQUOI. Le fichier est
auto-verifiant : il se termine par des assertions qui echouent si une
propriete pedagogique attendue n'est plus vraie.

Run: python3 03-exercises/solutions/20-distillation-synthetic-data-slms-medium.py
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
# Primitives numeriques A LA MAIN (pas de numpy)
# ============================================================================

def softmax(logits: list[float], T: float = 1.0) -> list[float]:
    """Softmax stable a temperature T (cf 02-code/20).
    -max : evite l'overflow de exp ; /T : lisse (T>1) ou pique (T<1)."""
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
    En distillation p = teacher (cible), q = student (on l'entraine a copier p)."""
    s = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            s += pi * math.log(pi / (qi + 1e-12))
    return max(s, 0.0)


# ============================================================================
# EXERCISE 4 — Loss de distillation KL complete (logits) from scratch
# ============================================================================
# But : implementer la vraie loss de Hinton 2015
#   loss = alpha * T^2 * KL(teacher_soft || student_soft) + (1-alpha) * CE(y, student)
# et montrer que distiller bat l'entrainement sur hard labels seuls.
#
# Cle du design : le teacher encode une STRUCTURE DE SIMILARITE (classe 0
# ressemble a 1, classe 2 a 3). Cette structure est la "dark knowledge" que
# le soft target transmet et que le hard label efface.

print("=" * 70)
print("EXERCISE 4 : loss de distillation KL complete (Hinton 2015) from scratch")
print("=" * 70)

K = 5            # classes
D = 6            # dimension des features
N_TRAIN = 500
N_TEST = 300
SIBLING = {0: 1, 1: 0, 2: 3, 3: 2, 4: 4}  # 0<->1, 2<->3, 4 isolee


def make_teacher_weights() -> list[list[float]]:
    """Poids teacher (K, D) avec structure de similarite : la ligne d'une
    classe partage une composante avec celle de sa soeur -> un input qui
    active la classe c active aussi (un peu) sa soeur -> la soeur sort 2e."""
    rng = random.Random(7)
    base = [[rng.gauss(0, 1.0) for _ in range(D)] for _ in range(K)]
    W = [[0.0] * D for _ in range(K)]
    for c in range(K):
        s = SIBLING[c]
        for j in range(D):
            # 70% propre + 30% de la soeur : correlation controlee.
            W[c][j] = 0.7 * base[c][j] + 0.3 * base[s][j]
    return W


W_teacher = make_teacher_weights()
# Facteur < 1 : ECRASE les logits -> teacher peu confiant -> distribution douce
# et informative (dark knowledge). Logits enormes => softmax ~ one-hot => rien
# a distiller.
TEACHER_CONFIDENCE = 0.7


def teacher_logits(x: list[float]) -> list[float]:
    return [TEACHER_CONFIDENCE * sum(W_teacher[c][j] * x[j] for j in range(D))
            for c in range(K)]


def sample_x(rng: random.Random, true_class: int) -> list[float]:
    """Feature gaussienne centree sur un prototype faible (S/B bas) -> exemples
    ambigus ou le teacher hesite -> dark knowledge riche."""
    proto = [(1.1 if (j % K) == true_class else 0.0) for j in range(D)]
    return [proto[j] + rng.gauss(0, 1.0) for j in range(D)]


def build_dataset(n: int, seed: int):
    """Renvoie [(x, teacher_logits, hard_label_bruite)]."""
    rng = random.Random(seed)
    data = []
    for _ in range(n):
        y_true = rng.randrange(K)
        x = sample_x(rng, y_true)
        zt = teacher_logits(x)
        hard = max(range(K), key=lambda c: zt[c])   # argmax teacher = pseudo-label
        # Bruit d'etiquetage : le teacher n'est pas parfait (cf 5% du 02-code).
        # 15% ici pour que le hard label perde de l'info que le soft conserve.
        if rng.random() < 0.15:
            hard = rng.randrange(K)
        data.append((x, zt, hard))
    return data


train4 = build_dataset(N_TRAIN, seed=1)
test4 = build_dataset(N_TEST, seed=2)


class LinearStudent:
    """logits = W x + b, softmax, SGD. Pas de numpy : listes + boucles."""

    def __init__(self, K: int, D: int, seed: int = 0):
        rng = random.Random(seed)
        # Memes inits pour hard et soft (meme seed) : seul le SIGNAL differe,
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
        # dL/dW[c][j] = grad_logits[c] * x[j] (chaine softmax).
        for c in range(self.K):
            g = grad_logits[c]
            self.b[c] -= lr * g
            row = self.W[c]
            for j in range(self.D):
                row[j] -= lr * g * x[j]

    def step_distill(self, x, teacher_soft, hard, T, alpha, lr):
        """Gradient de la loss combinee Hinton :
          alpha * T^2 * KL(teacher_soft || student_soft) + (1-alpha)*CE(hard).

        d/dz_c de T^2*KL(t || softmax(z/T)) = T*(q_c - t_c), q = softmax(z/T).
        POURQUOI le T^2 -> facteur NET T sur le gradient : il compense le
        1/T^2 du gradient des soft logits, sinon a T grand le signal soft
        s'evanouirait.
        d/dz_c de CE(hard) = (p_c - onehot(hard)), p = softmax(z) a T=1.
        """
        q = self.proba(x, T=T)                       # student soft (T)
        p = self.proba(x, T=1.0)                     # student dur (T=1) pour CE
        grad = []
        for c in range(self.K):
            g_soft = alpha * T * (q[c] - teacher_soft[c])          # terme T^2*KL
            g_hard = (1 - alpha) * (p[c] - (1.0 if c == hard else 0.0))  # terme CE
            grad.append(g_soft + g_hard)
        self._apply_grad(x, grad, lr)

    def step_hard(self, x, hard, lr):
        """Gradient pur cross-entropy hard (= distill avec alpha=0)."""
        p = self.proba(x, T=1.0)
        grad = [p[c] - (1.0 if c == hard else 0.0) for c in range(self.K)]
        self._apply_grad(x, grad, lr)


def train_hard(seed_init, epochs=60, lr=0.15):
    st = LinearStudent(K, D, seed=seed_init)
    idx = list(range(len(train4)))
    for _ in range(epochs):
        random.shuffle(idx)
        for i in idx:
            x, _zt, hard = train4[i]
            st.step_hard(x, hard, lr)
    return st


def train_distill(seed_init, T, alpha=0.7, epochs=60, lr=0.15):
    st = LinearStudent(K, D, seed=seed_init)
    idx = list(range(len(train4)))
    for _ in range(epochs):
        random.shuffle(idx)
        for i in idx:
            x, zt, hard = train4[i]
            st.step_distill(x, softmax(zt, T=T), hard, T, alpha, lr)
    return st


def accuracy(st, data):
    """Accuracy vs argmax teacher (la vraie cible, sans le bruit d'etiquetage)."""
    hits = sum(
        max(range(K), key=lambda c: st.logits(x)[c])
        == max(range(K), key=lambda c: zt[c])
        for x, zt, _ in data)
    return hits / len(data)


def mean_kl_to_teacher(st, data, T=1.0):
    """Calibration : KL(teacher || student) a T=1 sur le hold-out.
    Plus bas = le student imite mieux la DISTRIBUTION du teacher (et pas
    seulement sa classe gagnante)."""
    tot = sum(kl_divergence(softmax(zt, T=T), st.proba(x, T=T)) for x, zt, _ in data)
    return tot / len(data)


SEED_INIT = 123
T_MAIN, ALPHA = 3.0, 0.7
st_hard = train_hard(SEED_INIT)
st_dist = train_distill(SEED_INIT, T=T_MAIN, alpha=ALPHA)

acc_hard = accuracy(st_hard, test4)
acc_dist = accuracy(st_dist, test4)
kl_hard = mean_kl_to_teacher(st_hard, test4)
kl_dist = mean_kl_to_teacher(st_dist, test4)

print(f"\n  Setup : K={K} classes, structure de similarite (soeur = 2e du teacher)")
print(f"  alpha={ALPHA}, T={T_MAIN}, hold-out n={N_TEST}\n")
print(f"  Student HARD seul    : acc={acc_hard:.3f}  KL(teacher||student)={kl_hard:.4f}")
print(f"  Student DISTILLE     : acc={acc_dist:.3f}  KL(teacher||student)={kl_dist:.4f}")
print("  -> le student distille generalise >= hard ET imite mieux la")
print("     distribution du teacher (KL plus basse = meilleure calibration).")

# Ablation temperature : on attend une cloche sur l'accuracy.
print(f"\n  Ablation temperature (accuracy hold-out, plus haut = mieux) :")
print(f"    {'T':<6} {'accuracy':<10} {'KL(teacher||student)':<22}")
print("    " + "-" * 38)
acc_by_T = {}
for T in (1.0, 2.0, 3.0, 5.0, 10.0):
    st_T = train_distill(SEED_INIT, T=T, alpha=ALPHA)
    acc_by_T[T] = accuracy(st_T, test4)
    print(f"    {T:<6} {acc_by_T[T]:<10.3f} {mean_kl_to_teacher(st_T, test4):<22.4f}")
best_T = max(acc_by_T, key=acc_by_T.get)
print(f"  -> sweet spot ~ T={best_T} : T=1 perd la dark knowledge (soft ~ one-hot),")
print("     T tres grand rend la distribution quasi uniforme et noie le signal.")


# ============================================================================
# EXERCISE 5 — Pipeline de donnees synthetiques : l'impact du filtrage
# ============================================================================
# But : instrumenter le pipeline du 02-code, mesurer le rendement par etage de
# filtre, et comparer 4 students (seeds / + brut / + filtre / + filtre teacher
# mediocre). Plus une courbe volume vs qualite et une demo de contamination.

print("\n" + "=" * 70)
print("EXERCISE 5 : pipeline synthetique — l'impact du filtrage")
print("=" * 70)

SEEDS = [
    "Ce film etait incroyable, j'ai adore",
    "Produit defectueux, tres decu",
    "Service client excellent, je recommande",
    "Totalement inutile et cher",
    "Bonne experience dans l'ensemble",
    "Mauvais rapport qualite prix",
]

# Eval hold-out : vocabulaire choisi pour CHEVAUCHER les snippets generes
# (sinon le student ne peut rien transferer). Realiste : le synth doit couvrir
# le vocabulaire de la tache cible.
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
    """Teacher bruite : reformule le seed ; p_noise = proba de RETOURNER le
    label. p_noise eleve simule un teacher mediocre -> les filtres ont du
    travail (c'est ce que le judge devra rattraper)."""
    out = []
    is_pos = any(w in seed.lower() for w in ["adore", "excellent", "bon", "recommande"])
    for _ in range(n):
        if is_pos:
            words = rng.sample(POS_SNIPPETS, k=rng.randint(1, 3))
            text, label = f"Avis client : {' '.join(words)}.", 1
        else:
            words = rng.sample(NEG_SNIPPETS, k=rng.randint(1, 3))
            text, label = f"Avis client : {' '.join(words)}.", 0
        if rng.random() < p_noise:    # bruit teacher : label retourne
            label = 1 - label
        out.append((text, label))
    return out


def rule_based_ok(text):
    """Filtre 1 : longueur/format plausibles + au moins une ponctuation."""
    return 10 < len(text) < 500 and text.count(".") >= 1


def llm_judge(text, label):
    """Filtre 2 : juge LLM simule. Detecte un conflit texte/label (= les
    retournements du teacher). C'est le filtre qui neutralise le p_noise."""
    has_pos = any(w in text.lower() for w in POS_SNIPPETS)
    has_neg = any(w in text.lower() for w in NEG_SNIPPETS)
    if label == 1 and has_neg and not has_pos:
        return False
    if label == 0 and has_pos and not has_neg:
        return False
    return True


def minhash_dedup(examples):
    """Filtre 3 : dedup par hash de set-de-mots (approx minhash)."""
    seen, out = set(), []
    for text, label in examples:
        key = tuple(sorted(set(re.findall(r"[a-z]+", text.lower()))))
        h = hashlib.md5(str(key).encode()).hexdigest()[:12]
        if h not in seen:
            seen.add(h)
            out.append((text, label))
    return out


def contamination_check(synth, eval_set, thresh=0.6):
    """Nombre d'exemples synth quasi-identiques a un eval (Jaccard > thresh)."""
    eval_words = [set(re.findall(r"[a-z]+", t.lower())) for t, _ in eval_set]
    count = 0
    for text, _ in synth:
        s = set(re.findall(r"[a-z]+", text.lower()))
        if not s:
            continue
        if any(len(s & e) / len(s | e) > thresh for e in eval_words):
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


# --- 2) Rendement par etage de filtre (cascade complete) ---
def generate_raw(p_noise, n_per_seed, seed):
    rng = random.Random(seed)
    raw = []
    for s in SEEDS:
        raw.extend(teacher_generate(s, n=n_per_seed, p_noise=p_noise, rng=rng))
    return raw


P_NOISE = 0.30           # teacher franchement bruite : sans judge, ~30% faux
raw_demo = generate_raw(P_NOISE, n_per_seed=20, seed=100)
print(f"\n  Rendement par etage (p_noise={P_NOISE}, {len(raw_demo)} generes) :")
stage = list(raw_demo)
print(f"    genere            : {len(stage)}")
stage = [(t, l) for t, l in stage if rule_based_ok(t)]
print(f"    apres rule-based  : {len(stage)}  ({len(stage)/len(raw_demo):.0%} passent)")
after_rule = len(stage)
stage = [(t, l) for t, l in stage if llm_judge(t, l)]
print(f"    apres LLM-judge   : {len(stage)}  ({len(stage)/after_rule:.0%} du precedent)")
after_judge = len(stage)
stage = minhash_dedup(stage)
print(f"    apres dedup       : {len(stage)}  ({len(stage)/after_judge:.0%} du precedent)")
print("  -> le judge est l'etage qui jette le plus (il retire les labels")
print("     retournes par le teacher) ; le dedup compresse les doublons.")


# --- 3) Ablation des 4 students ---
def train_student(extra_examples):
    st = BagOfWordsLogReg()
    st.fit(list(SEEDS_LABELED) + list(extra_examples))
    return st.evaluate(EVAL_SET)


def cascade(raw):
    """Filtrage complet rule -> judge -> dedup."""
    d = [(t, l) for t, l in raw if rule_based_ok(t)]
    d = [(t, l) for t, l in d if llm_judge(t, l)]
    return minhash_dedup(d)


# Moyenne sur plusieurs seeds (SGD + tirage stochastiques -> un run = bruit).
N_SEEDS = 8
acc_seeds, acc_brut, acc_filtre, acc_mediocre = [], [], [], []
for s in range(N_SEEDS):
    raw = generate_raw(P_NOISE, n_per_seed=20, seed=200 + s)
    raw_bad = generate_raw(0.45, n_per_seed=20, seed=200 + s)  # teacher mediocre
    acc_seeds.append(train_student([]))                         # seeds seuls
    acc_brut.append(train_student(raw))                         # + synth brut
    acc_filtre.append(train_student(cascade(raw)))              # + synth filtre
    acc_mediocre.append(train_student(cascade(raw_bad)))        # filtre mais teacher nul

m_seeds = sum(acc_seeds) / N_SEEDS
m_brut = sum(acc_brut) / N_SEEDS
m_filtre = sum(acc_filtre) / N_SEEDS
m_mediocre = sum(acc_mediocre) / N_SEEDS

print(f"\n  Ablation 4 students (moyenne {N_SEEDS} seeds, accuracy hold-out) :")
print(f"    seeds seuls (baseline)         : {m_seeds:.3f}")
print(f"    + synth BRUT (sans filtre)     : {m_brut:.3f}")
print(f"    + synth FILTRE (cascade)       : {m_filtre:.3f}")
print(f"    + filtre mais teacher MEDIOCRE : {m_mediocre:.3f}  (p_noise=0.45)")
print("  -> filtre > brut quand le teacher est bruite ; un teacher mediocre")
print("     plombe le student MEME apres filtrage (garbage in, garbage out).")


# --- 4) Courbe volume vs qualite ---
print(f"\n  Volume vs qualite (moyenne {N_SEEDS} seeds) :")
print(f"    {'n/seed':<8} {'filtre':<10} {'brut':<10}")
print("    " + "-" * 28)
filtre_curve = []
for n in (2, 5, 10, 20):
    af, ar = [], []
    for s in range(N_SEEDS):
        raw = generate_raw(P_NOISE, n_per_seed=n, seed=300 + s)
        af.append(train_student(cascade(raw)))
        ar.append(train_student(raw))
    mf, mr = sum(af) / N_SEEDS, sum(ar) / N_SEEDS
    filtre_curve.append(mf)
    print(f"    {n:<8} {mf:<10.3f} {mr:<10.3f}")
print("  -> a volume egal, la QUALITE (filtrage) bat le VOLUME (brut) ; le gain")
print("     de volume SATURE (doubler le brut bruite n'achete presque rien).")


# --- 5) Contamination injectee ---
print(f"\n  Contamination injectee :")
rng_c = random.Random(999)
clean_pool = cascade(generate_raw(P_NOISE, n_per_seed=20, seed=999))
N_INJECT = 3
injected = EVAL_SET[:N_INJECT]                     # on TRICHE : eval dans le train
poisoned = clean_pool + [(t, l) for t, l in injected]
detected = contamination_check(poisoned, EVAL_SET)
acc_clean = train_student(clean_pool)
acc_poisoned = train_student(poisoned)
print(f"    exemples injectes      : {N_INJECT}")
print(f"    detectes par le check  : {detected}")
print(f"    accuracy SANS triche   : {acc_clean:.3f}")
print(f"    accuracy AVEC triche   : {acc_poisoned:.3f}  (gonflee artificiellement)")
print("  -> un eval set jamais publie est indispensable : si l'eval fuit dans")
print("     le train (via le teacher qui l'a lu), l'accuracy ment (piege #1).")


# ============================================================================
# EXERCISE 6 — Distillation de sequences (SFT sur outputs teacher) & mode collapse
# ============================================================================
# But : implementer la methode dominante 2026 (Type 2) — pas d'acces aux
# logits, on SFT le student sur les REPONSES completes du teacher — et montrer
# le mode collapse quand le teacher genere a temperature trop basse.
#
# Tache jouet : a partir d'un prompt = 1 token "categorie", le teacher produit
# une continuation (token suivant). A temp basse il sort presque toujours LA
# meme continuation par categorie ; a temp haute, plusieurs continuations
# valides. Le student est un bag-of-tokens -> next-token (table de comptage).

print("\n" + "=" * 70)
print("EXERCISE 6 : distillation de sequences (SFT outputs) & mode collapse")
print("=" * 70)

# Grammaire : chaque prompt-categorie a PLUSIEURS continuations VALIDES.
# C'est crucial : la diversite n'est "perdable" que s'il y a de la diversite
# a perdre. La regle valide = "la continuation appartient au set autorise".
GRAMMAR = {
    "A": ["a1", "a2", "a3"],     # 3 continuations valides
    "B": ["b1", "b2", "b3"],
    "C": ["c1", "c2", "c3"],
}
PROMPTS = list(GRAMMAR.keys())


def teacher_continue(prompt, temp, rng):
    """Teacher = regle stochastique. A temp basse il pique sur la 1ere
    continuation (mode collapse a la generation) ; a temp haute il echantillonne
    uniformement parmi les valides (diversite)."""
    options = GRAMMAR[prompt]
    # Logits jouets : 1ere option favorisee. temp basse -> softmax pique dessus.
    base_logits = [2.0, 1.0, 1.0][:len(options)]
    probs = softmax(base_logits, T=temp)
    r = rng.random()
    cum = 0.0
    for opt, p in zip(options, probs):
        cum += p
        if r <= cum:
            return opt
    return options[-1]


def generate_dataset(temp, k_per_prompt, seed):
    """Pour chaque prompt seed, le teacher genere K completions a temperature
    temp. C'est la distillation Type 2 : on ne garde QUE (prompt, completion),
    aucun acces aux logits du teacher."""
    rng = random.Random(seed)
    data = []
    for p in PROMPTS:
        for _ in range(k_per_prompt):
            data.append((p, teacher_continue(p, temp, rng)))
    return data


class NextTokenStudent:
    """Modele de sequence minimal : table prompt -> distribution des next tokens
    apprise par COMPTAGE (SFT next-token). A la generation, il echantillonne
    selon les frequences vues -> il ne peut produire que ce qu'il a vu."""

    def __init__(self):
        self.counts = defaultdict(Counter)

    def fit(self, pairs):
        for prompt, nxt in pairs:
            self.counts[prompt][nxt] += 1

    def sample(self, prompt, rng):
        c = self.counts.get(prompt)
        if not c:
            return None
        tokens, weights = zip(*c.items())
        total = sum(weights)
        r = rng.random() * total
        cum = 0.0
        for tok, w in zip(tokens, weights):
            cum += w
            if r <= cum:
                return tok
        return tokens[-1]


def evaluate_student(student, n_samples=120, seed=7):
    """Mesure (a) diversite = nb de continuations DISTINCTES produites par
    prompt (moyenne), (b) accuracy = fraction de continuations VALIDES."""
    rng = random.Random(seed)
    distinct_per_prompt = []
    valid = total = 0
    for p in PROMPTS:
        seen = set()
        for _ in range(n_samples):
            out = student.sample(p, rng)
            total += 1
            if out is not None:
                seen.add(out)
                valid += (out in GRAMMAR[p])
        distinct_per_prompt.append(len(seen))
    diversity = sum(distinct_per_prompt) / len(PROMPTS)
    accuracy = valid / total
    return diversity, accuracy


K_PER_PROMPT = 40
# Student entraine sur un dataset genere a temp BASSE (teacher peu diversifie).
ds_low = generate_dataset(temp=0.1, k_per_prompt=K_PER_PROMPT, seed=1)
st_low = NextTokenStudent()
st_low.fit(ds_low)
div_low, acc_low = evaluate_student(st_low)

# Student entraine sur un dataset genere a temp HAUTE (teacher diversifie).
ds_high = generate_dataset(temp=1.0, k_per_prompt=K_PER_PROMPT, seed=1)
st_high = NextTokenStudent()
st_high.fit(ds_high)
div_high, acc_high = evaluate_student(st_high)

# Diversite des DATASETS eux-memes (combien de continuations distinctes le
# teacher a produites par prompt).
def dataset_diversity(ds):
    by_prompt = defaultdict(set)
    for p, c in ds:
        by_prompt[p].add(c)
    return sum(len(v) for v in by_prompt.values()) / len(by_prompt)


print(f"\n  Teacher genere K={K_PER_PROMPT} completions/prompt ; student = SFT next-token.\n")
print(f"  {'Source':<28} {'diversite dataset':<19} {'diversite student':<19} {'accuracy':<10}")
print("  " + "-" * 76)
print(f"  {'temp=0.1 (peu diversifie)':<28} {dataset_diversity(ds_low):<19.2f} "
      f"{div_low:<19.2f} {acc_low:<10.2f}")
print(f"  {'temp=1.0 (diversifie)':<28} {dataset_diversity(ds_high):<19.2f} "
      f"{div_high:<19.2f} {acc_high:<10.2f}")
print("\n  -> temp BASSE cote teacher = completions quasi identiques -> le student")
print("     n'apprend qu'UNE facon de repondre (mode collapse) : faible")
print("     diversite de sortie, meme si chaque sortie reste valide.")
print("  -> temp HAUTE = teacher diversifie -> student plus varie (mais risque")
print("     de capter des completions limite si temp trop haute).")
print("  -> Compromis diversite/qualite (piege #2 du cours) : varier la")
print("     temperature de generation evite le mode collapse sans sacrifier")
print("     la validite (ici les continuations restent dans la grammaire).")


# ============================================================================
# ASSERTIONS — le fichier est auto-verifiant
# ============================================================================
print("\n" + "=" * 70)
print("ASSERTIONS (self-check)")
print("=" * 70)

# --- Exercise 4 ---
# (a) le student distille generalise au moins aussi bien que hard-only.
assert acc_dist >= acc_hard - 0.02, \
    f"distille ne tient pas l'accuracy : {acc_dist:.3f} vs hard {acc_hard:.3f}"
# (b) il imite mieux la distribution du teacher (calibration, dark knowledge).
assert kl_dist < kl_hard, \
    f"distille ne colle pas mieux : KL {kl_dist:.4f} vs hard {kl_hard:.4f}"
# (c) l'ablation T : le sweet spot n'est PAS a T=1 (sinon pas de dark knowledge).
assert best_T > 1.0, f"sweet spot a T={best_T} : pas de dark knowledge transmise"
print("  [Ex4] distille >= hard en accuracy, KL plus basse, sweet spot T>1 -> OK")

# --- Exercise 5 ---
# (a) le judge jette des exemples (il neutralise le bruit teacher).
assert after_judge < after_rule, "le judge devrait rejeter des labels conflictuels"
# (b) synth filtre > synth brut quand le teacher est bruite (moyennes).
assert m_filtre > m_brut, f"filtre ({m_filtre:.3f}) doit battre brut ({m_brut:.3f})"
# (c) un teacher mediocre plombe le student meme apres filtrage.
assert m_mediocre <= m_filtre, "teacher mediocre devrait <= teacher correct filtre"
# (d) la contamination injectee est detectee (au moins autant qu'injecte).
assert detected >= N_INJECT, f"contamination ratee : {detected} < {N_INJECT}"
# (e) la triche gonfle l'accuracy.
assert acc_poisoned >= acc_clean, "la contamination devrait gonfler l'accuracy"
print("  [Ex5] judge filtre ; filtre > brut ; contamination detectee+gonflee -> OK")

# --- Exercise 6 ---
# (a) sequence distillation SANS logits : le student n'a vu que des (prompt, out).
# (b) temp basse -> diversite student plus faible (mode collapse).
assert div_low < div_high, \
    f"temp basse devrait reduire la diversite : {div_low:.2f} >= {div_high:.2f}"
# (c) les sorties restent valides (la qualite ne s'effondre pas).
assert acc_low > 0.95 and acc_high > 0.95, "les continuations doivent rester valides"
print("  [Ex6] SFT sur outputs ; temp basse -> moins de diversite (collapse) -> OK")

print("\nDone (MEDIUM). Toutes les assertions passent.")
