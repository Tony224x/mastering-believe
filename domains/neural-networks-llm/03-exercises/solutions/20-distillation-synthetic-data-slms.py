"""
Solutions EASY — Jour 20 : Distillation, donnees synthetiques & SLMs
===================================================================
Exercices 1, 2, 3 (faciles). PUR PYTHON STDLIB (math seul), comme
02-code/20-distillation-synthetic-data-slms.py — PAS de numpy, pour rester
fidele au module et tourner sans dependance.

Chaque etape non triviale est commentee avec le POURQUOI. Le fichier est
auto-verifiant : il se termine par des assertions qui echouent si une
propriete pedagogique attendue n'est plus vraie.

Run: python3 03-exercises/solutions/20-distillation-synthetic-data-slms.py
"""

from __future__ import annotations
import sys
import io
import math

# Garde UTF-8 : certains terminaux Windows/CI sortent en cp1252 et plantent
# sur les accents francais. On force un wrapper UTF-8 tolerant.
if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# ============================================================================
# Petites primitives numeriques A LA MAIN (pas de numpy)
# ============================================================================

def softmax(logits: list[float], T: float = 1.0) -> list[float]:
    """Softmax numeriquement stable a temperature T.

    POURQUOI le -max : exp(grand) deborde en float ; soustraire le max ne
    change pas le resultat (invariance par translation du softmax) mais garde
    les exposants <= 0.
    POURQUOI /T : la temperature LISSE (T>1) ou PIQUE (T<1) la distribution.
    A T grand, softmax(z/T) tend vers l'uniforme ; a T->0, vers un one-hot.
    """
    scaled = [z / T for z in logits]
    m = max(scaled)
    exps = [math.exp(s - m) for s in scaled]
    Z = sum(exps)
    return [e / Z for e in exps]


def kl_divergence(p: list[float], q: list[float]) -> float:
    """KL(p || q) = sum_i p_i log(p_i / q_i). >= 0 (Gibbs), = 0 ssi p == q.

    En distillation p = teacher (cible fixe), q = student. C'est la mesure de
    "a quel point le student s'ecarte de la distribution du teacher".
    """
    s = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:                      # 0 * log(0) = 0 par convention
            s += pi * math.log(pi / (qi + 1e-12))
    return max(s, 0.0)                  # clamp : le bruit float peut donner -1e-17


def approx(a: float, b: float, tol: float = 0.02) -> bool:
    """Egalite a une tolerance absolue (pour comparer aux valeurs du .md)."""
    return abs(a - b) <= tol


# ============================================================================
# EXERCISE 1 — Distillation de logits : softmax a temperature a la main
# ============================================================================
# But : voir ce que le student "apprend en plus" via la distribution complete
# du teacher (dark knowledge) plutot que la seule classe gagnante, et le role
# de la temperature T. Les classes 1 et 2 (indices 1,2 dans z) sont des
# perdantes : leur ORDRE relatif encode "la classe 2 ressemble plus a la bonne
# reponse que la classe 3".

print("=" * 70)
print("EXERCISE 1 : softmax a temperature & dark knowledge")
print("=" * 70)

z_teacher = [4.0, 2.0, 1.0, 0.5]       # logits sur 4 classes (classe 0 gagne)

# 1) Softmax T=1 : la distribution "naturelle" du teacher.
p1 = softmax(z_teacher, T=1.0)
winner = max(range(len(p1)), key=lambda i: p1[i])
print(f"\n  z_teacher        = {z_teacher}")
print(f"  softmax(T=1)     = {[round(p, 3) for p in p1]}")
print(f"  classe gagnante  = {winner}  (proba {p1[winner]:.3f})")
# Le .md donne l'ordre de grandeur ~[0.78, 0.105, 0.039, 0.024] ; la valeur
# EXACTE de softmax([4,2,1,0.5]) est [0.823, 0.111, 0.041, 0.025] (la classe 0
# gagne nettement, ~82%). On asserte la realite calculee, pas l'approximation.
assert winner == 0, "la classe 0 doit gagner a T=1"
assert p1[0] > 0.75, f"la gagnante doit dominer (~80%), obtenu p[0]={p1[0]:.3f}"
assert approx(p1[0], 0.823, tol=0.005), f"p[0]={p1[0]:.3f}, attendu ~0.823 (exact)"

# 2) Softmax a temperature plus haute : la distribution se LISSE.
p2 = softmax(z_teacher, T=2.0)
p4 = softmax(z_teacher, T=4.0)
print(f"\n  softmax(T=2)     = {[round(p, 3) for p in p2]}")
print(f"  softmax(T=4)     = {[round(p, 3) for p in p4]}")
# Quand T monte : la gagnante BAISSE, les perdantes MONTENT.
print(f"\n  classe 0 (gagnante) : T=1 -> {p1[0]:.3f}  T=2 -> {p2[0]:.3f}  T=4 -> {p4[0]:.3f}  (baisse)")
for c in (1, 2, 3):
    print(f"  classe {c} (perdante) : T=1 -> {p1[c]:.3f}  T=2 -> {p2[c]:.3f}  T=4 -> {p4[c]:.3f}  (monte)")
assert p4[0] < p2[0] < p1[0], "la gagnante doit baisser quand T monte"
for c in (1, 2, 3):
    assert p4[c] > p1[c], f"la perdante {c} doit monter quand T monte"
print("\n  -> POURQUOI une distribution plus DOUCE transmet plus d'info qu'un")
print("     one-hot [1,0,0,0] : le one-hot dit seulement 'c'est la classe 0'.")
print("     Le soft target dit EN PLUS 'et la classe 1 est presque aussi")
print("     plausible, la classe 3 pas du tout' -> il transmet la SIMILARITE")
print("     entre classes (la dark knowledge), invisible dans le hard label.")

# 3) Dark knowledge : le rapport p(classe 1)/p(classe 2) encode la structure
#    relative des perdantes. Il vaut exp((z1 - z2)/T) car le softmax est une
#    exponentielle des logits : p_i/p_j = exp((z_i - z_j)/T).
ratio_T1 = p1[1] / p1[2]
ratio_T4 = p4[1] / p4[2]
gap = z_teacher[1] - z_teacher[2]      # 2.0 - 1.0 = 1.0
print(f"\n  rapport p(classe 1)/p(classe 2) :")
print(f"    T=1 : {ratio_T1:.3f}  (= exp({gap}/1) = {math.exp(gap / 1.0):.3f})")
print(f"    T=4 : {ratio_T4:.3f}  (= exp({gap}/4) = {math.exp(gap / 4.0):.3f})")
assert approx(ratio_T1, math.exp(gap / 1.0), tol=0.01)
assert approx(ratio_T4, math.exp(gap / 4.0), tol=0.01)
# L'ORDRE (classe 1 > classe 2) est conserve a toute T (le rapport reste > 1) :
# T compresse la structure mais ne l'inverse pas.
assert ratio_T1 > ratio_T4 > 1.0, "T eleve compresse le rapport mais le garde > 1"
print("  -> T eleve COMPRESSE le rapport (de ~2.72 vers ~1.28) mais le garde")
print("     > 1 : la structure relative des perdantes est preservee, juste")
print("     rendue plus lisible (gradients non nuls sur les perdantes).")

# 4) Hard label vs soft label : exemple concret.
print("\n  Hard vs soft (exemple 'chat' vs 'tigre' vs 'voiture') :")
print("    one-hot [chat=1, tigre=0, voiture=0] efface que tigre RESSEMBLE a")
print("    chat alors que voiture n'a rien a voir. Le soft target du teacher")
print("    (chat 0.78, tigre 0.10, voiture 0.02) transmet cette proximite")
print("    semantique -> le student apprend la geometrie des classes, pas")
print("    seulement la bonne reponse.")


# ============================================================================
# EXERCISE 2 — KL divergence teacher -> student
# ============================================================================
# But : implementer la loss de distillation de logits (KL(teacher || student))
# et voir qu'elle est minimale (=0) quand le student copie le teacher.

print("\n" + "=" * 70)
print("EXERCISE 2 : KL divergence teacher -> student")
print("=" * 70)

z_teacher2 = [3.0, 1.0, 0.2]
candidates = {
    "A (copie exacte)":      [3.0, 1.0, 0.2],
    "B (plus plate)":        [2.0, 1.5, 1.0],
    "C (plus piquee)":       [5.0, 0.0, -2.0],
}
T = 2.0

# 1) Distributions teacher/student a T=2.
p_teacher = softmax(z_teacher2, T=T)
print(f"\n  p_teacher (T={T}) = {[round(p, 3) for p in p_teacher]}")

# 2) KL(teacher || student) pour chaque candidat.
kls = {}
for name, z_student in candidates.items():
    p_student = softmax(z_student, T=T)
    kl = kl_divergence(p_teacher, p_student)
    kls[name] = kl
    print(f"  {name:<22} p_student={[round(p, 3) for p in p_student]}  KL={kl:.4f}")

# 3) Classement par KL croissante : A (copie) est le meilleur (KL ~ 0).
ranked = sorted(kls, key=kls.get)
print(f"\n  Classement par KL croissante : {ranked}")
assert approx(kls["A (copie exacte)"], 0.0, tol=1e-9), "A doit avoir KL ~ 0"
assert kls["B (plus plate)"] > 0.0 and kls["C (plus piquee)"] > 0.0, "B et C > 0"
assert ranked[0].startswith("A"), "le meilleur student doit etre A"
print("  -> A copie exactement le teacher -> KL = 0 (le meilleur).")
print("     B (trop plat) et C (sur-confiant) divergent -> KL > 0.")
print("  -> POURQUOI KL >= 0 toujours (inegalite de Gibbs) et = 0 SSI les")
print("     distributions sont identiques : KL mesure une 'distance' (non")
print("     symetrique) entre distributions, nulle seulement a l'egalite.")

# 4) Loss combinee : alpha * T^2 * KL + (1-alpha) * CE(hard_label, student).
print("\n  Loss combinee = alpha * T^2 * KL(teacher,student) + (1-alpha)*CE(hard,student)")
print("    - le terme CE sur le VRAI label ancre le student sur la bonne")
print("      reponse : le teacher peut se tromper, le hard label corrige.")
print("    - le facteur T^2 RESCALE le gradient : on a divise les logits par T,")
print("      donc le gradient des soft targets est ~1/T^2 ; multiplier la loss")
print("      par T^2 ramene le gradient soft a la meme echelle que le terme CE")
print("      (sinon, a T grand, le signal de distillation s'evanouirait).")


# ============================================================================
# EXERCISE 3 — Economie de la distillation : break-even
# ============================================================================
# But : refaire le calcul "distiller ou rester sur API frontier ?" du cours et
# trouver le volume de tokens au-dela duquel la distillation est rentable.
# Conventions du .md : V = millions de tokens / mois, couts exprimes en k$.

print("\n" + "=" * 70)
print("EXERCISE 3 : economie de la distillation — break-even")
print("=" * 70)

# NOTE sur les unites (conventions du .md) : on suit le cadrage du cours qui
# exprime tout en "k$" avec V en MILLIONS de tokens/mois. Les coefficients
# tombent alors sur des nombres ronds :
#   cout_api(V)     = 12 * V * 0.50 = 6 * V        (k$)
#   cout_distill(V) = 30 + 12 * V * 0.02 = 30 + 0.24 * V  (k$, le fixe = 30 k$)
# (Ce sont des chiffres jouets coherents avec le cours, pas un devis reel.)
PRICE_API = 0.50          # coeff API par M tokens/mois -> 12*0.50 = 6 k$/an/M
COST_DISTILL_FIXED_K = 30.0     # k$ one-shot (generation dataset + training)
PRICE_SELFHOST = 0.02     # coeff self-host par M tokens/mois -> 12*0.02 = 0.24
MONTHS = 12

SLOPE_API = MONTHS * PRICE_API          # 6   (k$ par M tokens/mois sur l'annee)
SLOPE_DISTILL = MONTHS * PRICE_SELFHOST  # 0.24


def cost_api_k(V: float) -> float:
    """Cout API sur V (millions tokens/mois) pendant 12 mois, en k$ : 6 * V."""
    return SLOPE_API * V


def cost_distill_k(V: float, maint_monthly_k: float = 0.0) -> float:
    """Cout distillation sur 12 mois, en k$ : fixe + inference + maintenance.
    30 + 0.24*V (+ 12*maint annuelle si fournie)."""
    return COST_DISTILL_FIXED_K + SLOPE_DISTILL * V + MONTHS * maint_monthly_k


# 1-2) Verification des formules en k$ (V en millions/mois).
V_test = 10.0
assert approx(cost_api_k(V_test), 6.0 * V_test, tol=1e-6)
assert approx(cost_distill_k(V_test), 30.0 + 0.24 * V_test, tol=1e-6)
print(f"\n  cout_api(V)     = {SLOPE_API:.0f} * V  (k$, V en M tokens/mois)")
print(f"  cout_distill(V) = 30 + {SLOPE_DISTILL:.2f} * V  (k$)")


def break_even(maint_monthly_k: float = 0.0) -> float:
    """Resout cost_api(V) = cost_distill(V) pour V.
    6*V = (30 + 12*maint) + 0.24*V  ->  V = fixe_total / (6 - 0.24)."""
    fixed_total = COST_DISTILL_FIXED_K + MONTHS * maint_monthly_k
    return fixed_total / (SLOPE_API - SLOPE_DISTILL)


# 3) Break-even SANS maintenance.
be = break_even()
print(f"\n  Break-even sans maintenance : V = 30 / (6 - 0.24) = 30 / 5.76 = {be:.2f}M tokens/mois")
assert approx(be, 30.0 / 5.76, tol=0.01)
assert 5.0 < be < 5.5, "break-even attendu ~5.2M tokens/mois (ordre de grandeur du cours)"

# 4) Qui gagne a V = 1M, 10M, 100M tokens/mois ?
print(f"\n  {'V (M/mois)':<12} {'cout_api(k$)':<14} {'cout_distill(k$)':<18} {'gagnant':<10}")
print("  " + "-" * 54)
for V in (1.0, 10.0, 100.0):
    ca, cd = cost_api_k(V), cost_distill_k(V)
    winner_eco = "API" if ca < cd else "DISTILL"
    print(f"  {V:<12.0f} {ca:<14.2f} {cd:<18.2f} {winner_eco:<10}")
# V=1M sous le seuil -> API ; V=10M et 100M au-dessus -> distillation gagne.
assert cost_api_k(1.0) < cost_distill_k(1.0), "a 1M tokens/mois l'API gagne"
assert cost_api_k(10.0) > cost_distill_k(10.0), "a 10M tokens/mois distiller gagne"
assert cost_api_k(100.0) > cost_distill_k(100.0), "a 100M tokens/mois distiller gagne"
print("  -> Recoupe le seuil '10-50M tokens/mois' du cours : sous le seuil,")
print("     rester sur API + caching ; au-dessus, distiller.")

# 5) Cout cache : ajouter 5000 $/mois = 5 k$/mois de maintenance (equipe ML).
be_maint = break_even(maint_monthly_k=5.0)
print(f"\n  Avec maintenance 5000 $/mois (+60 k$/an) :")
print(f"    break-even = (30 + 60) / 5.76 = 90 / 5.76 = {be_maint:.2f}M tokens/mois")
assert approx(be_maint, 90.0 / 5.76, tol=0.05)
assert be_maint > be, "la maintenance doit FAIRE MONTER le break-even"
print("  -> Le break-even MONTE (de ~5.2M a ~15.6M) : les couts caches")
print("     (maintenance) rendent la distillation rentable plus tard.")
print("  -> Lecon : ne jamais distiller sans compter la maintenance ; sous le")
print("     vrai seuil, API frontier + prompt caching + routeur reste gagnant.")


# ============================================================================
# Fin
# ============================================================================
print("\n" + "=" * 70)
print("Done (EASY). Toutes les assertions passent.")
print("=" * 70)
