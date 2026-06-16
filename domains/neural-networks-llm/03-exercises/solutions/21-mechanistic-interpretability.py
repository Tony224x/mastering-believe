"""
Solutions EASY — Jour 21 : Mechanistic interpretability
=======================================================
Exercices 1, 2, 3 (faciles). Pur NumPy + stdlib, comme
02-code/21-mechanistic-interpretability.py. Chaque etape non triviale est
commentee avec le POURQUOI. Le fichier est auto-verifiant : il se termine par
des assertions qui echouent si une propriete pedagogique attendue casse.

  1. Logit lens a la main : softmax + entropie par couche, entropie qui chute.
  2. Probing : selectivity (vraie - control) et pourquoi correlation != causalite.
  3. Superposition : capacite R^2, pentagone d'Elhage, role de la sparsity.

Run: python3 03-exercises/solutions/21-mechanistic-interpretability.py
"""

from __future__ import annotations
import sys
import io
import numpy as np

# Stdout en UTF-8 (Windows/CI-friendly), comme 02-code/21.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax stable 1D (subtract max), identique a 02-code/21 PART 2."""
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def entropy(p: np.ndarray) -> float:
    """Entropie de Shannon H = -sum p log p (nats). Max = log(K) (uniforme)."""
    return float(-np.sum(p * np.log(p + 1e-12)))


# ============================================================================
# EXERCISE 1 — Logit lens a la main : entropie qui chute
# ============================================================================
# But : le logit lens projette le residual stream sur le vocab a chaque couche.
# On mesure l'entropie de la distribution : elle TOMBE de ~uniforme (le modele
# "ne sait pas encore") vers ~piquee (le modele "s'est decide") en profondeur.

print("=" * 70)
print("EXERCISE 1 : logit lens — entropie qui chute couche par couche")
print("=" * 70)

# Logits projetes par couche (vocab de 4 tokens). Donnees de l'enonce.
layer_logits = {
    "Layer 1": np.array([0.1, 0.0, -0.1, 0.05]),   # quasi plat
    "Layer 2": np.array([1.5, 0.2, 0.1, 0.3]),
    "Layer 3": np.array([5.0, 0.5, 0.2, 0.1]),     # pique sur le token 0
}
H_MAX = np.log(4)        # entropie max sur 4 tokens ~ 1.386

print(f"\n  Entropie max (uniforme sur 4 tokens) = log(4) = {H_MAX:.3f}\n")
print(f"  {'Couche':<10} {'softmax(p)':<34} {'entropie':<10} {'top-1':<8}")
print("  " + "-" * 62)

probs_by_layer = {}
entropies = []
for name, logits in layer_logits.items():
    p = softmax(logits)
    h = entropy(p)
    probs_by_layer[name] = p
    entropies.append(h)
    top1 = int(np.argmax(p))
    p_str = "[" + ", ".join(f"{v:.3f}" for v in p) + "]"
    print(f"  {name:<10} {p_str:<34} {h:<10.3f} t{top1:<7}")

# Criteres du .md : L1 ~ [0.27, 0.25, 0.22, 0.26] (entropie ~1.38),
#                   L3 ~ [0.97, 0.01, 0.01, 0.01] (entropie << 1).
p1, p3 = probs_by_layer["Layer 1"], probs_by_layer["Layer 3"]
assert np.allclose(p1, [0.27, 0.25, 0.22, 0.26], atol=0.02), f"L1 p={p1}"
assert abs(entropies[0] - H_MAX) < 0.01, "L1 doit etre ~ uniforme (entropie ~ log 4)"
assert p3[0] > 0.95, f"L3 doit etre piquee sur t0, p[0]={p3[0]:.3f}"
assert entropies[-1] < 0.5, "L3 doit avoir une entropie << 1"

# L'entropie DECROIT strictement L1 -> L3 : le modele se decide en profondeur.
assert entropies[0] > entropies[1] > entropies[2], "l'entropie doit decroitre"
most_uniform = max(range(3), key=lambda i: entropies[i])
most_decided = min(range(3), key=lambda i: entropies[i])
print(f"\n  -> entropie L1={entropies[0]:.3f} > L2={entropies[1]:.3f} > L3={entropies[2]:.3f}")
print(f"  -> couche la plus proche de l'uniforme : Layer {most_uniform + 1}")
print(f"  -> couche la plus 'decidee' : Layer {most_decided + 1}")
print("  -> Lecture mech interp : l'entropie qui chute = le modele cristallise")
print("     sa prediction en profondeur (nostalgebraist, logit lens 2020).")

# Lecture mech interp : token apparait puis disparait = suppression circuit.
print("\n  Si le bon token (t0) etait top-1 en L2 puis DISPARAISSAIT en L3, on")
print("  soupconnerait un *suppression circuit* : une tete tardive qui ECRIT")
print("  une direction negative pour tuer un token que les couches precedentes")
print("  avaient deja propose (cf cours, debug par logit lens).")


# ============================================================================
# EXERCISE 2 — Probing : correlation n'est pas causalite
# ============================================================================
# But : distinguer probing (l'info est-elle PRESENTE ?) d'activation patching
# (l'info est-elle UTILISEE ?). La selectivity (Hewitt & Liang 2019) borne la
# confiance qu'on accorde a un probe.

print("\n" + "=" * 70)
print("EXERCISE 2 : probing — selectivity & correlation != causalite")
print("=" * 70)

acc_true = 0.92        # probe sur la vraie tache (singulier/pluriel)
acc_control = 0.88     # control task : MEMES hidden states, labels ALEATOIRES

# 1) Selectivity = acc_vraie - acc_control. Faible => suspect.
selectivity = acc_true - acc_control
print(f"\n  accuracy vraie tache   = {acc_true:.2%}")
print(f"  accuracy control task  = {acc_control:.2%}  (labels aleatoires)")
print(f"  selectivity = {acc_true:.2%} - {acc_control:.2%} = {selectivity:.2%} "
      f"({selectivity * 100:.0f} points)")
assert abs(selectivity - 0.04) < 1e-9, "selectivity = 92 - 88 = 4 points"
assert selectivity < 0.10, "une selectivity < 10 points est suspecte"
print("  -> selectivity TRES FAIBLE (4 points) : SUSPECT. Le probe atteint")
print("     presque le meme score sur des labels ALEATOIRES -> il memorise")
print("     plutot qu'il ne decode une vraie feature.")

# 2) Pourquoi un control a 88% est alarmant.
print("\n  POURQUOI un control eleve (88% sur du bruit) alarme : un probe capable")
print("  de coller a des labels aleatoires est TROP expressif (ou les activations")
print("  sont si riches qu'on y 'lit' n'importe quoi). Son score sur la vraie")
print("  tache ne prouve alors rien de specifique.")

# 3) Pourquoi un probe LINEAIRE (pas un MLP).
print("\n  POURQUOI un probe LINEAIRE et non un MLP : un MLP non-lineaire 'trouve")
print("  toujours quelque chose' (il peut fitter n'importe quelle frontiere) ->")
print("  ses conclusions ne sont plus interpretables. Le probe lineaire BORNE")
print("  l'expressivite : s'il decode l'info, c'est qu'elle est lineairement")
print("  presente dans le residual (une vraie 'direction').")

# 4) Le saut causal : meme une excellente selectivity ne prouve pas l'usage.
selectivity_good = 0.92 - 0.50
print(f"\n  Le saut causal : meme avec une selectivity excellente (ex 92% vraie,")
print(f"  50% control -> {selectivity_good:.0%}), le probe montre que l'info est")
print("  PRESENTE, PAS que le modele l'UTILISE pour sa prediction.")
print("  -> Pour la CAUSALITE il faut l'ACTIVATION PATCHING : substituer")
print("     l'activation (clean <-> corrupted) et mesurer l'effet sur la sortie.")
print("     Si patcher cette activation change la prediction, l'info est")
print("     causalement utilisee ; sinon elle n'est que correlee.")


# ============================================================================
# EXERCISE 3 — Superposition : pourquoi 5 features dans 2 dims ?
# ============================================================================
# But : le calcul de capacite qui explique la polysemy (1 neurone = plusieurs
# features) et le role de la sparsity (Elhage 2022).

print("\n" + "=" * 70)
print("EXERCISE 3 : superposition — 5 features dans 2 dimensions")
print("=" * 70)

n_features = 5
n_hidden = 2

# 1) Capacite orthogonale : dans R^d on tient au plus d directions orthogonales.
max_orthogonal = n_hidden          # R^2 -> 2 directions orthogonales max
must_superpose = n_features - max_orthogonal
print(f"\n  n_features={n_features}, n_hidden={n_hidden}")
print(f"  directions orthogonales max dans R^{n_hidden} = {max_orthogonal}")
print(f"  -> au moins {must_superpose} features DOIVENT se superposer "
      f"(partager des directions)")
assert max_orthogonal == 2, "R^2 tient 2 directions orthogonales"
assert must_superpose >= 3, "au moins 3 des 5 features doivent se superposer"

# 2) Le pentagone : 5 features placees aux sommets d'un pentagone regulier.
adjacent_angle = 360.0 / n_features            # 72 deg
cos_adjacent = float(np.cos(np.deg2rad(adjacent_angle)))
print(f"\n  Pentagone regulier (Elhage 2022) : 5 sommets dans R^2.")
print(f"  angle entre 2 features adjacentes = 360/5 = {adjacent_angle:.0f} deg")
print(f"  interference (cosinus) cos(72 deg) = {cos_adjacent:.3f}")
assert abs(adjacent_angle - 72.0) < 1e-9, "angle adjacent = 72 deg"
assert abs(cos_adjacent - 0.309) < 0.01, f"cos(72) ~ 0.309, obtenu {cos_adjacent:.3f}"
print("  -> interference FAIBLE (0.309) mais NON NULLE : les features adjacentes")
print("     se 'marchent' un peu dessus, c'est le prix de la superposition.")

# 3) Role de la sparsity : proba que 2 features données soient actives ENSEMBLE.
sparsity = 0.9
p_active = 1.0 - sparsity                       # proba qu'une feature soit active
p_coactive = p_active * p_active                # independance : 2 actives en meme temps
print(f"\n  sparsity={sparsity} -> une feature active avec proba {p_active:.2f}")
print(f"  co-activation de 2 features (independance) = {p_active:.2f} * {p_active:.2f} "
      f"= {p_coactive:.2f} ({p_coactive:.0%})")
assert abs(p_coactive - 0.01) < 1e-9, "co-activation = 0.1 * 0.1 = 0.01"
print("  -> POURQUOI la superposition est 'presque sans cout' : deux features ne")
print("     sont actives EN MEME TEMPS que ~1% du temps -> l'interference (cos72)")
print("     n'est 'payee' que rarement. Sous forte sparsity, stocker n > d")
print("     features est donc rationnel (Elhage 2022).")

# 4) Consequence pour mech interp.
print("\n  Consequence : si un neurone est une COMBINAISON de plusieurs features")
print("  superposees, 'un neurone = un concept' est FAUX (polysemy). Solution du")
print("  cours : les Sparse Autoencoders (SAE) qui projettent les activations")
print("  dans un espace plus grand pour DEMIXER les features mono-semantiques.")


# ============================================================================
# Fin
# ============================================================================
print("\n" + "=" * 70)
print("Done (EASY). Toutes les assertions passent.")
print("=" * 70)
