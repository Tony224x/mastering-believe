"""
Solutions EASY — Jour 16 : Mixture of Experts
=============================================
Exercices 1, 2, 3 (faciles). Pur NumPy, comme 02-code/16-mixture-of-experts.py.
Chaque etape non triviale est commentee avec le POURQUOI.

Run: python 03-exercises/solutions/16-mixture-of-experts.py
"""

import sys
import io
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def softmax(z):
    """Softmax numeriquement stable (on soustrait le max avant exp)."""
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


# ============================================================================
# Exercice 1 — Top-k routing a la main
# ============================================================================

def exercice_1():
    print("=" * 70)
    print("Exercice 1 : Top-k routing a la main (softmax + top-2 + renorm)")
    print("=" * 70)

    logits = np.array([2.0, 1.0, 3.0, 0.0])

    # 1) softmax brut sur les 4 experts.
    probs = softmax(logits)
    print(f"\n  logits        = {logits.tolist()}")
    print(f"  softmax       = {np.round(probs, 4).tolist()}  (somme = {probs.sum():.4f})")

    # 2) top-2 : on prend les 2 experts de plus forte proba.
    # argsort(-probs) trie en ordre decroissant ; on garde les 2 premiers.
    top2 = np.argsort(-probs)[:2]
    print(f"  top-2 experts = {top2.tolist()}  (experts 2 et 0, les 2 plus grands)")

    # 3) renormaliser les 2 poids choisis pour qu'ils somment a 1.
    # POURQUOI : sans renorm, w0 + w1 = 0.6652 + 0.2447 = 0.91 < 1 -> la sortie
    # serait attenuee. Renormaliser garde l'echelle de l'output stable quel que
    # soit k (le mix reste une moyenne ponderee convexe).
    top2_probs = probs[top2]
    w = top2_probs / top2_probs.sum()
    print(f"  poids bruts   = {np.round(top2_probs, 4).tolist()}")
    print(f"  poids renorm  = {np.round(w, 4).tolist()}  (somme = {w.sum():.4f})")

    # 4) sortie du MoE = somme ponderee des 2 experts choisis.
    # Les 2 experts choisis sont l'expert 2 (output [10,0]) puis l'expert 0 ([0,10]).
    # top2 = [2, 0], donc w[0] pondere expert 2 et w[1] pondere expert 0.
    expert_out = {2: np.array([10.0, 0.0]), 0: np.array([0.0, 10.0])}
    output = w[0] * expert_out[int(top2[0])] + w[1] * expert_out[int(top2[1])]
    print(f"  output        = {np.round(output, 2).tolist()}  "
          f"(= {w[0]:.3f}*[10,0] + {w[1]:.3f}*[0,10])")

    # 5) explication conceptuelle.
    print("\n  POURQUOI renormaliser : sans renorm, les poids ne somment pas a 1")
    print("  (ici 0.91) -> la sortie serait systematiquement attenuee. Renormaliser")
    print("  garde l'echelle de l'output constante quel que soit k.")

    # --- assertions de validation ---
    # NOTE : l'enonce annonce softmax ~ [0.2447, 0.09, 0.6652, 0.09] mais ces
    # chiffres sont approximatifs/errones (ils ne somment pas a 1). La vraie
    # valeur, avec le denominateur 31.19 donne dans l'enonce, est
    # exp([2,1,3,0])/31.19 = [0.2369, 0.0871, 0.6439, 0.0321]. C'est la
    # renormalisation top-2 [0.731, 0.269] et l'output [7.31, 2.69] qui comptent
    # (eux sont exacts car le ratio 0.6439/0.2369 = 0.6652/0.2447).
    assert np.allclose(probs, [0.2369, 0.0871, 0.6439, 0.0321], atol=1e-3)
    assert abs(probs.sum() - 1.0) < 1e-9
    assert sorted(top2.tolist()) == [0, 2]  # experts 2 et 0
    assert np.allclose(sorted(w), sorted([0.731, 0.269]), atol=1e-2)
    assert np.allclose(output, [7.31, 2.69], atol=1e-2)
    print("\n  [OK] Exercice 1")


# ============================================================================
# Exercice 2 — Comptabilite Mixtral : params totaux vs actifs
# ============================================================================

def exercice_2():
    print("\n" + "=" * 70)
    print("Exercice 2 : Comptabilite Mixtral (params totaux vs actifs)")
    print("=" * 70)

    d_model = 4096
    d_ff = 14336
    N = 8
    k = 2
    layers = 32

    # 1) params d'UN expert FFN (2 matrices up/down, on ignore SwiGLU).
    ffn_1_expert = 2 * d_model * d_ff
    print(f"\n  ffn_1_expert = 2 * {d_model} * {d_ff} = {ffn_1_expert / 1e6:.1f} M")

    # Briques partagees par couche.
    attn = 4 * d_model * d_model       # Q, K, V, O
    router = d_model * N               # 1 matmul pour le gating

    # 2) params TOTAUX d'une couche : attention partagee + N experts + routeur.
    total_layer = attn + N * ffn_1_expert + router

    # 3) params ACTIFS d'une couche : seuls k experts s'activent par token.
    active_layer = attn + k * ffn_1_expert + router

    # 4) extrapolation a toutes les couches (on ignore les embeddings ici).
    total = total_layer * layers
    active = active_layer * layers
    print(f"  total / layer  = {total_layer / 1e9:.3f} G  "
          f"(attn + 8 experts + router)")
    print(f"  actif / layer  = {active_layer / 1e9:.3f} G  "
          f"(attn + 2 experts + router)")
    print(f"  total ({layers} layers) = {total / 1e9:.2f} G")
    print(f"  actif ({layers} layers) = {active / 1e9:.2f} G")

    # 5) ratio de sparsite = total / actif.
    sparsity = total / active
    print(f"  ratio sparsite = {sparsity:.2f}x  (total / actif)")

    # 6) explication conceptuelle.
    print("\n  POURQUOI '8x7B' est trompeur : ca suggere 8*7B = 56B, mais les")
    print("  8 experts ne sont QUE les FFN ; l'attention, les embeddings et le")
    print("  LayerNorm sont PARTAGES -> ~47B reel (et non 56B).")
    print("  POURQUOI MoE n'economise PAS de VRAM : tous les experts doivent etre")
    print("  charges en memoire au cas ou un token les router. MoE economise les")
    print("  FLOPs (k/N), pas la VRAM.")
    print("  NOTE : ce calcul 2-matrices sous-estime le vrai Mixtral SwiGLU")
    print("  (3 matrices -> ~47B total / ~13B actifs).")

    # --- assertions de validation ---
    assert abs(ffn_1_expert - 117.4e6) < 0.2e6                # ~117.4 M
    assert abs(total / 1e9 - 32.0) < 1.0                      # ~32 G
    assert abs(active / 1e9 - 9.8) < 0.5                      # ~9.8 G
    assert abs(sparsity - 3.3) < 0.2                          # ~3.3x
    print("\n  [OK] Exercice 2")


# ============================================================================
# Exercice 3 — Load balancing loss : uniforme vs collapse
# ============================================================================

def load_balancing_loss(f, P, N):
    """
    Loss auxiliaire de Shazeer : L_aux = N * sum_i (f_i * P_i).
      f_i = fraction de tokens routes vers l'expert i (compte dur, non differentiable)
      P_i = proba softmax moyenne attribuee a l'expert i (differentiable)
    Minimale (=1) quand f et P sont uniformes ; maximale (=N) au collapse total.
    """
    f = np.asarray(f, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    return float(N * np.dot(f, P))


def exercice_3():
    print("\n" + "=" * 70)
    print("Exercice 3 : Load balancing loss (uniforme vs collapse)")
    print("=" * 70)

    N = 4

    # 1) Rappel : f_i = fraction de tokens vers l'expert i ; P_i = proba moyenne.
    print("\n  f_i = fraction de tokens routes vers l'expert i (dur)")
    print("  P_i = proba softmax moyenne attribuee a l'expert i (doux)")

    # Scenario A — uniforme : 1 token par expert, softmax moyen uniforme.
    f_A = [1 / 4, 1 / 4, 1 / 4, 1 / 4]
    P_A = [1 / 4, 1 / 4, 1 / 4, 1 / 4]
    L_A = load_balancing_loss(f_A, P_A, N)
    print(f"\n  Scenario A (uniforme) : f={f_A}, P={P_A}")
    print(f"    L_aux = 4 * sum(1/4 * 1/4) = 4 * 4 * 1/16 = {L_A:.4f}")

    # Scenario B — collapse : tous les tokens vers l'expert 0.
    f_B = [1.0, 0.0, 0.0, 0.0]
    P_B = [0.85, 0.05, 0.05, 0.05]
    L_B = load_balancing_loss(f_B, P_B, N)
    print(f"\n  Scenario B (collapse) : f={f_B}, P={P_B}")
    print(f"    L_aux = 4 * (1*0.85 + 0 + 0 + 0) = {L_B:.4f}")

    # 4) loss minimale en uniforme, grande en collapse ; max theorique = N.
    print(f"\n  Loss minimale (uniforme) = {L_A:.1f}  (= 1.0)")
    print(f"  Loss grande (collapse)   = {L_B:.1f}")
    print(f"  Max theorique            = N = {N}  (tout sur 1 expert avec P_0=1)")

    # 5) explication : pourquoi le produit f_i * P_i.
    print("\n  POURQUOI le produit f_i * P_i (et pas f_i ou P_i seul) :")
    print("  - f_i passe par un argmax/top-k -> NON differentiable (gradient nul).")
    print("  - P_i seul est doux mais ne contraint pas la charge REELLE dispatchee.")
    print("  - Leur produit accroche le gradient de P (differentiable) sur la")
    print("    distribution dure f : penaliser un expert sur-charge (f_i grand)")
    print("    pousse a baisser sa proba P_i. Genie de simplicite.")

    # --- assertions de validation ---
    assert abs(L_A - 1.0) < 1e-9
    assert abs(L_B - 3.4) < 1e-9
    # Max theorique = N quand f=[1,0,0,0] et P=[1,0,0,0].
    assert abs(load_balancing_loss([1, 0, 0, 0], [1, 0, 0, 0], N) - N) < 1e-9
    print("\n  [OK] Exercice 3")


if __name__ == "__main__":
    exercice_1()
    exercice_2()
    exercice_3()
    print("\nDone (EASY).")
