"""
Solutions EASY — Jour 17 : State Space Models
=============================================
Exercices 1, 2, 3 (faciles). Pur NumPy, comme 02-code/17-state-space-models.py.
Chaque etape non triviale est commentee avec le POURQUOI.

Run: python 03-exercises/solutions/17-state-space-models.py
"""

import sys
import io
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# ============================================================================
# Exercice 1 — Recurrence SSM a la main (mode recurrent / inference)
# ============================================================================

def ssm_scalar_recurrent(x, A, B, C, h0=0.0):
    """
    SSM scalaire (etat de dim 1) en mode recurrent :
      h_t = A * h_{t-1} + B * x_t
      y_t = C * h_t
    O(1) memoire par step : on ne garde QUE h (taille fixe), pas l'historique.
    """
    h = h0
    hs, ys = [], []
    for xt in x:
        h = A * h + B * xt            # mise a jour de l'etat
        hs.append(h)
        ys.append(C * h)              # readout
    return np.array(hs), np.array(ys)


def exercice_1():
    print("=" * 70)
    print("Exercice 1 : Recurrence SSM scalaire a la main")
    print("=" * 70)

    A, B, C = 0.5, 1.0, 2.0
    x = np.array([1.0, 0.0, 0.0, 1.0])

    # 1) + 2) derouler la recurrence (A=0.5, stable).
    h, y = ssm_scalar_recurrent(x, A, B, C)
    print(f"\n  A={A}, B={B}, C={C}, x={x.tolist()}")
    print(f"  h = {h.tolist()}")
    print(f"  y = {y.tolist()}")

    # 3) stabilite : A=1.5 instable.
    h_unst, y_unst = ssm_scalar_recurrent(x, 1.5, B, C)
    print(f"\n  Avec A=1.5 (instable) :")
    print(f"    h = {np.round(h_unst, 3).tolist()}  -> croit")
    print(f"    y = {np.round(y_unst, 3).tolist()}  -> explose")
    print("  POURQUOI |A|<1 : h_t = sum A^k B x_{t-k}. Si |A|>=1, A^k ne decroit")
    print("  pas et l'etat diverge ; |A|<1 garantit une serie geometrique bornee.")

    # 4) memoire : contribution de x_1=1 a h_4.
    # h_4 = A^3 B x_1 + A^2 B x_2 + A B x_3 + B x_4. Le terme de x_1 est A^3 B x_1.
    contrib_x1 = A ** 3 * B * x[0]
    print(f"\n  Contribution de x_1=1 a h_4 = A^3 * B * x_1 = {A}^3 = {contrib_x1}")
    print("  -> l'influence d'un token ancien s'efface exponentiellement (A^k).")
    print("  C'est pourquoi l'etat de taille fixe est un COMPRESSEUR LOSSY :")
    print("  il ne garde qu'une trace decroissante du passe.")

    # 5) memoire recurrente.
    print("\n  Memoire pour calculer y_t en recurrent = O(D) (on stocke juste h,")
    print("  taille fixe), INDEPENDANTE de la longueur N de la sequence.")

    # --- assertions de validation ---
    assert np.allclose(h, [1.0, 0.5, 0.25, 1.125])
    assert np.allclose(y, [2.0, 1.0, 0.5, 2.25])
    assert h_unst[-1] > h[-1]                       # A=1.5 -> etat plus grand
    assert abs(contrib_x1 - 0.125) < 1e-12
    print("\n  [OK] Exercice 1")


# ============================================================================
# Exercice 2 — Equivalence recurrent / convolutionnel
# ============================================================================

def ssm_kernel_scalar(A, B, C, N):
    """Kernel SSM scalaire : K[k] = C * A^k * B (reponse impulsionnelle au lag k)."""
    k = np.arange(N)
    return C * (A ** k) * B


def ssm_conv_scalar(x, K):
    """
    Convolution causale : y_t = sum_{k=0..t-1} K[k] * x[t-k].
    np.convolve(x, K) en 'full' donne la convolution complete ; on tronque a N
    pour garder le prefixe causal (les memes y_t que la recurrence).
    """
    N = len(x)
    return np.convolve(x, K)[:N]


def exercice_2():
    print("\n" + "=" * 70)
    print("Exercice 2 : Equivalence recurrent / convolutionnel")
    print("=" * 70)

    A, B, C = 0.5, 1.0, 2.0
    x = np.array([1.0, 0.0, 0.0, 1.0])
    N = len(x)

    # 1) kernel SSM.
    K = ssm_kernel_scalar(A, B, C, N)
    print(f"\n  K = C*A^k*B = {K.tolist()}  (= 2 * 0.5^k)")

    # 2) convolution causale.
    y_conv = ssm_conv_scalar(x, K)
    print(f"  y (conv) = {y_conv.tolist()}")

    # 3) comparaison avec le mode recurrent.
    _, y_rec = ssm_scalar_recurrent(x, A, B, C)
    diff = float(np.max(np.abs(y_conv - y_rec)))
    print(f"  y (rec)  = {y_rec.tolist()}")
    print(f"  max|y_conv - y_rec| = {diff:.2e}  (les deux modes sont IDENTIQUES)")

    # 4) explication parallelisme.
    print("\n  POURQUOI le mode conv est parallelisable au training :")
    print("  - conv = pas de dependance temporelle (chaque y_t est une somme")
    print("    independante) -> on peut tout calculer d'un coup, par FFT O(N log N).")
    print("  - recurrence = h_t depend de h_{t-1} -> sequentiel, pas parallelisable.")

    # 5) bonus : verification numerique sur sequence aleatoire (D=1 scalaire).
    rng = np.random.default_rng(17)
    x2 = rng.standard_normal(64)
    K2 = ssm_kernel_scalar(A, B, C, len(x2))
    _, y_rec2 = ssm_scalar_recurrent(x2, A, B, C)
    y_conv2 = ssm_conv_scalar(x2, K2)
    diff2 = float(np.max(np.abs(y_rec2 - y_conv2)))
    print(f"\n  Bonus (N=64 aleatoire) : max|rec - conv| = {diff2:.2e}")

    # --- assertions de validation ---
    assert np.allclose(K, [2.0, 1.0, 0.5, 0.25])
    assert np.allclose(y_conv, [2.0, 1.0, 0.5, 2.25])
    assert np.allclose(y_conv, y_rec)
    assert diff2 < 1e-10
    print("\n  [OK] Exercice 2")


# ============================================================================
# Exercice 3 — SSM vs RNN vs Transformer : la table de complexite
# ============================================================================

def exercice_3():
    print("\n" + "=" * 70)
    print("Exercice 3 : SSM vs RNN vs Transformer (table de complexite)")
    print("=" * 70)

    # 1) Table de complexite (cf cadre mental du cours).
    table = {
        "RNN (LSTM)": {
            "compute_train": "O(N)",
            "mem_step_inf": "O(1)",
            "parallel_train": "NON",
            "recall": "faible",
        },
        "Transformer": {
            "compute_train": "O(N^2)",
            "mem_step_inf": "O(N)",          # KV cache croit avec N
            "parallel_train": "OUI",
            "recall": "excellent",
        },
        "SSM (Mamba)": {
            "compute_train": "O(N log N)",   # FFT (S4) ou parallel scan
            "mem_step_inf": "O(1)",          # etat fixe
            "parallel_train": "OUI",
            "recall": "faible/moyen",
        },
    }
    hdr = f"  {'Archi':<16}{'compute train':<16}{'mem/step inf':<16}" \
          f"{'parallel?':<12}{'recall':<14}"
    print("\n" + hdr)
    print("  " + "-" * (len(hdr) - 2))
    for arch, r in table.items():
        print(f"  {arch:<16}{r['compute_train']:<16}{r['mem_step_inf']:<16}"
              f"{r['parallel_train']:<12}{r['recall']:<14}")

    # 2) classement memoire a N=100_000.
    N = 100_000
    D = 64                                   # taille d'etat fixe (ordre de grandeur)
    mem_attn = N * N                          # matrice d'attention N x N
    mem_kv = N                                # KV cache (lineaire en N)
    mem_state = D                             # etat SSM / RNN fixe
    print(f"\n  A N={N:,} (memoire, ordre de grandeur) :")
    print(f"    SSM / RNN (etat fixe)        ~ O(D)   = {mem_state}")
    print(f"    Transformer (KV cache)       ~ O(N)   = {mem_kv:,}")
    print(f"    Transformer (matrice N x N)  ~ O(N^2) = {mem_attn:,}")
    print("    Classement (moins -> plus) : SSM ~= RNN  <<  Transformer.")

    # 3) pourquoi RNN non parallelisable mais SSM oui.
    print("\n  POURQUOI un RNN n'est PAS parallelisable mais un SSM lineaire oui :")
    print("  - RNN : h_t = tanh(W h_{t-1} + ...) -> la NON-LINEARITE dans la")
    print("    recurrence interdit de derouler en convolution. Sequentiel obligatoire.")
    print("  - SSM lineaire : h_t = A h_{t-1} + B x_t est LINEAIRE en h -> se")
    print("    deroule en convolution (kernel K = C A^k B), donc parallelisable.")

    # 4) choix de backbone par workload.
    choix = {
        "(a) audio 200k samples": "Mamba/SSM pur (long context, throughput, recall peu critique)",
        "(b) agent code, recall def 40k tokens": "Transformer (recall associatif dense)",
        "(c) LM prod 256k + throughput": "Hybride Jamba-like (Mamba + rares attn)",
    }
    print("\n  Choix de backbone :")
    for w, c in choix.items():
        print(f"    {w:<40} -> {c}")

    # --- assertions de validation ---
    assert table["RNN (LSTM)"]["parallel_train"] == "NON"
    assert table["SSM (Mamba)"]["parallel_train"] == "OUI"
    assert table["Transformer"]["compute_train"] == "O(N^2)"
    assert mem_state < mem_kv < mem_attn          # classement memoire
    print("\n  [OK] Exercice 3")


if __name__ == "__main__":
    exercice_1()
    exercice_2()
    exercice_3()
    print("\nDone (EASY).")
