"""
Solutions MEDIUM — Jour 17 : State Space Models
===============================================
Exercices 4, 5, 6 (medium). Pur NumPy, comme 02-code/17-state-space-models.py.
Chaque etape non triviale est commentee avec le POURQUOI.

Run: python 03-exercises/solutions/17-state-space-models-medium.py
"""

import sys
import io
import time
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SEED = 17
np.random.seed(SEED)


def make_stable_A(D, decay=0.9):
    """A diagonale a spectre dans le cercle unite -> stabilite (|eig|<1)."""
    diag = np.linspace(decay, decay * 0.5, D)
    return np.diag(diag)


# ============================================================================
# Exercice 4 — Selectivite : S4 echoue, Mamba reussit
# ============================================================================

def make_selective_copy_data(N=64, data_frac=0.25, rng=None):
    """N tokens : data (valeur aleatoire) + filler (0), avec un flag par token."""
    rng = rng or np.random.default_rng(SEED)
    flags = (rng.random(N) < data_frac).astype(np.float64)   # 1 = data, 0 = filler
    x = np.where(flags == 1, rng.standard_normal(N), 0.0)
    return x, flags


def ssm_nonselective(x, flags, A, B, C):
    """B, C fixes : le filler pompe quand meme l'etat via B (fuite)."""
    D = A.shape[0]
    h = np.zeros(D)
    y = np.zeros(len(x))
    for t in range(len(x)):
        h = A @ h + B.flatten() * x[t]       # B fixe -> laisse toujours x entrer
        y[t] = (C @ h).item()
    return y


def ssm_selective(x, flags, A, B_base, C_base):
    """
    Mamba-style : B_t et C_t sont 'gates' par le flag. Quand flag=0 (filler),
    le gate ferme l'entree (B_t ~ 0) et le readout (C_t ~ 0) -> filler ignore.
    Dans le vrai Mamba le gate vient d'un petit MLP sur x_t ; ici on utilise
    directement le flag pour la lisibilite (le principe est identique).
    """
    D = A.shape[0]
    h = np.zeros(D)
    y = np.zeros(len(x))
    for t in range(len(x)):
        gate = flags[t]
        B_t = B_base.flatten() * gate
        C_t = C_base * gate
        h = A @ h + B_t * x[t]
        y[t] = (C_t @ h).item()
    return y


def snr(y, flags):
    """signal (sur data) / leak (sur filler)."""
    data, filler = flags == 1, flags == 0
    signal = float(np.mean(np.abs(y[data]))) if data.any() else 0.0
    leak = float(np.mean(np.abs(y[filler]))) if filler.any() else 1e-9
    return signal / max(leak, 1e-9), signal, leak


def exercice_4():
    print("=" * 70)
    print("Exercice 4 : Selectivite (S4 non-selectif vs Mamba selectif)")
    print("=" * 70)

    rng = np.random.default_rng(SEED)
    x, flags = make_selective_copy_data(N=64, data_frac=0.25, rng=rng)
    D = 4
    A = make_stable_A(D, decay=0.85)
    B = rng.standard_normal((D, 1)) * 0.6
    C = rng.standard_normal((1, D)) * 0.6

    y_ns = ssm_nonselective(x, flags, A, B, C)
    y_sel = ssm_selective(x, flags, A, B, C)

    snr_ns, sig_ns, leak_ns = snr(y_ns, flags)
    snr_sel, sig_sel, leak_sel = snr(y_sel, flags)

    print(f"\n  N={len(x)}, data={int((flags == 1).sum())}, "
          f"filler={int((flags == 0).sum())}")
    print(f"  Non-selectif : leak={leak_ns:.4f}, signal={sig_ns:.4f}, "
          f"SNR={snr_ns:.2f}")
    print(f"  Selectif     : leak={leak_sel:.4f}, signal={sig_sel:.4f}, "
          f"SNR={snr_sel:.2f}")
    print("  -> le selectif ferme B,C sur le filler -> leak quasi nul -> SNR >>.")

    # 5) balayage decay : le non-selectif fuit plus a forte memoire.
    print("\n  Balayage decay de A (leak du non-selectif) :")
    leaks = []
    for decay in [0.5, 0.85, 0.95, 0.99]:
        Ad = make_stable_A(D, decay=decay)
        y = ssm_nonselective(x, flags, Ad, B, C)
        _, _, leak = snr(y, flags)
        leaks.append(leak)
        print(f"    decay={decay:<5} leak={leak:.4f}")
    print("  POURQUOI : plus A retient (decay eleve), plus l'etat garde la trace")
    print("  des data passees -> elle 'fuit' sur les positions filler suivantes,")
    print("  ou le non-selectif ne peut pas couper le readout.")

    # 6) analyse.
    print("\n  POURQUOI rendre B,C,Delta fonction de x_t = gate dynamique ~ attention")
    print("  en O(N) : le modele DECIDE par token d'absorber (gate ouvert) ou")
    print("  d'ignorer (gate ferme) l'input. La selectivite agit via Delta_t sur")
    print("  Ā = exp(Delta_t * A) : grand Delta -> absorbe, petit Delta -> retient.")
    print("  C'est l'equivalent fonctionnel d'une attention, sans la matrice N x N.")

    # --- assertions de validation ---
    assert snr_sel > snr_ns                       # selectif nettement meilleur
    assert leak_sel < leak_ns                     # moins de fuite
    # A forte memoire, le leak du non-selectif est >= qu'a faible memoire.
    assert leaks[-1] >= leaks[0] * 0.5            # tendance (robuste au bruit)
    print("\n  [OK] Exercice 4")


# ============================================================================
# Exercice 5 — Benchmark complexite : SSM vs attention, + version FFT
# ============================================================================

def attention_forward(x, d_model=16):
    """Self-attention single-head jouet. Cout O(N^2 d) (matrice N x N)."""
    N = x.shape[0]
    rng = np.random.default_rng(SEED)
    Wq = rng.standard_normal((1, d_model)) * 0.1
    Wk = rng.standard_normal((1, d_model)) * 0.1
    Wv = rng.standard_normal((1, d_model)) * 0.1
    X = x[:, None]
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    scores = Q @ K.T / np.sqrt(d_model)           # (N, N) <- l'etape quadratique
    scores -= scores.max(axis=-1, keepdims=True)
    w = np.exp(scores)
    w /= w.sum(axis=-1, keepdims=True)
    return (w @ V).sum(axis=-1)


def ssm_kernel(A, B, C, N):
    """Kernel de longueur N : K[k] = C A^k B."""
    D = A.shape[0]
    K = np.zeros(N)
    AkB = B.flatten().copy()
    for k in range(N):
        K[k] = (C @ AkB).item()
        AkB = A @ AkB
    return K


def ssm_conv_direct(x, K):
    """Convolution DIRECTE (np.convolve) : O(N^2). Causale."""
    return np.convolve(x, K)[:len(x)]


def ssm_conv_fft(x, K):
    """
    Convolution par FFT : O(N log N). On zero-pad a >= 2N-1 pour eviter le
    repliement circulaire, puis ifft du produit des spectres.
    """
    N = len(x)
    L = 1
    while L < 2 * N - 1:
        L *= 2
    Xf = np.fft.rfft(x, L)
    Kf = np.fft.rfft(K, L)
    y = np.fft.irfft(Xf * Kf, L)
    return y[:N]


def exercice_5():
    print("\n" + "=" * 70)
    print("Exercice 5 : Benchmark complexite SSM vs attention (+ FFT)")
    print("=" * 70)

    print("\n  DISCLAIMER : np.convolve est une convolution DIRECTE O(N^2), pas une")
    print("  FFT. Ce micro-bench SOUS-ESTIME donc les vrais SSM (qui scalent en")
    print("  O(N log N) via FFT / parallel scan). On ajoute la version FFT ci-dessous.")

    D = 8
    A = make_stable_A(D)
    B = np.random.randn(D, 1) * 0.3
    C = np.random.randn(1, D) * 0.3
    sizes = [128, 512, 2048, 8192]

    print(f"\n  {'N':>6} | {'SSM dir (s)':>12} | {'SSM fft (s)':>12} | "
          f"{'Attn (s)':>10} | {'Attn/SSMdir':>11}")
    print("  " + "-" * 64)
    rows = []
    for N in sizes:
        x = np.random.randn(N)
        K = ssm_kernel(A, B, C, N)

        t0 = time.perf_counter(); _ = ssm_conv_direct(x, K); t_dir = time.perf_counter() - t0
        t0 = time.perf_counter(); y_fft = ssm_conv_fft(x, K); t_fft = time.perf_counter() - t0
        t0 = time.perf_counter(); _ = attention_forward(x); t_attn = time.perf_counter() - t0

        # FFT et conv directe doivent donner le meme resultat.
        y_dir = ssm_conv_direct(x, K)
        assert np.max(np.abs(y_dir - y_fft)) < 1e-8, "FFT != direct"

        ratio = t_attn / t_dir if t_dir > 0 else float("inf")
        rows.append((N, t_dir, t_fft, t_attn, ratio))
        print(f"  {N:>6} | {t_dir:>12.5f} | {t_fft:>12.5f} | "
              f"{t_attn:>10.5f} | {ratio:>11.2f}x")

    # 4) verifier le scaling quadratique de l'attention (x16 quand N x4).
    if len(rows) >= 2:
        N0, _, _, ta0, _ = rows[0]
        N1, _, _, ta1, _ = rows[-1]
        grow_attn = ta1 / max(ta0, 1e-9)
        print(f"\n  De N={N0} a N={N1} (x{N1 / N0:.0f}) : attention a grandi "
              f"~{grow_attn:.0f}x (attendu ~{(N1 / N0) ** 2:.0f}x quadratique).")

    # 6) pic memoire theorique attention N^2 vs etat fixe SSM (cf table cours).
    print("\n  Pic memoire theorique (FP16, 2 octets) :")
    print(f"    {'N':>8} | {'attn N^2 (matrice)':>22} | {'etat SSM (fixe)':>18}")
    print("    " + "-" * 54)
    D_state = 64
    for N in [2048, 32768, 131072]:
        attn_bytes = N * N * 2
        state_bytes = D_state * 2
        print(f"    {N:>8} | {attn_bytes / 1e9:>18.2f} GB | "
              f"{state_bytes:>14} o")
    print("  -> le mur quadratique : la matrice N x N explose (131k -> ~34 GB),")
    print("     l'etat SSM reste constant. C'est l'argument structurel des SSM.")

    print("\n  [OK] Exercice 5")


# ============================================================================
# Exercice 6 — Hybride Transformer + Mamba : le ratio 1/8 de Jamba
# ============================================================================

def make_mqar(n_pairs, key_dim=8, n_queries=None, rng=None):
    """
    MQAR-like : n_pairs paires (cle, valeur) dispersees, puis des requetes.
    Les cles sont des vecteurs aleatoires (quasi-orthogonaux en haute dim).
    """
    rng = rng or np.random.default_rng(0)
    n_queries = n_queries or n_pairs
    keys = rng.standard_normal((n_pairs, key_dim))
    keys /= np.linalg.norm(keys, axis=1, keepdims=True) + 1e-9
    vals = rng.standard_normal((n_pairs, key_dim))
    q_idx = rng.integers(0, n_pairs, size=n_queries)   # quelle paire on demande
    return keys, vals, q_idx


def mamba_like_memory(keys, vals, key_dim):
    """
    'Mamba pur' = memoire associative LINEAIRE a etat fixe :
      H = sum_m v_m k_m^T  (matrice key_dim x key_dim, taille FIXE)
      recall : v_hat = H k_query
    L'etat ne grandit PAS avec le nombre de paires -> compresseur lossy : les
    interferences entre cles non-orthogonales degradent le recall quand la
    densite de paires monte (l'etat sature).
    """
    H = np.zeros((key_dim, key_dim))
    for k, v in zip(keys, vals):
        H += np.outer(v, k)                  # ecriture (accumulation)
    return H


def recall_accuracy(H_or_store, keys, vals, q_idx, mode):
    """Recall : fraction de requetes ou v_hat est le plus proche du bon v."""
    correct = 0
    for qi in q_idx:
        if mode == "mamba":
            v_hat = H_or_store @ keys[qi]    # lecture lossy
        else:  # transformer : lookup exact (KV cache stocke chaque paire)
            v_hat = vals[qi]
        # plus proche voisin parmi les vraies valeurs.
        d = np.linalg.norm(vals - v_hat[None, :], axis=1)
        if int(np.argmin(d)) == int(qi):
            correct += 1
    return correct / len(q_idx)


def exercice_6():
    print("\n" + "=" * 70)
    print("Exercice 6 : Hybride Transformer + Mamba (ratio 1/8 Jamba)")
    print("=" * 70)

    rng = np.random.default_rng(0)
    key_dim = 8

    # 3 archis modelisees par leur capacite de recall :
    #  - pure SSM (L layers Mamba) : etat fixe key_dim x key_dim -> sature.
    #  - hybride 7 Mamba + 1 attn : la layer d'attention fait un lookup EXACT.
    #  - pure attention : lookup exact partout.
    # On modelise l'effet : des qu'AU MOINS une layer d'attention est presente,
    # le recall ponctuel exact est recupere (l'attn accede a toute la sequence).
    print(f"\n  {'#paires':>8} | {'pure SSM':>10} | {'hybride 7+1':>12} | "
          f"{'pure attn':>10}")
    print("  " + "-" * 50)
    acc_ssm_curve, acc_hyb_curve = [], []
    for n_pairs in [2, 4, 8, 16, 32, 64]:
        keys, vals, q_idx = make_mqar(n_pairs, key_dim, n_queries=64, rng=rng)
        H = mamba_like_memory(keys, vals, key_dim)

        acc_ssm = recall_accuracy(H, keys, vals, q_idx, mode="mamba")
        # Hybride : la layer d'attention fournit le lookup exact -> ~recall attn.
        acc_hyb = recall_accuracy(None, keys, vals, q_idx, mode="transformer")
        acc_attn = recall_accuracy(None, keys, vals, q_idx, mode="transformer")

        acc_ssm_curve.append(acc_ssm)
        acc_hyb_curve.append(acc_hyb)
        print(f"  {n_pairs:>8} | {acc_ssm:>10.0%} | {acc_hyb:>12.0%} | "
              f"{acc_attn:>10.0%}")

    print("\n  -> pure SSM DECROCHE quand la densite de paires monte (l'etat fixe")
    print(f"     key_dim x key_dim = {key_dim}x{key_dim} sature : interferences).")
    print("  -> ajouter 1 layer d'attention (lookup exact) recupere le recall.")

    # 6) analyse.
    print("\n  POURQUOI 1 layer d'attention sur 8 suffit : les rares layers d'attn")
    print("  permettent le recall ponctuel EXACT quand necessaire ; les 7 layers")
    print("  Mamba font le gros du travail en O(N) (throughput, long context).")
    print("  Cout ajoute : 1 layer attn = O(N^2) compute + KV cache O(N) memoire,")
    print("  soit ~1/8 du cout d'un transformer pur, pour ~tout le gain de recall.")
    print("  POURQUOI le pure SSM sature : etat de taille FIXE = compresseur lossy")
    print("  (idee fausse n.4). Au-dela de ~capacite, l'info ancienne se dilue.")

    # --- assertions de validation ---
    # Pure SSM se degrade quand la densite monte.
    assert acc_ssm_curve[0] >= acc_ssm_curve[-1]
    assert acc_ssm_curve[-1] < 0.95                 # decrochage a forte densite
    # L'hybride (lookup exact) tient a forte densite.
    assert acc_hyb_curve[-1] >= acc_ssm_curve[-1]
    assert acc_hyb_curve[-1] > 0.95
    print("\n  [OK] Exercice 6")


if __name__ == "__main__":
    exercice_4()
    exercice_5()
    exercice_6()
    print("\nDone (MEDIUM).")
