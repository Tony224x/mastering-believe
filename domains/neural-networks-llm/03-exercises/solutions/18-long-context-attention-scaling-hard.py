"""
Solutions HARD — Jour 18 : Long context (Flash Attention, RoPE scaling, ring attention)
======================================================================================
Exercices 7, 8, 9 (hard). Pur NumPy, comme 02-code/18-long-context-attention-scaling.py.
Chaque etape non triviale est commentee avec le POURQUOI.

Run: python 03-exercises/solutions/18-long-context-attention-scaling-hard.py
"""

import sys
import io
import math
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

np.random.seed(42)


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def rope_frequencies(d, base=10000.0):
    """theta_i = 1 / base^(2i/d) pour d/2 paires (identique a 02-code/18)."""
    i = np.arange(0, d, 2, dtype=np.float64)
    return 1.0 / (base ** (i / d))


def rope_yarn_frequencies(d, base, scale, alpha=1.0, beta=32.0, L_train=4096):
    """YaRN : scaling par bande (identique a 02-code/18 et a la solution MEDIUM)."""
    inv_freq = rope_frequencies(d, base)
    inv_freq_pi = inv_freq / scale
    wavelen = 2 * math.pi / inv_freq
    ratio = L_train / wavelen
    ramp = np.clip((ratio - alpha) / (beta - alpha), 0.0, 1.0)
    return (1 - ramp) * inv_freq_pi + ramp * inv_freq


# ============================================================================
# EXERCISE 7 — YaRN end-to-end + propriete de position relative
# ============================================================================

print("=" * 70)
print("EXERCISE 7 : YaRN end-to-end + propriete q_m . k_n = f(m-n)")
print("=" * 70)


def apply_rope(x, pos, inv_freq):
    """
    Applique RoPE a un vecteur x (dim d) a la position pos.
    Chaque paire (x[2i], x[2i+1]) est rotee de l'angle pos*inv_freq[i] :
      x'[2i]   = x[2i]*cos - x[2i+1]*sin
      x'[2i+1] = x[2i]*sin + x[2i+1]*cos
    POURQUOI : une rotation 2D preserve la norme ; le produit scalaire de deux
    vecteurs rotes ne depend que de la DIFFERENCE d'angles -> invariance relative.
    """
    angles = pos * inv_freq            # (d/2,)
    cos, sin = np.cos(angles), np.sin(angles)
    x = np.asarray(x, dtype=np.float64)
    x_even = x[0::2]
    x_odd = x[1::2]
    out = np.empty_like(x)
    out[0::2] = x_even * cos - x_odd * sin
    out[1::2] = x_even * sin + x_odd * cos
    return out


d, base, L_train, scale = 64, 10000.0, 4096, 8.0
f_orig = rope_frequencies(d, base)
f_yarn = rope_yarn_frequencies(d, base, scale, L_train=L_train)

# Verifier que apply_rope preserve la norme (controle de sanity).
qv = np.random.randn(d)
norm_err = abs(np.linalg.norm(apply_rope(qv, 1234, f_orig)) - np.linalg.norm(qv))
print(f"\n  apply_rope preserve la norme : ecart = {norm_err:.2e}  "
      f"({'OK' if norm_err < 1e-9 else 'FAIL'})")


def dot_at(q, k, m, n, inv_freq):
    """Produit scalaire de q (pos m) et k (pos n) apres RoPE."""
    return float(apply_rope(q, m, inv_freq) @ apply_rope(k, n, inv_freq))


def rel_property_variance(q, k, inv_freq, deltas, base_positions):
    """
    Pour chaque delta = m-n, on calcule q_m.k_n a plusieurs positions de base
    (donc plusieurs m,n de meme delta) et on mesure la variance des produits.
    Si la propriete relative tient, la variance a delta fixe est ~0.
    On retourne la variance MOYENNE sur les deltas.
    """
    variances = []
    for delta in deltas:
        vals = [dot_at(q, k, n + delta, n, inv_freq) for n in base_positions]
        variances.append(np.var(vals))
    return float(np.mean(variances))


q = np.random.randn(d)
k = np.random.randn(d)
deltas = [0, 5, 17, 42]

# --- IN-RANGE : positions dans la plage d'entrainement (RoPE base) ---
base_in = [0, 100, 500, 1000, 2000, 3000]  # toutes < L_train
var_in = rel_property_variance(q, k, f_orig, deltas, base_in)
print(f"\n  Propriete relative q_m.k_n = f(m-n) (base RoPE, positions in-range) :")
print(f"    variance de q_m.k_n a delta fixe = {var_in:.2e}  "
      f"({'OK' if var_in < 1e-3 else 'FAIL'})")
assert var_in < 1e-3, "La propriete relative doit tenir (var ~0)"
print("    -> q_m.k_n ne depend (quasi) que de m-n : invariance par translation.")

# --- OUT-OF-RANGE : pourquoi le naif casse et pas YaRN ---
# CAVEAT HONNETE : l'invariance par translation (q_m.k_n = f(m-n)) est une
# propriete EXACTE de la rotation 2D ; elle tient pour N'IMPORTE QUELLES
# frequences, y compris a des positions enormes (la variance a delta fixe reste
# du bruit machine ~1e-24 dans les deux cas). Mesurer cette variance out-of-range
# ne separe donc PAS naif et YaRN : ce serait comparer du bruit numerique.
#
# Ce qui casse REELLEMENT en extrapolation naive n'est pas l'algebre des
# rotations, mais le fait que le modele voit des ANGLES DE ROTATION jamais
# rencontres a l'entrainement (out-of-distribution). A l'entrainement, pour
# chaque bande i, l'angle pos*theta_i a parcouru au plus [0, L_train*theta_i] :
# c'est l'enveloppe d'angles "vue". On mesure donc, par bande, de combien
# l'angle a une position out-of-range DEPASSE cette enveloppe.
print(f"\n  Angles de rotation out-of-range vs enveloppe vue a l'entrainement :")
print(f"  (in-range, bande i : angle parcouru au plus L_train*theta_i)")

pos_oor = 30000                                  # position >> L_train
env_band = L_train * f_orig                       # enveloppe d'angle vue, par bande
ang_naif = pos_oor * f_orig                        # angle naif a pos_oor, par bande
ang_yarn = pos_oor * f_yarn                        # angle YaRN a pos_oor, par bande

# "Excess OOD" par bande : de combien l'angle out-of-range deborde l'enveloppe,
# normalise par l'enveloppe. 0 = reste dans la plage vue ; >0 = hors-distribution.
excess_naif = np.maximum(0.0, ang_naif - env_band) / env_band
excess_yarn = np.maximum(0.0, ang_yarn - env_band) / env_band

# On regarde les BASSES frequences a LONGUE PORTEE = le dernier quart du spectre.
# Ce sont precisement les bandes que YaRN ramene en mode PI (ramp ~ 0 : freqs
# divisees par scale). Les hautes freqs locales sont laissees intactes PAR DESIGN
# (resolution locale) ; leur angle cycle mod 2*pi a chaque position, donc leur
# "depassement" n'est pas le vrai probleme d'extrapolation -> on ne les compte pas
# ici. Le vrai danger d'extrapolation est sur les basses freqs, dont l'angle croit
# de facon monotone (pas de wrap) et sort donc franchement de la plage vue.
low = slice(3 * (d // 2) // 4, None)
ood_naif = float(np.mean(excess_naif[low]))
ood_yarn = float(np.mean(excess_yarn[low]))
print(f"    excess OOD moyen (dernier quart du spectre = longue portee) :")
print(f"      naif (extrapolation) = {ood_naif:.3f}   (angles ~{1+ood_naif:.1f}x hors plage)")
print(f"      YaRN (freqs scalees) = {ood_yarn:.3f}   (reste dans la plage vue)")

# Assertions VRAIES et pertinentes (et non du bruit) :
#  - le naif sort franchement de la distribution vue sur les bandes longue portee ;
#  - YaRN ramene ces memes bandes dans l'enveloppe d'entrainement (excess ~0).
assert ood_naif > 1.0, "naif : les basses freqs sortent largement de la plage vue"
assert ood_yarn < 1e-6, "YaRN : les bandes longue portee restent dans la plage vue"
assert ood_yarn < ood_naif, "YaRN garde les angles bien plus proches de la plage vue"
print("    -> les basses freqs naives debordent l'enveloppe (OOD) ; YaRN les garde dedans.")
print("  POURQUOI : en naif, basse freq * position enorme = angle jamais vu a")
print("  l'entrainement -> le modele extrapole hors-distribution. YaRN comprime ces")
print("  basses freqs (theta_i -> theta_i/scale en bas du spectre) pour ramener leurs")
print("  angles dans la plage vue, tout en laissant les hautes freqs (resolution")
print("  locale) intactes. C'est exactement l'argument de Peng et al. 2023 (YaRN).")


# ============================================================================
# EXERCISE 8 — Ring attention / sequence parallelism
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 8 : ring attention (sequence parallelism) == full attention")
print("=" * 70)


def full_attention(Q, K, V):
    """Attention complete (non causale) de reference."""
    d = Q.shape[-1]
    S = Q @ K.T / math.sqrt(d)
    return softmax(S, axis=-1) @ V


def ring_attention(chunks_Q, chunks_K, chunks_V):
    """
    Simulation de Ring Attention (Liu et al. 2023).
    Chaque "GPU" p detient un chunk (Q_p, K_p, V_p). A chaque step de l'anneau,
    il recoit le chunk K/V d'un voisin et fusionne son attention en online softmax.
    Apres P steps, chaque GPU a vu tous les K/V -> attention complete, mais
    chaque GPU ne garde qu'UN chunk K/V a la fois en memoire (O(N/P)).
    """
    P = len(chunks_Q)
    d = chunks_Q[0].shape[-1]
    outputs = []
    for p in range(P):
        Qp = chunks_Q[p]
        Op = np.zeros_like(Qp)
        mp = np.full((Qp.shape[0],), -np.inf)   # running max par query
        lp = np.zeros((Qp.shape[0],))           # running denom par query
        # L'anneau : a chaque step, on recoit le chunk du GPU (p - step) mod P.
        for step in range(P):
            src = (p - step) % P
            Kj, Vj = chunks_K[src], chunks_V[src]
            Sij = Qp @ Kj.T / math.sqrt(d)
            # Online softmax (meme rescale que Flash : correction du max global).
            mij = np.max(Sij, axis=-1)
            m_new = np.maximum(mp, mij)
            alpha = np.exp(mp - m_new)
            Pij = np.exp(Sij - m_new[:, None])
            lp = alpha * lp + np.sum(Pij, axis=-1)
            Op = alpha[:, None] * Op + Pij @ Vj
            mp = m_new
        outputs.append(Op / lp[:, None])
    return np.concatenate(outputs, axis=0)


N, d, P = 128, 32, 4
Q = np.random.randn(N, d).astype(np.float64)
K = np.random.randn(N, d).astype(np.float64)
V = np.random.randn(N, d).astype(np.float64)

chunk = N // P
chunks_Q = [Q[i:i + chunk] for i in range(0, N, chunk)]
chunks_K = [K[i:i + chunk] for i in range(0, N, chunk)]
chunks_V = [V[i:i + chunk] for i in range(0, N, chunk)]

O_full = full_attention(Q, K, V)
O_ring = ring_attention(chunks_Q, chunks_K, chunks_V)
max_diff = float(np.max(np.abs(O_full - O_ring)))
print(f"\n  N={N}, d={d}, P={P} GPU (chunks de {chunk} tokens)")
print(f"  max|O_full - O_ring| = {max_diff:.2e}  ({'OK' if max_diff < 1e-5 else 'FAIL'})")
assert max_diff < 1e-5, "Ring attention doit egaler full attention"
print("  -> le sequence parallelism ne change PAS le resultat, seulement la")
print("     repartition memoire/compute.")
print(f"\n  Memoire K/V par GPU a un instant : 1 chunk = {chunk} tokens (O(N/P))")
print(f"  vs full : {N} tokens (O(N)) -> gain x{P}. L'overlap comm/compute (envoyer")
print("  le chunk suivant pendant qu'on calcule le courant) vise un overhead ~0.")


# ============================================================================
# EXERCISE 9 — Attention sinks (StreamingLLM)
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 9 : attention sinks (StreamingLLM) vs sliding-only")
print("=" * 70)


def make_sliding_mask(N, W):
    mask = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        mask[i, max(0, i - W + 1):i + 1] = 1.0
    return mask


def attention_with_mask(Q, K, V, mask):
    d = Q.shape[-1]
    S = Q @ K.T / math.sqrt(d)
    S = np.where(mask > 0, S, -1e9)
    P = softmax(S, axis=-1)
    return P, P @ V


# Sequence synthetique : on fabrique 4 sinks emergents (comme 02-code/18 part 4).
N, d = 256, 32
Q = np.random.randn(N, d).astype(np.float64) * 0.5
K = np.random.randn(N, d).astype(np.float64) * 0.5
V = np.random.randn(N, d).astype(np.float64)

# Les 4 premiers tokens : K alignes sur une direction commune (sinks).
sink_dir = np.random.randn(d)
for i in range(4):
    K[i] = sink_dir + 0.05 * np.random.randn(d)
# Tous les Q penchent vers sink_dir -> les tokens "ennuyeux" drainent ici.
for i in range(N):
    Q[i] = Q[i] + 0.4 * sink_dir

W, n_sinks = 32, 4
causal = np.tril(np.ones((N, N), dtype=np.float64))

# A) Full causal (oracle).
P_full, O_full = attention_with_mask(Q, K, V, causal)
# B) Sliding only (premiers tokens jetes).
sw_mask = make_sliding_mask(N, W)
P_sw, O_sw = attention_with_mask(Q, K, V, sw_mask)
# C) Sliding + sinks (les n_sinks premiers tokens toujours visibles).
sink_mask = sw_mask.copy()
sink_mask[:, :n_sinks] = 1.0
sink_mask = sink_mask * causal  # rester causal : i ne voit jamais > i
P_sink, O_sink = attention_with_mask(Q, K, V, sink_mask)

last = N - 1
print(f"\n  N={N}, fenetre W={W}, n_sinks={n_sinks}")

# Claim NUMERIQUEMENT EXACT : en sliding-only, l'attention hors-fenetre = 0.
far_sw = float(P_sw[last, :N - W].sum())
print(f"\n  Attention hors-fenetre du dernier token (sliding only) = {far_sw:.2e}  "
      f"({'OK' if far_sw < 1e-7 else 'FAIL'})")
assert far_sw < 1e-7, "Hors fenetre, l'attention sliding doit etre exactement nulle"

# Masse d'attention deposee sur les sinks par le dernier token.
sink_mass_full = float(P_full[last, :n_sinks].sum())
sink_mass_sink = float(P_sink[last, :n_sinks].sum())
print(f"\n  Masse d'attention du dernier token sur les {n_sinks} premiers tokens :")
print(f"    full causal     : {sink_mass_full:.4f}")
print(f"    sliding + sinks : {sink_mass_sink:.4f}  (les sinks absorbent une part)")

# Derive L2 de la sortie du dernier token vs l'oracle full-causal.
drift_sw = float(np.linalg.norm(O_sw[last] - O_full[last]))
drift_sink = float(np.linalg.norm(O_sink[last] - O_full[last]))
print(f"\n  Derive L2 du dernier token vs oracle full-causal :")
print(f"    sliding only    : {drift_sw:.4f}")
print(f"    sliding + sinks : {drift_sink:.4f}")

# Caveat HONNETE (calque sur 02-code/18 part 4) : pas de sur-assertion.
print("\n  CAVEAT (calque sur 02-code/18) : dans ce setup jouet (single-pass,")
print("  single-layer), NI sliding NI sliding+sinks ne reproduit l'oracle exactement")
print("  (les deux derivent). Le phenomene reel de StreamingLLM (Xiao 2023) se")
print("  manifeste en generation autoregressive sur de nombreuses couches et des")
print("  millions de tokens, ou le collapse softmax-doit-sommer-a-1 se compose")
print("  couche apres couche. Une attention numpy single-pass ne reproduit pas cette")
print("  dynamique. Voir Xiao et al. 2023, figure 4, pour l'explosion de perplexite")
print("  reelle quand on jette les premiers tokens sur de vrais decoders LLM.")

print("\nDone (HARD).")
