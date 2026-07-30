"""
Solutions HARD — Jour 9 : LLMs modernes (RoPE, RMSNorm, SwiGLU, GQA)
===================================================================
Exercices 7, 8, 9 (hard). Pur NumPy.

7. RoPE complexe + preuve relative + extension de contexte (PI, NTK).
8. Mini-bloc LLaMA complet (RMSNorm + RoPE + GQA + SwiGLU) en NumPy.

Run: python 03-exercises/solutions/09-llms-modernes-architectures-hard.py
"""

import numpy as np

np.random.seed(42)


def precompute_rope_frequencies(head_dim, max_seq_len, base=10000.0):
    freqs = 1.0 / (base ** (np.arange(0, head_dim, 2) / head_dim))
    angles = np.outer(np.arange(max_seq_len), freqs)
    return np.cos(angles), np.sin(angles), freqs


def apply_rope(x, cos, sin):
    x_a, x_b = x[:, 0::2], x[:, 1::2]
    out = np.empty_like(x)
    out[:, 0::2] = x_a * cos - x_b * sin
    out[:, 1::2] = x_a * sin + x_b * cos
    return out


# ============================================================================
# EXERCISE 7: RoPE complexe + preuve relative + PI / NTK
# ============================================================================

print("=" * 70)
print("EXERCISE 7: RoPE complexe + extension de contexte (PI, NTK)")
print("=" * 70)

head_dim, max_seq = 8, 64
cos, sin, freqs = precompute_rope_frequencies(head_dim, max_seq)


def apply_rope_complex(x, pos, freqs):
    """
    RoPE via complexes : paire (x_2j, x_2j+1) -> z_j = x_2j + i x_2j+1,
    rotation z_j * e^{i pos theta_j}. Renvoie le vecteur reel reconstruit.
    """
    z = x[0::2] + 1j * x[1::2]                   # (head_dim/2,)
    rot = z * np.exp(1j * pos * freqs)
    out = np.empty_like(x)
    out[0::2] = rot.real
    out[1::2] = rot.imag
    return out


# Verif : complexe == reel.
x = np.random.randn(head_dim)
pos = 7
via_real = apply_rope(x.reshape(1, -1), cos[pos:pos + 1], sin[pos:pos + 1])[0]
via_cplx = apply_rope_complex(x, pos, freqs)
print(f"\n  apply_rope_complex == apply_rope reel : max diff = {np.max(np.abs(via_real - via_cplx)):.2e}")

# Preuve relative : Re(<RoPE(q,m), conj(RoPE(k,n))>) depend de (m-n).
# <q e^{im theta}, conj(k e^{in theta})> = q*conj(k)*e^{i(m-n)theta} -> phase relative.
print("\n  Propriete relative (phases e^{i(m-n)theta}):")
q = np.random.randn(head_dim)
k = np.random.randn(head_dim)


def complex_inner(q, k, m, n, freqs):
    zq = (q[0::2] + 1j * q[1::2]) * np.exp(1j * m * freqs)
    zk = (k[0::2] + 1j * k[1::2]) * np.exp(1j * n * freqs)
    return np.real(np.sum(zq * np.conj(zk)))


for (m, n) in [(3, 5), (10, 12), (20, 22)]:     # tous distance -2
    print(f"    m={m:2d} n={n:2d} (m-n={m - n}): Re<.,.> = {complex_inner(q, k, m, n, freqs):+.6f}")
print("  -> identique a m-n fixe : seules les phases relatives comptent.")

# Position Interpolation (PI) : pos' = pos * L_train / L_new.
L_train, L_new = 4096, 32768
print(f"\n  Position Interpolation {L_train} -> {L_new}:")
print(f"    facteur de compression = L_train/L_new = {L_train / L_new:.4f}")
pos_new = 5000
pos_pi = pos_new * L_train / L_new
print(f"    position {pos_new} -> position effective {pos_pi:.1f} (dans la plage vue au training)")

# NTK-aware : base' = base * (L_new/L_train)^(d/(d-2)).
d = head_dim
base_ntk = 10000.0 * (L_new / L_train) ** (d / (d - 2))
print(f"\n  NTK-aware scaling: base {10000.0:.0f} -> {base_ntk:.0f}")
_, _, freqs_ntk = precompute_rope_frequencies(head_dim, 4, base=base_ntk)
print("    longueurs d'onde (2*pi/freq), hautes (locales) vs basses (longue portee):")
print(f"      RoPE base 10000 : haute={2 * np.pi / freqs[0]:.1f}  basse={2 * np.pi / freqs[-1]:.1f}")
print(f"      NTK             : haute={2 * np.pi / freqs_ntk[0]:.1f}  basse={2 * np.pi / freqs_ntk[-1]:.1f}")
print("    -> NTK etire surtout les basses frequences (longue portee), preserve les hautes (details).")
print("\n  RoPE est une FONCTION CONTINUE de la position -> on peut interpoler/extrapoler.")
print("  Les embeddings APPRIS (GPT-2) sont une table finie : aucune position > L_train n'existe.")


# ============================================================================
# EXERCISE 8: Mini-bloc LLaMA complet en NumPy
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 8: Mini-bloc LLaMA (RMSNorm + RoPE + GQA + SwiGLU)")
print("=" * 70)


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def rms_norm(x, gamma, eps=1e-6):
    rms = np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)
    return x / rms * gamma


def silu(x):
    return x / (1.0 + np.exp(-x))


def split_heads(M, n_h, hd):
    seq = M.shape[0]
    return M.reshape(seq, n_h, hd).transpose(1, 0, 2)   # (n_h, seq, hd)


def gqa_attention(x, p, n_heads, n_kv_heads):
    seq, d_model = x.shape
    hd = d_model // n_heads
    n_rep = n_heads // n_kv_heads
    q = split_heads(x @ p['W_q'], n_heads, hd)
    k = split_heads(x @ p['W_k'], n_kv_heads, hd)
    v = split_heads(x @ p['W_v'], n_kv_heads, hd)
    cos, sin, _ = precompute_rope_frequencies(hd, seq)
    q = np.stack([apply_rope(q[h], cos, sin) for h in range(n_heads)])
    k = np.stack([apply_rope(k[h], cos, sin) for h in range(n_kv_heads)])
    # Repeat K/V pour matcher les queries.
    k = np.repeat(k, n_rep, axis=0)
    v = np.repeat(v, n_rep, axis=0)
    mask = np.triu(np.ones((seq, seq)), k=1)
    outs = []
    for h in range(n_heads):
        s = q[h] @ k[h].T / np.sqrt(hd)
        s = np.where(mask == 1, -np.inf, s)
        outs.append(softmax(s, -1) @ v[h])
    concat = np.concatenate(outs, axis=-1)
    return concat @ p['W_o'], n_rep


def round_to_multiple(x, mult):
    return ((x + mult - 1) // mult) * mult


def swiglu_ffn(x, W_gate, W_up, W_down):
    return (silu(x @ W_gate) * (x @ W_up)) @ W_down


def make_llama_block(d_model, n_heads, n_kv_heads, seed=0, mult=8):
    rng = np.random.RandomState(seed)
    hd = d_model // n_heads
    d_ff = round_to_multiple(int(8 * d_model / 3), mult)
    s = 0.02
    return dict(
        g1=np.ones(d_model), g2=np.ones(d_model),
        W_q=rng.randn(d_model, n_heads * hd) * s,
        W_k=rng.randn(d_model, n_kv_heads * hd) * s,
        W_v=rng.randn(d_model, n_kv_heads * hd) * s,
        W_o=rng.randn(n_heads * hd, d_model) * s,
        W_gate=rng.randn(d_model, d_ff) * s,
        W_up=rng.randn(d_model, d_ff) * s,
        W_down=rng.randn(d_ff, d_model) * s,
        d_ff=d_ff,
    )


def llama_block(x, p, n_heads, n_kv_heads):
    a, n_rep = gqa_attention(rms_norm(x, p['g1']), p, n_heads, n_kv_heads)
    x = x + a
    x = x + swiglu_ffn(rms_norm(x, p['g2']), p['W_gate'], p['W_up'], p['W_down'])
    return x, n_rep


d_model, n_heads, n_kv_heads, seq = 64, 8, 2, 12
p = make_llama_block(d_model, n_heads, n_kv_heads, seed=1)
x = np.random.randn(seq, d_model) * 0.5
out, n_rep = llama_block(x, p, n_heads, n_kv_heads)
hd = d_model // n_heads
print(f"\n  Shapes (d_model={d_model}, n_heads={n_heads}, n_kv_heads={n_kv_heads}, seq={seq}):")
print(f"    head_dim = {hd}, n_rep = {n_rep}, d_ff = {p['d_ff']}")
print(f"    q (avant attention) : ({n_heads}, {seq}, {hd})")
print(f"    k/v avant repeat    : ({n_kv_heads}, {seq}, {hd})")
print(f"    k/v apres repeat    : ({n_heads}, {seq}, {hd})")
print(f"    output bloc         : {out.shape}")
assert n_rep == 4

# Causalite end-to-end.
t = 7
x2 = x.copy(); x2[t] += np.random.randn(d_model)
out2, _ = llama_block(x2, p, n_heads, n_kv_heads)
print(f"\n  Causalite: max|out[:t]-out2[:t]| = {np.max(np.abs(out[:t] - out2[:t])):.2e} (~0), "
      f"max|out[t:]-out2[t:]| = {np.max(np.abs(out[t:] - out2[t:])):.2e} (>0)")

# Compte de parametres LLaMA-block vs GPT-2-block (meme d_model).
print("\n  Parametres LLaMA-block vs GPT-2-block (d_model=%d):" % d_model)
attn_llama = p['W_q'].size + p['W_k'].size + p['W_v'].size + p['W_o'].size
ffn_llama = p['W_gate'].size + p['W_up'].size + p['W_down'].size
norm_llama = 2 * d_model                          # 2 RMSNorm (scale seulement)
total_llama = attn_llama + ffn_llama + norm_llama

d_ff_gpt = 4 * d_model
attn_gpt = 3 * d_model * d_model + d_model * d_model    # c_attn + c_proj
ffn_gpt = d_model * d_ff_gpt + d_ff_gpt * d_model       # 2 matrices
norm_gpt = 2 * (2 * d_model)                            # 2 LayerNorm (scale + bias)
total_gpt = attn_gpt + ffn_gpt + norm_gpt
print(f"    LLaMA : attn={attn_llama:>6} ffn={ffn_llama:>6} norm={norm_llama:>4} -> total {total_llama:,}")
print(f"    GPT-2 : attn={attn_gpt:>6} ffn={ffn_gpt:>6} norm={norm_gpt:>4} -> total {total_gpt:,}")
print("    -> LLaMA economise sur l'attention (GQA : K/V plus petits) mais")
print("       depense plus en FFN (SwiGLU = 3 matrices au lieu de 2).")

print("\n  Impact des 4 ameliorations:")
print("    (a) qualite          : SwiGLU + RMSNorm (gains de loss)")
print("    (b) vitesse inference: GQA (moins de K/V -> cache plus petit, decode plus rapide)")
print("    (c) memoire          : GQA (cache reduit) + RoPE (pas de table positionnelle)")

# ===========================================================================
# EXERCICE 9 - SiTU-GLU : borner une activation pour survivre au FP4
# ===========================================================================
print("\n" + "=" * 70)
print("EXERCICE 9 - SiTU-GLU (Kimi K3, 2026)")
print("=" * 70)

B1, B2 = 4.0, 25.0
FP8_E4M3_MAX = 448.0


def _sigmoid(x):
    # clip avant exp : sans lui, exp(710) deborde et renvoie inf. On borne
    # l'ARGUMENT, pas le resultat, donc la valeur retournee reste exacte.
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def softcap(x, beta):
    """beta * tanh(x / beta) : identite pres de 0, plafonne a beta a l'infini."""
    return beta * np.tanh(x / beta)


def swiglu(gate_pre, up_pre):
    """Swish(gate) * up. Rien ne borne la sortie : elle suit les outliers."""
    return gate_pre * _sigmoid(gate_pre) * up_pre


def situ_glu(gate_pre, up_pre, b1=B1, b2=B2):
    """[b1*tanh(g/b1) * Sigmoid(g)] * [b2*tanh(u/b2)]."""
    gate_branch = softcap(gate_pre, b1) * _sigmoid(gate_pre)
    up_branch = softcap(up_pre, b2)
    return gate_branch * up_branch


# --- 1. Borne theorique ----------------------------------------------------
print("\n  1. Borne theorique")
print(f"     |b1*tanh(g/b1)| <= b1 = {B1}   et   0 < Sigmoid(g) < 1")
print(f"     |b2*tanh(u/b2)| <= b2 = {B2}")
print(f"     -> |y| <= b1 * b2 = {B1 * B2:.0f}")
print("     Le facteur Sigmoid <= 1 ne peut que REDUIRE la borne, jamais")
print("     l'augmenter : la borne dure est donc 100.")
print("     SwiGLU, lui, n'est borne par rien : Swish(g) ~ g quand g grandit,")
print("     donc la sortie croit comme le PRODUIT de deux pre-activations.")

# --- 3. Regime normal ------------------------------------------------------
print("\n  3. Regime normal (gate, up ~ N(0,1))")
rng9 = np.random.default_rng(42)
gate = rng9.standard_normal((512, 1024))
up = rng9.standard_normal((512, 1024))

y_swi = swiglu(gate, up)
y_situ = situ_glu(gate, up)
rel = np.abs(y_situ - y_swi) / (np.abs(y_swi) + 1e-9)
print(f"     ecart relatif median = {np.median(rel):.2%}")
print(f"     ecart relatif p95    = {np.percentile(rel, 95):.2%}")
print("     -> dans le regime ou vivent 99 % des activations, SiTU-GLU est une")
print("        reparametrisation quasi transparente de SwiGLU.")
print("        C'est une condition NECESSAIRE : si la fonction changeait")
print("        partout, on ne bornerait pas SwiGLU, on le remplacerait par")
print("        autre chose - et tout ce que l'on sait de son comportement")
print("        (gains de loss, dynamique d'entrainement) serait a revalider.")

# --- 4. Regime outlier -----------------------------------------------------
print("\n  4. Regime outlier (20 canaux x60 : les 'massive activations')")
gate_o, up_o = gate.copy(), up.copy()
bad = rng9.choice(1024, size=20, replace=False)
gate_o[:, bad] *= 60.0
up_o[:, bad] *= 60.0

max_swi = float(np.max(np.abs(swiglu(gate_o, up_o))))
max_situ = float(np.max(np.abs(situ_glu(gate_o, up_o))))
print(f"     max|SwiGLU|   = {max_swi:10.2f}   "
      f"({'DEBORDE' if max_swi > FP8_E4M3_MAX else 'ok'} FP8 E4M3, max "
      f"{FP8_E4M3_MAX:.0f})")
print(f"     max|SiTU-GLU| = {max_situ:10.2f}   "
      f"({'DEBORDE' if max_situ > FP8_E4M3_MAX else 'ok'} FP8 E4M3)")
print(f"     depassement SwiGLU : x{max_swi / FP8_E4M3_MAX:.1f} le maximum")
print("     representable. En FP8 ces valeurs deviennent des inf, et un inf")
print("     dans une activation contamine tout le reste du forward.")

# --- 5. Soft cap vs hard clamp --------------------------------------------
print("\n  5. Soft cap vs hard clamp : la derivee en x = 100")
h = 1e-4
x0 = 100.0
# Difference AVANT (a droite). Le clip a un point anguleux exactement en 100 :
# une difference centree l'enjamberait et renverrait 0.5, une moyenne entre
# les deux pentes qui n'existe nulle part. C'est la derivee a droite qui
# decrit ce que vit un neurone sature.
d_soft = (softcap(x0 + h, B2) - softcap(x0, B2)) / h
d_clip = (np.clip(x0 + h, -100, 100) - np.clip(x0, -100, 100)) / h
print(f"     d/dx [b2*tanh(x/b2)] en x=100+ = {d_soft:.6f}   (b2={B2:.0f})")
print(f"     d/dx [clip(x,-100,100)] en x=100+ = {d_clip:.6f}")
print("     -> le clip donne un gradient EXACTEMENT nul. Un neurone qui sature")
print("        ne recoit plus aucun signal : il ne peut plus jamais revenir")
print("        dans la zone utile, meme si c'est ce que la loss demande.")
print("        Le soft cap garde un gradient minuscule mais non nul, donc une")
print("        voie de retour. Meme borne, dynamique d'apprentissage opposee.")

# --- 6. Synthese -----------------------------------------------------------
print("\n  6. Synthese : un format numerique a modifie la fonction du reseau")
print("     Autre cas du meme type, jour 17 : le gate de decroissance de KDA.")
print("     Kimi K3 le borne par g_min = -5 non pas pour une raison")
print("     mathematique, mais pour que 1/Gamma tienne dans la dynamique du")
print("     BF16 sur une tuile de 16 tokens - ce qui permet de calculer TOUTES")
print("     les tuiles causales, diagonales comprises, en matmul dense sur")
print("     Tensor Cores.")
print("     Jour 19, meme logique : le format MX fixe un bloc de 32 elements et")
print("     une echelle en puissance de deux, moins precis que NF4, mais")
print("     dequantifiable dans le datapath du Tensor Core.")
print("     Dans les trois cas la contrainte materielle remonte jusqu'au design")
print("     mathematique. A l'echelle frontier, ce n'est plus l'exception.")


print("\n" + "=" * 70)
print("FIN DES SOLUTIONS HARD (Jour 9)")
print("=" * 70)
