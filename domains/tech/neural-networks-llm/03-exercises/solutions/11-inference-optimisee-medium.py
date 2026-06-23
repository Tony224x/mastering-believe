"""
Solutions MEDIUM — Jour 11 : Inference optimisee
=================================================
Exercices 4, 5, 6 (medium).

Pur NumPy (comme 02-code/11-inference-optimisee.py). Aucun framework requis.
Chaque etape non triviale est commentee avec le POURQUOI.

Run: python 03-exercises/solutions/11-inference-optimisee-medium.py
"""

import sys
import io
import time
import math
import numpy as np

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

np.random.seed(42)

d_model = 32
n_heads = 4
head_dim = d_model // n_heads


# ============================================================================
# EXERCICE 4 : KV cache vs naif
# ============================================================================

print("=" * 70)
print("EXERCICE 4 : KV cache vs naif — equivalence + scaling")
print("=" * 70)


def attention_no_cache(x, W_q, W_k, W_v, W_o):
    """Attention causale standard, recalcul complet (voir 02-code)."""
    seq_len = x.shape[0]
    Q = (x @ W_q).reshape(seq_len, n_heads, head_dim)
    K = (x @ W_k).reshape(seq_len, n_heads, head_dim)
    V = (x @ W_v).reshape(seq_len, n_heads, head_dim)
    out = np.zeros_like(Q)
    for h in range(n_heads):
        scores = Q[:, h] @ K[:, h].T / math.sqrt(head_dim)
        mask = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)
        scores = scores + mask
        w = np.exp(scores - scores.max(axis=-1, keepdims=True))
        w = w / w.sum(axis=-1, keepdims=True)
        out[:, h] = w @ V[:, h]
    return (out.reshape(seq_len, d_model)) @ W_o


class AttentionWithCache:
    """Attention avec KV cache (un token a la fois)."""

    def __init__(self, Wq, Wk, Wv, Wo):
        self.W_q, self.W_k, self.W_v, self.W_o = Wq, Wk, Wv, Wo
        self.reset()

    def reset(self):
        self.K_cache = None
        self.V_cache = None

    def forward_step(self, x_new):
        q = (x_new @ self.W_q).reshape(n_heads, head_dim)
        k = (x_new @ self.W_k).reshape(n_heads, head_dim)
        v = (x_new @ self.W_v).reshape(n_heads, head_dim)
        if self.K_cache is None:
            self.K_cache = k[np.newaxis]
            self.V_cache = v[np.newaxis]
        else:
            self.K_cache = np.concatenate([self.K_cache, k[np.newaxis]], axis=0)
            self.V_cache = np.concatenate([self.V_cache, v[np.newaxis]], axis=0)
        out = np.zeros((n_heads, head_dim))
        for h in range(n_heads):
            scores = self.K_cache[:, h] @ q[h] / math.sqrt(head_dim)
            w = np.exp(scores - scores.max())
            w = w / w.sum()
            out[h] = w @ self.V_cache[:, h]
        return out.reshape(d_model) @ self.W_o


W_q = np.random.randn(d_model, d_model) * 0.1
W_k = np.random.randn(d_model, d_model) * 0.1
W_v = np.random.randn(d_model, d_model) * 0.1
W_o = np.random.randn(d_model, d_model) * 0.1

# Equivalence : a chaque prefix, la sortie cache du dernier token == sortie naive
seq = np.random.randn(12, d_model)
attn = AttentionWithCache(W_q, W_k, W_v, W_o)
max_diff = 0.0
for t in range(seq.shape[0]):
    out_cache = attn.forward_step(seq[t])               # token t avec cache
    out_naive = attn_naive_last = attention_no_cache(seq[:t + 1], W_q, W_k, W_v, W_o)[-1]
    max_diff = max(max_diff, np.max(np.abs(out_cache - out_naive)))
print(f"\nEquivalence cache vs naif : ecart max = {max_diff:.2e}")
print("  -> exact (pas approx) : le cache stocke EXACTEMENT les memes K,V deja calcules.")


def attn_flops_per_token(t, d):
    """FLOPs ~ de l'attention pour generer le token t (ordre de grandeur)."""
    naive = (t ** 2) * d   # recalcul scores sur toute la sequence
    cached = t * d         # q courant contre t cles
    return naive, cached


print("\nFLOPs par token (ratio naif/cache):")
for t in [100, 1000, 10000]:
    n, c = attn_flops_per_token(t, d_model)
    print(f"  t={t:>6d} : naif={n:.2e}  cache={c:.2e}  ratio={n/c:.0f}")

# Cout cumule pour n tokens : sum t^2 vs sum t
n = 1000
cum_naive = sum(t ** 2 for t in range(1, n + 1))
cum_cache = sum(t for t in range(1, n + 1))
print(f"\nCout cumule n={n}: naif={cum_naive:.2e}, cache={cum_cache:.2e}, "
      f"ratio={cum_naive/cum_cache:.0f} (≈2n/3={2*n/3:.0f})")

# Mesure de temps
def gen_naive(initial, k):
    toks = initial.copy()
    for _ in range(k):
        out = attention_no_cache(toks, W_q, W_k, W_v, W_o)[-1]
        nxt = out / (np.linalg.norm(out) + 1e-8)
        toks = np.vstack([toks, nxt[np.newaxis]])
    return toks


def gen_cache(initial, k):
    attn.reset()
    for t in range(initial.shape[0]):
        out = attn.forward_step(initial[t])
    cur = out
    for _ in range(k):
        nxt = cur / (np.linalg.norm(cur) + 1e-8)
        cur = attn.forward_step(nxt)
    return cur


prompt = np.random.randn(20, d_model)
for k in [50, 200]:
    t0 = time.perf_counter(); gen_naive(prompt, k); tn = time.perf_counter() - t0
    t0 = time.perf_counter(); gen_cache(prompt, k); tc = time.perf_counter() - t0
    print(f"  k={k:>4d} nouveaux tokens : naif={tn*1e3:6.1f}ms  cache={tc*1e3:6.1f}ms  "
          f"speedup={tn/max(tc,1e-9):.1f}x")

print("""
Analyse: le cache transforme O(n^3) total en O(n^2) car chaque token ne re-attend
que UNE fois aux cles deja calculees au lieu de tout recalculer. Prix paye: la memoire
du cache (2 * n_layers * n_kv_heads * head_dim * seq), qui grandit lineairement en seq.
""")


# ============================================================================
# EXERCICE 5 : Quantization int8/int4, per-tensor vs per-channel
# ============================================================================

print("=" * 70)
print("EXERCICE 5 : Quantization int8/int4, per-tensor vs per-channel")
print("=" * 70)


def quantize(W, bits=8, mode="per_tensor"):
    """Quantization symetrique. Renvoie (q, scale)."""
    qmax = 2 ** (bits - 1) - 1  # int8 -> 127, int4 -> 7
    if mode == "per_tensor":
        max_abs = np.abs(W).max()
        scale = max_abs / qmax
        q = np.clip(np.round(W / scale), -qmax, qmax)
    elif mode == "per_channel":
        # une scale par ligne (channel de sortie)
        max_abs = np.abs(W).max(axis=1, keepdims=True)
        scale = max_abs / qmax            # (rows, 1)
        q = np.clip(np.round(W / scale), -qmax, qmax)
    else:
        raise ValueError(mode)
    return q, scale


def dequantize(q, scale):
    return q * scale  # broadcast pour per_channel (scale (rows,1))


# Matrice realiste avec outliers
W = np.random.randn(256, 256) * 0.05
# Injecter quelques outliers 10x plus grands sur certaines lignes
out_rows = [3, 17, 100, 200]
W[out_rows, :] += np.random.randn(len(out_rows), 256) * 0.5
x = np.random.randn(8, 256)
y_ref = x @ W.T  # reference matmul (x @ W.T comme une couche Linear)

print(f"\n{'bits':>4s} {'mode':>12s} {'err_moy':>10s} {'err_max':>10s} {'matmul_rel':>11s}")
for bits in [8, 4]:
    for mode in ["per_tensor", "per_channel"]:
        q, scale = quantize(W, bits, mode)
        Wd = dequantize(q, scale)
        err = np.abs(W - Wd)
        y_q = x @ Wd.T
        rel = np.linalg.norm(y_ref - y_q) / np.linalg.norm(y_ref)
        print(f"{bits:>4d} {mode:>12s} {err.mean():>10.2e} {err.max():>10.2e} "
              f"{rel*100:>10.3f}%")

# Ratio int4/int8 sur l'erreur (per_tensor)
q8, s8 = quantize(W, 8, "per_tensor"); e8 = np.abs(W - dequantize(q8, s8)).mean()
q4, s4 = quantize(W, 4, "per_tensor"); e4 = np.abs(W - dequantize(q4, s4)).mean()
print(f"\nRatio erreur int4/int8 (per_tensor) = {e4/e8:.1f}x "
      f"(attendu ~16 = 127/7 ; le pas de quant. passe de max/127 a max/7)")

# Cout des scales per-channel
rows = W.shape[0]
print(f"Cout scales per-channel : {rows} floats vs {rows*W.shape[1]} poids "
      f"-> {100*rows/(rows*W.shape[1]):.3f}% (negligeable)")

print("""
Analyse:
- Per-channel < per-tensor en erreur : un outlier ne gonfle que la scale de SA ligne,
  pas celle de tout le tenseur (qui ecraserait toutes les petites valeurs).
- int4 ~16x plus bruite qu'int8 : utile quand la memoire prime (edge, gros modeles)
  et tolerable surtout avec per-channel + group-wise quantization.
""")


# ============================================================================
# EXERCICE 6 : Speculative decoding (draft + verify)
# ============================================================================

print("=" * 70)
print("EXERCICE 6 : Speculative decoding")
print("=" * 70)


def speculative_step(p, q, K, rng):
    """
    Un pass de speculative decoding (Leviathan et al., 2023) sur un vocab discret.
    p = distribution target (vraie), q = distribution draft (approx).
    Renvoie la liste des tokens produits (acceptes + 1 token final).
    """
    # Draft propose K tokens i.i.d. selon q (cas simplifie : meme distrib a chaque pos)
    proposals = rng.choice(len(q), size=K, p=q)
    produced = []
    for x in proposals:
        r = rng.random()
        if r < min(1.0, p[x] / q[x]):
            produced.append(x)           # accepte
        else:
            # rejet : echantillonner depuis la distribution residuelle normalisee
            resid = np.maximum(p - q, 0)
            resid = resid / resid.sum()
            produced.append(rng.choice(len(p), p=resid))
            return produced              # stop au premier rejet
    # Tous acceptes : token bonus depuis p
    produced.append(rng.choice(len(p), p=p))
    return produced


V = 6
p = np.random.dirichlet(np.ones(V) * 2)
q = np.random.dirichlet(np.ones(V) * 2)
rng = np.random.default_rng(0)

# 1) Correction du sampling : la distribution du PREMIER token produit == p
counts = np.zeros(V)
N = 200000
for _ in range(N):
    toks = speculative_step(p, q, K=4, rng=rng)
    counts[toks[0]] += 1
emp = counts / N
l1 = np.abs(emp - p).sum()
print(f"\nDistribution empirique (1er token) vs p_target : L1 = {l1:.4f}")
print(f"  p     : {np.round(p, 3)}")
print(f"  empir : {np.round(emp, 3)}")
print("  -> identique (a l'echantillonnage pres) : speculative decoding ne biaise PAS p.")

# 2) Acceptance rate vs proximite draft/target
print("\nAcceptance (tokens acceptes / pass) vs proximite draft->target:")
for alpha_mix in [0.0, 0.5, 0.9, 1.0]:
    q_mix = (1 - alpha_mix) * q + alpha_mix * p   # rapproche q de p
    q_mix /= q_mix.sum()
    accepted = []
    for _ in range(20000):
        toks = speculative_step(p, q_mix, K=4, rng=rng)
        accepted.append(len(toks) - 1)  # -1 car le dernier token est le correction/bonus
    print(f"  mix={alpha_mix:.1f} (q->p) : ~{np.mean(accepted):.2f} tokens acceptes / pass")

# 3) Modele de speedup
print("\nSpeedup theorique (c=0.1 = draft 10x moins cher) vs alpha (tokens/pass):")
c = 0.1
for alpha in [1.0, 2.0, 3.0, 4.0]:
    # cout par token genere ~ (1 pass target + K passes draft) / alpha
    # speedup ~ alpha / (1 + K*c) avec K~alpha
    K = alpha
    speedup = alpha / (1 + K * c)
    print(f"  alpha={alpha:.0f} : speedup ≈ {speedup:.2f}x")

print("""
Analyse:
- Pas de perte de qualite : la regle accept (min(1, p/q)) + correction residuelle
  garantit mathematiquement que la distribution produite == p_target.
- Si le draft est mauvais (q loin de p), l'acceptance s'effondre : on rejette tot,
  on a paye les K passes draft pour presque rien -> le gain disparait.
""")

print("=" * 70)
print("Fin solutions medium Jour 11.")
print("=" * 70)
