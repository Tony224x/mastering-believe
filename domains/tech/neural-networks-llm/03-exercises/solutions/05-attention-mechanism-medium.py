"""
Solutions MEDIUM — Jour 5 : Attention Mechanism
===============================================
Exercices 4, 5, 6 (medium). Pur NumPy (comme 02-code/05-attention-mechanism.py).

Run: python 03-exercises/solutions/05-attention-mechanism-medium.py
"""

import numpy as np

np.random.seed(42)


def softmax(x, axis=-1):
    """Softmax stable le long d'un axe (shift-invariant)."""
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V.
    mask : (n_q, n_k), convention cours : 1 = bloque, 0 = autorise.
    Renvoie output (n_q, d_v) et weights (n_q, n_k).
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)         # (n_q, n_k)
    if mask is not None:
        scores = np.where(mask == 1, -np.inf, scores)
    weights = softmax(scores, axis=-1)
    output = weights @ V
    return output, weights


def causal_mask(n):
    """1 dans le triangle superieur strict (positions futures bloquees)."""
    return np.triu(np.ones((n, n), dtype=np.int32), k=1)


# ============================================================================
# EXERCISE 4: Self-attention layer + proprietes
# ============================================================================

print("=" * 70)
print("EXERCISE 4: Self-attention layer + proprietes")
print("=" * 70)


class SelfAttention:
    def __init__(self, d_model, d_k=None, seed=0):
        rng = np.random.RandomState(seed)
        self.d_k = d_k or d_model
        scale = 1.0 / np.sqrt(d_model)
        self.W_Q = rng.randn(d_model, self.d_k) * scale
        self.W_K = rng.randn(d_model, self.d_k) * scale
        self.W_V = rng.randn(d_model, self.d_k) * scale

    def forward(self, X, mask=None):
        Q, K, V = X @ self.W_Q, X @ self.W_K, X @ self.W_V
        return scaled_dot_product_attention(Q, K, V, mask)


d_model, seq = 8, 5
X = np.random.randn(seq, d_model) * 0.5
sa = SelfAttention(d_model, seed=1)
out, w = sa.forward(X)

# Propriete 1 : lignes stochastiques.
row_sums = w.sum(axis=1)
print(f"\n  Sommes des lignes des poids : {np.round(row_sums, 8)}")
print(f"  Max ecart a 1 : {np.max(np.abs(row_sums - 1.0)):.2e}")

# Propriete 2 : masque causal annule le triangle superieur strict.
out_c, w_c = sa.forward(X, mask=causal_mask(seq))
upper = w_c[np.triu_indices(seq, k=1)]
print(f"\n  Masque causal: max poids dans le triangle superieur strict = {np.max(np.abs(upper)):.2e}")
print("  -> 0 : aucun token ne regarde le futur.")

# Propriete 3 : invariance par permutation (sans masque ni PE).
perm = np.array([2, 0, 4, 1, 3])
out_perm, _ = sa.forward(X[perm])
# L'output des tokens permutes = permutation de l'output original.
print(f"\n  Invariance par permutation: max |out[perm] - out_perm| = "
      f"{np.max(np.abs(out[perm] - out_perm)):.2e}")
print("  -> ~0 : l'attention seule est permutation-equivariante -> il FAUT un positional encoding.")

# Question : justification du sqrt(d_k).
print("\n  Justification du sqrt(d_k) (Var(Q.K) = d_k):")
for d_k in [4, 64, 512]:
    q = np.random.randn(100000, d_k)
    k = np.random.randn(100000, d_k)
    dots = np.sum(q * k, axis=1)            # produits scalaires q_i . k_i
    print(f"    d_k={d_k:>4}: Var(Q.K) empirique = {dots.var():.2f} (~ d_k), "
          f"Var apres /sqrt(d_k) = {(dots / np.sqrt(d_k)).var():.2f} (~1)")


# ============================================================================
# EXERCISE 5: Multi-head attention + cout memoire
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 5: Multi-head attention + cout memoire")
print("=" * 70)


class MultiHeadAttention:
    def __init__(self, d_model, n_heads, seed=0):
        assert d_model % n_heads == 0
        self.d_model, self.n_heads = d_model, n_heads
        self.d_head = d_model // n_heads
        rng = np.random.RandomState(seed)
        scale = 1.0 / np.sqrt(d_model)
        self.W_Q = rng.randn(d_model, d_model) * scale
        self.W_K = rng.randn(d_model, d_model) * scale
        self.W_V = rng.randn(d_model, d_model) * scale
        self.W_O = rng.randn(d_model, d_model) * scale

    def forward(self, X, mask=None):
        seq = X.shape[0]
        Q, K, V = X @ self.W_Q, X @ self.W_K, X @ self.W_V
        # (seq, d_model) -> (n_heads, seq, d_head)
        def split(M):
            return M.reshape(seq, self.n_heads, self.d_head).transpose(1, 0, 2)
        Qh, Kh, Vh = split(Q), split(K), split(V)
        outs, ws = [], []
        for h in range(self.n_heads):
            o, wt = scaled_dot_product_attention(Qh[h], Kh[h], Vh[h], mask)
            outs.append(o); ws.append(wt)
        concat = np.concatenate(outs, axis=-1)   # (seq, d_model)
        return concat @ self.W_O, np.stack(ws, 0)


d_model, n_heads, seq = 64, 8, 10
X = np.random.randn(seq, d_model) * 0.5
mha = MultiHeadAttention(d_model, n_heads, seed=2)
out, w = mha.forward(X)

print(f"\n  Shapes (d_model={d_model}, n_heads={n_heads}, seq={seq}):")
print(f"    Input X        : {X.shape}")
print(f"    Q = X @ W_Q    : {(X @ mha.W_Q).shape}")
print(f"    Split en tetes : ({n_heads}, {seq}, {mha.d_head})  d_head={mha.d_head}")
print(f"    Poids attention: {w.shape}  (n_heads, seq, seq)")
print(f"    Output         : {out.shape}")

# Equivalence : grosse projection + reshape vs tetes separees.
Q = X @ mha.W_Q
head0_via_reshape = Q.reshape(seq, n_heads, mha.d_head).transpose(1, 0, 2)[0]
# Tete 0 = colonnes 0..d_head-1 de W_Q.
head0_direct = X @ mha.W_Q[:, 0:mha.d_head]
print(f"\n  Equivalence reshape vs sous-bloc colonnes (tete 0): "
      f"max diff = {np.max(np.abs(head0_via_reshape - head0_direct)):.2e}")

# Cout memoire de la matrice d'attention (n_heads * L^2 floats * 4 bytes).
print("\n  Memoire de la matrice d'attention (n_heads=12, float32):")
for L in [512, 2048, 8192]:
    mem_mb = 12 * L * L * 4 / 1e6
    print(f"    L={L:>5}: {mem_mb:>10.1f} MB  (croit en O(L^2))")
print("  -> doubler L quadruple la memoire d'attention.")

print("\n  Params: single-head et multi-head ont le MEME nombre de params")
print("  (projections d_model x d_model dans les 2 cas). Multi-head permet la specialisation.")


# ============================================================================
# EXERCISE 6: Cross-attention
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 6: Cross-attention et role de Q vs K/V")
print("=" * 70)


def cross_attention(X_query, X_context, W_Q, W_K, W_V):
    """Q depuis la source query, K/V depuis le contexte."""
    Q = X_query @ W_Q          # (n_q, d_k)
    K = X_context @ W_K        # (n_c, d_k)
    V = X_context @ W_V        # (n_c, d_v)
    return scaled_dot_product_attention(Q, K, V)


d_model, d_k = 6, 6
rng = np.random.RandomState(3)
W_Q = rng.randn(d_model, d_k) * 0.3
W_K = rng.randn(d_model, d_k) * 0.3
W_V = rng.randn(d_model, d_k) * 0.3

n_q, n_c = 3, 5
X_query = np.random.randn(n_q, d_model)
X_context = np.random.randn(n_c, d_model)
out, w = cross_attention(X_query, X_context, W_Q, W_K, W_V)
print(f"\n  Shapes asymetriques: X_query {X_query.shape}, X_context {X_context.shape}")
print(f"    Matrice d'attention : {w.shape}  (n_q, n_c) = ({n_q}, {n_c})")
print(f"    Output              : {out.shape}  (suit la query, pas le contexte)")

# Self-attention = cross-attention avec X_query == X_context.
out_self_via_cross, _ = cross_attention(X_query, X_query, W_Q, W_K, W_V)
Q, K, V = X_query @ W_Q, X_query @ W_K, X_query @ W_V
out_self_direct, _ = scaled_dot_product_attention(Q, K, V)
print(f"\n  Self == cross(X,X): max diff = {np.max(np.abs(out_self_via_cross - out_self_direct)):.2e}")

# Mini-retrieval : query alignee avec une seule key.
print("\n  Mini-retrieval (1 query alignee avec key #2):")
Q_r = np.array([[0.0, 0.0, 10.0]])             # query
K_r = np.eye(3) * 1.0                           # keys = base canonique
V_r = np.array([[1.0], [2.0], [99.0]])         # values distinctes
out_r, w_r = scaled_dot_product_attention(Q_r, K_r, V_r)
print(f"    poids d'attention : {np.round(w_r[0], 4)}  (concentre sur key #2)")
print(f"    output            : {np.round(out_r[0], 4)}  (~ value #2 = 99)")

print("\n  Cross-attention dans un encoder-decoder (traduction): a chaque pas de")
print("  generation, le decoder 'regarde' les representations de la phrase source.")

print("\n" + "=" * 70)
print("FIN DES SOLUTIONS MEDIUM (Jour 5)")
print("=" * 70)
