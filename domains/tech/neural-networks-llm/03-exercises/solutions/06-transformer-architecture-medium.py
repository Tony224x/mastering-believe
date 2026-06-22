"""
Solutions MEDIUM — Jour 6 : Transformer Architecture
====================================================
Exercices 4, 5, 6 (medium). Pur NumPy.

4. Bloc Transformer complet (forward, pre-norm).
5. LayerNorm backward + gradient check.
6. Comptage exact des parametres.

Run: python 03-exercises/solutions/06-transformer-architecture-medium.py
"""

import math
import numpy as np

np.random.seed(42)


def softmax_rows(S):
    S = S - np.max(S, axis=-1, keepdims=True)
    e = np.exp(S)
    return e / np.sum(e, axis=-1, keepdims=True)


def gelu(x):
    """GeLU approximee (formule tanh, comme GPT-2 / 02-code/09)."""
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def layer_norm(x, gamma, beta, eps=1e-5):
    """LayerNorm sur la derniere dimension (features)."""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)        # population (ddof=0)
    x_hat = (x - mu) / np.sqrt(var + eps)
    return gamma * x_hat + beta


# ============================================================================
# EXERCISE 4: Bloc Transformer complet (forward, pre-norm)
# ============================================================================

print("=" * 70)
print("EXERCISE 4: Bloc Transformer (forward, pre-norm)")
print("=" * 70)


def self_attention(x, W_Q, W_K, W_V, W_O):
    """Single-head self-attention (suffisant pour la pedagogie)."""
    d_k = W_Q.shape[1]
    Q, K, V = x @ W_Q, x @ W_K, x @ W_V
    S = Q @ K.T / np.sqrt(d_k)
    A = softmax_rows(S)
    return (A @ V) @ W_O


def feed_forward(x, W1, b1, W2, b2):
    """Linear -> GELU -> Linear (position-wise)."""
    return gelu(x @ W1 + b1) @ W2 + b2


def make_block_params(d_model, d_ff, seed=0):
    rng = np.random.RandomState(seed)
    sc = 1.0 / np.sqrt(d_model)
    return {
        'W_Q': rng.randn(d_model, d_model) * sc,
        'W_K': rng.randn(d_model, d_model) * sc,
        'W_V': rng.randn(d_model, d_model) * sc,
        'W_O': rng.randn(d_model, d_model) * sc,
        'W1': rng.randn(d_model, d_ff) * sc, 'b1': np.zeros(d_ff),
        'W2': rng.randn(d_ff, d_model) * sc, 'b2': np.zeros(d_model),
        'g1': np.ones(d_model), 'be1': np.zeros(d_model),
        'g2': np.ones(d_model), 'be2': np.zeros(d_model),
    }


def transformer_block(x, p):
    """Pre-norm: x = x + Attn(LN(x)); x = x + FFN(LN(x))."""
    a = layer_norm(x, p['g1'], p['be1'])
    x = x + self_attention(a, p['W_Q'], p['W_K'], p['W_V'], p['W_O'])
    b = layer_norm(x, p['g2'], p['be2'])
    x = x + feed_forward(b, p['W1'], p['b1'], p['W2'], p['b2'])
    return x


d_model, d_ff, seq = 16, 64, 6
x = np.random.randn(seq, d_model)
p = make_block_params(d_model, d_ff, seed=1)
y = transformer_block(x, p)
print(f"\n  Input shape : {x.shape}")
print(f"  Output shape: {y.shape}  (preservee -> empilable)")

# Empiler 4 blocs, verifier que la norme reste bornee.
blocks = [make_block_params(d_model, d_ff, seed=s) for s in range(4)]
h = x.copy()
print("\n  Empilement de 4 blocs (norme moyenne de l'activation):")
print(f"    entree : {np.linalg.norm(h) / np.sqrt(h.size):.4f}")
for i, bp in enumerate(blocks):
    h = transformer_block(h, bp)
    print(f"    bloc {i}: {np.linalg.norm(h) / np.sqrt(h.size):.4f}")
print("  -> bornee (les LayerNorm renormalisent a chaque sous-couche).")
print("\n  Pre-norm: le chemin residuel reste 'propre' (somme d'identites + corrections),")
print("  le gradient circule sans renormalisation -> stack profond entrainable.")


# ============================================================================
# EXERCISE 5: LayerNorm backward + gradient check
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 5: LayerNorm backward + gradient check")
print("=" * 70)


def layernorm_forward(x, gamma, beta, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    std_inv = 1.0 / np.sqrt(var + eps)
    x_hat = (x - mu) * std_inv
    y = gamma * x_hat + beta
    cache = (x, x_hat, mu, var, std_inv, gamma, eps)
    return y, cache


def layernorm_backward(dy, cache):
    """
    Backward LayerNorm. mu et var dependent de tous les x_i -> dx a 3 termes.
    Par ligne, D = nombre de features.
    """
    x, x_hat, mu, var, std_inv, gamma, eps = cache
    D = x.shape[-1]

    # Parametres (somme sur le batch = toutes les lignes).
    dgamma = np.sum(dy * x_hat, axis=0)
    dbeta = np.sum(dy, axis=0)

    dx_hat = dy * gamma
    # dvar et dmu redescendent a travers les statistiques.
    dvar = np.sum(dx_hat * (x - mu) * -0.5 * std_inv ** 3, axis=-1, keepdims=True)
    dmu = (np.sum(dx_hat * -std_inv, axis=-1, keepdims=True)
           + dvar * np.mean(-2.0 * (x - mu), axis=-1, keepdims=True))
    dx = dx_hat * std_inv + dvar * 2.0 * (x - mu) / D + dmu / D
    return dx, dgamma, dbeta


B, D = 4, 8
x = np.random.randn(B, D) * 2.0 + 1.0
gamma = np.random.randn(D) * 0.5 + 1.0
beta = np.random.randn(D) * 0.5

y, cache = layernorm_forward(x, gamma, beta)
dy = y.copy()                                  # loss = 0.5*||y||^2 -> dy = y
dx, dgamma, dbeta = layernorm_backward(dy, cache)

eps = 1e-5
print("\n  Gradient check LayerNorm (eps=1e-5):")
all_ok = True
for name, M, dM in [('x', x, dx), ('gamma', gamma, dgamma), ('beta', beta, dbeta)]:
    gnum = np.zeros_like(M)
    it = np.nditer(M, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index; orig = M[idx]
        M[idx] = orig + eps; yp, _ = layernorm_forward(x, gamma, beta); lp = 0.5*np.sum(yp**2)
        M[idx] = orig - eps; ym, _ = layernorm_forward(x, gamma, beta); lm = 0.5*np.sum(ym**2)
        M[idx] = orig
        gnum[idx] = (lp - lm) / (2 * eps)
        it.iternext()
    denom = np.abs(dM) + np.abs(gnum) + 1e-8
    rel = np.max(np.abs(dM - gnum) / denom)
    ok = rel < 1e-5; all_ok = all_ok and ok
    print(f"    d{name:>5}: max rel error = {rel:.2e}  [{'PASS' if ok else 'FAIL'}]")
print(f"  -> {'LAYERNORM BACKWARD CORRECT' if all_ok else 'WRONG'}")
print("  dx n'est PAS juste dy*gamma/std : il manquerait les termes dmu et dvar")
print("  (mu et var sont fonctions de tous les x_i).")


# ============================================================================
# EXERCISE 6: Comptage exact des parametres
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 6: Comptage exact des parametres")
print("=" * 70)


def count_block(D, d_ff=None):
    """Params d'un bloc : attention (4D^2, no bias) + FFN (8D^2+5D) + 2 LN (4D)."""
    if d_ff is None:
        d_ff = 4 * D
    attn = 4 * D * D
    ffn = (D * d_ff + d_ff) + (d_ff * D + D)    # W1+b1 + W2+b2
    norms = 2 * (2 * D)                          # 2 LayerNorm (gamma+beta)
    return attn, ffn, norms


def count_gpt(V, D, n_layers, P, tied_head=True):
    attn, ffn, norms = count_block(D)
    per_block = attn + ffn + norms
    stack = n_layers * per_block
    emb = V * D + P * D                          # token + pos
    final_ln = 2 * D
    head = 0 if tied_head else V * D
    return per_block, stack, emb, final_ln, head, stack + emb + final_ln + head


D = 768
attn, ffn, norms = count_block(D)
print(f"\n  Par bloc (D={D}, d_ff=4D):")
print(f"    Attention : {attn:>12,}  (4 D^2)")
print(f"    FFN       : {ffn:>12,}  (8 D^2 + 5D)")
print(f"    2 LayerNorm: {norms:>11,}  (4D)")
print(f"    Ratio FFN/attention = {ffn / attn:.2f}x  (d_ff=4D -> 2 matrices de 4D^2)")

per_block, stack, emb, fln, head, total = count_gpt(50257, 768, 12, 1024, tied_head=True)
print(f"\n  GPT-2 small (V=50257, D=768, L=12, P=1024, head tied):")
print(f"    Par bloc      : {per_block:>13,}")
print(f"    Stack (12)    : {stack:>13,}")
print(f"    Embeddings    : {emb:>13,}")
print(f"    Final LN+head : {fln + head:>13,}")
print(f"    TOTAL         : {total:>13,}  (~124M reels)")
frac_emb = emb / total
print(f"    Fraction embeddings: {frac_emb:.1%}")
print("\n  Quand le modele grandit (D, L), le stack en O(L*D^2) domine les")
print("  embeddings en O(V*D) -> la fraction embeddings chute (GPT-3 175B: <1%).")

print("\n" + "=" * 70)
print("FIN DES SOLUTIONS MEDIUM (Jour 6)")
print("=" * 70)
