"""
Solutions HARD — Jour 7 : Mini-Transformer (Capstone Week 1)
===========================================================
Exercices 7, 8 (hard). Pur NumPy.

7. KV-cache from scratch : equivalence EXACTE prefill vs decode + cout.
8. Backward du bloc (softmax-CE, LayerNorm, residual) + gradient check + weight tying.

Run: python 03-exercises/solutions/07-mini-transformer-hard.py
"""

import numpy as np

np.random.seed(42)


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


# ============================================================================
# EXERCISE 7: KV-cache from scratch — equivalence prefill vs decode
# ============================================================================

print("=" * 70)
print("EXERCISE 7: KV-cache — equivalence EXACTE prefill vs decode")
print("=" * 70)


def split_heads(M, n_head):
    """(seq, d_model) -> (n_head, seq, head_dim)."""
    seq, d_model = M.shape
    head_dim = d_model // n_head
    return M.reshape(seq, n_head, head_dim).transpose(1, 0, 2)


def attn_full(X, W_qkv, W_o, n_head):
    """Attention causale complete (prefill) sur toute la sequence."""
    seq, d_model = X.shape
    head_dim = d_model // n_head
    q, k, v = np.split(X @ W_qkv, 3, axis=-1)
    qh, kh, vh = split_heads(q, n_head), split_heads(k, n_head), split_heads(v, n_head)
    mask = np.triu(np.ones((seq, seq)), k=1)
    outs = []
    for h in range(n_head):
        s = qh[h] @ kh[h].T / np.sqrt(head_dim)
        s = np.where(mask == 1, -np.inf, s)
        outs.append(softmax(s, -1) @ vh[h])
    return np.concatenate(outs, axis=-1) @ W_o


def attn_step(x_t, cache, W_qkv, W_o, n_head):
    """
    Pas incremental : x_t (d_model,) est le SEUL token courant.
    cache : dict {'k': (n_head, t, head_dim), 'v': ...} des tokens precedents.
    On append k_t/v_t puis q_t attend sur tout le cache (0..t). Pas de masque
    necessaire : les futurs ne sont pas dans le cache.
    Renvoie out_t (d_model,) et le cache mis a jour.
    """
    d_model = x_t.shape[0]
    head_dim = d_model // n_head
    q, k, v = np.split(x_t @ W_qkv, 3, axis=-1)         # chacun (d_model,)
    qh = q.reshape(n_head, head_dim)
    kh = k.reshape(n_head, head_dim)
    vh = v.reshape(n_head, head_dim)
    # Append au cache.
    if cache['k'] is None:
        cache['k'] = kh[:, None, :]                     # (n_head, 1, head_dim)
        cache['v'] = vh[:, None, :]
    else:
        cache['k'] = np.concatenate([cache['k'], kh[:, None, :]], axis=1)
        cache['v'] = np.concatenate([cache['v'], vh[:, None, :]], axis=1)
    outs = []
    for h in range(n_head):
        s = qh[h] @ cache['k'][h].T / np.sqrt(head_dim)  # (t+1,)
        w = softmax(s, -1)
        outs.append(w @ cache['v'][h])                   # (head_dim,)
    concat = np.concatenate(outs)                        # (d_model,)
    return concat @ W_o, cache


d_model, n_head, T = 32, 4, 9
X = np.random.randn(T, d_model) * 0.5
rng = np.random.RandomState(1)
W_qkv = rng.randn(d_model, 3 * d_model) * 0.1
W_o = rng.randn(d_model, d_model) * 0.1

O_full = attn_full(X, W_qkv, W_o, n_head)

cache = {'k': None, 'v': None}
O_inc = np.zeros_like(O_full)
for t in range(T):
    O_inc[t], cache = attn_step(X[t], cache, W_qkv, W_o, n_head)

print(f"\n  max |O_full - O_inc| = {np.max(np.abs(O_full - O_inc)):.2e}")
print("  -> < 1e-10 : le KV-cache est EXACT, pas une approximation.")

# Cout des projections Q/K/V.
print("\n  Cout des projections Q/K/V pour generer T tokens:")
for Tn in [128, 512, 2048]:
    naif = sum(t for t in range(1, Tn + 1))     # re-projete 1+2+...+T tokens
    cache_cost = Tn                             # projete 1 token par pas
    print(f"    T={Tn:>5}: naif={naif:>9} (O(T^2)) | cache={cache_cost:>5} (O(T)) "
          f"| gain x{naif / cache_cost:.0f}")
print("  -> sans cache on RE-projete tous les tokens passes a chaque pas (O(T^2)).")
print("     avec cache, on ne projette que le nouveau token (O(T)).")
print("\n  Le decode devient memory-bound : a chaque pas un tout petit matmul")
print("  (1,d)x(d,3d), mais il faut RELIRE tout le cache (t,d) -> la memoire domine.")


# ============================================================================
# EXERCISE 8: Backward du bloc + gradient check + weight tying
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 8: Backward (softmax-CE, LayerNorm, residual) + weight tying")
print("=" * 70)


def softmax_ce_backward(logits, target):
    """dL/dz = softmax(z) - onehot(target), pour une seule ligne (vocab,)."""
    p = softmax(logits)
    p[target] -= 1.0
    return p


def layernorm_forward(x, gamma, beta, eps=1e-5):
    """x (n, D). Renvoie y et cache pour le backward."""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    x_hat = (x - mu) * inv
    y = x_hat * gamma + beta
    return y, (x_hat, inv, gamma)


def layernorm_backward(dy, cache):
    """
    Backward du LayerNorm sur le dernier axe (dimension D).
      dx = inv * (dx_hat - mean(dx_hat) - x_hat * mean(dx_hat * x_hat))
    Renvoie dx, dgamma, dbeta.
    """
    x_hat, inv, gamma = cache
    D = x_hat.shape[-1]
    dbeta = dy.sum(axis=0)
    dgamma = (dy * x_hat).sum(axis=0)
    dx_hat = dy * gamma
    mean_dxh = dx_hat.mean(axis=-1, keepdims=True)
    mean_dxh_xhat = (dx_hat * x_hat).mean(axis=-1, keepdims=True)
    dx = inv * (dx_hat - mean_dxh - x_hat * mean_dxh_xhat)
    return dx, dgamma, dbeta


# --- Verif LayerNorm seul par difference finie ---
n, D = 3, 6
x = np.random.randn(n, D)
gamma = np.random.randn(D) * 0.5 + 1.0
beta = np.random.randn(D) * 0.1
y, cache = layernorm_forward(x, gamma, beta)
dy = np.random.randn(n, D)                       # gradient amont arbitraire
dx, dgamma, dbeta = layernorm_backward(dy, cache)

eps = 1e-5


def num_grad_ln(param, idx, which):
    orig = param[idx]
    param[idx] = orig + eps
    yp, _ = layernorm_forward(x, gamma, beta)
    lp = np.sum(yp * dy)
    param[idx] = orig - eps
    ym, _ = layernorm_forward(x, gamma, beta)
    lm = np.sum(ym * dy)
    param[idx] = orig
    return (lp - lm) / (2 * eps)


# dx
gnum_x = np.zeros_like(x)
for i in range(n):
    for j in range(D):
        o = x[i, j]
        x[i, j] = o + eps; yp, _ = layernorm_forward(x, gamma, beta); lp = np.sum(yp * dy)
        x[i, j] = o - eps; ym, _ = layernorm_forward(x, gamma, beta); lm = np.sum(ym * dy)
        x[i, j] = o
        gnum_x[i, j] = (lp - lm) / (2 * eps)
print(f"\n  LayerNorm dx     : max rel err = {np.max(np.abs(dx - gnum_x) / (np.abs(dx) + np.abs(gnum_x) + 1e-8)):.2e}")
gnum_g = np.array([num_grad_ln(gamma, i, 'g') for i in range(D)])
gnum_b = np.array([num_grad_ln(beta, i, 'b') for i in range(D)])
print(f"  LayerNorm dgamma : max rel err = {np.max(np.abs(dgamma - gnum_g) / (np.abs(dgamma) + np.abs(gnum_g) + 1e-8)):.2e}")
print(f"  LayerNorm dbeta  : max rel err = {np.max(np.abs(dbeta - gnum_b) / (np.abs(dbeta) + np.abs(gnum_b) + 1e-8)):.2e}")


# --- Mini-reseau bout-en-bout : z = LN(x) @ W + b ; loss = CE(softmax(z), y) ---
print("\n  Mini-reseau z = LN(x)@W + b, loss = cross-entropy:")
D, V = 6, 5
x = np.random.randn(2, D)
gamma = np.ones(D); beta = np.zeros(D)
W = np.random.randn(D, V) * 0.3
b = np.random.randn(V) * 0.1
targets = np.array([2, 4])


def forward_loss(x, gamma, beta, W, b, residual=False):
    h, cache = layernorm_forward(x, gamma, beta)
    z = h @ W + b
    if residual:
        # residual sur les D premieres composantes (V peut differ de D : on
        # ne fait le residual que si V == D pour rester bien defini).
        z = z + (x if x.shape[-1] == z.shape[-1] else 0.0)
    loss = 0.0
    for i in range(len(targets)):
        p = softmax(z[i])
        loss += -np.log(p[targets[i]] + 1e-12)
    return loss / len(targets), (cache, h, z)


loss, (cache, h, z) = forward_loss(x, gamma, beta, W, b)
# Backward analytique.
dz = np.zeros_like(z)
for i in range(len(targets)):
    dz[i] = softmax_ce_backward(z[i], targets[i]) / len(targets)
dW = h.T @ dz
db = dz.sum(axis=0)
dh = dz @ W.T
dx, dgamma, dbeta = layernorm_backward(dh, cache)


def check(name, analytic, tensor):
    g = np.zeros_like(tensor)
    it = np.nditer(tensor, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index; o = tensor[idx]
        tensor[idx] = o + eps; lp, _ = forward_loss(x, gamma, beta, W, b)
        tensor[idx] = o - eps; lm, _ = forward_loss(x, gamma, beta, W, b)
        tensor[idx] = o
        g[idx] = (lp - lm) / (2 * eps)
        it.iternext()
    rel = np.max(np.abs(analytic - g) / (np.abs(analytic) + np.abs(g) + 1e-8))
    print(f"    d{name}: max rel err = {rel:.2e}  [{'PASS' if rel < 1e-5 else 'FAIL'}]")


check('x', dx, x)
check('W', dW, W)
check('b', db, b)
check('gamma', dgamma, gamma)
check('beta', dbeta, beta)


# --- Residual : dx recoit DEUX contributions (D == V pour bien definir) ---
print("\n  Residual out = x + LN(x)@W (D == V):")
D = V = 5
x = np.random.randn(2, D)
gamma = np.ones(D); beta = np.zeros(D)
W = np.random.randn(D, V) * 0.3
b = np.random.randn(V) * 0.1
targets = np.array([1, 3])
loss, (cache, h, z) = forward_loss(x, gamma, beta, W, b, residual=True)
dz = np.zeros_like(z)
for i in range(len(targets)):
    dz[i] = softmax_ce_backward(z[i], targets[i]) / len(targets)
dW = h.T @ dz
dh = dz @ W.T
dx_layer, _, _ = layernorm_backward(dh, cache)
dx_total = dx_layer + dz                          # +dz vient du chemin residual direct
gnum = np.zeros_like(x)
it = np.nditer(x, flags=['multi_index'])
while not it.finished:
    idx = it.multi_index; o = x[idx]
    x[idx] = o + eps; lp, _ = forward_loss(x, gamma, beta, W, b, residual=True)
    x[idx] = o - eps; lm, _ = forward_loss(x, gamma, beta, W, b, residual=True)
    x[idx] = o
    gnum[idx] = (lp - lm) / (2 * eps)
    it.iternext()
rel = np.max(np.abs(dx_total - gnum) / (np.abs(dx_total) + np.abs(gnum) + 1e-8))
print(f"    dx (2 chemins) : max rel err = {rel:.2e}  [{'PASS' if rel < 1e-5 else 'FAIL'}]")
print("    -> le residual ajoute l'identite au gradient : il ne s'eteint jamais.")


# --- Weight tying : embedding partage avec la tete ---
print("\n  Weight tying (GPT-2): table embedding = tete de sortie (transposee):")
vocab, d_model = 50257, 768
saved = vocab * d_model
print(f"    params economises = vocab*d_model = {vocab}*{d_model} = {saved:,}")
# Quand W est utilise comme embedding ET comme tete, dW = dW_embed + dW_head.
W_tie = np.random.randn(4, 3) * 0.2     # (vocab=4, d_model=3) ; tete = W_tie.T
ids = np.array([0, 2])
emb = W_tie[ids]                         # usage 1 : embedding lookup
hidden = np.random.randn(2, 3)
logits = hidden @ W_tie.T               # usage 2 : projection de sortie (W_tie.T)
# Gradients amont arbitraires sur chaque usage.
d_emb = np.random.randn(*emb.shape)
d_logits = np.random.randn(*logits.shape)
# Gradient via usage embedding (scatter-add sur les lignes utilisees).
dW_embed = np.zeros_like(W_tie)
for r, i in enumerate(ids):
    dW_embed[i] += d_emb[r]
# Gradient via usage tete : logits = hidden @ W_tie.T -> dW_tie = d_logits.T @ hidden.
dW_head = d_logits.T @ hidden
dW_total = dW_embed + dW_head
# Verif numerique : perturber W_tie et mesurer l'effet sur la somme des 2 usages.
gnum = np.zeros_like(W_tie)
it = np.nditer(W_tie, flags=['multi_index'])
while not it.finished:
    idx = it.multi_index; o = W_tie[idx]
    W_tie[idx] = o + eps
    Lp = np.sum(W_tie[ids] * d_emb) + np.sum((hidden @ W_tie.T) * d_logits)
    W_tie[idx] = o - eps
    Lm = np.sum(W_tie[ids] * d_emb) + np.sum((hidden @ W_tie.T) * d_logits)
    W_tie[idx] = o
    gnum[idx] = (Lp - Lm) / (2 * eps)
    it.iternext()
rel = np.max(np.abs(dW_total - gnum) / (np.abs(dW_total) + np.abs(gnum) + 1e-8))
print(f"    dW = dW_embed + dW_head : max rel err = {rel:.2e}  [{'PASS' if rel < 1e-5 else 'FAIL'}]")

print("\n" + "=" * 70)
print("FIN DES SOLUTIONS HARD (Jour 7)")
print("=" * 70)
