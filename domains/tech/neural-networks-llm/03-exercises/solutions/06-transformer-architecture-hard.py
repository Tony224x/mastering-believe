"""
Solutions HARD — Jour 6 : Transformer Architecture
==================================================
Exercices 7, 8 (hard). Pur NumPy (forward ET backward a la main).

7. Bloc Transformer pre-norm entrainable : attention + FFN + LayerNorm + 2
   residuals, forward + backward complets, gradient check de bout en bout (<1e-4),
   et un mini-entrainement qui fait descendre la loss.
8. Pre-norm vs post-norm : etude de stabilite en profondeur (norme des activations
   et des gradients vs N), tableau recap, analyse du chemin residuel.

Run: python 03-exercises/solutions/06-transformer-architecture-hard.py
"""

import sys
import io
import math
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

np.random.seed(0)


# ============================================================================
# BRIQUES : forward + backward (chacune cache ce dont son backward a besoin)
# ============================================================================

def linear_forward(x, W, b=None):
    """y = x @ W (+ b). x:(N,Din) W:(Din,Dout). Cache pour le backward."""
    y = x @ W
    if b is not None:
        y = y + b
    return y, (x, W, b)


def linear_backward(dy, cache):
    """
    POURQUOI : y = xW -> dx = dy W^T, dW = x^T dy, db = somme sur les lignes.
    (regle produit matriciel, transposees pour aligner les shapes).
    """
    x, W, b = cache
    dx = dy @ W.T
    dW = x.T @ dy
    db = dy.sum(axis=0) if b is not None else None
    return dx, dW, db


def gelu_forward(x):
    """GeLU approx tanh (comme GPT-2 / 02-code/06)."""
    c = math.sqrt(2.0 / math.pi)
    inner = c * (x + 0.044715 * x ** 3)
    y = 0.5 * x * (1.0 + np.tanh(inner))
    return y, (x, inner, c)


def gelu_backward(dy, cache):
    """
    Derivee de 0.5 x (1+tanh(g(x))) avec g(x)=c(x+0.044715 x^3).
    POURQUOI le terme en sech^2 : d/dx tanh(g) = (1-tanh^2(g)) * g'(x).
    """
    x, inner, c = cache
    t = np.tanh(inner)
    dgi = c * (1.0 + 3 * 0.044715 * x ** 2)          # g'(x)
    dydx = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * dgi
    return dy * dydx


def layernorm_forward(x, gamma, beta, eps=1e-5):
    """LayerNorm sur la derniere dim. Cache tout pour le backward a 3 termes."""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)              # population (ddof=0)
    std_inv = 1.0 / np.sqrt(var + eps)
    x_hat = (x - mu) * std_inv
    y = gamma * x_hat + beta
    return y, (x, x_hat, mu, var, std_inv, gamma, eps)


def layernorm_backward(dy, cache):
    """
    POURQUOI 3 termes : mu et var dependent de TOUS les x_i de la ligne, donc dx
    recoit un terme direct + une correction via dmu + une via dvar.
    """
    x, x_hat, mu, var, std_inv, gamma, eps = cache
    D = x.shape[-1]
    dgamma = np.sum(dy * x_hat, axis=0)
    dbeta = np.sum(dy, axis=0)
    dx_hat = dy * gamma
    dvar = np.sum(dx_hat * (x - mu) * -0.5 * std_inv ** 3, axis=-1, keepdims=True)
    dmu = (np.sum(dx_hat * -std_inv, axis=-1, keepdims=True)
           + dvar * np.mean(-2.0 * (x - mu), axis=-1, keepdims=True))
    dx = dx_hat * std_inv + dvar * 2.0 * (x - mu) / D + dmu / D
    return dx, dgamma, dbeta


def softmax_rows(S):
    S = S - np.max(S, axis=-1, keepdims=True)        # stabilite numerique
    e = np.exp(S)
    return e / np.sum(e, axis=-1, keepdims=True)


def attention_forward(x, W_Q, W_K, W_V, W_O):
    """
    Single-head self-attention (suffit pour la pedagogie du backward).
    Cache Q,K,V,A et les poids pour reconstruire chaque gradient.
    """
    d_k = W_Q.shape[1]
    Q, K, V = x @ W_Q, x @ W_K, x @ W_V
    scale = 1.0 / np.sqrt(d_k)
    S = (Q @ K.T) * scale
    A = softmax_rows(S)                               # poids d'attention (seq,seq)
    ctx = A @ V                                       # contexte
    y = ctx @ W_O
    cache = (x, W_Q, W_K, W_V, W_O, Q, K, V, A, ctx, scale)
    return y, cache


def attention_backward(dy, cache):
    """
    Backward complet de l'attention single-head.
    POURQUOI le jacobien softmax par ligne : dS_i = A_i * (dCtx·V^T_i - somme),
    implemente via la forme compacte dS = A * (G - somme_ligne(A*G)).
    """
    x, W_Q, W_K, W_V, W_O, Q, K, V, A, ctx, scale = cache

    # ctx @ W_O
    dctx = dy @ W_O.T
    dW_O = ctx.T @ dy

    # ctx = A @ V
    dA = dctx @ V.T
    dV = A.T @ dctx

    # A = softmax(S) par ligne : dS = A * (dA - rowsum(A*dA))
    rowsum = np.sum(A * dA, axis=-1, keepdims=True)
    dS = A * (dA - rowsum)

    # S = (Q K^T) * scale
    dQ = (dS @ K) * scale
    dK = (dS.T @ Q) * scale

    # Q,K,V = x @ W_*
    dW_Q = x.T @ dQ
    dW_K = x.T @ dK
    dW_V = x.T @ dV
    dx = dQ @ W_Q.T + dK @ W_K.T + dV @ W_V.T         # x alimente Q, K ET V

    grads = {"W_Q": dW_Q, "W_K": dW_K, "W_V": dW_V, "W_O": dW_O}
    return dx, grads


# ============================================================================
# EXERCISE 7 : Bloc Transformer pre-norm (forward + backward + grad check)
# ============================================================================

print("=" * 70)
print("EXERCISE 7 : Bloc Transformer entrainable (forward + backward)")
print("=" * 70)


def make_block_params(d_model, d_ff, seed=0):
    rng = np.random.RandomState(seed)
    sc = 1.0 / np.sqrt(d_model)
    return {
        "W_Q": rng.randn(d_model, d_model) * sc,
        "W_K": rng.randn(d_model, d_model) * sc,
        "W_V": rng.randn(d_model, d_model) * sc,
        "W_O": rng.randn(d_model, d_model) * sc,
        "W1": rng.randn(d_model, d_ff) * sc, "b1": np.zeros(d_ff),
        "W2": rng.randn(d_ff, d_model) * sc, "b2": np.zeros(d_model),
        "g1": np.ones(d_model), "be1": np.zeros(d_model),
        "g2": np.ones(d_model), "be2": np.zeros(d_model),
    }


def block_forward(x, p):
    """
    Pre-norm :
      a = LN1(x); attn = Attn(a); x1 = x + attn          (residual 1)
      b = LN2(x1); ffn = FFN(b);  y = x1 + ffn           (residual 2)
    On cache TOUTES les valeurs intermediaires pour le backward.
    """
    a, ln1_cache = layernorm_forward(x, p["g1"], p["be1"])
    attn, attn_cache = attention_forward(a, p["W_Q"], p["W_K"], p["W_V"], p["W_O"])
    x1 = x + attn                                        # residual 1

    b, ln2_cache = layernorm_forward(x1, p["g2"], p["be2"])
    h, lin1_cache = linear_forward(b, p["W1"], p["b1"])
    g, gelu_cache = gelu_forward(h)
    ffn, lin2_cache = linear_forward(g, p["W2"], p["b2"])
    y = x1 + ffn                                         # residual 2

    cache = (ln1_cache, attn_cache, ln2_cache, lin1_cache, gelu_cache, lin2_cache)
    return y, cache


def block_backward(dy, cache):
    """
    Backward du bloc. POINT CLE des residuals : x1 = x + attn -> dx ET dattn
    recoivent TOUS LES DEUX dx1 (le gradient se DUPLIQUE sur les deux chemins).
    """
    ln1_cache, attn_cache, ln2_cache, lin1_cache, gelu_cache, lin2_cache = cache

    # --- residual 2 : y = x1 + ffn ---
    dx1 = dy.copy()                                      # chemin direct du residual
    dffn = dy.copy()                                     # chemin sous-module FFN

    # FFN backward : ffn = lin2(gelu(lin1(b)))
    dg, dW2, db2 = linear_backward(dffn, lin2_cache)
    dh = gelu_backward(dg, gelu_cache)
    db, dW1, db1 = linear_backward(dh, lin1_cache)
    # b = LN2(x1) -> le gradient de LN2 retourne sur x1
    dx1_from_ln2, dg2, dbe2 = layernorm_backward(db, ln2_cache)
    dx1 = dx1 + dx1_from_ln2                             # somme des deux contributions a x1

    # --- residual 1 : x1 = x + attn ---
    dx = dx1.copy()                                      # chemin direct du residual 1
    dattn = dx1.copy()                                   # chemin sous-module attention

    da, attn_grads = attention_backward(dattn, attn_cache)
    # a = LN1(x) -> gradient de LN1 retourne sur x
    dx_from_ln1, dg1, dbe1 = layernorm_backward(da, ln1_cache)
    dx = dx + dx_from_ln1                                # somme des deux contributions a x

    grads = {
        "W_Q": attn_grads["W_Q"], "W_K": attn_grads["W_K"],
        "W_V": attn_grads["W_V"], "W_O": attn_grads["W_O"],
        "W1": dW1, "b1": db1, "W2": dW2, "b2": db2,
        "g1": dg1, "be1": dbe1, "g2": dg2, "be2": dbe2,
    }
    return dx, grads


def test_exercise_7():
    d_model, d_ff, seq = 12, 32, 5
    x = np.random.randn(seq, d_model)
    p = make_block_params(d_model, d_ff, seed=1)

    y, cache = block_forward(x, p)
    assert y.shape == x.shape, "le bloc preserve la shape -> empilable"

    # loss = 0.5*||y||^2 -> dy = y
    dy = y.copy()
    dx, grads = block_backward(dy, cache)

    eps = 1e-5
    print("\n  Gradient check de bout en bout (loss=0.5||y||^2) :")

    def loss_of(params, xin):
        yy, _ = block_forward(xin, params)
        return 0.5 * np.sum(yy ** 2)

    all_ok = True

    # 1) gradient par rapport a l'entree x
    gnum_x = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps; lp = loss_of(p, x)
        x[idx] = orig - eps; lm = loss_of(p, x)
        x[idx] = orig
        gnum_x[idx] = (lp - lm) / (2 * eps)
        it.iternext()
    rel = np.max(np.abs(dx - gnum_x) / (np.abs(dx) + np.abs(gnum_x) + 1e-8))
    ok = rel < 1e-4; all_ok = all_ok and ok
    print(f"    d{'x':>4}: max rel err = {rel:.2e}  [{'PASS' if ok else 'FAIL'}]")

    # 2) gradient par rapport a chaque parametre
    for name in ["W_Q", "W_K", "W_V", "W_O", "W1", "b1", "W2", "b2",
                 "g1", "be1", "g2", "be2"]:
        M = p[name]
        dM = grads[name]
        gnum = np.zeros_like(M)
        it = np.nditer(M, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = M[idx]
            M[idx] = orig + eps; lp = loss_of(p, x)
            M[idx] = orig - eps; lm = loss_of(p, x)
            M[idx] = orig
            gnum[idx] = (lp - lm) / (2 * eps)
            it.iternext()
        rel = np.max(np.abs(dM - gnum) / (np.abs(dM) + np.abs(gnum) + 1e-8))
        ok = rel < 1e-4; all_ok = all_ok and ok
        print(f"    d{name:>4}: max rel err = {rel:.2e}  [{'PASS' if ok else 'FAIL'}]")

    assert all_ok, "le gradient check doit passer pour TOUS les parametres"
    print("  -> tous les gradients sont corrects (residuals dupliques OK).")

    # --- Mini-entrainement : prouver que les gradients sont exploitables ---
    # Tache jouet : la tete lineaire doit predire la moyenne des tokens d'entree.
    print("\n  Mini-entrainement (le bloc + tete lineaire apprennent une cible) :")
    rng = np.random.RandomState(7)
    d_model, d_ff, seq = 8, 16, 4
    p = make_block_params(d_model, d_ff, seed=3)
    W_head = rng.randn(d_model, 1) * 0.1                 # tete : (seq,d)->(seq,1)
    x_train = rng.randn(seq, d_model)
    target = x_train.mean(axis=1, keepdims=True)         # cible jouet (seq,1)

    lr = 0.05
    first_loss = None
    for step in range(200):
        y, cache = block_forward(x_train, p)
        pred = y @ W_head                                # (seq,1)
        diff = pred - target
        loss = 0.5 * np.sum(diff ** 2)
        if first_loss is None:
            first_loss = loss
        # backward : loss -> pred -> (W_head, y) -> bloc
        dpred = diff
        dW_head = y.T @ dpred
        dy = dpred @ W_head.T
        _, grads = block_backward(dy, cache)
        # SGD
        W_head -= lr * dW_head
        for k in p:
            p[k] -= lr * grads[k]
        if step % 50 == 0:
            print(f"    step {step:3d} : loss = {loss:.5f}")
    print(f"    final    : loss = {loss:.5f}")
    assert loss < first_loss * 0.5, "la loss doit nettement descendre"
    print("  -> la loss descend : les gradients sont corrects ET utilisables.")


test_exercise_7()


# ============================================================================
# EXERCISE 8 : Pre-norm vs Post-norm — stabilite en profondeur
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 8 : Pre-norm vs Post-norm (stabilite en profondeur)")
print("=" * 70)


def block_forward_postnorm(x, p):
    """
    Post-norm (2017) : x = LN(x + Sublayer(x)).
    Le LayerNorm est APRES le residual -> il renormalise le chemin identite.
    """
    attn, attn_cache = attention_forward(x, p["W_Q"], p["W_K"], p["W_V"], p["W_O"])
    x1, ln1_cache = layernorm_forward(x + attn, p["g1"], p["be1"])     # norm apres residual

    h, lin1_cache = linear_forward(x1, p["W1"], p["b1"])
    g, gelu_cache = gelu_forward(h)
    ffn, lin2_cache = linear_forward(g, p["W2"], p["b2"])
    y, ln2_cache = layernorm_forward(x1 + ffn, p["g2"], p["be2"])      # norm apres residual

    cache = (attn_cache, ln1_cache, lin1_cache, gelu_cache, lin2_cache, ln2_cache)
    return y, cache


def block_backward_postnorm(dy, cache):
    """Backward post-norm : le LN englobe le residual -> on traverse LN d'abord."""
    attn_cache, ln1_cache, lin1_cache, gelu_cache, lin2_cache, ln2_cache = cache

    # y = LN2(x1 + ffn)
    d_sum2, dg2, dbe2 = layernorm_backward(dy, ln2_cache)
    dx1 = d_sum2.copy()                                  # chemin direct (residual)
    dffn = d_sum2.copy()                                 # chemin FFN
    dg, dW2, db2 = linear_backward(dffn, lin2_cache)
    dh = gelu_backward(dg, gelu_cache)
    dx1_ffn, dW1, db1 = linear_backward(dh, lin1_cache)
    dx1 = dx1 + dx1_ffn

    # x1 = LN1(x + attn)
    d_sum1, dg1, dbe1 = layernorm_backward(dx1, ln1_cache)
    dx = d_sum1.copy()                                   # chemin direct
    dattn = d_sum1.copy()                                # chemin attention
    da, attn_grads = attention_backward(dattn, attn_cache)
    dx = dx + da

    grads = {**attn_grads, "W1": dW1, "b1": db1, "W2": dW2, "b2": db2,
             "g1": dg1, "be1": dbe1, "g2": dg2, "be2": dbe2}
    return dx, grads


def stack_stats(variant, N, d_model=16, d_ff=64, seq=8):
    """
    Empile N blocs, mesure la norme moyenne de l'activation de sortie ET la
    norme du gradient a l'entree (dy=ones propage jusqu'a x_0).
    """
    rng = np.random.RandomState(123)
    x0 = rng.randn(seq, d_model)
    params = [make_block_params(d_model, d_ff, seed=s) for s in range(N)]

    fwd = block_forward if variant == "pre" else block_forward_postnorm
    bwd = block_backward if variant == "pre" else block_backward_postnorm

    h = x0
    caches = []
    for p in params:
        h, c = fwd(h, p)
        caches.append(c)
    act_norm = np.linalg.norm(h) / math.sqrt(h.size)    # RMS de l'activation

    # backward : dy = ones a la sortie, propage jusqu'a l'entree
    dy = np.ones_like(h)
    for c in reversed(caches):
        dy, _ = bwd(dy, c)
    grad_in_norm = np.linalg.norm(dy)
    return act_norm, grad_in_norm


def test_exercise_8():
    Ns = [2, 8, 32, 64]
    print("\n  Tableau recap (RMS activation sortie / ||grad entree||) :")
    print(f"    {'N':>3} | {'act post':>10} {'act pre':>10} | "
          f"{'grad post':>11} {'grad pre':>11}")
    print("    " + "-" * 56)
    rows = []
    for N in Ns:
        a_post, g_post = stack_stats("post", N)
        a_pre, g_pre = stack_stats("pre", N)
        rows.append((N, a_post, a_pre, g_post, g_pre))
        print(f"    {N:>3} | {a_post:>10.3f} {a_pre:>10.3f} | "
              f"{g_post:>11.2e} {g_pre:>11.2e}")

    # Verifs : en profondeur le pre-norm PRESERVE mieux le gradient d'entree.
    # On compare le plus profond (N=64) : ratio grad_pre / grad_post >> 1.
    _, _, _, g_post_64, g_pre_64 = rows[-1]
    assert g_pre_64 > g_post_64, "pre-norm preserve mieux le gradient en profondeur"
    print("\n  Observation : le gradient d'entree du POST-norm s'attenue bien plus")
    print("  vite avec la profondeur que celui du PRE-norm.")

    print("\n  4. Analyse du chemin residuel :")
    print("     PRE-norm : x_{l+1} = x_l + Sublayer(LN(x_l)). En deroulant,")
    print("       sortie = x_0 + somme(corrections) -> d(sortie)/d(x_0) contient un")
    print("       terme IDENTITE -> le gradient remonte ~intact jusqu'a x_0.")
    print("     POST-norm : x_{l+1} = LN(x_l + Sublayer(x_l)). Chaque residual est")
    print("       RE-NORMALISE -> le LayerNorm 'casse' le chemin identite et attenue")
    print("       le signal couche apres couche -> gradient qui decroit en profondeur.")

    print("\n  5. Variantes tres profondes (DeepNorm / ReZero / sandwich-norm) :")
    print("     Le pre-norm souffre d'une CROISSANCE de la variance du chemin")
    print("     residuel (chaque sous-couche ajoute sans renormaliser le tronc).")
    print("     DeepNorm/sandwich-norm rescalent/renorment ce chemin pour stabiliser")
    print("     les stacks de 100+ couches.")


test_exercise_8()


print("\n" + "=" * 70)
print("FIN DES SOLUTIONS HARD (Jour 6) — gradient checks + ablations OK")
print("=" * 70)
