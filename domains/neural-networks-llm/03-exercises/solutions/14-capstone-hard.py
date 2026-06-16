"""
Solutions HARD — Jour 14 : Capstone (extensions de mini-LLaMA)
=============================================================
Exercices 7, 8 (hard). Pur NumPy (forward ET backward a la main), runnable sans
framework. Le code de reference 02-code/14-capstone.py est en PyTorch ; ici on
reimplemente les composants en NumPy.

7. RoPE (convention interleaved-pairs, comme 02-code/14-capstone.py) : implem +
   PREUVE numerique de l'invariance par translation + derivation + spectre de
   frequences + extrapolation au-dela de max_seq_len.
8. Bloc mini-LLaMA NumPy entrainable (RMSNorm + attention causale + MLP +
   residuals pre-norm), backward complet, gradient check (<1e-4), entrainement
   char-level, et les 3 ablations (RMSNorm/LayerNorm, pre/post-norm, +/- residual).

Run: python 03-exercises/solutions/14-capstone-hard.py
"""

import sys
import io
import math
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

np.random.seed(0)


# ============================================================================
# EXERCISE 7 : RoPE — implem, invariance par translation, frequences
# ============================================================================

print("=" * 70)
print("EXERCISE 7 : RoPE (invariance par translation + frequences)")
print("=" * 70)


def precompute_rope(head_dim, max_seq_len, base=10000.0):
    """
    cos, sin de shape (seq, head_dim/2). theta_j = base^(-2j/head_dim).
    POURQUOI precompute : les angles ne dependent que de (position, paire),
    pas des valeurs de q/k -> calcule une fois, reutilise. (cf 02-code/14)
    """
    assert head_dim % 2 == 0, "RoPE exige head_dim pair"
    j = np.arange(0, head_dim, 2)                     # 0,2,4,... -> indices de paires
    freqs = 1.0 / (base ** (j / head_dim))           # theta_j
    positions = np.arange(max_seq_len)
    angles = np.outer(positions, freqs)              # (seq, head_dim/2)
    return np.cos(angles), np.sin(angles)


def apply_rope(x, cos, sin):
    """
    Convention INTERLEAVED-PAIRS : on tourne les paires (0,1),(2,3),...
    rot1 = x1 cos - x2 sin ; rot2 = x1 sin + x2 cos. (identique a 02-code/14)
    x: (..., seq, head_dim) ; cos/sin: (seq, head_dim/2).
    """
    x1 = x[..., 0::2]                                 # composantes paires
    x2 = x[..., 1::2]                                 # composantes impaires
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    out = np.empty_like(x)
    out[..., 0::2] = rot1
    out[..., 1::2] = rot2
    return out


def apply_rope_at(vec, position, cos_tab, sin_tab):
    """Applique RoPE a un seul vecteur a une position absolue donnee."""
    return apply_rope(vec[None, :], cos_tab[position:position + 1],
                      sin_tab[position:position + 1])[0]


def test_exercise_7():
    head_dim, max_seq = 8, 128
    cos, sin = precompute_rope(head_dim, max_seq)
    assert cos.shape == (max_seq, head_dim // 2)

    # --- Invariance par translation : <RoPE(q,m), RoPE(k,n)> ne depend que de m-n.
    print("\n  Invariance par translation (meme delta -> meme score) :")
    rng = np.random.RandomState(1)
    ok = True
    for _ in range(5):                                # plusieurs paires (q,k)
        q = rng.randn(head_dim)
        k = rng.randn(head_dim)
        for delta in [0, 2, 5]:                       # plusieurs decalages
            scores = []
            for n in [3, 8, 20, 50]:                  # on fait glisser (m,n)
                m = n + delta
                if m >= max_seq:
                    continue
                qm = apply_rope_at(q, m, cos, sin)
                kn = apply_rope_at(k, n, cos, sin)
                scores.append(float(qm @ kn))
            spread = max(scores) - min(scores)
            ok = ok and spread < 1e-9
            print(f"    delta={delta} : spread du score sur (m,n) glissants = {spread:.2e}")
    assert ok, "le score ne doit dependre que de m-n"
    print("  -> le produit scalaire ne depend QUE de la position relative m-n.")

    # --- Derivation (en commentaire) ---
    # Pour une paire 2D, RoPE applique la rotation R(theta) d'angle position*theta.
    # <R(m theta) q, R(n theta) k> = q^T R(m theta)^T R(n theta) k.
    # Or R est une rotation : R(a)^T = R(-a), et R(-a) R(b) = R(b-a).
    # Donc R(m theta)^T R(n theta) = R((n-m) theta) -> le produit ne depend
    # que de (n-m). CQFD : l'attention ne voit que la position RELATIVE.
    print("\n  Derivation : R(m.theta)^T R(n.theta) = R((n-m).theta)  (rotation 2D)")
    print("    -> <RoPE(q,m),RoPE(k,n)> = q^T R((n-m)theta) k : depend de n-m seul.")

    # --- Spectre de frequences ---
    j = np.arange(0, head_dim, 2)
    thetas = 1.0 / (10000.0 ** (j / head_dim))
    print("\n  Spectre des frequences theta_j (paire -> angle/position) :")
    for idx, th in enumerate(thetas):
        print(f"    paire {idx} : theta = {th:.5f}")
    assert thetas[0] > thetas[-1], "premieres paires = haute frequence (tournent vite)"
    print("  -> 1eres paires : haute frequence (position LOCALE) ; dernieres : basse")
    print("     frequence (position GLOBALE). Meme idee que les embeddings sinusoidaux.")

    # --- Extrapolation au-dela de max_seq_len ---
    short = 16
    cos_s, sin_s = precompute_rope(head_dim, short)   # 'entraine' jusqu'a 16
    cos_l, sin_l = precompute_rope(head_dim, 2 * short)  # teste a 32
    # Les angles restent bien definis (RoPE extrapole mecaniquement) :
    assert np.allclose(cos_l[:short], cos_s)          # la table coincide sur [0,16)
    far = apply_rope_at(rng.randn(head_dim), 2 * short - 1, cos_l, sin_l)
    assert np.all(np.isfinite(far))                   # angles bien definis hors range
    print("\n  Extrapolation : RoPE applique a des positions > max_seq_len reste bien")
    print("  defini (angles finis) MAIS les scores deviennent hors-distribution ->")
    print("  motivation du NTK-aware / YaRN scaling (cours J11).")


test_exercise_7()


# ============================================================================
# EXERCISE 8 : Bloc mini-LLaMA NumPy entrainable + ablations
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 8 : mini-LLaMA NumPy (forward+backward) + ablations")
print("=" * 70)


# --- Briques avec backward --------------------------------------------------

def rmsnorm_forward(x, gamma, eps=1e-5):
    """
    RMSNorm : x / sqrt(mean(x^2)+eps) * gamma. PAS de centrage (vs LayerNorm).
    """
    ms = np.mean(x ** 2, axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(ms + eps)
    x_hat = x * inv
    y = x_hat * gamma
    return y, (x, x_hat, inv, gamma, eps)


def rmsnorm_backward(dy, cache):
    """
    POURQUOI : r = mean(x^2) lie tous les x_i -> dx a un terme direct + une
    correction via d(mean(x^2)). gamma agit elementwise.
    """
    x, x_hat, inv, gamma, eps = cache
    D = x.shape[-1]
    dgamma = np.sum(dy * x_hat, axis=0)
    dxhat = dy * gamma
    # x_hat = x * inv ; inv = (mean(x^2)+eps)^(-1/2)
    dms = np.sum(dxhat * x, axis=-1, keepdims=True) * (-0.5) * inv ** 3
    dx = dxhat * inv + dms * (2.0 * x / D)
    return dx, dgamma


def layernorm_forward(x, gamma, beta, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    x_hat = (x - mu) * inv
    return x_hat * gamma + beta, (x, x_hat, mu, var, inv, gamma, eps)


def linear_forward(x, W, b):
    return x @ W + b, (x, W, b)


def linear_backward(dy, cache):
    x, W, b = cache
    return dy @ W.T, x.T @ dy, dy.sum(axis=0)


def relu_forward(x):
    return np.maximum(0, x), x


def relu_backward(dy, x):
    return dy * (x > 0)


def softmax_rows(S):
    S = S - np.max(S, axis=-1, keepdims=True)
    e = np.exp(S)
    return e / np.sum(e, axis=-1, keepdims=True)


def causal_attention_forward(x, Wq, Wk, Wv, Wo):
    """
    Self-attention causale single-head (sans RoPE pour un backward simple).
    Masque triangulaire : un token ne voit que le passe.
    """
    T, d = x.shape
    Q, K, V = x @ Wq, x @ Wk, x @ Wv
    scale = 1.0 / math.sqrt(d)
    S = (Q @ K.T) * scale
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)  # True = futur a masquer
    S = np.where(mask, -1e30, S)
    A = softmax_rows(S)
    ctx = A @ V
    y = ctx @ Wo
    return y, (x, Wq, Wk, Wv, Wo, Q, K, V, A, ctx, scale, mask)


def causal_attention_backward(dy, cache):
    x, Wq, Wk, Wv, Wo, Q, K, V, A, ctx, scale, mask = cache
    dctx = dy @ Wo.T
    dWo = ctx.T @ dy
    dA = dctx @ V.T
    dV = A.T @ dctx
    rowsum = np.sum(A * dA, axis=-1, keepdims=True)
    dS = A * (dA - rowsum)
    dS = np.where(mask, 0.0, dS)                       # pas de gradient sur le futur
    dQ = (dS @ K) * scale
    dK = (dS.T @ Q) * scale
    dWq = x.T @ dQ
    dWk = x.T @ dK
    dWv = x.T @ dV
    dx = dQ @ Wq.T + dK @ Wk.T + dV @ Wv.T
    return dx, {"Wq": dWq, "Wk": dWk, "Wv": dWv, "Wo": dWo}


# --- Bloc parametre par le type de norm et le placement (pre/post, residual) -

def make_block(d_model, d_ff, seed):
    rng = np.random.RandomState(seed)
    sc = 1.0 / math.sqrt(d_model)
    return {
        "Wq": rng.randn(d_model, d_model) * sc,
        "Wk": rng.randn(d_model, d_model) * sc,
        "Wv": rng.randn(d_model, d_model) * sc,
        "Wo": rng.randn(d_model, d_model) * sc,
        "W1": rng.randn(d_model, d_ff) * sc, "b1": np.zeros(d_ff),
        "W2": rng.randn(d_ff, d_model) * sc, "b2": np.zeros(d_model),
        "g1": np.ones(d_model), "be1": np.zeros(d_model),     # norm 1
        "g2": np.ones(d_model), "be2": np.zeros(d_model),     # norm 2
    }


def block_forward(x, p, norm="rms", placement="pre", residual=True):
    """
    norm      : 'rms' ou 'ln'
    placement : 'pre' (x + f(norm(x))) ou 'post' (norm(x + f(x)))
    residual  : True/False (ablation sans residual)
    """
    def norm_fwd(z, g, be):
        if norm == "rms":
            y, c = rmsnorm_forward(z, g)
        else:
            y, c = layernorm_forward(z, g, be)
        return y, c

    if placement == "pre":
        a, n1 = norm_fwd(x, p["g1"], p["be1"])
        attn, ac = causal_attention_forward(a, p["Wq"], p["Wk"], p["Wv"], p["Wo"])
        x1 = (x + attn) if residual else attn
        b, n2 = norm_fwd(x1, p["g2"], p["be2"])
        h, l1 = linear_forward(b, p["W1"], p["b1"])
        r, rc = relu_forward(h)
        ffn, l2 = linear_forward(r, p["W2"], p["b2"])
        y = (x1 + ffn) if residual else ffn
    else:  # post-norm
        attn, ac = causal_attention_forward(x, p["Wq"], p["Wk"], p["Wv"], p["Wo"])
        x1, n1 = norm_fwd((x + attn) if residual else attn, p["g1"], p["be1"])
        h, l1 = linear_forward(x1, p["W1"], p["b1"])
        r, rc = relu_forward(h)
        ffn, l2 = linear_forward(r, p["W2"], p["b2"])
        y, n2 = norm_fwd((x1 + ffn) if residual else ffn, p["g2"], p["be2"])

    cache = (norm, placement, residual, n1, ac, n2, l1, rc, l2)
    return y, cache


def _norm_bwd(norm, dy, c):
    if norm == "rms":
        dx, dg = rmsnorm_backward(dy, c)
        return dx, dg, np.zeros_like(dg)     # pas de beta en RMSNorm
    else:
        # LayerNorm backward (3 termes) inline
        x, x_hat, mu, var, inv, gamma, eps = c
        D = x.shape[-1]
        dgamma = np.sum(dy * x_hat, axis=0)
        dbeta = np.sum(dy, axis=0)
        dxhat = dy * gamma
        dvar = np.sum(dxhat * (x - mu) * -0.5 * inv ** 3, axis=-1, keepdims=True)
        dmu = (np.sum(dxhat * -inv, axis=-1, keepdims=True)
               + dvar * np.mean(-2.0 * (x - mu), axis=-1, keepdims=True))
        dx = dxhat * inv + dvar * 2.0 * (x - mu) / D + dmu / D
        return dx, dgamma, dbeta


def block_backward(dy, cache):
    norm, placement, residual, n1, ac, n2, l1, rc, l2 = cache

    if placement == "pre":
        # residual 2 : y = x1 + ffn (ou ffn seul)
        dx1 = dy.copy() if residual else np.zeros_like(dy)
        dffn = dy.copy()
        dr, dW2, db2 = linear_backward(dffn, l2)
        dh = relu_backward(dr, rc)
        db, dW1, db1 = linear_backward(dh, l1)
        dx1_n2, dg2, dbe2 = _norm_bwd(norm, db, n2)
        dx1 = dx1 + dx1_n2
        # residual 1 : x1 = x + attn (ou attn seul)
        dx = dx1.copy() if residual else np.zeros_like(dx1)
        dattn = dx1.copy()
        da, ag = causal_attention_backward(dattn, ac)
        dx_n1, dg1, dbe1 = _norm_bwd(norm, da, n1)
        dx = dx + dx_n1
    else:  # post-norm
        d_sum2, dg2, dbe2 = _norm_bwd(norm, dy, n2)
        dx1 = d_sum2.copy() if residual else np.zeros_like(d_sum2)
        dffn = d_sum2.copy()
        dr, dW2, db2 = linear_backward(dffn, l2)
        dh = relu_backward(dr, rc)
        dx1_f, dW1, db1 = linear_backward(dh, l1)
        dx1 = dx1 + dx1_f
        d_sum1, dg1, dbe1 = _norm_bwd(norm, dx1, n1)
        dx = d_sum1.copy() if residual else np.zeros_like(d_sum1)
        dattn = d_sum1.copy()
        da, ag = causal_attention_backward(dattn, ac)
        dx = dx + da

    grads = {**ag, "W1": dW1, "b1": db1, "W2": dW2, "b2": db2,
             "g1": dg1, "be1": dbe1, "g2": dg2, "be2": dbe2}
    return dx, grads


def test_gradient_check():
    """Difference finie sur un petit bloc (RMSNorm, pre-norm, residual)."""
    print("\n  Gradient check de bout en bout (loss=0.5||y||^2) :")
    d_model, d_ff, T = 6, 12, 4
    x = np.random.randn(T, d_model)
    p = make_block(d_model, d_ff, seed=2)

    y, cache = block_forward(x, p, norm="rms", placement="pre", residual=True)
    dy = y.copy()
    dx, grads = block_backward(dy, cache)

    def loss_of(params, xin):
        yy, _ = block_forward(xin, params, "rms", "pre", True)
        return 0.5 * np.sum(yy ** 2)

    eps = 1e-5
    all_ok = True
    # x
    gnum = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index; o = x[idx]
        x[idx] = o + eps; lp = loss_of(p, x)
        x[idx] = o - eps; lm = loss_of(p, x)
        x[idx] = o
        gnum[idx] = (lp - lm) / (2 * eps); it.iternext()
    rel = np.max(np.abs(dx - gnum) / (np.abs(dx) + np.abs(gnum) + 1e-8))
    all_ok = all_ok and rel < 1e-4
    print(f"    d{'x':>4}: max rel err = {rel:.2e}  [{'PASS' if rel < 1e-4 else 'FAIL'}]")
    for name in ["Wq", "Wk", "Wv", "Wo", "W1", "b1", "W2", "b2", "g1", "g2"]:
        M = p[name]; dM = grads[name]
        gnum = np.zeros_like(M)
        it = np.nditer(M, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index; o = M[idx]
            M[idx] = o + eps; lp = loss_of(p, x)
            M[idx] = o - eps; lm = loss_of(p, x)
            M[idx] = o
            gnum[idx] = (lp - lm) / (2 * eps); it.iternext()
        rel = np.max(np.abs(dM - gnum) / (np.abs(dM) + np.abs(gnum) + 1e-8))
        all_ok = all_ok and rel < 1e-4
        print(f"    d{name:>4}: max rel err = {rel:.2e}  [{'PASS' if rel < 1e-4 else 'FAIL'}]")
    assert all_ok, "le gradient check doit passer pour tous les parametres"
    print("  -> backward du bloc mini-LLaMA correct (RMSNorm + attn causale + MLP).")


test_gradient_check()


# --- Modele complet char-level : embedding + 1 bloc + final norm + lm_head ---

def build_corpus():
    """Petit corpus char-level repetitif (apprenable rapidement)."""
    text = "hello world. " * 20
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = np.array([stoi[c] for c in text])
    return text, chars, stoi, itos, data


def train_mini_llama(norm="rms", placement="pre", residual=True,
                     steps=300, seed=0, verbose=False):
    """
    Modele : token embedding (tied lm_head) + 1 bloc + final norm.
    Forward + backward + Adam NumPy. Renvoie la liste des loss.
    POURQUOI tied head : lm_head = embedding^T -> moins de params, comme LLaMA.
    """
    text, chars, stoi, itos, data = build_corpus()
    V = len(chars)
    d_model, d_ff, T = 16, 32, 12
    rng = np.random.RandomState(seed)

    emb = rng.randn(V, d_model) * 0.1                  # token embedding (tied lm_head)
    block = make_block(d_model, d_ff, seed=seed + 1)
    gf = np.ones(d_model)                              # final norm gamma

    # Adam state
    params = {"emb": emb, "gf": gf, **{f"b_{k}": v for k, v in block.items()}}
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v_ = {k: np.zeros_like(v) for k, v in params.items()}
    lr, b1, b2, eps = 0.01, 0.9, 0.999, 1e-8

    losses = []
    for step in range(1, steps + 1):
        i = rng.randint(0, len(data) - T - 1)
        idx = data[i:i + T]                            # entree
        tgt = data[i + 1:i + 1 + T]                    # cible = next char

        # forward
        x = params["emb"][idx]                         # (T, d)
        block_p = {k[2:]: params[k] for k in params if k.startswith("b_")}
        h, cache = block_forward(x, block_p, norm, placement, residual)
        if norm == "rms":
            hn, ncf = rmsnorm_forward(h, params["gf"])
        else:
            hn, ncf = layernorm_forward(h, params["gf"], np.zeros_like(params["gf"]))
        logits = hn @ params["emb"].T                  # tied lm_head -> (T, V)
        probs = softmax_rows(logits)
        loss = -np.mean(np.log(probs[np.arange(T), tgt] + 1e-9))
        losses.append(loss)

        # backward
        dlogits = probs.copy()
        dlogits[np.arange(T), tgt] -= 1.0
        dlogits /= T
        dhn = dlogits @ params["emb"]                  # via lm_head
        demb = dlogits.T @ hn                          # grad emb from lm_head side
        if norm == "rms":
            dh, dgf = rmsnorm_backward(dhn, ncf)
        else:
            dh, dgf, _ = _norm_bwd("ln", dhn, ncf)
        dx, gblock = block_backward(dh, cache)
        # grad emb from the input lookup side (scatter-add)
        demb_in = np.zeros_like(params["emb"])
        np.add.at(demb_in, idx, dx)
        demb = demb + demb_in

        grads = {"emb": demb, "gf": dgf,
                 **{f"b_{k}": gblock[k] for k in gblock}}

        # Adam update
        for k in params:
            g = grads[k]
            m[k] = b1 * m[k] + (1 - b1) * g
            v_[k] = b2 * v_[k] + (1 - b2) * g * g
            mhat = m[k] / (1 - b1 ** step)
            vhat = v_[k] / (1 - b2 ** step)
            params[k] -= lr * mhat / (np.sqrt(vhat) + eps)

        if verbose and step % 100 == 0:
            print(f"    step {step:3d} : loss={loss:.4f}  ppl={math.exp(loss):.2f}")

    return losses


def deep_stack_input_grad(residual, N, d_model=16, d_ff=32, T=8, seed=5):
    """
    Empile N blocs identiques, injecte dy=ones a la sortie et propage jusqu'a
    l'entree. Renvoie ||grad entree||. Avec residual le chemin identite preserve
    le gradient ; sans residual il s'attenue (vanishing) en profondeur.
    """
    rng = np.random.RandomState(seed)
    x = rng.randn(T, d_model)
    blocks = [make_block(d_model, d_ff, seed=seed + i) for i in range(N)]
    h, caches = x, []
    for p in blocks:
        h, c = block_forward(h, p, norm="rms", placement="pre", residual=residual)
        caches.append(c)
    dy = np.ones_like(h)
    for c in reversed(caches):
        dy, _ = block_backward(dy, c)
    return float(np.linalg.norm(dy))


def test_training_and_ablations():
    print("\n  Entrainement char-level (RMSNorm, pre-norm, residual) :")
    losses = train_mini_llama(verbose=True)
    print(f"    loss initiale {losses[0]:.4f} -> finale {losses[-1]:.4f}  "
          f"(ppl {math.exp(losses[0]):.1f} -> {math.exp(losses[-1]):.1f})")
    assert losses[-1] < losses[0] * 0.6, "la loss/perplexite doit descendre"
    print("  -> la loss ET la perplexite descendent : modele entrainable.")

    # --- Ablations ---
    print("\n  Ablations (3) :")
    final = {}

    # 1) RMSNorm vs LayerNorm
    l_rms = train_mini_llama(norm="rms", steps=300)[-1]
    l_ln = train_mini_llama(norm="ln", steps=300)[-1]
    final["rms"], final["ln"] = l_rms, l_ln
    print(f"    [norm]      RMSNorm loss={l_rms:.4f}  |  LayerNorm loss={l_ln:.4f}")
    assert l_rms < 1.0 and l_ln < 1.0                  # les deux convergent

    # 2) pre-norm vs post-norm (sur un bloc profond simule via plus de steps)
    l_pre = train_mini_llama(placement="pre", steps=300)[-1]
    l_post = train_mini_llama(placement="post", steps=300)[-1]
    final["pre"], final["post"] = l_pre, l_post
    print(f"    [placement] pre-norm loss={l_pre:.4f}  |  post-norm loss={l_post:.4f}")

    # 3) avec vs sans residual : sur 1 bloc la loss converge dans les deux cas
    #    (trop peu profond). Le VRAI effet des residuals est sur le FLUX DU
    #    GRADIENT en PROFONDEUR -> on empile N blocs et on mesure ||grad entree||.
    #    POURQUOI : sans residual, le gradient doit traverser N couches de
    #    matmuls/normalisations et s'attenue (vanishing) ; le residual ajoute un
    #    chemin identite qui le preserve.
    g_res = deep_stack_input_grad(residual=True, N=12)
    g_nores = deep_stack_input_grad(residual=False, N=12)
    final["residual"], final["no_residual"] = g_res, g_nores
    print(f"    [residual]  ||grad entree|| (12 blocs) avec={g_res:.3e}  |  "
          f"sans={g_nores:.3e}")
    assert g_res > g_nores, "sans residual le gradient s'attenue en profondeur"

    # --- Tableau de synthese ---
    print("\n  Tableau de synthese (metrique finale + lien LLaMA) :")
    print(f"    {'ablation':<28} {'metrique':>10}  observation / decision LLaMA")
    print("    " + "-" * 70)
    print(f"    {'RMSNorm (loss)':<28} {final['rms']:>10.4f}  ~equivalent a LN mais +")
    print(f"    {'LayerNorm (loss)':<28} {final['ln']:>10.4f}  simple/rapide -> RMSNorm")
    print(f"    {'pre-norm (loss)':<28} {final['pre']:>10.4f}  stable en profondeur ->")
    print(f"    {'post-norm (loss)':<28} {final['post']:>10.4f}  LLaMA choisit pre-norm")
    print(f"    {'avec residual (||grad||)':<28} {final['residual']:>10.3e}  preserve le")
    print(f"    {'sans residual (||grad||)':<28} {final['no_residual']:>10.3e}  gradient -> indispensable")


test_training_and_ablations()


print("\n" + "=" * 70)
print("FIN DES SOLUTIONS HARD (Jour 14) — RoPE + mini-LLaMA + ablations OK")
print("=" * 70)
