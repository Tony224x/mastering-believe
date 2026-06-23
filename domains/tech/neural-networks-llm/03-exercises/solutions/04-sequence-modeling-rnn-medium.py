"""
Solutions MEDIUM — Jour 4 : Sequence Modeling (RNN, LSTM, GRU)
==============================================================
Exercices 4, 5, 6 (medium).

Pur NumPy (comme 02-code/04-sequence-modeling-rnn.py). Aucun framework.
Chaque etape non triviale est commentee avec le POURQUOI.

Run: python 03-exercises/solutions/04-sequence-modeling-rnn-medium.py
"""

import numpy as np

np.random.seed(42)


def sigmoid(z):
    """Sigmoid numeriquement stable (clip pour eviter overflow de exp)."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def softmax(z):
    """Softmax stable sur un vecteur 1D."""
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


# ============================================================================
# EXERCISE 4: RNN cell complete (forward + backward) + gradient check
# ============================================================================

print("=" * 70)
print("EXERCISE 4: RNN cell forward + backward + gradient check")
print("=" * 70)


def rnn_cell_forward(x_t, h_prev, params):
    """
    Un pas de RNN vanilla.
      h_t = tanh(W_xh @ x_t + W_hh @ h_prev + b_h)
      y_t = W_hy @ h_t + b_y

    Shapes:
      x_t (D,), h_prev (H,), W_xh (H,D), W_hh (H,H), W_hy (O,H), b_h (H,), b_y (O,)
    """
    W_xh, W_hh, W_hy, b_h, b_y = (params['W_xh'], params['W_hh'],
                                  params['W_hy'], params['b_h'], params['b_y'])
    # Pre-activation : combinaison de l'input et de l'etat precedent.
    z_h = W_xh @ x_t + W_hh @ h_prev + b_h
    # tanh : borne l'etat dans (-1, 1) -> empeche l'explosion de l'etat dans le temps.
    h_t = np.tanh(z_h)
    # Projection de sortie (lineaire ; softmax applique plus tard a la loss).
    y_t = W_hy @ h_t + b_y
    cache = (x_t, h_prev, h_t)  # h_t suffit : d/dz tanh = 1 - h_t^2
    return h_t, y_t, cache


def rnn_cell_backward(dh_t, dy_t, cache, params):
    """
    Backward d'un pas de RNN.
      dh_t : gradient de la loss venant du futur (h_{t+1})
      dy_t : gradient de la loss venant de la sortie y_t locale
    """
    W_hh, W_hy = params['W_hh'], params['W_hy']
    x_t, h_prev, h_t = cache

    # Gradient sur h_t venant de la sortie y_t : dL/dh_t (via y) = W_hy^T @ dy_t
    dh_from_y = W_hy.T @ dy_t
    # Gradient total sur h_t : du futur + de la sortie locale.
    dh_total = dh_t + dh_from_y

    # Backprop a travers tanh : d/dz tanh(z) = 1 - tanh(z)^2 = 1 - h_t^2.
    # POURQUOI 1 - h_t^2 : on reutilise h_t deja calcule (pas besoin de z_h).
    dz_h = dh_total * (1.0 - h_t ** 2)

    # Gradients des parametres.
    dW_xh = np.outer(dz_h, x_t)     # d/dW_xh de (W_xh @ x_t) = outer(dz, x)
    dW_hh = np.outer(dz_h, h_prev)  # d/dW_hh de (W_hh @ h_prev) = outer(dz, h_prev)
    db_h = dz_h
    dW_hy = np.outer(dy_t, h_t)     # d/dW_hy de (W_hy @ h_t) = outer(dy, h)
    db_y = dy_t

    # Gradient qui remonte vers h_{t-1} : c'est le terme recurrent.
    # POURQUOI ce terme cause vanishing/exploding : applique sur T pas, il
    # multiplie W_hh^T a chaque fois -> (W_hh^T)^T grandit/decroit exponentiellement.
    dh_prev = W_hh.T @ dz_h

    return {'dW_xh': dW_xh, 'dW_hh': dW_hh, 'dW_hy': dW_hy,
            'db_h': db_h, 'db_y': db_y, 'dh_prev': dh_prev}


def make_rnn_params(D, H, O, seed=0):
    rng = np.random.RandomState(seed)
    return {
        'W_xh': rng.randn(H, D) * 0.1,
        'W_hh': rng.randn(H, H) * 0.1,
        'W_hy': rng.randn(O, H) * 0.1,
        'b_h': rng.randn(H) * 0.1,
        'b_y': rng.randn(O) * 0.1,
    }


# --- Gradient check sur une cellule isolee ---
D, H, O = 3, 4, 2
params = make_rnn_params(D, H, O, seed=1)
x_t = np.random.randn(D)
h_prev = np.random.randn(H)
y_target = np.random.randn(O)


def loss_from_params(params):
    """Loss MSE sur la sortie y_t d'une cellule isolee (h_prev fixe)."""
    h_t, y_t, _ = rnn_cell_forward(x_t, h_prev, params)
    return 0.5 * np.sum((y_t - y_target) ** 2)


# Backward analytique : dL/dy = (y - target), pas de gradient du futur.
h_t, y_t, cache = rnn_cell_forward(x_t, h_prev, params)
dy = (y_t - y_target)               # derivee de 0.5*||y-target||^2
dh_future = np.zeros(H)             # cellule isolee : pas de futur
grads = rnn_cell_backward(dh_future, dy, cache, params)

eps = 1e-5
print("\n  Gradient check (analytique vs numerique, eps=1e-5):")
all_ok = True
for name in ['W_xh', 'W_hh', 'W_hy', 'b_h', 'b_y']:
    param = params[name]
    grad_num = np.zeros_like(param)
    it = np.nditer(param, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        orig = param[idx]
        param[idx] = orig + eps
        lp = loss_from_params(params)
        param[idx] = orig - eps
        lm = loss_from_params(params)
        param[idx] = orig
        grad_num[idx] = (lp - lm) / (2 * eps)
        it.iternext()
    grad_ana = grads['d' + name]
    denom = np.abs(grad_ana) + np.abs(grad_num) + 1e-8
    rel_err = np.max(np.abs(grad_ana - grad_num) / denom)
    ok = rel_err < 1e-5
    all_ok = all_ok and ok
    print(f"    {name:>5}: max rel error = {rel_err:.2e}  [{'PASS' if ok else 'FAIL'}]")
print(f"  -> {'ALL GRADIENTS CORRECT' if all_ok else 'SOME GRADIENTS WRONG'}")


# ============================================================================
# EXERCISE 5: BPTT complet + monitoring du gradient
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 5: BPTT complet + monitoring vanishing")
print("=" * 70)


def train_char_rnn(corpus, hidden_dim=32, seq_len=15, lr=0.1, n_iters=400, seed=0):
    """Entrainement char-level avec BPTT, clipping, et monitoring."""
    chars = sorted(list(set(corpus)))
    V = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}

    rng = np.random.RandomState(seed)
    params = {
        'W_xh': rng.randn(hidden_dim, V) * 0.01,
        'W_hh': rng.randn(hidden_dim, hidden_dim) * 0.01,
        'W_hy': rng.randn(V, hidden_dim) * 0.01,
        'b_h': np.zeros(hidden_dim),
        'b_y': np.zeros(V),
    }
    loss_hist, gnorm_hist = [], []

    for it in range(n_iters):
        start = rng.randint(0, len(corpus) - seq_len - 1)
        inputs = [stoi[c] for c in corpus[start:start + seq_len]]
        targets = [stoi[c] for c in corpus[start + 1:start + seq_len + 1]]

        # --- Forward : derouler le RNN ---
        h_prev = np.zeros(hidden_dim)
        caches, probs_list = [], []
        loss = 0.0
        for t in range(seq_len):
            x_t = np.zeros(V)
            x_t[inputs[t]] = 1.0
            h_prev, y_t, cache = rnn_cell_forward(x_t, h_prev, params)
            caches.append(cache)
            p = softmax(y_t)
            probs_list.append(p)
            loss += -np.log(max(p[targets[t]], 1e-12))
        loss /= seq_len

        # --- Backward (BPTT) : accumuler les gradients ---
        # POURQUOI accumuler (somme) : les memes parametres sont reutilises a
        # chaque pas. Par la regle de la somme sur les chemins, le gradient
        # total d'un parametre partage = somme de ses gradients a chaque usage.
        dparams = {k: np.zeros_like(v) for k, v in params.items()}
        dh_next = np.zeros(hidden_dim)
        for t in reversed(range(seq_len)):
            dy = probs_list[t].copy()
            dy[targets[t]] -= 1.0   # gradient softmax+CE : probs - one_hot
            dy /= seq_len           # car la loss a ete moyennee
            g = rnn_cell_backward(dh_next, dy, caches[t], params)
            for k in ['W_xh', 'W_hh', 'W_hy', 'b_h', 'b_y']:
                dparams[k] += g['d' + k]
            dh_next = g['dh_prev']

        # --- Gradient clipping par norme globale ---
        total_norm = np.sqrt(sum(np.sum(g ** 2) for g in dparams.values()))
        gnorm_hist.append(total_norm)
        clip = 5.0
        if total_norm > clip:
            for k in dparams:
                dparams[k] *= clip / (total_norm + 1e-6)

        # --- SGD ---
        for k in params:
            params[k] -= lr * dparams[k]
        loss_hist.append(loss)

    return params, loss_hist, gnorm_hist


CORPUS = "le chat dort sur le tapis. le chien court vite. " * 12

print("\n  seq_len | loss finale | grad_norm moyen (50 derniers)")
print("  " + "-" * 55)
for sl in [5, 15, 30]:
    _, lh, gh = train_char_rnn(CORPUS, hidden_dim=32, seq_len=sl, n_iters=400, seed=0)
    print(f"  {sl:>7d} | {np.mean(lh[-50:]):>11.4f} | {np.mean(gh[-50:]):>10.4f}")


def vanishing_profile(T=40, hidden_dim=16, scale=0.5, seed=0):
    """
    Profil de ||dL/dh_t|| en remontant le temps t = T..0.
    On simule une trajectoire et on propage un gradient ones depuis la fin.
    """
    rng = np.random.RandomState(seed)
    W_hh = rng.randn(hidden_dim, hidden_dim)
    # Normaliser pour fixer le rayon spectral approx a `scale`.
    s = np.linalg.svd(W_hh, compute_uv=False)[0]
    W_hh = W_hh / s * scale

    h = rng.randn(hidden_dim) * 0.1
    hs = []
    for _ in range(T):
        h = np.tanh(W_hh @ h)
        hs.append(h)

    dh = np.ones(hidden_dim)
    norms = [np.linalg.norm(dh)]
    for t in reversed(range(T)):
        dh = (1.0 - hs[t] ** 2) * (W_hh.T @ dh)
        norms.append(np.linalg.norm(dh))
    return norms  # norms[0] = a t=T, norms[-1] = a t=0


print("\n  Profil ||dL/dh_t|| (scale W_hh = 0.5, T=40), echantillons:")
prof = vanishing_profile(T=40, scale=0.5)
for k in [0, 10, 20, 30, 40]:
    print(f"    {40 - k:>3d} pas avant la fin | ||grad|| = {prof[k]:.6e}")
print("  -> decroissance exponentielle vers t=0 : c'est le vanishing gradient.")


# ============================================================================
# EXERCISE 6: LSTM cell from scratch (forward) + verification des gates
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 6: LSTM cell forward + verification des gates")
print("=" * 70)


def lstm_cell_forward(x_t, h_prev, c_prev, params):
    """
    Forward LSTM. On concatene [h_prev, x_t] et on a une matrice par gate.
      f = sigmoid(W_f @ z + b_f), i = sigmoid(W_i @ z + b_i)
      o = sigmoid(W_o @ z + b_o), c~ = tanh(W_c @ z + b_c)
      c_t = f * c_prev + i * c~
      h_t = o * tanh(c_t)
    z = concat([h_prev, x_t]) de taille (H + D,)
    """
    z = np.concatenate([h_prev, x_t])
    f = sigmoid(params['W_f'] @ z + params['b_f'])
    i = sigmoid(params['W_i'] @ z + params['b_i'])
    o = sigmoid(params['W_o'] @ z + params['b_o'])
    c_tilde = np.tanh(params['W_c'] @ z + params['b_c'])
    c_t = f * c_prev + i * c_tilde
    h_t = o * np.tanh(c_t)
    cache = (z, f, i, o, c_tilde, c_prev, c_t)
    return h_t, c_t, cache


def make_lstm_params(D, H, seed=0):
    rng = np.random.RandomState(seed)
    sc = 0.1
    return {
        'W_f': rng.randn(H, H + D) * sc, 'b_f': np.zeros(H),
        'W_i': rng.randn(H, H + D) * sc, 'b_i': np.zeros(H),
        'W_o': rng.randn(H, H + D) * sc, 'b_o': np.zeros(H),
        'W_c': rng.randn(H, H + D) * sc, 'b_c': np.zeros(H),
    }


D, H = 3, 4
p = make_lstm_params(D, H, seed=2)
h, c = np.zeros(H), np.zeros(H)
print("\n  Deroulement sur 4 pas (normes de c_t et h_t):")
for t in range(4):
    x = np.random.randn(D)
    h, c, _ = lstm_cell_forward(x, h, c, p)
    print(f"    pas {t}: ||c_t|| = {np.linalg.norm(c):.4f}, ||h_t|| = {np.linalg.norm(h):.4f}")

# --- Regime "memoire figee" : forget ~ 1, input ~ 0 ---
p_keep = make_lstm_params(D, H, seed=2)
p_keep['b_f'] = np.full(H, 10.0)    # sigmoid(10) ~ 1
p_keep['b_i'] = np.full(H, -10.0)   # sigmoid(-10) ~ 0
c_prev = np.array([2.0, -1.0, 0.5, 3.0])
x = np.random.randn(D)
_, c_new, _ = lstm_cell_forward(x, np.zeros(H), c_prev, p_keep)
print("\n  Memoire figee (b_f=+10, b_i=-10):")
print(f"    c_prev = {np.round(c_prev, 4)}")
print(f"    c_t    = {np.round(c_new, 4)}  (doit ~= c_prev)")
print(f"    max |c_t - c_prev| = {np.max(np.abs(c_new - c_prev)):.4e}")

# --- Regime "reset complet" : forget ~ 0 ---
p_reset = make_lstm_params(D, H, seed=2)
p_reset['b_f'] = np.full(H, -10.0)  # forget ~ 0
_, c_new2, cache2 = lstm_cell_forward(x, np.zeros(H), c_prev, p_reset)
_, f2, i2, o2, c_tilde2, _, _ = cache2
expected = i2 * c_tilde2
print("\n  Reset complet (b_f=-10):")
print(f"    c_t       = {np.round(c_new2, 4)}")
print(f"    i_t * c~_t = {np.round(expected, 4)}  (doit ~= c_t)")
print(f"    max |c_t - i*c~| = {np.max(np.abs(c_new2 - expected)):.4e}")

# --- Gradient du cell state : dc_t/dc_{t-1} = f_t ---
print("\n  Gradient du cell state dc_t/dc_{t-1}:")
print(f"    f_t (forget gate, memoire figee) ~= {np.round(f := sigmoid(p_keep['W_f'] @ np.concatenate([np.zeros(H), x]) + p_keep['b_f']), 4)}")
print("    -> dc_t/dc_{t-1} = f_t ~= 1 : le gradient passe quasi intact.")
print("    Contrairement au RNN ou dh_t/dh_{t-1} = diag(1-h^2) @ W_hh (vanish).")

print("\n" + "=" * 70)
print("FIN DES SOLUTIONS MEDIUM (Jour 4)")
print("=" * 70)
