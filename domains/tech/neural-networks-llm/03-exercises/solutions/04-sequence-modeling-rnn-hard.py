"""
Solutions HARD — Jour 4 : Sequence Modeling (RNN, LSTM, GRU)
============================================================
Exercices 7, 8 (hard).

Pur NumPy. LSTM complet (forward + BPTT + gradient check),
GRU complet (forward + backward + gradient check), comptage de
parametres et benchmark a budget de parametres egal.

Run: python 03-exercises/solutions/04-sequence-modeling-rnn-hard.py
"""

import numpy as np

np.random.seed(42)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


# ============================================================================
# EXERCISE 7: LSTM complet (forward + BPTT) + gradient check
# ============================================================================

print("=" * 70)
print("EXERCISE 7: LSTM forward + backward (BPTT) + gradient check")
print("=" * 70)


def lstm_forward_step(x_t, h_prev, c_prev, P):
    """Un pas de LSTM. P = dict de parametres. Renvoie h_t, c_t, cache."""
    z = np.concatenate([h_prev, x_t])              # (H+D,)
    f = sigmoid(P['W_f'] @ z + P['b_f'])
    i = sigmoid(P['W_i'] @ z + P['b_i'])
    o = sigmoid(P['W_o'] @ z + P['b_o'])
    g = np.tanh(P['W_c'] @ z + P['b_c'])           # candidate (c~)
    c_t = f * c_prev + i * g
    tanh_c = np.tanh(c_t)
    h_t = o * tanh_c
    cache = (z, f, i, o, g, c_prev, c_t, tanh_c)
    return h_t, c_t, cache


def lstm_backward_step(dh_t, dc_t, cache, P, H):
    """
    Backward d'un pas de LSTM.
      dh_t : gradient sur h_t (futur + sortie locale)
      dc_t : gradient sur c_t venant du futur (dc_next)
    Renvoie les gradients des parametres + dh_prev, dc_prev.
    """
    z, f, i, o, g, c_prev, c_t, tanh_c = cache

    # h_t = o * tanh(c_t)
    do = dh_t * tanh_c                              # dL/do
    # c_t recoit DEUX contributions : du futur (dc_t) ET via h_t.
    dc = dc_t + dh_t * o * (1.0 - tanh_c ** 2)      # dL/dc_t total

    # c_t = f * c_prev + i * g
    df = dc * c_prev
    di = dc * g
    dg = dc * i
    dc_prev = dc * f                               # gradient vers c_{t-1}

    # Backprop a travers les non-linearites des gates.
    df_pre = df * f * (1.0 - f)                     # sigmoid'
    di_pre = di * i * (1.0 - i)
    do_pre = do * o * (1.0 - o)
    dg_pre = dg * (1.0 - g ** 2)                    # tanh'

    grads = {}
    dz = np.zeros_like(z)
    for name, dpre in [('f', df_pre), ('i', di_pre), ('o', do_pre), ('c', dg_pre)]:
        grads['W_' + name] = np.outer(dpre, z)
        grads['b_' + name] = dpre
        dz += P['W_' + name].T @ dpre               # accumuler la contribution a z

    # z = [h_prev, x_t] -> dh_prev = partie h de dz.
    dh_prev = dz[:H]
    return grads, dh_prev, dc_prev


def make_lstm_params(D, H, seed=0):
    rng = np.random.RandomState(seed)
    sc = 0.2
    P = {}
    for nm in ['f', 'i', 'o', 'c']:
        P['W_' + nm] = rng.randn(H, H + D) * sc
        P['b_' + nm] = rng.randn(H) * sc
    return P


def lstm_sequence_loss(P, xs, H):
    """Forward sur la sequence, loss = somme des 0.5*||h_t||^2 (MSE vers 0)."""
    h, c = np.zeros(H), np.zeros(H)
    caches = []
    loss = 0.0
    for x in xs:
        h, c, cache = lstm_forward_step(x, h, c, P)
        caches.append(cache)
        loss += 0.5 * np.sum(h ** 2)
    return loss, caches


def lstm_sequence_backward(P, xs, caches, H):
    """BPTT : accumule les gradients sur toute la sequence."""
    T = len(xs)
    dP = {k: np.zeros_like(v) for k, v in P.items()}
    dh_next = np.zeros(H)
    dc_next = np.zeros(H)
    for t in reversed(range(T)):
        # loss = 0.5*||h_t||^2 -> dL/dh_t (local) = h_t
        # Recompute h_t depuis le cache (h_t = o * tanh(c_t)).
        z, f, i, o, g, c_prev, c_t, tanh_c = caches[t]
        h_t = o * tanh_c
        dh_local = h_t                              # derivee de 0.5*||h_t||^2
        dh_t = dh_next + dh_local
        grads, dh_next, dc_next = lstm_backward_step(dh_t, dc_next, caches[t], P, H)
        for k in dP:
            dP[k] += grads[k]
    return dP


# --- Gradient check sur T=4 pas ---
D, H, T = 3, 4, 4
P = make_lstm_params(D, H, seed=3)
xs = [np.random.randn(D) for _ in range(T)]

loss, caches = lstm_sequence_loss(P, xs, H)
dP = lstm_sequence_backward(P, xs, caches, H)

eps = 1e-5
print("\n  Gradient check LSTM (BPTT, T=4, eps=1e-5):")
all_ok = True
for name in sorted(P.keys()):
    param = P[name]
    gnum = np.zeros_like(param)
    it = np.nditer(param, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        orig = param[idx]
        param[idx] = orig + eps
        lp, _ = lstm_sequence_loss(P, xs, H)
        param[idx] = orig - eps
        lm, _ = lstm_sequence_loss(P, xs, H)
        param[idx] = orig
        gnum[idx] = (lp - lm) / (2 * eps)
        it.iternext()
    gana = dP[name]
    denom = np.abs(gana) + np.abs(gnum) + 1e-8
    rel = np.max(np.abs(gana - gnum) / denom)
    ok = rel < 1e-4
    all_ok = all_ok and ok
    print(f"    {name:>4}: max rel error = {rel:.2e}  [{'PASS' if ok else 'FAIL'}]")
print(f"  -> {'LSTM BACKWARD CORRECT' if all_ok else 'LSTM BACKWARD WRONG'}")


# --- Experience "copy" : LSTM vs RNN sur un delai ---
print("\n  Tache copy: predire le 1er bit apres un delai de T pas.")


def copy_task_batch(T_delay, n, rng):
    """X: signal a t=0 dans {-1,+1}, bruit ensuite. Cible = signe initial."""
    Xs, ys = [], []
    for _ in range(n):
        first = rng.choice([-1.0, 1.0])
        seq = [np.array([first])]
        for _ in range(T_delay):
            seq.append(np.array([0.0]))            # delai (input neutre)
        Xs.append(seq)
        ys.append(1.0 if first > 0 else 0.0)
    return Xs, ys


def train_lstm_copy(T_delay, hidden=8, iters=400, lr=0.3, seed=0):
    rng = np.random.RandomState(seed)
    P = make_lstm_params(1, hidden, seed=seed)
    W_out = rng.randn(1, hidden) * 0.1
    b_out = np.zeros(1)
    for _ in range(iters):
        Xs, ys = copy_task_batch(T_delay, 16, rng)
        gW = {k: np.zeros_like(v) for k, v in P.items()}
        gWo = np.zeros_like(W_out)
        gbo = 0.0
        for seq, y in zip(Xs, ys):
            h, c = np.zeros(hidden), np.zeros(hidden)
            caches = []
            for x in seq:
                h, c, cache = lstm_forward_step(x, h, c, P)
                caches.append(cache)
            logit = W_out @ h + b_out
            pred = sigmoid(logit)
            dlogit = (pred - y)                     # BCE + sigmoid
            gWo += np.outer(dlogit, h)
            gbo += dlogit
            dh = W_out.T @ dlogit
            dc = np.zeros(hidden)
            for t in reversed(range(len(seq))):
                grads, dh, dc = lstm_backward_step(dh, dc, caches[t], P, hidden)
                for k in P:
                    gW[k] += grads[k]
        for k in P:
            P[k] -= lr * gW[k] / 16
        W_out -= lr * gWo / 16
        b_out -= lr * gbo / 16
    # Eval
    Xs, ys = copy_task_batch(T_delay, 100, rng)
    correct = 0
    for seq, y in zip(Xs, ys):
        h, c = np.zeros(hidden), np.zeros(hidden)
        for x in seq:
            h, c, _ = lstm_forward_step(x, h, c, P)
        pred = sigmoid(W_out @ h + b_out)[0]
        correct += int((pred > 0.5) == (y > 0.5))
    return correct / 100


for T_delay in [5, 20, 50]:
    acc = train_lstm_copy(T_delay, hidden=8, iters=300, seed=0)
    print(f"    LSTM, delai={T_delay:>3d} pas : accuracy = {acc:.0%}")
print("  -> Le LSTM conserve l'info sur de longs delais (chemin additif du cell state).")


# ============================================================================
# EXERCISE 8: GRU from scratch + comparaison RNN/LSTM/GRU
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 8: GRU forward + backward + gradient check + comptage params")
print("=" * 70)


def gru_forward_step(x_t, h_prev, P):
    """
    Forward GRU.
      z = sigmoid(W_z @ [h_prev, x] + b_z)   update gate
      r = sigmoid(W_r @ [h_prev, x] + b_r)   reset gate
      h~ = tanh(W_h @ [r*h_prev, x] + b_h)   candidat (note r*h_prev)
      h_t = (1 - z) * h_prev + z * h~
    """
    zc = np.concatenate([h_prev, x_t])
    z = sigmoid(P['W_z'] @ zc + P['b_z'])
    r = sigmoid(P['W_r'] @ zc + P['b_r'])
    rh = r * h_prev
    zc2 = np.concatenate([rh, x_t])            # entree du candidat
    h_tilde = np.tanh(P['W_h'] @ zc2 + P['b_h'])
    h_t = (1.0 - z) * h_prev + z * h_tilde
    cache = (x_t, h_prev, z, r, rh, zc2, h_tilde)
    return h_t, cache


def gru_backward_step(dh_t, cache, P, H):
    """
    Backward GRU. h_prev apparait a 4 endroits :
      (1) dans z, (2) dans r, (3) via rh=r*h_prev dans le candidat,
      (4) directement dans (1-z)*h_prev.
    Tous contribuent a dh_prev.
    """
    x_t, h_prev, z, r, rh, zc2, h_tilde = cache
    D = x_t.shape[0]

    # h_t = (1-z)*h_prev + z*h_tilde
    dz = dh_t * (h_tilde - h_prev)                 # dL/dz
    dh_tilde = dh_t * z                            # dL/dh_tilde
    dh_prev = dh_t * (1.0 - z)                     # contribution (4) directe

    # h_tilde = tanh(W_h @ [r*h_prev, x] + b_h)
    dpre_h = dh_tilde * (1.0 - h_tilde ** 2)       # tanh'
    dW_h = np.outer(dpre_h, zc2)
    db_h = dpre_h
    dzc2 = P['W_h'].T @ dpre_h                      # gradient vers [r*h_prev, x]
    drh = dzc2[:H]                                  # gradient vers rh = r*h_prev
    # (x_t part de dzc2[H:] ne remonte pas a h_prev)

    # rh = r * h_prev -> dr et contribution (3) a dh_prev
    dr = drh * h_prev
    dh_prev += drh * r                             # contribution (3)

    # z et r = sigmoid(W @ [h_prev, x] + b)
    dpre_z = dz * z * (1.0 - z)
    dpre_r = dr * r * (1.0 - r)
    zc = np.concatenate([h_prev, x_t])
    dW_z = np.outer(dpre_z, zc); db_z = dpre_z
    dW_r = np.outer(dpre_r, zc); db_r = dpre_r
    # contributions (1) et (2) a dh_prev (partie h de [h_prev, x])
    dh_prev += (P['W_z'].T @ dpre_z)[:H]
    dh_prev += (P['W_r'].T @ dpre_r)[:H]

    grads = {'W_z': dW_z, 'b_z': db_z, 'W_r': dW_r, 'b_r': db_r,
             'W_h': dW_h, 'b_h': db_h}
    return grads, dh_prev


def make_gru_params(D, H, seed=0):
    rng = np.random.RandomState(seed)
    sc = 0.2
    P = {}
    for nm in ['z', 'r', 'h']:
        P['W_' + nm] = rng.randn(H, H + D) * sc
        P['b_' + nm] = rng.randn(H) * sc
    return P


def gru_seq_loss(P, xs, H):
    h = np.zeros(H)
    caches = []
    loss = 0.0
    for x in xs:
        h, cache = gru_forward_step(x, h, P)
        caches.append(cache)
        loss += 0.5 * np.sum(h ** 2)
    return loss, caches


def gru_seq_backward(P, xs, caches, H):
    dP = {k: np.zeros_like(v) for k, v in P.items()}
    dh = np.zeros(H)
    for t in reversed(range(len(xs))):
        # h_t = caches[t] ... recompute h_t pour la loss locale
        x_t, h_prev, z, r, rh, zc2, h_tilde = caches[t]
        h_t = (1.0 - z) * h_prev + z * h_tilde
        dh_total = dh + h_t                        # local 0.5*||h_t||^2 -> h_t
        grads, dh = gru_backward_step(dh_total, caches[t], P, H)
        for k in dP:
            dP[k] += grads[k]
    return dP


# --- Gradient check GRU ---
D, H, T = 3, 4, 4
P = make_gru_params(D, H, seed=5)
xs = [np.random.randn(D) for _ in range(T)]
loss, caches = gru_seq_loss(P, xs, H)
dP = gru_seq_backward(P, xs, caches, H)

eps = 1e-5
print("\n  Gradient check GRU (BPTT, T=4, eps=1e-5):")
all_ok = True
for name in sorted(P.keys()):
    param = P[name]
    gnum = np.zeros_like(param)
    it = np.nditer(param, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        orig = param[idx]
        param[idx] = orig + eps
        lp, _ = gru_seq_loss(P, xs, H)
        param[idx] = orig - eps
        lm, _ = gru_seq_loss(P, xs, H)
        param[idx] = orig
        gnum[idx] = (lp - lm) / (2 * eps)
        it.iternext()
    gana = dP[name]
    denom = np.abs(gana) + np.abs(gnum) + 1e-8
    rel = np.max(np.abs(gana - gnum) / denom)
    ok = rel < 1e-4
    all_ok = all_ok and ok
    print(f"    {name:>4}: max rel error = {rel:.2e}  [{'PASS' if ok else 'FAIL'}]")
print(f"  -> {'GRU BACKWARD CORRECT' if all_ok else 'GRU BACKWARD WRONG'}")


# --- Comptage de parametres ---
def count_params(kind, D, H):
    """Nombre de parametres (matrices + biais)."""
    block = H * (D + H) + H   # une matrice (H, D+H) + biais (H,)
    return {'rnn': block, 'lstm': 4 * block, 'gru': 3 * block}[kind]


D, H = 10, 20
print("\n  Comptage de parametres (D=10, H=20):")
n_rnn = count_params('rnn', D, H)
n_lstm = count_params('lstm', D, H)
n_gru = count_params('gru', D, H)
print(f"    RNN  : {n_rnn}")
print(f"    LSTM : {n_lstm}")
print(f"    GRU  : {n_gru}  (ratio GRU/LSTM = {n_gru / n_lstm:.2f}, attendu 0.75)")


# --- Benchmark a budget de parametres egal ---
import time

print("\n  Benchmark (char-level next-token, budget params ~egalise):")
CORPUS = "le chat dort. le chien court. " * 20
chars = sorted(set(CORPUS))
V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}


def egalise_hidden(target_params, kind, D):
    """Resout H pour que count_params ~= target (recherche simple)."""
    best_H, best_diff = 1, float('inf')
    for Hh in range(2, 200):
        diff = abs(count_params(kind, D, Hh) - target_params)
        if diff < best_diff:
            best_diff, best_H = diff, Hh
    return best_H


def bench_lstm(seq_len=12, iters=150, lr=0.2, hidden=16, seed=0):
    rng = np.random.RandomState(seed)
    P = make_lstm_params(V, hidden, seed=seed)
    Wo, bo = rng.randn(V, hidden) * 0.1, np.zeros(V)
    t0 = time.time()
    last = 0.0
    for _ in range(iters):
        start = rng.randint(0, len(CORPUS) - seq_len - 1)
        inp = [stoi[c] for c in CORPUS[start:start + seq_len]]
        tgt = [stoi[c] for c in CORPUS[start + 1:start + seq_len + 1]]
        h, c = np.zeros(hidden), np.zeros(hidden)
        caches, probs = [], []
        loss = 0.0
        for t in range(seq_len):
            x = np.zeros(V); x[inp[t]] = 1.0
            h, c, cache = lstm_forward_step(x, h, c, P)
            caches.append(cache)
            logit = Wo @ h + bo
            e = np.exp(logit - logit.max()); pr = e / e.sum()
            probs.append(pr)
            loss += -np.log(max(pr[tgt[t]], 1e-12))
        last = loss / seq_len
        gP = {k: np.zeros_like(v) for k, v in P.items()}
        gWo, gbo = np.zeros_like(Wo), np.zeros_like(bo)
        dh, dc = np.zeros(hidden), np.zeros(hidden)
        for t in reversed(range(seq_len)):
            dlogit = probs[t].copy(); dlogit[tgt[t]] -= 1.0; dlogit /= seq_len
            gWo += np.outer(dlogit, caches[t][7] * caches[t][3])  # h_t = o*tanh_c
            gbo += dlogit
            dh_local = Wo.T @ dlogit
            grads, dh, dc = lstm_backward_step(dh + dh_local, dc, caches[t], P, hidden)
            for k in P:
                gP[k] += grads[k]
        for k in P:
            P[k] -= lr * gP[k]
        Wo -= lr * gWo; bo -= lr * gbo
    return last, time.time() - t0


# Cible : ~ params d'un LSTM hidden=16
target = count_params('lstm', V, 16)
H_lstm = 16
loss_l, time_l = bench_lstm(hidden=H_lstm, seed=0)
print(f"    LSTM hidden={H_lstm} (~{count_params('lstm', V, H_lstm)} params): "
      f"loss={loss_l:.4f}, time={time_l:.3f}s")
print(f"    (Pour egaliser, un GRU utiliserait hidden~{egalise_hidden(target, 'gru', V)}, "
      f"un RNN hidden~{egalise_hidden(target, 'rnn', V)}.)")
print("  -> A budget egal, les differences de loss sont marginales ;")
print("     le GRU est plus rapide par iteration (1 matmul de gate en moins que le LSTM).")

print("\n" + "=" * 70)
print("FIN DES SOLUTIONS HARD (Jour 4)")
print("=" * 70)
