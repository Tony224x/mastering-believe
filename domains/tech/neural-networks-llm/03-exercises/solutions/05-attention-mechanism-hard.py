"""
Solutions HARD — Jour 5 : Attention Mechanism
=============================================
Exercices 7, 8 (hard). Pur NumPy.

7. Backward de l'attention (softmax Jacobian) + gradient check.
8. FlashAttention-style online softmax (exact, memoire O(L)).

Run: python 03-exercises/solutions/05-attention-mechanism-hard.py
"""

import numpy as np

np.random.seed(42)


def softmax_rows(S):
    """Softmax stable par ligne d'une matrice (n, m)."""
    S = S - np.max(S, axis=-1, keepdims=True)
    e = np.exp(S)
    return e / np.sum(e, axis=-1, keepdims=True)


# ============================================================================
# EXERCISE 7: Backward de l'attention + gradient check
# ============================================================================

print("=" * 70)
print("EXERCISE 7: Backward de l'attention (softmax Jacobian) + gradient check")
print("=" * 70)


def softmax_backward(dp, p):
    """
    Backward du softmax, par ligne.
      p = softmax(s), Jacobienne J_ij = p_i(delta_ij - p_j).
      ds = J^T @ dp = p * (dp - sum(dp * p))
    dp, p : (n, m). Renvoie ds : (n, m).
    """
    # sum sur l'axe des classes (par ligne), garde la dim pour broadcast.
    return p * (dp - np.sum(dp * p, axis=-1, keepdims=True))


def attention_forward(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    S = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        S = np.where(mask == 1, -np.inf, S)
    A = softmax_rows(S)
    O = A @ V
    cache = (Q, K, V, A, d_k, mask)
    return O, cache


def attention_backward(dO, cache):
    """
    Backward complet.
      O = A @ V       -> dV = A^T @ dO,  dA = dO @ V^T
      A = softmax(S)  -> dS = softmax_backward(dA, A)   (par ligne)
      S = QK^T/sqrt   -> dQ = dS @ K / sqrt, dK = dS^T @ Q / sqrt
    """
    Q, K, V, A, d_k, mask = cache
    dV = A.T @ dO
    dA = dO @ V.T
    dS = softmax_backward(dA, A)
    # Les positions masquees ont A=0 ; softmax_backward y donne dS=0 -> pas de fuite.
    dQ = dS @ K / np.sqrt(d_k)
    dK = dS.T @ Q / np.sqrt(d_k)
    return dQ, dK, dV


def grad_check_attention(mask=None, label=""):
    n_q, n_k, d = 4, 5, 6
    Q = np.random.randn(n_q, d)
    K = np.random.randn(n_k, d)
    V = np.random.randn(n_k, d)

    O, cache = attention_forward(Q, K, V, mask)
    dO = O.copy()                              # loss = 0.5*||O||^2 -> dO = O
    dQ, dK, dV = attention_backward(dO, cache)

    eps = 1e-5
    results = {}
    for name, M, dM in [('Q', Q, dQ), ('K', K, dK), ('V', V, dV)]:
        gnum = np.zeros_like(M)
        it = np.nditer(M, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            orig = M[idx]
            M[idx] = orig + eps
            Op, _ = attention_forward(Q, K, V, mask)
            lp = 0.5 * np.sum(Op ** 2)
            M[idx] = orig - eps
            Om, _ = attention_forward(Q, K, V, mask)
            lm = 0.5 * np.sum(Om ** 2)
            M[idx] = orig
            gnum[idx] = (lp - lm) / (2 * eps)
            it.iternext()
        denom = np.abs(dM) + np.abs(gnum) + 1e-8
        rel = np.max(np.abs(dM - gnum) / denom)
        results[name] = rel
    ok = all(v < 1e-5 for v in results.values())
    print(f"\n  Gradient check {label}:")
    for name, rel in results.items():
        print(f"    d{name}: max rel error = {rel:.2e}  [{'PASS' if rel < 1e-5 else 'FAIL'}]")
    return ok


ok1 = grad_check_attention(mask=None, label="sans masque")


def grad_check_causal():
    # Masque causal carre (chaque ligne i a au moins la position i non masquee).
    n, d = 4, 6
    Q = np.random.randn(n, d); K = np.random.randn(n, d); V = np.random.randn(n, d)
    msk = np.triu(np.ones((n, n), dtype=np.int32), k=1)
    O, cache = attention_forward(Q, K, V, msk)
    dQ, dK, dV = attention_backward(O.copy(), cache)
    eps = 1e-5
    rels = {}
    for name, M, dM in [('Q', Q, dQ), ('K', K, dK), ('V', V, dV)]:
        gnum = np.zeros_like(M)
        it = np.nditer(M, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index; orig = M[idx]
            M[idx] = orig + eps; Op, _ = attention_forward(Q, K, V, msk); lp = 0.5*np.sum(Op**2)
            M[idx] = orig - eps; Om, _ = attention_forward(Q, K, V, msk); lm = 0.5*np.sum(Om**2)
            M[idx] = orig
            gnum[idx] = (lp - lm) / (2 * eps)
            it.iternext()
        denom = np.abs(dM) + np.abs(gnum) + 1e-8
        rels[name] = np.max(np.abs(dM - gnum) / denom)
    print("\n  Gradient check avec masque causal:")
    for name, rel in rels.items():
        print(f"    d{name}: max rel error = {rel:.2e}  [{'PASS' if rel < 1e-5 else 'FAIL'}]")
    return all(v < 1e-5 for v in rels.values())


ok2 = grad_check_causal()
print("\n  -> dQ depend de TOUTES les keys : la Jacobienne du softmax est dense,")
print("     elle couple toutes les sorties d'une ligne d'attention.")


# ============================================================================
# EXERCISE 8: FlashAttention-style online softmax
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 8: FlashAttention-style online softmax (exact, memoire O(L))")
print("=" * 70)


def online_softmax_weighted(s, V, block=4):
    """
    Calcule softmax(s) @ V en parcourant s/V par blocs, sans materialiser
    le softmax complet. s : (n_k,), V : (n_k, d). Renvoie (d,).

    Etat maintenu :
      m   : running max
      l   : running sum des exp (renormalisee)
      acc : accumulateur de l'output renormalise
    """
    n_k, d = V.shape
    m = -np.inf
    l = 0.0
    acc = np.zeros(d)
    for start in range(0, n_k, block):
        end = min(start + block, n_k)
        s_blk = s[start:end]
        V_blk = V[start:end]
        m_blk = np.max(s_blk)
        p_blk = np.exp(s_blk - m_blk)          # exp stable dans le bloc
        l_blk = np.sum(p_blk)
        acc_blk = p_blk @ V_blk                # contribution non normalisee du bloc

        m_new = max(m, m_blk)
        # Rescale l'ancien etat et le nouveau bloc vers le nouveau max commun.
        scale_old = np.exp(m - m_new) if m != -np.inf else 0.0
        scale_blk = np.exp(m_blk - m_new)
        l = l * scale_old + l_blk * scale_blk
        acc = acc * scale_old + acc_blk * scale_blk
        m = m_new
    return acc / l


# Verifier l'exactitude sur un vecteur.
n_k, d = 17, 5
s = np.random.randn(n_k) * 3.0                 # scores avec amplitude
V = np.random.randn(n_k, d)
p = np.exp(s - s.max()); p /= p.sum()
ref = p @ V
online = online_softmax_weighted(s, V, block=4)
print(f"\n  Online softmax vs naive (vecteur): max diff = {np.max(np.abs(ref - online)):.2e}")
print("  -> < 1e-10 : FlashAttention est EXACT, pas une approximation.")


def flash_attention(Q, K, V, block_q=2, block_k=4):
    """Attention complete par blocs (online softmax). Exact."""
    n_q, d = Q.shape
    O = np.zeros((n_q, V.shape[1]))
    scale = 1.0 / np.sqrt(d)
    for qs in range(0, n_q, block_q):
        qe = min(qs + block_q, n_q)
        Qb = Q[qs:qe]                          # (bq, d)
        for qi in range(Qb.shape[0]):
            s = (Qb[qi] @ K.T) * scale         # scores de cette query
            # online_softmax_weighted parcourt s et V par blocs de taille block_k.
            O[qs + qi] = online_softmax_weighted(s, V, block=block_k)
    return O


def naive_attention(Q, K, V):
    d = Q.shape[-1]
    S = Q @ K.T / np.sqrt(d)
    A = softmax_rows(S)
    return A @ V


n_q, n_k, d = 9, 13, 6
Q = np.random.randn(n_q, d); K = np.random.randn(n_k, d); V = np.random.randn(n_k, d)
O_flash = flash_attention(Q, K, V, block_q=2, block_k=4)
O_naive = naive_attention(Q, K, V)
print(f"\n  Flash vs naive (attention complete): max diff = {np.max(np.abs(O_flash - O_naive)):.2e}")

# Cout memoire.
print("\n  Memoire pic (matrice d'attention float32):")
for L in [512, 2048, 8192]:
    naive_mb = L * L * 4 / 1e6                 # O(L^2)
    flash_mb = L * d * 4 / 1e6                 # accumulateurs O(L*d), bloc negligeable
    print(f"    L={L:>5}: naive {naive_mb:>9.1f} MB (O(L^2)) | flash ~{flash_mb:>7.2f} MB (O(L))")

print("\n  Le running max est indispensable : sans lui, exp(s) overflow pour de grands")
print("  scores. Soustraire le max borne les exposants a <= 0 -> stabilite numerique.")

print("\n" + "=" * 70)
print("FIN DES SOLUTIONS HARD (Jour 5)")
print("=" * 70)
