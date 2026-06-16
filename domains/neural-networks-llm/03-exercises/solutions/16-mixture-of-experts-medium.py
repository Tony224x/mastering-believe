"""
Solutions MEDIUM — Jour 16 : Mixture of Experts
===============================================
Exercices 4, 5, 6 (medium). Pur NumPy, comme 02-code/16-mixture-of-experts.py.
Chaque etape non triviale est commentee avec le POURQUOI.

Run: python 03-exercises/solutions/16-mixture-of-experts-medium.py
"""

import sys
import io
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Les comportements MoE (collapse, equilibre) sont tres sensibles au seed ;
# on le fixe pour des chiffres reproductibles.
np.random.seed(42)


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def relu(x):
    return np.maximum(0.0, x)


# ============================================================================
# Exercice 4 — Forward MoE complet + FLOPs dense vs sparse
# ============================================================================

def top_k_router(x, W_g, k):
    """Routeur = 1 matmul + softmax + top-k + renormalisation des poids."""
    full_probs = softmax(x @ W_g, axis=-1)               # (B, N)
    top_k_indices = np.argsort(-full_probs, axis=-1)[:, :k]  # (B, k)
    rows = np.arange(x.shape[0])[:, None]
    top_k_probs = full_probs[rows, top_k_indices]        # (B, k)
    top_k_weights = top_k_probs / top_k_probs.sum(axis=-1, keepdims=True)
    return top_k_indices, top_k_weights, full_probs


class MoELayer:
    """N experts FFN (2 matrices up/down + ReLU), routeur top-k renormalise."""

    def __init__(self, d_model, d_ff, N, k=2, seed=0):
        rng = np.random.default_rng(seed)
        self.N, self.k, self.d_model, self.d_ff = N, k, d_model, d_ff
        self.W_g = rng.standard_normal((d_model, N)).astype(np.float64) * 0.1
        # N copies des matrices FFN : c'est la ou vivent les "params sparse".
        self.W_up = rng.standard_normal((N, d_model, d_ff)).astype(np.float64) * 0.1
        self.W_dn = rng.standard_normal((N, d_ff, d_model)).astype(np.float64) * 0.1

    def expert_forward(self, x, e):
        return relu(x @ self.W_up[e]) @ self.W_dn[e]

    def forward(self, x):
        idx, w, _ = top_k_router(x, self.W_g, self.k)
        out = np.zeros_like(x)
        # Boucle naive O(B*k) appels d'expert : sur GPU c'est un grouped-matmul,
        # mais la boucle rend la dependance limpide.
        for t in range(x.shape[0]):
            for j in range(self.k):
                e = int(idx[t, j])
                out[t] += w[t, j] * self.expert_forward(x[t:t + 1], e)[0]
        return out

    def n_params_experts(self):
        return self.N * 2 * self.d_model * self.d_ff

    def n_params_router(self):
        return self.d_model * self.N


class DenseFFN:
    """FFN dense de reference : d_ff_dense = N * d_ff -> meme capacite stockage."""

    def __init__(self, d_model, d_ff_dense, seed=0):
        rng = np.random.default_rng(seed + 99)
        self.d_model, self.d_ff = d_model, d_ff_dense
        self.W_up = rng.standard_normal((d_model, d_ff_dense)).astype(np.float64) * 0.1
        self.W_dn = rng.standard_normal((d_ff_dense, d_model)).astype(np.float64) * 0.1

    def forward(self, x):
        return relu(x @ self.W_up) @ self.W_dn

    def n_params(self):
        return 2 * self.d_model * self.d_ff


def exercice_4():
    print("=" * 70)
    print("Exercice 4 : Forward MoE complet + FLOPs dense vs sparse")
    print("=" * 70)

    B, d_model, d_ff, N, k = 32, 64, 128, 8, 2
    x = np.random.randn(B, d_model)

    moe = MoELayer(d_model, d_ff, N, k, seed=1)
    # 2) dense de meme capacite totale : d_ff_dense = N * d_ff.
    d_ff_dense = N * d_ff
    dense = DenseFFN(d_model, d_ff_dense, seed=1)

    y_moe = moe.forward(x)
    y_dense = dense.forward(x)

    # 3) memes shapes de sortie.
    print(f"\n  shapes : MoE {y_moe.shape}, dense {y_dense.shape}")
    assert y_moe.shape == (B, d_model) == y_dense.shape

    # 4) FLOPs par token (1 multiply-add = 1 FLOP). On ignore le routeur (negligeable).
    flops_dense = 2 * d_model * d_ff_dense * 2          # up + down
    flops_moe = k * (2 * d_model * d_ff * 2) + d_model * N
    ratio_flops = flops_dense / flops_moe
    print(f"\n  FLOPs/token dense = {flops_dense:,}")
    print(f"  FLOPs/token MoE   = {flops_moe:,}  (k={k} experts actifs + routeur)")
    print(f"  ratio FLOPs dense/MoE = {ratio_flops:.2f}x  (~ N/k = {N / k:.0f}x)")

    # 5) params : meme ordre au total, mais actifs = k/N des experts.
    p_experts = moe.n_params_experts()
    p_router = moe.n_params_router()
    p_moe_total = p_experts + p_router
    p_dense = dense.n_params()
    p_moe_active = (k / N) * p_experts + p_router
    print(f"\n  params MoE (experts+router) = {p_moe_total:,}")
    print(f"  params dense                = {p_dense:,}  (meme ordre)")
    print(f"  params actifs MoE/token     = {p_moe_active:,.0f}  "
          f"(= k/N * experts + router)")

    # 6) balayage k : ratio de sparsite total/actif. A k=N -> dense.
    print("\n  Balayage k (N=8 fixe) :")
    print(f"    {'k':<4} {'actifs':<14} {'sparsite (total/actif)':<24}")
    for kk in [1, 2, 4, 8]:
        active = (kk / N) * p_experts + p_router
        sparsity = p_moe_total / active
        tag = "  <- dense (aucun gain)" if kk == N else ""
        print(f"    {kk:<4} {active:<14,.0f} {sparsity:<24.2f}{tag}")
    print("  -> a k=N le MoE active TOUS les experts : sparsite ~1, il redevient dense.")

    # --- assertions de validation ---
    assert abs(ratio_flops - N / k) < 0.05               # ~4x a k=2, N=8
    assert abs(p_moe_active - ((k / N) * p_experts + p_router)) < 1e-6
    sparsity_kN = p_moe_total / ((N / N) * p_experts + p_router)
    assert abs(sparsity_kN - 1.0) < 0.01                 # k=N -> sparsite ~1
    print("\n  [OK] Exercice 4")


# ============================================================================
# Exercice 5 — Expert collapse et load balancing loss en action
# ============================================================================

def load_balancing_loss(top_k_indices, full_probs, N):
    """L_aux = N * sum_i (f_i * P_i). =1 en uniforme, =N au collapse total."""
    B, k = top_k_indices.shape
    f = np.zeros(N)
    for row in top_k_indices:
        for e in row:
            f[int(e)] += 1.0
    f /= (B * k)
    P = full_probs.mean(axis=0)
    return float(N * np.dot(f, P)), f, P


def entropy(p):
    """Entropie de Shannon (nats). Haute = equilibre, basse = collapse."""
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def exercice_5():
    print("\n" + "=" * 70)
    print("Exercice 5 : Expert collapse vs load balancing loss")
    print("=" * 70)

    N, d_model, B, steps = 8, 32, 256, 80
    rng = np.random.default_rng(0)
    X = rng.standard_normal((B, d_model))

    def run(use_aux, lr=0.5, lam=1.0):
        # Routeur initialise petit (proche de l'uniforme au depart).
        W_g = rng.standard_normal((d_model, N)) * 0.05
        top1_history = []
        for _ in range(steps):
            logits = X @ W_g                             # (B, N)
            probs = softmax(logits, axis=-1)            # (B, N)
            assign = np.argmax(probs, axis=-1)           # top-1
            f = np.bincount(assign, minlength=N).astype(np.float64) / B
            top1_history.append(float(f.max()))

            # --- Driver de collapse : un VRAI gradient (rich-get-richer) ---
            # On MAXIMISE la popularite : sum_b sum_i probs[b,i] * f_i, ou f_i
            # est la charge actuelle de l'expert i. Cela pousse chaque token
            # vers les experts deja populaires -> boucle de feedback positif
            # "le routeur prefere ce qu'il connait" = la cause documentee du
            # collapse. Gradient (a maximiser) de sum probs.f via Jacobien
            # softmax : g = probs * (f - <f,probs[b]>). On DESCEND -g.
            fp = np.sum(f[None, :] * probs, axis=-1, keepdims=True)
            g_pop = probs * (f[None, :] - fp)            # d(popularite)/dlogits
            g_logits = -g_pop                            # on MAXIMISE -> descend -g

            if use_aux:
                # Gradient REEL de la load balancing loss L_aux = N * sum(f_i P_i)
                # p/r aux logits, via P_i = mean_b softmax(logits)_i (f traite
                # comme constante, non differentiable). dP/dlogits = Jacobien
                # softmax ; dL_aux/dlogits[b] = N * probs[b] * (f - <f,probs[b]>).
                # C'est EXACTEMENT l'oppose du driver de popularite (au facteur N
                # pres) : l'aux loss penalise precisement ce que le collapse
                # renforce. lam>=1/N suffit donc a inverser la dynamique.
                g_aux = N * probs * (f[None, :] - fp)
                g_logits = g_logits + lam * g_aux        # +grad(L_aux) = descend L_aux

            W_g -= lr * (X.T @ g_logits)
        return W_g, np.array(top1_history)

    # Sans regularisation -> collapse (le sharpening concentre tout le trafic).
    Wg_no, hist_no = run(use_aux=False)
    probs_no = softmax(X @ Wg_no, axis=-1)
    assign_no = np.argmax(probs_no, axis=-1)
    f_no = np.bincount(assign_no, minlength=N) / B
    dead_no = int(np.sum(f_no < 1e-6))

    # Avec aux loss -> charge ~uniforme : le gradient de L_aux contre le
    # sharpening en penalisant les experts sur-charges (P_i baisse quand f_i^).
    Wg_aux, hist_aux = run(use_aux=True, lam=1.0)
    probs_aux = softmax(X @ Wg_aux, axis=-1)
    assign_aux = np.argmax(probs_aux, axis=-1)
    f_aux = np.bincount(assign_aux, minlength=N) / B
    dead_aux = int(np.sum(f_aux < 1e-6))

    # 1) verifier que la loss vaut 1 en uniforme et N en collapse.
    L_unif, _, _ = load_balancing_loss(
        np.array([[i % N] for i in range(B)]),
        np.ones((B, N)) / N, N)
    coll_idx = np.zeros((B, 1), dtype=int)
    coll_probs = np.zeros((B, N)); coll_probs[:, 0] = 1.0
    L_coll, _, _ = load_balancing_loss(coll_idx, coll_probs, N)
    print(f"\n  load_balancing_loss : uniforme = {L_unif:.3f} (cible 1.0), "
          f"collapse = {L_coll:.3f} (max N={N})")

    print(f"\n  SANS aux loss :")
    print(f"    top-1 fraction : debut {hist_no[0]:.2f} -> fin {hist_no[-1]:.2f}")
    print(f"    distribution f = {np.round(f_no, 3).tolist()}")
    print(f"    entropie(f) = {entropy(f_no):.3f}  (max = {np.log(N):.3f})")
    print(f"    experts morts = {dead_no} / {N}")

    print(f"\n  AVEC aux loss (lambda=1.0) :")
    print(f"    top-1 fraction : debut {hist_aux[0]:.2f} -> fin {hist_aux[-1]:.2f}")
    print(f"    distribution f = {np.round(f_aux, 3).tolist()}")
    print(f"    entropie(f) = {entropy(f_aux):.3f}  (max = {np.log(N):.3f})")
    print(f"    experts morts = {dead_aux} / {N}")

    print("\n  POURQUOI f_i * P_i : f_i passe par argmax (NON differentiable) ;")
    print("  P_i seul ne contraint pas la charge dure dispatchee. Le produit")
    print("  accroche le gradient de P (doux) sur la distribution dure f.")

    # --- assertions de validation ---
    assert abs(L_unif - 1.0) < 1e-6
    assert abs(L_coll - N) < 1e-6
    # Le collapse grimpe : la fraction top-1 finit nettement plus haute.
    assert hist_no[-1] > hist_no[0]
    assert hist_no[-1] > 0.5                       # collapse marque (winner-take-all)
    # L'aux loss garde une charge plus equilibree (entropie plus haute,
    # fraction top-1 plus basse, pas plus d'experts morts).
    assert entropy(f_aux) > entropy(f_no)
    assert hist_aux[-1] < hist_no[-1]
    assert dead_aux <= dead_no                     # moins (ou autant) d'experts morts
    print("\n  [OK] Exercice 5")


# ============================================================================
# Exercice 6 — Capacity factor et token dropping
# ============================================================================

def dispatch_with_capacity(assignments, N, capacity):
    """
    Remplit le buffer de chaque expert jusqu'a `capacity` dans l'ordre des
    tokens. Drop (marque False) les tokens en exces une fois le buffer plein.
    Retourne un masque booleen (True = traite, False = droppe).
    """
    processed = np.zeros(len(assignments), dtype=bool)
    load = np.zeros(N, dtype=int)
    for t, e in enumerate(assignments):
        e = int(e)
        if load[e] < capacity:
            load[e] += 1
            processed[t] = True
        # sinon : buffer plein -> token droppe (passe par le residual seul).
    return processed


def capacity_of(CF, B, N):
    """capacity = ceil(CF * tokens_par_batch / N)."""
    return int(np.ceil(CF * B / N))


def exercice_6():
    print("\n" + "=" * 70)
    print("Exercice 6 : Capacity factor et token dropping")
    print("=" * 70)

    N, B = 8, 512
    rng = np.random.default_rng(7)
    CFs = [1.0, 1.25, 1.5, 2.0]

    # 3) Scenario equilibre : routage uniforme.
    assign_unif = rng.integers(0, N, size=B)

    # 4) Scenario desequilibre : 50% des tokens vers les 2 premiers experts.
    n_skew = B // 2
    assign_skew = np.empty(B, dtype=int)
    assign_skew[:n_skew] = rng.integers(0, 2, size=n_skew)        # experts 0,1
    assign_skew[n_skew:] = rng.integers(2, N, size=B - n_skew)    # 2..N-1
    rng.shuffle(assign_skew)

    def drop_rate(assign, CF):
        cap = capacity_of(CF, B, N)
        processed = dispatch_with_capacity(assign, N, cap)
        return 1.0 - processed.mean(), cap

    print(f"\n  N={N}, B={B}")
    print(f"  {'CF':<6} {'capacity':<10} {'drop uniforme':<16} {'drop skewed':<14}")
    print("  " + "-" * 48)
    drops_unif, drops_skew = {}, {}
    for CF in CFs:
        d_u, cap = drop_rate(assign_unif, CF)
        d_s, _ = drop_rate(assign_skew, CF)
        drops_unif[CF], drops_skew[CF] = d_u, d_s
        print(f"  {CF:<6} {cap:<10} {d_u:<16.2%} {d_s:<14.2%}")

    # 5) constats.
    print(f"\n  - CF=1.0 uniforme : drop = {drops_unif[1.0]:.1%} "
          f"(remplissage discret/aleatoire, comme le NOTE du cours ~1-3%).")
    print(f"  - CF>=1.5 uniforme : drop = {drops_unif[1.5]:.1%} (quasi nul).")
    print(f"  - Desequilibre a CF=1.0 : drop = {drops_skew[1.0]:.1%} (explose),")
    print(f"    chute a {drops_skew[2.0]:.1%} a CF=2.0 -- mais capacite x2 = "
          f"VRAM x2.")

    # 6) analyse inference + tradeoff.
    print("\n  POURQUOI pas de drop a l'inference : pas de buffer fixe -> on")
    print("  attend que chaque expert traite TOUS ses tokens. Le drop est un")
    print("  artefact du training (buffers GPU statiques pour batcher).")
    print("  Tradeoff CF : CF^ -> moins de drop (meilleure qualite/stabilite)")
    print("  mais memoire allouee proportionnelle a CF (capacity = CF*B/N).")

    # --- assertions de validation ---
    # Uniforme : drop faible a CF=1.0, ~0 a CF>=1.5.
    assert drops_unif[1.0] < 0.10
    assert drops_unif[1.5] < 0.01
    # Desequilibre : drop bien plus eleve a bas CF, decroissant avec CF.
    assert drops_skew[1.0] > drops_unif[1.0]
    assert drops_skew[2.0] <= drops_skew[1.0]
    # capacity = ceil(CF*B/N).
    assert capacity_of(1.0, B, N) == B // N
    print("\n  [OK] Exercice 6")


if __name__ == "__main__":
    exercice_4()
    exercice_5()
    exercice_6()
    print("\nDone (MEDIUM).")
