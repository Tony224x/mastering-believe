"""
Solutions HARD — Jour 16 : Mixture of Experts
=============================================
Exercices 7, 8, 9 (hard). Pur NumPy, comme 02-code/16-mixture-of-experts.py.
Chaque etape non triviale est commentee avec le POURQUOI.

Run: python 03-exercises/solutions/16-mixture-of-experts-hard.py
"""

import sys
import io
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

np.random.seed(42)


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


# ============================================================================
# Exercice 7 — Token-choice vs expert-choice routing
# ============================================================================

def token_choice_dispatch(scores, k, capacity):
    """
    Token-choice (top-k) : chaque token choisit ses k experts (plus grands
    scores), on dispatche dans des buffers de taille `capacity` par expert,
    dans l'ordre des tokens. Un (token, slot) est DROPPE si le buffer cible
    est plein.

    Retourne :
      load   : (N,) nb de slots reellement traites par expert
      dropped_slots : nb de (token, slot) droppes
      token_dropped : (B,) True si AU MOINS un slot du token a ete droppe
    """
    B, N = scores.shape
    top_k = np.argsort(-scores, axis=-1)[:, :k]          # (B, k)
    load = np.zeros(N, dtype=int)
    dropped_slots = 0
    token_dropped = np.zeros(B, dtype=bool)
    # On parcourt token par token (ordre de la sequence) -> les premiers
    # tokens "reservent" les places ; un expert populaire deborde sur la fin.
    for t in range(B):
        for e in top_k[t]:
            e = int(e)
            if load[e] < capacity:
                load[e] += 1
            else:
                dropped_slots += 1
                token_dropped[t] = True
    return load, dropped_slots, token_dropped


def expert_choice_dispatch(scores, M):
    """
    Expert-choice (top-M) : chaque EXPERT choisit ses M tokens preferes
    (ses M plus grands scores dans sa colonne). Equilibre parfait par
    construction (chaque expert traite exactement M tokens), mais certains
    tokens ne sont choisis par AUCUN expert -> orphelins.

    Retourne :
      per_expert : (N,) = M partout (par construction)
      n_unseen   : nb de tokens jamais selectionnes
      seen       : (B,) True si le token a ete pris par >=1 expert
    """
    B, N = scores.shape
    seen = np.zeros(B, dtype=bool)
    per_expert = np.zeros(N, dtype=int)
    for e in range(N):
        chosen = np.argsort(-scores[:, e])[:M]           # M meilleurs tokens
        seen[chosen] = True
        per_expert[e] = len(chosen)
    n_unseen = int(np.sum(~seen))
    return per_expert, n_unseen, seen


def make_scores(B, N, kind, rng):
    """Genere une matrice de scores (softmax de logits) selon 3 regimes."""
    if kind == "uniform":
        logits = rng.standard_normal((B, N))
    elif kind == "skewed":
        # Quelques experts intrinsequement plus attractifs (biais sur colonnes).
        bias = np.zeros(N)
        bias[:2] = 3.0
        logits = rng.standard_normal((B, N)) + bias
    elif kind == "collapse":
        # 1 expert domine fortement.
        bias = np.zeros(N)
        bias[0] = 6.0
        logits = rng.standard_normal((B, N)) * 0.5 + bias
    else:
        raise ValueError(kind)
    return softmax(logits, axis=-1)


def exercice_7():
    print("=" * 70)
    print("Exercice 7 : Token-choice vs expert-choice routing")
    print("=" * 70)

    B, N, k = 512, 8, 2
    CF = 1.0
    capacity = int(CF * B * k / N)       # buffer token-choice
    M = B * k // N                        # tokens par expert en expert-choice
    print(f"\n  B={B}, N={N}, k={k}, CF={CF} -> capacity={capacity}, M={M}")
    rng = np.random.default_rng(1)

    results = {}
    print(f"\n  {'regime':<12} {'TC drop':<10} {'TC load std':<14} "
          f"{'EC unseen':<12} {'EC load (min/max)':<18}")
    print("  " + "-" * 66)
    for kind in ["uniform", "skewed", "collapse"]:
        scores = make_scores(B, N, kind, rng)

        # Token-choice.
        load_tc, dropped, tok_dropped = token_choice_dispatch(scores, k, capacity)
        tc_drop_rate = float(np.mean(tok_dropped))
        tc_std = float(np.std(load_tc))

        # Expert-choice.
        per_expert, n_unseen, seen = expert_choice_dispatch(scores, M)
        ec_unseen_rate = n_unseen / B

        results[kind] = (tc_drop_rate, tc_std, ec_unseen_rate, per_expert)
        print(f"  {kind:<12} {tc_drop_rate:<10.2%} {tc_std:<14.2f} "
              f"{ec_unseen_rate:<12.2%} "
              f"{per_expert.min():>3}/{per_expert.max():<14}")

        # Expert-choice equilibre PARFAIT : exactement M par expert.
        assert np.all(per_expert == M)

    # 5) analyse.
    print("\n  POURQUOI ces resultats :")
    print("  - Token-choice DEBORDE un expert populaire sous routage skewed/collapse")
    print("    (drops eleves a bas CF) ; le token sacrifie est celui EN EXCES sur un")
    print("    expert deja plein (souvent un token a preference forte mais tardif).")
    print("  - Expert-choice garantit M tokens/expert (equilibre parfait) mais laisse")
    print("    des tokens ORPHELINS : un token 'ambivalent' (faible partout) n'entre")
    print("    dans le top-M d'aucune colonne -> jamais traite.")
    print("  - DeepSeek-V3 = 3e voie : auxiliary-loss-free balancing. On garde le")
    print("    token-choice mais on ajoute un BIAIS par expert aux logits du routeur,")
    print("    ajuste dynamiquement (expert surcharge -> biais baisse). On equilibre")
    print("    sans loss auxiliaire a tuner ni drop garanti.")

    # --- assertions de validation ---
    # Token-choice : drop quasi nul en uniforme, eleve en collapse.
    assert results["uniform"][0] < 0.10
    assert results["collapse"][0] > results["uniform"][0]
    # Expert-choice : pas d'orphelins en uniforme (~tout le monde vu), beaucoup
    # plus quand le routage est skewed/collapse.
    assert results["collapse"][2] > results["uniform"][2]
    print("\n  [OK] Exercice 7")


# ============================================================================
# Exercice 8 — Specialisation des experts (mini-MoE entrainable)
# ============================================================================

def exercice_8():
    print("\n" + "=" * 70)
    print("Exercice 8 : Specialisation des experts (mini-MoE entrainable)")
    print("=" * 70)

    rng = np.random.default_rng(0)
    G = 4          # nombre de clusters latents
    d = 16         # dimension d'entree
    d_out = 8      # dimension de sortie
    N = 4          # nombre d'experts
    n_per = 200    # points par cluster
    B = G * n_per

    # 1) tache jouet a clusters : chaque cluster a son centre + sa cible lineaire.
    centers = rng.standard_normal((G, d)) * 4.0
    target_W = rng.standard_normal((G, d, d_out))         # 1 matrice cible / cluster
    X, Y, cluster_of = [], [], []
    for g in range(G):
        Xg = centers[g] + rng.standard_normal((n_per, d)) * 0.5
        Yg = Xg @ target_W[g]
        X.append(Xg); Y.append(Yg); cluster_of += [g] * n_per
    X = np.vstack(X); Y = np.vstack(Y)
    cluster_of = np.array(cluster_of)
    # On melange : le modele ne voit jamais le cluster d'origine.
    perm = rng.permutation(B)
    X, Y, cluster_of = X[perm], Y[perm], cluster_of[perm]

    def train_moe(use_aux, steps=500, lr=0.02, lam=0.05, seed=1):
        r = np.random.default_rng(seed)
        # N experts lineaires + routeur.
        We = r.standard_normal((N, d, d_out)) * 0.1
        Wg = r.standard_normal((d, N)) * 0.1
        losses = []
        for _ in range(steps):
            logits = X @ Wg                              # (B, N)
            probs = softmax(logits, axis=-1)            # (B, N) routing doux
            # Forward soft : sortie = sum_i probs_i * expert_i(x). Le soft-routing
            # rend le routeur differentiable (le hard top-1 sert juste a mesurer).
            expert_out = np.einsum("bd,ndo->bno", X, We)  # (B, N, d_out)
            y_hat = np.einsum("bn,bno->bo", probs, expert_out)  # (B, d_out)

            err = y_hat - Y                              # (B, d_out)
            mse = float(np.mean(err ** 2))
            losses.append(mse)

            # --- gradients manuels ---
            # dL/dy_hat = 2/B/d_out * err
            g_y = (2.0 / (B * d_out)) * err              # (B, d_out)
            # Experts : dL/dWe_n = sum_b probs[b,n] * x_b^T g_y_b
            g_We = np.einsum("bn,bd,bo->ndo", probs, X, g_y)
            # Routeur via les probs : dL/dprobs[b,n] = <g_y_b, expert_out[b,n]>
            g_probs = np.einsum("bo,bno->bn", g_y, expert_out)  # (B, N)
            # Backprop softmax : dL/dlogits = probs * (g_probs - sum(probs*g_probs))
            dot = np.sum(probs * g_probs, axis=-1, keepdims=True)
            g_logits = probs * (g_probs - dot)          # (B, N)

            if use_aux:
                # Load balancing : f_i (dur, top-1) et P_i (doux). Gradient de
                # L_aux = N * sum(f_i P_i) p/r aux logits, via P_i = mean softmax.
                top1 = np.argmax(probs, axis=-1)
                f = np.bincount(top1, minlength=N) / B
                P = probs.mean(axis=0)
                # dP_i/dlogits propage comme un softmax moyen ; on approxime le
                # gradient par un push proportionnel a (N * f_i) sur chaque logit.
                g_aux = lam * (N * f)[None, :] * (probs * (1 - probs))  # (B,N)
                g_logits = g_logits + g_aux

            g_Wg = X.T @ g_logits                        # (d, N)

            We -= lr * g_We
            Wg -= lr * g_Wg
        return We, Wg, losses

    def specialisation(Wg):
        """Matrice de confusion cluster -> expert (top-1) + score de purete."""
        top1 = np.argmax(X @ Wg, axis=-1)
        conf = np.zeros((G, N), dtype=int)
        for c, e in zip(cluster_of, top1):
            conf[c, int(e)] += 1
        # Purete : pour chaque EXPERT, fraction de ses tokens issus du cluster
        # dominant. Moyenne ponderee = a quel point un expert = un cluster.
        col_sums = conf.sum(axis=0)
        purity = 0.0
        for e in range(N):
            if col_sums[e] > 0:
                purity += conf[:, e].max()
        purity /= max(conf.sum(), 1)
        return conf, purity

    # Avec load balancing.
    We_a, Wg_a, loss_a = train_moe(use_aux=True)
    conf_a, purity_a = specialisation(Wg_a)
    mse_a = loss_a[-1]

    # Sans load balancing.
    We_n, Wg_n, loss_n = train_moe(use_aux=False)
    conf_n, purity_n = specialisation(Wg_n)
    mse_n = loss_n[-1]

    print(f"\n  AVEC load balancing : loss {loss_a[0]:.3f} -> {mse_a:.4f}")
    print(f"    confusion cluster->expert :\n{conf_a}")
    print(f"    purete = {purity_a:.2%}  (1 expert ~ 1 cluster si proche de 1)")
    n_active_a = int(np.sum(conf_a.sum(axis=0) > 0))
    print(f"    experts utilises = {n_active_a} / {N}")

    print(f"\n  SANS load balancing : loss {loss_n[0]:.3f} -> {mse_n:.4f}")
    print(f"    confusion cluster->expert :\n{conf_n}")
    print(f"    purete = {purity_n:.2%}")
    n_active_n = int(np.sum(conf_n.sum(axis=0) > 0))
    print(f"    experts utilises = {n_active_n} / {N}")

    # 6) analyse.
    print("\n  POURQUOI la specialisation par domaine emerge ici : la tache a une")
    print("  structure latente PROPRE (clusters separes + cible distincte). Sur du")
    print("  langage reel, Mixtral specialise plutot par patterns SYNTAXIQUES")
    print("  (ponctuation, tokens numeriques) ; DeepSeek-V3 (fine-grained, 256")
    print("  experts) capture des niches plus SEMANTIQUES car chaque petit expert")
    print("  peut isoler un sous-domaine que les 8 gros experts de Mixtral diluent.")
    print("  Si N < G : un expert doit servir >1 cluster -> il moyenne des cibles")
    print("  incompatibles -> MSE plancher non nul, specialisation impossible.")

    # --- assertions de validation ---
    assert loss_a[-1] < loss_a[0]                  # converge
    assert loss_n[-1] < loss_n[0]                  # converge aussi
    # Le load balancing utilise au moins autant d'experts (moins de collapse).
    assert n_active_a >= n_active_n
    # Avec equilibre, la purete est elevee (specialisation nette).
    assert purity_a > 0.6
    print("\n  [OK] Exercice 8")


# ============================================================================
# Exercice 9 — Comptabilite complete MoE + cout all-to-all
# ============================================================================

def moe_accounting(layers, d_model, d_ff, N, k, vocab, n_shared=0):
    """
    Comptabilite params d'un MoE (SwiGLU = 3 matrices/expert), avec shared
    expert toujours actif (style DeepSeek).
      - attention partagee : 4 * d_model^2 (Q,K,V,O)
      - 1 expert SwiGLU    : 3 * d_model * d_ff (gate, up, down)
      - router             : d_model * N
      - embeddings         : vocab * d_model
    """
    attn = 4 * d_model * d_model
    expert = 3 * d_model * d_ff
    router = d_model * N
    emb = vocab * d_model

    per_layer_total = attn + (N + n_shared) * expert + router
    per_layer_active = attn + (k + n_shared) * expert + router

    total = per_layer_total * layers + emb
    active = per_layer_active * layers + emb
    return {
        "total": total,
        "active": active,
        "sparsity": total / active,
        "expert": expert,
    }


def exercice_9():
    print("\n" + "=" * 70)
    print("Exercice 9 : Comptabilite MoE complete + cout all-to-all")
    print("=" * 70)

    # 1) validation des ordres de grandeur.
    # Mixtral-like : d_model=4096, d_ff=14336, N=8, k=2, 32 layers -> ~47B.
    mix = moe_accounting(layers=32, d_model=4096, d_ff=14336,
                         N=8, k=2, vocab=32000, n_shared=0)
    # DeepSeek-V3-like : backbone propre (61 layers, d_model=7168), 256 experts
    # fine-grained (d_ff=2048), top-8, 1 shared -> ~671B total / ~37B actifs.
    ds = moe_accounting(layers=61, d_model=7168, d_ff=2048,
                        N=256, k=8, vocab=129280, n_shared=1)

    def gb(n):  # params -> milliards
        return n / 1e9

    print(f"\n  Mixtral-like (SwiGLU, N=8, k=2, 32 layers)")
    print(f"    total  = {gb(mix['total']):.1f} B  (cible ~47 B)")
    print(f"    actifs = {gb(mix['active']):.1f} B  (cible ~13 B)")
    print(f"    sparsite = {mix['sparsity']:.2f}x")

    print(f"\n  DeepSeek-V3-like (SwiGLU fine-grained, N=256, k=8, 1 shared, 61 layers)")
    print(f"    total  = {gb(ds['total']):.0f} B  (cible ~671 B)")
    print(f"    actifs = {gb(ds['active']):.0f} B  (cible ~37 B)")
    print(f"    sparsite = {ds['sparsity']:.2f}x")

    # 2) VRAM des poids (FP16 = 2 octets/param).
    def vram_gb(params):
        return params * 2 / 1e9
    print(f"\n  VRAM poids (FP16, 2 o/param) :")
    print(f"    Mixtral-like   = {vram_gb(mix['total']):.0f} GB  "
          f"(TOUS les experts, pas seulement k)")
    print(f"    DeepSeek-like  = {vram_gb(ds['total']):.0f} GB")
    print("    -> idee fausse n.2 confirmee : MoE prend la VRAM de tous les experts.")

    # 3) FLOPs/token = 2 * params_actifs ; gain vs dense de meme total.
    print(f"\n  FLOPs/token = 2 * params_actifs :")
    for name, a in [("Mixtral-like", mix), ("DeepSeek-like", ds)]:
        flops_moe = 2 * a["active"]
        flops_dense = 2 * a["total"]   # un dense de meme capacite stockage
        print(f"    {name:<14} MoE = {flops_moe / 1e9:.1f} GFLOP, "
              f"dense equiv = {flops_dense / 1e9:.0f} GFLOP, "
              f"gain x{flops_dense / flops_moe:.1f}")

    # 4) cout communication all-to-all.
    # Par layer MoE : 2 all-to-all transferent d_model floats par token (FP16).
    def comm_fraction(tokens, d_model, active_params, BW_GBs,
                      gpu_flops=1e15):
        comm_bytes = 2 * tokens * d_model * 2            # 2 all-to-all, FP16
        t_comm = comm_bytes / (BW_GBs * 1e9)
        # compute du layer = 2 * params_actifs_FFN * tokens (regle FLOPs).
        flops_layer = 2 * active_params * tokens
        t_compute = flops_layer / gpu_flops
        return t_comm / (t_comm + t_compute), t_comm, t_compute

    # Params actifs FFN par layer DeepSeek (k+shared experts SwiGLU).
    active_ffn_layer = (8 + 1) * ds["expert"]
    tokens = 4096
    print(f"\n  5) Balayage interconnect (DeepSeek-like, {tokens} tokens/layer) :")
    print(f"     {'BW (GB/s)':<12} {'frac comm':<12}")
    for BW in [600, 200, 50, 25]:
        frac, tc, tk = comm_fraction(tokens, 7168, active_ffn_layer, BW)
        print(f"     {BW:<12} {frac:<12.1%}")
    print("     -> a haute BW (NVLink 600) la comm reste contenue ; a basse BW")
    print("        (Ethernet 25) elle domine -> les gains MoE s'evaporent.")

    # 6) analyse.
    print("\n  POURQUOI MoE est un investissement infra : sur multi-GPU, l'all-to-all")
    print("  peut prendre 20-50% du temps total sur les gros MoE distribues. C'est")
    print("  pourquoi DeepSeek a eu besoin de DualPipe (overlap compute/comm) et de")
    print("  communication FP8 (moins d'octets a transferer). Sans NVLink/IB, le")
    print("  MoE perd ses gains : le k/N de FLOPs economises est mange par le reseau.")

    # --- assertions de validation ---
    assert 40 < gb(mix["total"]) < 55          # ~47 B
    assert 600 < gb(ds["total"]) < 750         # ~671 B
    assert 30 < gb(ds["active"]) < 45          # ~37 B
    assert mix["sparsity"] > 2.5
    assert ds["sparsity"] > 10                 # DeepSeek bien plus sparse
    # La fraction comm augmente quand la BW baisse.
    f600, _, _ = comm_fraction(tokens, 7168, active_ffn_layer, 600)
    f25, _, _ = comm_fraction(tokens, 7168, active_ffn_layer, 25)
    assert f25 > f600
    print("\n  [OK] Exercice 9")


if __name__ == "__main__":
    exercice_7()
    exercice_8()
    exercice_9()
    print("\nDone (HARD).")
