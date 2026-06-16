"""
Solutions MEDIUM — Jour 21 : Mechanistic interpretability
=========================================================
Exercices 4, 5, 6 (medium). Pur NumPy + stdlib, comme
02-code/21-mechanistic-interpretability.py. Chaque etape non triviale est
commentee avec le POURQUOI. Le fichier est auto-verifiant (assertions finales).

  4. Activation patching (causal mediation, denoising) : carte (layer x pos)
     de recovery + contraste probe (correlation) vs patching (causalite).
  5. Induction head : prev-token head (L1) + induction head (L2) composes via
     le residual stream, le matching exprime comme une ATTENTION (softmax),
     pas un 'if' ; effet de la longueur de contexte.
  6. SAE minimal L1 vs TopK : recovery mono-semantique, dead features,
     shrinkage L1 vs sparsite exacte TopK (Gao 2024).

Run: python3 03-exercises/solutions/21-mechanistic-interpretability-medium.py
"""

from __future__ import annotations
import sys
import io
import numpy as np

# Stdout en UTF-8 (Windows/CI-friendly), comme 02-code/21.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

np.random.seed(42)


def softmax_rows(x: np.ndarray) -> np.ndarray:
    """Softmax stable par ligne (axis=-1) pour les scores d'attention."""
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


# ============================================================================
# EXERCISE 4 — Activation patching : le test causal from scratch
# ============================================================================
# But : un fait vit a un (layer, position) connu du residual stream. Le patching
# DENOISING (copier l'activation clean dans le run corrupted) ne fait remonter
# le logit cible QU'A ce site -> on produit une carte causale. Un linear probe
# peut etre eleve a un endroit SANS effet causal (correlation != causalite).

print("=" * 70)
print("EXERCISE 4 : activation patching — carte causale (layer x position)")
print("=" * 70)

D_MODEL = 20
N_LAYERS = 4
N_POS = 4
VOCAB = 10
L_STAR = 1               # couche d'injection du fait
P_STAR = N_POS - 1       # position porteuse = celle LUE pour les logits finaux
T_CLEAN = 2              # token cible du run clean
T_CORR = 7              # token cible du run corrupted
FACT_STRENGTH = 4.0     # force d'injection : domine la sortie a P*

_rng = np.random.default_rng(7)
# Unembedding (d_model, vocab) : une colonne = direction d'un token.
W_U = _rng.normal(0, 1.0, size=(D_MODEL, VOCAB))


def token_dir(tok: int) -> np.ndarray:
    """Direction-fait = colonne d'unembedding normalisee : ecrire cette
    direction dans le residual POUSSE le logit du token vers le haut (effet
    causal a la sortie)."""
    v = W_U[:, tok]
    return v / (np.linalg.norm(v) + 1e-8)


d_clean, d_corr = token_dir(T_CLEAN), token_dir(T_CORR)

# Bruit de fond COMMUN aux deux runs : identique => non discriminant (il ne
# porte aucune info clean-vs-corrupted). Le seul ecart sera la direction-fait.
_rng_bg = np.random.default_rng(123)
base_contrib = _rng_bg.normal(0, 0.25, size=(N_LAYERS, N_POS, D_MODEL))
embed0 = np.random.default_rng(99).normal(0, 0.2, size=(N_POS, D_MODEL))


def run_decoder(fact_dir, patch=None):
    """Toy decoder additif traceable. h[pos] part de embed0[pos], chaque couche
    ajoute base_contrib (bruit commun) ; a (L_STAR, P_STAR) on injecte le fait.

    patch : dict {(L, P): vecteur} -> apres la couche L on REMPLACE h[P] par le
    vecteur fourni puis on continue le forward (coeur du denoising : on injecte
    une activation saine dans un run corrompu et on observe l'effet en aval).
    Renvoie (acts: snapshots APRES chaque couche, logits a la derniere position)."""
    patch = patch or {}
    h = embed0.copy()
    acts = [h.copy()]                        # acts[0] = embedding initial
    for L in range(N_LAYERS):
        h = h + base_contrib[L]              # bruit commun
        if L == L_STAR:                      # injection du fait : une seule couche
            h[P_STAR] = h[P_STAR] + FACT_STRENGTH * fact_dir
        for (pL, pP), pvec in patch.items():  # denoising eventuel
            if pL == L:
                h[pP] = pvec
        acts.append(h.copy())
    logits_final = acts[-1][-1] @ W_U        # logits a la derniere position
    return acts, logits_final


# --- Runs clean & corrupted : verifier qu'ils donnent A/B en top-1 ---
acts_clean, logits_clean = run_decoder(d_clean)
acts_corr, logits_corr = run_decoder(d_corr)
top_clean, top_corr = int(np.argmax(logits_clean)), int(np.argmax(logits_corr))
print(f"\n  clean top-1     = t{top_clean} (attendu t{T_CLEAN})")
print(f"  corrupted top-1 = t{top_corr} (attendu t{T_CORR})")
assert top_clean == T_CLEAN and top_corr == T_CORR, "clean/corrupted doivent donner A/B"

lc, lr = float(logits_clean[T_CLEAN]), float(logits_corr[T_CLEAN])
denom = (lc - lr) if abs(lc - lr) > 1e-9 else 1e-9

# --- Patching denoising sur toute la grille (layer x position) ---
# Recovery normalise : (logit_A_patched - logit_A_corrupted) / (clean - corrupted).
# 1.0 = totalement restaure, 0.0 = inerte.
recovery = np.zeros((N_LAYERS + 1, N_POS))
for L in range(1, N_LAYERS + 1):             # acts[0] = embed, pas une couche
    for P in range(N_POS):
        clean_act = acts_clean[L][P]         # snapshot APRES la couche L
        _, logits_p = run_decoder(d_corr, patch={(L - 1, P): clean_act})
        recovery[L, P] = (float(logits_p[T_CLEAN]) - lr) / denom

print(f"\n  Carte de recovery (lignes = snapshot layer, colonnes = position)")
print(f"  site porteur attendu : L>={L_STAR + 1} a P={P_STAR}\n")
print("        " + " ".join(f"P{p}".center(8) for p in range(N_POS)))
for L in range(N_LAYERS + 1):
    name = "embed" if L == 0 else f"L{L}"
    row = " ".join(f"{recovery[L, p]:+.3f}".center(8) for p in range(N_POS))
    print(f"  {name:<6} {row}")

hot = recovery[L_STAR + 1, P_STAR]
print(f"\n  recovery au site porteur (L{L_STAR + 1}, P{P_STAR}) = {hot:+.3f}")
# Site porteur : recovery fort. Positions != P_STAR : ~0 (aucune info-fait).
assert hot > 0.5, "le site porteur doit avoir un recovery > 0.5 (causal)"
for P in range(N_POS):
    if P == P_STAR:
        continue
    assert float(np.max(np.abs(recovery[1:, P]))) < 0.1, f"P{P} non porteuse doit ~0"
print("  -> patcher l'activation clean ne restaure A QU'a (L*, P*) : c'est la")
print("     que le fait est porte CAUSALEMENT (denoising).")

# --- Probe (correlation) vs patching (causalite) ---
# On entraine un linear probe a chaque couche, a la position porteuse, pour
# detecter clean(1)/corrupted(0). Le probe est eleve des que le fait est
# present (correlation), MAIS patcher avant l'injection n'a aucun effet causal.
def collect_probe_feats(L_probe):
    """Active a (L_probe, P_STAR), label 1 si clean. Le jitter rend la frontiere
    apprise non triviale."""
    rng = np.random.default_rng(2024)
    feats, labels = [], []
    for i in range(200):
        is_clean = (i % 2 == 0)
        fact = d_clean if is_clean else d_corr
        jitter = rng.normal(0, 0.05, size=(N_POS, D_MODEL))
        h = embed0 + jitter
        for L in range(N_LAYERS):
            h = h + base_contrib[L]
            if L == L_STAR:
                h[P_STAR] = h[P_STAR] + FACT_STRENGTH * fact
            if L == L_probe - 1:
                feats.append(h[P_STAR].copy())
                labels.append(1 if is_clean else 0)
                break
    return np.array(feats), np.array(labels)


def linear_probe_acc(X, y, n_iters=300, lr=0.3):
    """Logistic regression LINEAIRE (GD), accuracy train. On garde le probe
    lineaire (regle du cours) : sinon il 'trouve toujours quelque chose'."""
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    yf = y.astype(np.float64)
    for _ in range(n_iters):
        probs = 1.0 / (1.0 + np.exp(-np.clip(X @ w + b, -30, 30)))
        w -= lr * (X.T @ (probs - yf) / n)
        b -= lr * float(np.mean(probs - yf))
    return float(np.mean((X @ w + b > 0).astype(int) == y))


# Probe a une couche AVANT l'injection (L_STAR) vs APRES (L_STAR+1).
fX_pre, fy_pre = collect_probe_feats(L_STAR)        # avant injection : info absente
fX_post, fy_post = collect_probe_feats(L_STAR + 1)  # apres injection : info presente
acc_pre = linear_probe_acc(fX_pre, fy_pre)
acc_post = linear_probe_acc(fX_post, fy_post)
recovery_pre = float(np.max(np.abs(recovery[1:L_STAR + 1, P_STAR])))  # patch avant injection
print(f"\n  Probe (correlation) a P{P_STAR} :")
print(f"    avant injection (L{L_STAR})   accuracy = {acc_pre:.3f}  | patching recovery = {recovery_pre:+.3f}")
print(f"    apres injection (L{L_STAR + 1})   accuracy = {acc_post:.3f}  | patching recovery = {hot:+.3f}")
assert acc_post > 0.9, "le probe doit decoder l'info une fois injectee"
print("  -> denoising vs noising (Heimersheim & Nanda 2024) : ici on fait du")
print("     DENOISING (clean -> corrupted, 'qu'est-ce qui SUFFIT a restaurer ?').")
print("     Le noising (corrupted -> clean) demande 'qu'est-ce qui est NECESSAIRE")
print("     ?'. Les deux ne donnent pas la meme carte ; on normalise le recovery")
print("     pour comparer des sites a echelles de logits differentes.")


# ============================================================================
# EXERCISE 5 — Induction head : circuit a 2 layers & effet du contexte
# ============================================================================
# But : prev-token head (ecrit le token t-1 dans un canal du residual) +
# induction head (prefix matching exprime comme une ATTENTION : scores + softmax,
# pas un 'if'). Puis l'accuracy in-context qui monte avec le nb de repetitions.

print("\n" + "=" * 70)
print("EXERCISE 5 : induction head — circuit 2-layers (attention) & contexte")
print("=" * 70)

V5 = 6                     # vocab
# Residual = [ token_onehot | prev_signal | position ] (sous-espaces disjoints).
D_TOK = V5
D_PREV = V5
D_POS = 8
OFF_TOK, OFF_PREV, OFF_POS = 0, D_TOK, D_TOK + D_PREV
D_RES = D_TOK + D_PREV + D_POS


def embed_tokens(tokens):
    """Residual initial (T, D_RES) : one-hot token + encodage positionnel sin/cos."""
    T = len(tokens)
    res = np.zeros((T, D_RES))
    for t, tok in enumerate(tokens):
        res[t, OFF_TOK + tok] = 1.0
        for k in range(D_POS):
            freq = 1.0 / (10000 ** (2 * (k // 2) / D_POS))
            res[t, OFF_POS + k] = np.sin(t * freq) if k % 2 == 0 else np.cos(t * freq)
    return res


def causal_mask(T):
    """Masque causal additif (T, T) : -inf au-dessus de la diagonale."""
    m = np.zeros((T, T))
    m[np.triu_indices(T, k=1)] = -1e9
    return m


def prev_token_head(res):
    """LAYER 1 — previous-token head, comme une VRAIE attention.
    Q/K positionnels biaises pour cibler s = t-1 (softmax, pas un 'if') ; V = le
    bloc token one-hot ; W_O ECRIT le token deplace dans le bloc PREV du residual
    de t (composition via le residual stream que la head 2 lira)."""
    T = res.shape[0]
    pos = res[:, OFF_POS:OFF_POS + D_POS]
    scores = pos @ pos.T / np.sqrt(D_POS)        # terme QK positionnel
    shift = np.zeros((T, T))
    for t in range(1, T):
        shift[t, t - 1] = 10.0                    # biais net 'regarde t-1'
    attn = softmax_rows(scores + shift + causal_mask(T))  # softmax = vraie attention
    moved = attn @ res[:, OFF_TOK:OFF_TOK + D_TOK]         # transporte le token de t-1
    out = res.copy()
    out[:, OFF_PREV:OFF_PREV + D_PREV] += moved          # OV circuit -> bloc PREV
    return out


def induction_head(res):
    """LAYER 2 — induction head, comme une VRAIE attention.
    Q = token courant ; K = prev_signal (ecrit par la head 1) ; score[t,s] est
    maximal quand tokens[s-1] == token courant (prefix matching, via softmax).
    V = token de s ; W_O ecrit la copie dans le bloc TOKEN de t (lu par
    l'unembedding) -> on predit le token qui SUIVAIT l'occurrence precedente."""
    T = res.shape[0]
    Q = res[:, OFF_TOK:OFF_TOK + D_TOK]
    K = res[:, OFF_PREV:OFF_PREV + D_PREV]
    attn = softmax_rows(5.0 * (Q @ K.T) + causal_mask(T))  # matching = scores+softmax
    copied = attn @ res[:, OFF_TOK:OFF_TOK + D_TOK]
    out = res.copy()
    out[:, OFF_TOK:OFF_TOK + D_TOK] += copied
    return out


def induction_forward(tokens):
    """Forward 2-layers ; logits a la derniere position = bloc token de la sortie."""
    res = embed_tokens(tokens)
    res = prev_token_head(res)        # L1 ecrit dans PREV
    res = induction_head(res)         # L2 lit PREV, ecrit dans TOK
    return res[-1, OFF_TOK:OFF_TOK + D_TOK]


def expected_target(tokens):
    """Cible d'induction : token suivant la derniere occurrence anterieure du courant."""
    last = tokens[-1]
    for p in range(len(tokens) - 2, 0, -1):
        if tokens[p - 1] == last:
            return tokens[p]
    return last


# --- Validation sur quelques sequences repetees ---
test_seqs = [
    [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3],   # -> attend 4
    [0, 5, 2, 0, 5, 2, 0, 5],            # -> attend 2
    [3, 1, 4, 3, 1, 4, 3, 1],            # -> attend 4
]
print(f"\n  {'Sequence':<32} {'pred':<6} {'expected':<10} {'OK'}")
print("  " + "-" * 54)
n_circuit_ok = 0
for seq in test_seqs:
    pred = int(np.argmax(induction_forward(seq)))
    exp = expected_target(seq)
    ok = pred == exp
    n_circuit_ok += ok
    print(f"  {str(seq)[:30]:<32} {pred:<6} {exp:<10} {'OK' if ok else 'FAIL'}")
    assert ok, f"l'induction head doit predire {exp} sur {seq}"

# Baselines : copie naive (token courant) et bigram (token le plus frequent apres).
def naive_copy_pred(tokens):
    return tokens[-1]


def bigram_pred(tokens):
    """Token le plus frequent apres le courant DANS la sequence (hors derniere pos)."""
    last = tokens[-1]
    from collections import Counter
    nxt = Counter()
    for p in range(len(tokens) - 1):
        if tokens[p] == last:
            nxt[tokens[p + 1]] += 1
    return nxt.most_common(1)[0][0] if nxt else last


naive_ok = sum(naive_copy_pred(s) == expected_target(s) for s in test_seqs)
bigram_ok = sum(bigram_pred(s) == expected_target(s) for s in test_seqs)
print(f"\n  Circuit induction : {n_circuit_ok}/{len(test_seqs)} corrects")
print(f"  Baseline copie naive du token courant : {naive_ok}/{len(test_seqs)}")
print(f"  Baseline bigram (plus frequent apres) : {bigram_ok}/{len(test_seqs)}")
assert n_circuit_ok > naive_ok, "le circuit doit battre la copie naive"

# --- Effet de la longueur de contexte (nb de repetitions du motif) ---
print(f"\n  Effet du contexte (accuracy in-context vs nb de repetitions du motif) :")
print(f"    {'repetitions':<14} {'accuracy':<10}")
print("    " + "-" * 26)


def context_accuracy(n_reps, n_seq=60, seed=8):
    """Genere des sequences = motif repete n_reps fois + debut du motif, et
    mesure l'accuracy d'induction. Plus de repetitions = matching plus fiable."""
    rng = np.random.default_rng(seed)
    correct = 0
    for _ in range(n_seq):
        base_len = int(rng.integers(2, 4))
        pattern = list(rng.integers(0, V5, size=base_len))
        seq = pattern * n_reps
        cut = int(rng.integers(1, base_len + 1))
        seq = seq + pattern[:cut]
        if len(seq) < 3:
            continue
        if int(np.argmax(induction_forward(seq))) == expected_target(seq):
            correct += 1
    return correct / n_seq


ctx_accs = {}
for n_reps in (1, 2, 3, 4):
    ctx_accs[n_reps] = context_accuracy(n_reps)
    print(f"    {n_reps:<14} {ctx_accs[n_reps]:<10.3f}")
# Avec 1 seule occurrence, pas de motif anterieur a matcher -> bas. Avec 2+
# repetitions, le prefix matching trouve une cible -> l'accuracy monte.
assert ctx_accs[3] > ctx_accs[1] and ctx_accs[4] >= ctx_accs[2], \
    "plus de repetitions -> meilleure copie in-context"
print("  -> plus le contexte contient de repetitions, plus le matching est")
print("     fiable : c'est la racine de l'in-context learning (Olsson 2022).")

print("\n  POURQUOI 2 layers (et pas 1) : la head 2 doit comparer le token COURANT")
print("  au token PRECEDANT chaque position passee. Une seule head ne peut pas a")
print("  la fois (a) calculer 'quel est le token precedent' ET (b) s'en servir")
print("  pour le matching. Le residual stream sert de MEMOIRE/canal : la head 1")
print("  y ECRIT le prev-token, la head 2 l'y LIT -> composition de circuits.")


# ============================================================================
# EXERCISE 6 — Sparse autoencoder minimal : L1 vs TopK
# ============================================================================
# But : entrainer un SAE L1 sur des activations superposees, mesurer recovery +
# dead features, puis comparer a un SAE TopK (Gao 2024) qui evite le shrinkage.

print("\n" + "=" * 70)
print("EXERCISE 6 : sparse autoencoder minimal — L1 vs TopK")
print("=" * 70)

# --- Generer des activations superposees (5 features dans 2 dims, sparses) ---
# Reprend le setup superposition : m features quasi-orthogonales packees dans d<m.
M_FEAT, D_ACT, K_TRUE, N_SAMP = 5, 2, 2, 4000
_rng6 = np.random.default_rng(9)
D_true = _rng6.normal(0, 1.0, size=(M_FEAT, D_ACT))
D_true = D_true / (np.linalg.norm(D_true, axis=1, keepdims=True) + 1e-9)
codes = np.zeros((N_SAMP, M_FEAT))
for i in range(N_SAMP):
    active = _rng6.choice(M_FEAT, size=K_TRUE, replace=False)  # sparsite controlee
    codes[i, active] = _rng6.uniform(0.5, 1.0, size=K_TRUE)
acts = codes @ D_true                          # (N_SAMP, D_ACT) superposition reelle
N_SAE = 8                                       # sur-completion (n_sae > m)
print(f"\n  Activations : {M_FEAT} features sparses dans {D_ACT} dims, "
      f"k={K_TRUE}/exemple ; n_sae={N_SAE}")


def normalize_dec(W_dec):
    """Normalise chaque ligne (colonne de dictionnaire) a norm 1.
    POURQUOI (trick Anthropic 2023) : sans ca le modele triche en gonflant W_dec
    et reduisant f pour CONTOURNER la penalite de sparsite."""
    return W_dec / (np.linalg.norm(W_dec, axis=1, keepdims=True) + 1e-8)


def topk_mask(f, k):
    """Garde les k plus grandes activations par ligne, le reste a 0.
    POURQUOI TopK : la sparsite est imposee EXACTEMENT par k (pas par une
    penalite qui retrecit les magnitudes) -> pas de shrinkage."""
    if k >= f.shape[1]:
        return f
    thresh_idx = np.argpartition(f, -k, axis=1)[:, :-k]   # (n_sae - k) plus petites
    out = f.copy()
    out[np.arange(f.shape[0])[:, None], thresh_idx] = 0.0
    return out


def train_sae(acts, n_sae, mode, l1_lambda=0.05, k=K_TRUE,
              n_iters=4000, lr=0.03, seed=3):
    """SAE from scratch. mode='l1' : ReLU + penalite L1. mode='topk' : ReLU +
    masque top-K (pas de L1). Backward manuel (MSE [+ L1]) + normalisation des
    colonnes du decoder a chaque step."""
    n, d = acts.shape
    rng = np.random.default_rng(seed)
    W_enc = rng.normal(0, 0.3, size=(d, n_sae))
    b_enc = np.zeros(n_sae)
    if mode == "topk":
        # Init decoder ALIGNEE sur des activations reelles (anti dead-feature).
        W_dec = acts[rng.choice(n, size=n_sae, replace=True)].copy() \
            + rng.normal(0, 0.05, size=(n_sae, d))
    else:
        W_dec = rng.normal(0, 0.3, size=(n_sae, d))
    b_dec = np.zeros(d)
    for _ in range(n_iters):
        W_dec = normalize_dec(W_dec)
        pre = acts @ W_enc + b_enc
        f_full = np.maximum(0, pre)
        f = topk_mask(f_full, k) if mode == "topk" else f_full
        recon = f @ W_dec + b_dec
        err = recon - acts
        grad_recon = 2 * err / n
        grad_b_dec = np.sum(grad_recon, axis=0)
        grad_W_dec = f.T @ grad_recon
        grad_f = grad_recon @ W_dec.T
        if mode == "l1":
            grad_f += l1_lambda * (f > 0).astype(np.float64) / n   # sous-gradient L1
            grad_pre = grad_f * (pre > 0)
        else:
            # TopK : le gradient ne passe que par les entrees survivantes du topk.
            grad_pre = grad_f * (f > 0) * (pre > 0)
        grad_W_enc = acts.T @ grad_pre
        grad_b_enc = np.sum(grad_pre, axis=0)
        W_enc -= lr * grad_W_enc
        b_enc -= lr * grad_b_enc
        W_dec -= lr * grad_W_dec
        b_dec -= lr * grad_b_dec
    return W_enc, b_enc, normalize_dec(W_dec), b_dec


def evaluate_sae(W_enc, b_enc, W_dec, b_dec, mode):
    """Renvoie (n_recovered, n_dead, mag_ratio, recon_loss).
    n_recovered : features ground-truth matchees 1-1 par cosine > 0.5 (et active).
    mag_ratio : ||recon|| / ||acts|| moyen (proxy shrinkage : <1 = sous-estime)."""
    pre = acts @ W_enc + b_enc
    f = topk_mask(np.maximum(0, pre), K_TRUE) if mode == "topk" else np.maximum(0, pre)
    recon = f @ W_dec + b_dec
    active_rate = np.mean(f > 1e-4, axis=0)
    n_dead = int(np.sum(active_rate < 1e-3))
    # Matching 1-1 par cosine (decroissant, sans reuse).
    pairs = []
    for s in range(W_dec.shape[0]):
        if active_rate[s] < 1e-3:
            continue
        sv = W_dec[s]
        sn = np.linalg.norm(sv) + 1e-9
        for g in range(M_FEAT):
            cos = float(np.dot(sv, D_true[g]) / (sn * (np.linalg.norm(D_true[g]) + 1e-9)))
            pairs.append((cos, s, g))
    pairs.sort(reverse=True)
    used_s, used_g, recovered = set(), set(), set()
    for cos, s, g in pairs:
        if cos < 0.5:
            break
        if s in used_s or g in used_g:
            continue
        used_s.add(s)
        used_g.add(g)
        recovered.add(g)
    mag_ratio = float(np.mean(np.linalg.norm(recon, axis=1) /
                              (np.linalg.norm(acts, axis=1) + 1e-9)))
    recon_loss = float(np.mean(np.sum((recon - acts) ** 2, axis=1)))
    return len(recovered), n_dead, mag_ratio, recon_loss


l1_params = train_sae(acts, N_SAE, mode="l1", l1_lambda=0.05)
topk_params = train_sae(acts, N_SAE, mode="topk", k=K_TRUE)
rec_l1, dead_l1, mag_l1, loss_l1 = evaluate_sae(*l1_params, mode="l1")
rec_tk, dead_tk, mag_tk, loss_tk = evaluate_sae(*topk_params, mode="topk")

print(f"\n  {'SAE':<10} {'recovered':<12} {'dead':<8} {'mag_ratio':<12} {'recon_loss':<12}")
print("  " + "-" * 56)
print(f"  {'L1 naif':<10} {rec_l1}/{M_FEAT:<10} {dead_l1}/{N_SAE:<6} {mag_l1:<12.3f} {loss_l1:<12.4f}")
print(f"  {'TopK':<10} {rec_tk}/{M_FEAT:<10} {dead_tk}/{N_SAE:<6} {mag_tk:<12.3f} {loss_tk:<12.4f}")

# Inegalites attendues par construction.
assert rec_tk >= rec_l1, "TopK doit recuperer AU MOINS autant de features que L1"
assert dead_tk <= dead_l1, "TopK doit avoir AU PLUS autant de dead features que L1"
assert mag_tk >= mag_l1 - 1e-6, "TopK ne doit pas sous-estimer plus que L1 (shrinkage)"
print(f"\n  -> TopK recupere >= de features ({rec_tk} vs {rec_l1}) et a <= de dead")
print(f"     features ({dead_tk} vs {dead_l1}) que L1.")
print(f"  -> Shrinkage : magnitude reconstruite L1={mag_l1:.3f} < TopK={mag_tk:.3f}")
print("     (1.0 = pas de shrinkage). POURQUOI : la penalite L1 pousse TOUTES")
print("     les activations vers 0 -> elle SOUS-ESTIME les magnitudes et tue")
print("     des unites (dead features). TopK impose la sparsite par K sans")
print("     penaliser la magnitude -> pas de shrinkage (Gao/OpenAI 2024).")
print("\n  Caveat honnete (comme 02-code) : toy minimal. On reproduit le MECANISME")
print("  (TopK > L1 sur shrinkage + dead features), pas un resultat a l'echelle.")


# ============================================================================
# Fin
# ============================================================================
print("\n" + "=" * 70)
print("Done (MEDIUM). Toutes les assertions passent.")
print("=" * 70)
