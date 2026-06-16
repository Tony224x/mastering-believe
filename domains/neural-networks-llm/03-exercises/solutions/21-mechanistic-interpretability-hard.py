"""
Solutions HARD — Jour 21 : Mechanistic interpretability
=======================================================
Exercices 7, 8, 9 (hard). Pur NumPy + stdlib, comme
02-code/21-mechanistic-interpretability.py. Chaque etape non triviale est
commentee avec le POURQUOI. La ou un resultat est numeriquement exact (par
construction du toy), on pose une assertion.

  7. Causal mediation complet : carte de recovery (layer x position) par
     activation patching denoising. Assertions : site porteur > 0.5, sites
     non pertinents ~ 0. Demonstration probe (correlation) != patching
     (causalite).
  8. Induction head en VRAIE attention Q/K/V : prev-token head (layer 1) +
     induction head (layer 2) composees via le residual stream, puis phase
     transition (accuracy vs strength).
  9. TopK SAE (Gao/OpenAI 2024) vs L1 naif : sur un ground-truth controle,
     TopK recupere >= de features et a <= de dead features (inegalites
     asserties).

Run: python 03-exercises/solutions/21-mechanistic-interpretability-hard.py
"""

from __future__ import annotations
import sys
import io
import numpy as np

# Stdout en UTF-8 (Windows-friendly), comme 02-code/21.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Seed global. Tout est deterministe.
np.random.seed(42)


def softmax_vec(x: np.ndarray) -> np.ndarray:
    """Softmax stable 1D (subtract max). Identique a 02-code/21 PART 2."""
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def softmax_rows(x: np.ndarray) -> np.ndarray:
    """Softmax stable applique par ligne (axis=-1) pour les scores d'attention."""
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


# ============================================================================
# EXERCISE 7 — Causal mediation complet : carte de recovery (layer x position)
#   But : un fait vit a un (L*, P*) connu du residual stream. Le patching
#   denoising (copier l'activation clean dans le run corrupted) ne doit faire
#   remonter le logit cible QU'A ce site. Probe eleve partout != effet causal.
# ============================================================================
print("=" * 70)
print("EXERCISE 7 : causal mediation — carte de recovery (layer x position)")
print("=" * 70)

# --- Geometrie du toy decoder ---
D_MODEL = 24
N_LAYERS = 5          # couches 0..4 ; le residual additif h += contrib
N_POS = 4             # positions (sequence courte traceable)
VOCAB = 12
L_STAR = 2            # couche a partir de laquelle le fait est injecte
P_STAR = N_POS - 1    # position porteuse = la position LUE pour les logits finaux
T_CLEAN = 3           # token cible du run "clean"
T_CORR = 8            # token cible du run "corrupted"
P_DISTRACT = 0        # position distractrice : info presente mais inerte

_rng7 = np.random.default_rng(7)
# Unembedding (d_model, vocab) : une colonne = direction d'un token.
W_U = _rng7.normal(0, 1.0, size=(D_MODEL, VOCAB))

# Direction-fait = colonne d'unembedding normalisee du token cible. Ecrire
# cette direction dans le residual *pousse* le logit du token vers le haut :
# c'est ce qui rend le fait causalement effectif a la sortie.
def token_dir(tok: int) -> np.ndarray:
    v = W_U[:, tok]
    return v / (np.linalg.norm(v) + 1e-8)

d_clean = token_dir(T_CLEAN)
d_corr = token_dir(T_CORR)

# Bruit de fond commun aux deux runs : identique => il ne porte AUCUNE info
# discriminante sur le fait (clean vs corrupted). On le seed une fois.
_rng_bg = np.random.default_rng(123)
# base_contrib[layer][pos] = vecteur ajoute au residual a chaque (layer, pos).
# Meme tenseur pour clean et corrupted (le seul ecart sera la direction-fait).
base_contrib = _rng_bg.normal(0, 0.25, size=(N_LAYERS, N_POS, D_MODEL))

# Embedding initial du residual (position 0 du forward = "embed"), commun.
_rng_emb = np.random.default_rng(99)
embed0 = _rng_emb.normal(0, 0.2, size=(N_POS, D_MODEL))

# Force d'injection du fait : assez grande pour dominer la sortie a P*.
FACT_STRENGTH = 4.0


def run_decoder(fact_dir: np.ndarray, patch=None):
    """
    Forward du toy decoder additif et trace des activations.

    Residual h[pos] commence a embed0[pos], puis a chaque couche L on ajoute :
      - base_contrib[L][pos]  (bruit commun, non informatif),
      - a (L == L_STAR, P_STAR) UNIQUEMENT : FACT_STRENGTH * fact_dir.
        Le fait est injecte a UNE seule couche -> il est localise a (L*, P*),
        ce qui rend la carte causale tranchante (le patching restaure tout ou
        rien selon qu'on patche avant ou apres l'injection).

    Args:
      fact_dir : direction-fait injectee a P* a partir de L* (clean ou corr).
      patch : optionnel dict {(L, P): vecteur} -> apres avoir calcule le
              residual a la couche L et position P, on REMPLACE h[P] par le
              vecteur fourni, puis on continue le forward. C'est le coeur du
              denoising : on injecte une activation saine dans un run corrompu.

    Returns:
      acts : liste de longueur N_LAYERS+1 ; acts[L] est (N_POS, D_MODEL),
             snapshot du residual APRES la couche L (acts[0] = embed initial).
      logits_final : (VOCAB,) logits a la derniere position (h_final[-1] @ W_U).
    """
    patch = patch or {}
    h = embed0.copy()                       # (N_POS, D_MODEL)
    acts = [h.copy()]                        # acts[0] = embedding initial
    for L in range(N_LAYERS):
        # 1) bruit commun a toutes les positions
        h = h + base_contrib[L]
        # 2) injection du fait a (== L_STAR, P_STAR) : UNE seule couche
        if L == L_STAR:
            h[P_STAR] = h[P_STAR] + FACT_STRENGTH * fact_dir
        # 3) DENOISING : si un patch cible cette couche, on remplace
        #    l'activation a la position visee APRES recalcul de la couche.
        #    Re-tourner le forward a partir de ce point = les couches > L
        #    continueront avec l'activation patchee (effet causal en aval).
        for (pL, pP), pvec in patch.items():
            if pL == L:
                h[pP] = pvec
        acts.append(h.copy())
    logits_final = acts[-1][-1] @ W_U        # logits a la derniere position
    return acts, logits_final


# --- Runs clean & corrupted ---
acts_clean, logits_clean = run_decoder(d_clean)
acts_corr, logits_corr = run_decoder(d_corr)

top_clean = int(np.argmax(logits_clean))
top_corr = int(np.argmax(logits_corr))
print(f"\n  clean top-1   = t{top_clean} (attendu t{T_CLEAN})")
print(f"  corrupted top-1 = t{top_corr} (attendu t{T_CORR})")
# Le fait domine la sortie : clean->T_CLEAN, corrupted->T_CORR (par construction).
assert top_clean == T_CLEAN, "clean doit predire T_CLEAN en top-1"
assert top_corr == T_CORR, "corrupted doit predire T_CORR en top-1"

# logit du token cible (T_CLEAN) dans chaque run de reference.
lc = float(logits_clean[T_CLEAN])
lr = float(logits_corr[T_CLEAN])
denom = (lc - lr)
denom = denom if abs(denom) > 1e-9 else 1e-9
print(f"  logit(T_CLEAN) clean={lc:+.3f}, corrupted={lr:+.3f}")

# --- Patching denoising sur toute la grille (layer x position) ---
# Recovery normalise : on injecte l'activation CLEAN au site (L,P) dans le run
# corrompu, on re-tourne le forward, et on mesure de combien le logit de
# T_CLEAN remonte vers sa valeur clean. 1.0 = totalement restaure, 0.0 = inerte.
recovery = np.zeros((N_LAYERS + 1, N_POS))
for L in range(1, N_LAYERS + 1):          # acts[0] = embed (pas une "couche")
    for P in range(N_POS):
        clean_act = acts_clean[L][P]      # snapshot APRES la couche L
        # patch indexe par la couche L-1 dans run_decoder (snapshot acts[L]
        # = etat APRES execution de la couche d'indice L-1).
        _, logits_p = run_decoder(d_corr, patch={(L - 1, P): clean_act})
        lp = float(logits_p[T_CLEAN])
        recovery[L, P] = (lp - lr) / denom

print(f"\n  Carte de recovery (lignes = layer snapshot, colonnes = position)")
print(f"  site porteur attendu : L>={L_STAR+1} a P={P_STAR}\n")
header = "        " + " ".join(f"P{p}".center(8) for p in range(N_POS))
print(header)
for L in range(N_LAYERS + 1):
    name = "embed" if L == 0 else f"L{L}"
    row = " ".join(f"{recovery[L, p]:+.3f}".center(8) for p in range(N_POS))
    print(f"  {name:<6} {row}")

# --- Assertions causales ---
# Site porteur : a P_STAR, des que la couche snapshot >= L_STAR+1 (le fait a
# ete injecte), patcher l'activation clean restaure le logit -> recovery > 0.5.
hot = recovery[L_STAR + 1, P_STAR]
print(f"\n  recovery au site porteur (L{L_STAR+1}, P{P_STAR}) = {hot:+.3f}")
assert hot > 0.5, "le site porteur doit avoir un recovery > 0.5 (causal)"

# Positions != P_STAR : aucune info-fait -> recovery ~ 0 a toutes les couches.
for P in range(N_POS):
    if P == P_STAR:
        continue
    max_off = float(np.max(np.abs(recovery[1:, P])))
    assert max_off < 0.1, f"position P{P} (non porteuse) doit rester ~0"
print("  positions non porteuses : recovery < 0.1 partout (OK)")

# Couches AVANT l'injection a P_STAR (snapshot < L_STAR+1) : le fait n'y est
# pas encore -> patcher l'activation clean n'apporte rien -> recovery ~ 0.
pre = float(np.max(np.abs(recovery[1:L_STAR + 1, P_STAR])))
assert pre < 0.1, "avant l'injection (couches < L*), recovery doit etre ~0"
print(f"  couches < L* a P{P_STAR} : recovery < 0.1 (le fait n'y est pas encore)")
print("  -> Le patching localise PRECISEMENT le fait : un seul (L*, P*) chaud.")


# --- Probe (correlation) vs patching (causalite) ---
# On fabrique un mini-dataset binaire : moitie clean (label 1 = T_CLEAN porte),
# moitie corrupted (label 0). On regarde les activations a une couche tardive
# pour (a) la position porteuse P_STAR et (b) une position DISTRACTRICE ou l'on
# COPIE la direction-fait dans le residual SANS la router vers la sortie
# (feature presente mais causalement inerte : le forward final ne la lit pas).
def collect_probe_data(L_probe: int, P_probe: int, inert_copy: bool):
    """
    Genere n_samp exemples (moitie clean, moitie corrupted) et renvoie
    (features, labels) = activations a (L_probe, P_probe), label 1 si clean.

    inert_copy=True : on injecte en plus la direction-fait correcte a P_probe
    (donc l'info est LINEAIREMENT presente -> le probe la verra), mais comme
    P_probe != P_STAR cette direction n'influence PAS le logit final dans
    run_decoder -> elle est causalement inerte. C'est le piege du cours.
    """
    rng = np.random.default_rng(2024)
    n_samp = 200
    feats, labels = [], []
    for i in range(n_samp):
        is_clean = (i % 2 == 0)
        fact = d_clean if is_clean else d_corr
        # Petit jitter par exemple pour que le probe apprenne une vraie frontiere.
        jitter = rng.normal(0, 0.05, size=(N_POS, D_MODEL))
        # Forward custom avec jitter et eventuelle copie inerte.
        h = embed0 + jitter
        for L in range(N_LAYERS):
            h = h + base_contrib[L]
            if L == L_STAR:                          # meme injection que run_decoder
                h[P_STAR] = h[P_STAR] + FACT_STRENGTH * fact
            if inert_copy and L == L_STAR:
                # Copie la MEME direction-fait a P_probe : info presente,
                # mais P_probe != P_STAR donc ignoree par la sortie (le forward
                # ne lit QUE la position P_STAR pour les logits) -> inerte.
                sign = 1.0 if is_clean else -1.0
                h[P_probe] = h[P_probe] + sign * FACT_STRENGTH * d_clean
            if L == L_probe - 1:
                feats.append(h[P_probe].copy())
                labels.append(1 if is_clean else 0)
                break
    return np.array(feats), np.array(labels)


def linear_probe_acc(features: np.ndarray, labels: np.ndarray,
                     n_iters: int = 300, lr: float = 0.3) -> float:
    """Logistic regression lineaire (GD), accuracy train. Cf 02-code/21 PART 1.

    On garde le probe LINEAIRE (pas de MLP) : c'est la regle du cours, sinon il
    'trouve toujours quelque chose' et la conclusion n'est plus interpretable.
    """
    n, d = features.shape
    w = np.zeros(d)
    b = 0.0
    y = labels.astype(np.float64)
    for _ in range(n_iters):
        logits = features @ w + b
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        w -= lr * (features.T @ (probs - y) / n)
        b -= lr * float(np.mean(probs - y))
    preds = (features @ w + b > 0).astype(np.int64)
    return float(np.mean(preds == labels))

L_PROBE = N_LAYERS                       # couche tardive
# (a) probe a la position porteuse P_STAR.
fX, fy = collect_probe_data(L_PROBE, P_STAR, inert_copy=False)
acc_carrier = linear_probe_acc(fX, fy)
# (b) probe a la position distractrice avec copie inerte de la direction-fait.
dX, dy = collect_probe_data(L_PROBE, P_DISTRACT, inert_copy=True)
acc_distract = linear_probe_acc(dX, dy)

print(f"\n  Probe (correlation) :")
print(f"    accuracy a P{P_STAR} (porteuse)      = {acc_carrier:.3f}")
print(f"    accuracy a P{P_DISTRACT} (distractrice inerte) = {acc_distract:.3f}")
# Le probe est ELEVE aux deux positions (l'info est lineairement presente).
assert acc_carrier > 0.9, "probe doit decoder l'info a la position porteuse"
assert acc_distract > 0.9, "probe doit decoder l'info copiee a la distractrice"

print(f"  Patching (causalite) :")
print(f"    recovery a P{P_STAR} = {recovery[L_STAR+1, P_STAR]:+.3f} (effet fort)")
print(f"    recovery a P{P_DISTRACT} = {float(np.max(np.abs(recovery[1:, P_DISTRACT]))):+.3f} (nul)")
print("  -> CONCLUSION : le probe trouve l'info aux DEUX positions (correlation),")
print("     mais seul P* a un effet causal au patching. Probing != causalite.")


# ============================================================================
# EXERCISE 8 — Induction head en VRAIE attention Q/K/V + phase transition
#   But : prev-token head (L1) + induction head (L2) composees via le residual
#   stream, implementees comme de vraies attentions (Q/K/V/W_O, softmax causal),
#   pas le 'if' du 02-code ni un simple softmax sur un canal. Puis phase
#   transition : accuracy de copie in-context qui saute avec la force du circuit.
# ============================================================================
print("\n" + "=" * 70)
print("EXERCISE 8 : induction head en VRAIE attention Q/K/V + phase transition")
print("=" * 70)

V8 = 6           # taille du vocab
D_TOK = V8       # bloc "token courant" = one-hot du token (dim V8)
D_PREV = V8      # bloc "previous-token signal" ecrit par la head 1 (dim V8)
D_OUT = V8       # bloc "sortie d'induction" ecrit par la head 2 (dim V8)
D_POS = 8        # bloc positionnel (pour que la head 1 cible t-1)
# Residual = [ token_onehot | prev_signal | out_signal | position ]
#   -> sous-espaces DISJOINTS. Le bloc OUT est crucial : l'unembedding lit la
#   sortie ECRITE par l'OV de la head 2 (le token copie), PAS le bloc token
#   courant. Lire le bloc token courant reviendrait a une "copie naive" du token
#   present et ferait toujours gagner le token courant -> on doit router la
#   prediction par l'OV de l'induction head, comme dans un vrai circuit.
OFF_TOK = 0
OFF_PREV = D_TOK
OFF_OUT = D_TOK + D_PREV
OFF_POS = D_TOK + D_PREV + D_OUT
D_RES = D_TOK + D_PREV + D_OUT + D_POS

# Gain de l'OV/QK de l'induction head. Une vraie head a un produit QK appris de
# grande amplitude : sur des cles one-hot (produit scalaire = 1 au match, 0
# sinon), il faut amplifier les scores pour que le softmax PIQUE sur la (les)
# position(s) qui matchent. A strength faible le softmax reste plat (~hasard) ;
# a strength fort il devient quasi one-hot -> copie fiable. C'est le scalaire
# qui pilote la phase transition d'Olsson.
INDUCTION_QK_GAIN = 8.0

# Embedding token : one-hot dans le bloc token. Unembedding : lit le bloc token
# de la sortie -> les logits sont l'identite du token copie.
def embed_tokens(tokens: list[int], max_pos: int) -> np.ndarray:
    """Construit le residual initial (T, D_RES) : one-hot token + encodage pos."""
    T = len(tokens)
    res = np.zeros((T, D_RES))
    for t, tok in enumerate(tokens):
        res[t, OFF_TOK + tok] = 1.0                       # bloc token courant
        # Encodage positionnel simple : sin/cos sur le bloc position.
        for k in range(D_POS):
            freq = 1.0 / (10000 ** (2 * (k // 2) / D_POS))
            res[t, OFF_POS + k] = np.sin(t * freq) if k % 2 == 0 else np.cos(t * freq)
    return res


def causal_mask(T: int) -> np.ndarray:
    """Masque causal additif (T, T) : -inf au-dessus de la diagonale."""
    m = np.zeros((T, T))
    idx = np.triu_indices(T, k=1)
    m[idx] = -1e9
    return m


def prev_token_head(res: np.ndarray, strength: float) -> np.ndarray:
    """
    LAYER 1 — previous-token head (attention reelle).

    La position t doit attendre la position t-1, puis ECRIRE l'identite du
    token t-1 dans le bloc 'prev_signal' du residual de t.

    Mecanisme attentionnel :
      - Q/K bases sur le bloc POSITION : on construit des scores ou score[t, s]
        est maximal pour s = t-1 (biais de decalage de 1). On l'implemente via
        une matrice de scores positionnels explicite (equivalent d'un QK appris
        qui aurait converge vers 'regarde une position en arriere').
      - V = bloc TOKEN one-hot de la position attendue ; W_O ecrit cette valeur
        dans le bloc PREV du residual de t (OV circuit).

    POURQUOI une vraie attention : QK = OU regarder (ici t-1), OV = QUOI ecrire
    (l'identite du token la-bas). C'est le circuit d'Olsson, pas un 'if'.
    """
    T = res.shape[0]
    # Scores positionnels : favorise s = t-1. On part du produit scalaire des
    # encodages positionnels (Q=pos[t], K=pos[s]) + un biais de decalage net.
    pos = res[:, OFF_POS:OFF_POS + D_POS]            # (T, D_POS)
    scores = pos @ pos.T / np.sqrt(D_POS)            # (T, T) terme QK positionnel
    shift_bias = np.zeros((T, T))
    for t in range(T):
        if t - 1 >= 0:
            shift_bias[t, t - 1] = 10.0 * strength   # cible franchement t-1
    scores = scores + shift_bias + causal_mask(T)
    attn = softmax_rows(scores)                       # (T, T) poids causaux
    # V = bloc token one-hot ; on transporte le token de la position attendue.
    V = res[:, OFF_TOK:OFF_TOK + D_TOK]              # (T, D_TOK)
    moved = attn @ V                                  # (T, D_TOK)
    out = res.copy()
    # W_O : ecrit la valeur deplacee dans le bloc PREV (additif, comme un
    # residual stream). C'est la composition : la head 2 lira ce bloc.
    out[:, OFF_PREV:OFF_PREV + D_PREV] += moved
    return out


def induction_head(res: np.ndarray, strength: float) -> np.ndarray:
    """
    LAYER 2 — induction head (attention reelle).

    A la position t (courante), on cherche la position s du passe dont le
    PREV_SIGNAL (ecrit par la head 1) == token courant. C'est le prefix
    matching : 's tel que tokens[s-1] == tokens[t]'. On copie alors le token
    a la position s (= le token qui SUIVAIT l'occurrence precedente).

    Mecanisme attentionnel :
      - Q = bloc TOKEN courant de la position t.
      - K = bloc PREV_SIGNAL de chaque position s (ecrit par la head 1).
      - score[t, s] = <token_t, prev_signal_s> -> maximal quand le token
        precedant s est egal au token courant t. C'est le QK matching. On
        amplifie par INDUCTION_QK_GAIN * strength pour piquer le softmax (sinon
        des scores 0/1 donnent une attention trop plate -> la masse fuit).
      - V = bloc TOKEN de s ; W_O ecrit la copie dans le bloc OUT de t (sous-
        espace DEDIE lu par l'unembedding). POURQUOI un bloc separe et pas le
        bloc TOKEN : ecrire dans le bloc TOKEN melange la prediction avec
        l'identite du token courant (poids 1.0 garanti) -> la 'copie naive' du
        token present gagnerait toujours. En routant l'OV vers OUT, la
        prediction provient UNIQUEMENT de l'attention d'induction (vrai OV).
    """
    T = res.shape[0]
    Q = res[:, OFF_TOK:OFF_TOK + D_TOK]             # token courant (one-hot)
    K = res[:, OFF_PREV:OFF_PREV + D_PREV]          # prev-token signal
    scores = INDUCTION_QK_GAIN * strength * (Q @ K.T)  # (T, T) prefix matching
    scores = scores + causal_mask(T)                 # ne regarde que le passe
    attn = softmax_rows(scores)
    V = res[:, OFF_TOK:OFF_TOK + D_TOK]             # token a copier
    copied = attn @ V                                # (T, D_TOK)
    out = res.copy()
    # W_O : ecrit la copie dans le bloc OUT (sous-espace de sortie dedie).
    out[:, OFF_OUT:OFF_OUT + D_OUT] = out[:, OFF_OUT:OFF_OUT + D_OUT] + copied
    return out


def induction_forward(tokens: list[int], strength: float) -> np.ndarray:
    """Forward complet 2-layers et logits a la derniere position (sur le bloc token)."""
    res = embed_tokens(tokens, max_pos=len(tokens))
    res = prev_token_head(res, strength=strength)
    res = induction_head(res, strength=strength)
    # Unembedding = lecture du bloc OUT (ecrit par l'OV de l'induction head)
    # -> logits sur V8 tokens. On NE lit PAS le bloc token courant : la
    # prediction doit venir de l'attention d'induction, pas du token present.
    logits_last = res[-1, OFF_OUT:OFF_OUT + D_OUT]
    return logits_last


def expected_induction_target(tokens: list[int]) -> int:
    """Cible d'induction : token suivant la derniere occurrence anterieure du courant."""
    last = tokens[-1]
    for p in range(len(tokens) - 2, 0, -1):
        if tokens[p - 1] == last:
            return tokens[p]
    return last


# --- Validation sur sequences repetees ---
test_seqs = [
    [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3],   # ABCD ABCD ABC -> attend 4
    [0, 5, 2, 0, 5, 2, 0, 5],            # ABC ABC AB -> attend 2
    [3, 1, 4, 3, 1, 4, 3, 1],            # -> attend 4
]
print(f"\n  {'Sequence':<34} {'pred':<6} {'expected':<10} {'OK':<4}")
print("  " + "-" * 56)
STRONG = 1.0
for seq in test_seqs:
    logits = induction_forward(seq, strength=STRONG)
    pred = int(np.argmax(logits))
    exp = expected_induction_target(seq)
    ok = "OK" if pred == exp else "FAIL"
    print(f"  {str(seq)[:32]:<34} {pred:<6} {exp:<10} {ok:<4}")
    # Le circuit en VRAIE attention doit predire la cible d'induction.
    assert pred == exp, f"induction head doit predire {exp} sur {seq}"

# Baseline naive : copier le token courant (echoue sur l'induction).
naive_ok = sum(1 for s in test_seqs if s[-1] == expected_induction_target(s))
print(f"\n  Baseline 'copie naive du token courant' : {naive_ok}/{len(test_seqs)} corrects")
print("  -> l'induction head bat la copie naive (elle copie le token SUIVANT")
print("     l'occurrence precedente, pas le token courant).")


# --- Phase transition : accuracy de copie in-context vs strength ---
def batch_induction_accuracy(strength: float, n_seq: int = 60, seed: int = 8) -> float:
    """Accuracy de prediction d'induction sur un batch de sequences repetees.

    On tire des motifs a tokens DISTINCTS (pas de doublon dans le motif) pour
    que la cible d'induction soit non-ambigue : ainsi un circuit parfait atteint
    ~1.0 et la phase transition est nette par construction.
    """
    rng = np.random.default_rng(seed)
    correct = 0
    for _ in range(n_seq):
        # Motif de tokens DISTINCTS (longueur 3-4), repete 2-3 fois.
        base_len = int(rng.integers(3, 5))
        pattern = list(rng.choice(V8, size=base_len, replace=False))
        reps = int(rng.integers(2, 4))
        seq = pattern * reps
        cut = int(rng.integers(1, base_len))   # finit au milieu du motif
        seq = seq + pattern[:cut]              # -> derniere position = induction
        if len(seq) < 3:
            continue
        logits = induction_forward(seq, strength=strength)
        pred = int(np.argmax(logits))
        if pred == expected_induction_target(seq):
            correct += 1
    return correct / n_seq


print(f"\n  Phase transition — accuracy in-context vs strength du circuit :")
print(f"    {'strength':<10} {'accuracy':<10}")
print("    " + "-" * 22)
strengths = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0]
accs = []
for s in strengths:
    a = batch_induction_accuracy(s)
    accs.append(a)
    print(f"    {s:<10.2f} {a:<10.3f}")

acc_low = accs[0]       # strength ~ 0 : pas de circuit
acc_high = accs[-1]     # strength fort : circuit actif
print(f"\n  acc(strength=0) = {acc_low:.3f}  ->  acc(strength=4) = {acc_high:.3f}")
# A strength 0 le QK matching est ecrase (softmax quasi-uniforme) -> hasard.
# A strength fort, le matching pique -> copie fiable. Saut net (Olsson 2022).
assert acc_low < 0.5, "a faible strength, l'accuracy doit etre ~aleatoire"
assert acc_high > 0.9, "a forte strength, le circuit copie quasi parfaitement"
assert acc_high - acc_low > 0.4, "la transition doit etre nette (saut)"
print("  -> Saut net = phase transition (Olsson 2022) : le circuit d'induction")
print("     emerge brutalement et coincide avec l'in-context learning.")
print("  -> Mecanisme = ATTENTION (QK choisit ou regarder, OV quoi copier),")
print("     pas le 'if' algorithmique du 02-code.")


# ============================================================================
# EXERCISE 9 — TopK SAE (Gao 2024) recupere PLUS de features que le L1 naif
#   But : ground-truth controle (m features sparses superposees dans d<m), on
#   entraine un SAE L1 naif et un SAE TopK ; TopK recupere >= de features et a
#   <= de dead features (inegalites asserties), et evite le shrinkage L1.
# ============================================================================
print("\n" + "=" * 70)
print("EXERCISE 9 : TopK SAE vs L1 naif (recovery de features ground-truth)")
print("=" * 70)

# --- Ground-truth controle ---
M_FEAT = 6          # nombre de features atomiques (verite terrain)
D_ACT = 4           # dimension observee : d < m -> superposition reelle
K_TRUE = 2          # nb de features actives par exemple (sparsite controlee)
N_SAMP = 4000
_rng9 = np.random.default_rng(9)

# Directions ground-truth quasi-orthogonales (normalisees). Avec m>d elles ne
# peuvent pas etre toutes orthogonales -> superposition, exactement le cadre
# d'Elhage 2022.
D_true = _rng9.normal(0, 1.0, size=(M_FEAT, D_ACT))
D_true = D_true / (np.linalg.norm(D_true, axis=1, keepdims=True) + 1e-9)

# Codes sparses : chaque exemple active K_TRUE features (magnitude positive).
codes = np.zeros((N_SAMP, M_FEAT))
for i in range(N_SAMP):
    active = _rng9.choice(M_FEAT, size=K_TRUE, replace=False)
    codes[i, active] = _rng9.uniform(0.5, 1.0, size=K_TRUE)

# Activations observees = superposition lineaire des features actives.
acts = codes @ D_true                                  # (N_SAMP, D_ACT)
print(f"\n  Ground-truth : m={M_FEAT} features dans d={D_ACT} dims, "
      f"k_true={K_TRUE} actives/exemple")
print(f"  Activations observees : shape {acts.shape} (superposition reelle)")

N_SAE = 12          # sur-completion (n_sae > m)


def normalize_dec_cols(W_dec: np.ndarray) -> np.ndarray:
    """Normalise chaque ligne de W_dec (une 'colonne de dictionnaire') a norm 1.

    POURQUOI (trick Anthropic 2023) : sans ca, le modele triche en gonflant
    W_dec et en reduisant f, ce qui contourne la penalite de sparsite.
    """
    norms = np.linalg.norm(W_dec, axis=1, keepdims=True) + 1e-8
    return W_dec / norms


def train_l1_sae(acts, n_sae, l1_lambda=0.05, n_iters=4000, lr=0.03, seed=3):
    """
    SAE L1 naif (baseline 02-code) : ReLU + penalite L1, normalisation des
    colonnes du decoder. C'est la version qui SOUFFRE de shrinkage (L1 retrecit
    les magnitudes) et de dead features (le gradient L1 pousse a 0 et certaines
    unites ne reviennent jamais).
    """
    n, d = acts.shape
    rng = np.random.default_rng(seed)
    W_enc = rng.normal(0, 0.3, size=(d, n_sae))
    b_enc = np.zeros(n_sae)
    W_dec = rng.normal(0, 0.3, size=(n_sae, d))
    b_dec = np.zeros(d)
    for _ in range(n_iters):
        W_dec = normalize_dec_cols(W_dec)
        pre = acts @ W_enc + b_enc
        f = np.maximum(0, pre)                          # codes sparses (ReLU)
        recon = f @ W_dec + b_dec
        err = recon - acts
        # Backward manuel (MSE + L1), cf 02-code/21 PART 5.
        grad_recon = 2 * err / n
        grad_b_dec = np.sum(grad_recon, axis=0)
        grad_W_dec = f.T @ grad_recon
        grad_f = grad_recon @ W_dec.T
        grad_f += l1_lambda * (f > 0).astype(np.float64) / n   # sous-gradient L1
        grad_pre = grad_f * (pre > 0)                  # ReLU backward
        grad_W_enc = acts.T @ grad_pre
        grad_b_enc = np.sum(grad_pre, axis=0)
        W_enc -= lr * grad_W_enc
        b_enc -= lr * grad_b_enc
        W_dec -= lr * grad_W_dec
        b_dec -= lr * grad_b_dec
    W_dec = normalize_dec_cols(W_dec)
    return W_enc, b_enc, W_dec, b_dec


def topk_mask(f: np.ndarray, k: int) -> np.ndarray:
    """Garde les k plus grandes activations par ligne, met le reste a 0.

    POURQUOI TopK (Gao/OpenAI 2024) : la sparsite est imposee EXACTEMENT par k
    (pas par une penalite qui retrecit les magnitudes) -> pas de shrinkage. Le
    gradient ne passe que par les entrees survivantes (les autres sont gelees a
    0 ce step), ce qui evite l'erosion L1 systematique de toutes les unites.
    """
    if k >= f.shape[1]:
        return f
    # Indices des (n_sae - k) plus PETITES valeurs par ligne -> a annuler.
    thresh_idx = np.argpartition(f, -k, axis=1)[:, :-k]
    out = f.copy()
    rows = np.arange(f.shape[0])[:, None]
    out[rows, thresh_idx] = 0.0
    return out


def train_topk_sae(acts, n_sae, k, n_iters=5000, lr=0.03, seed=3,
                   resample_every=500, dead_thresh=1e-4):
    """
    SAE TopK (Gao 2024) : pas de L1. A chaque forward on ne garde que les k
    plus grandes activations (topk_mask). Anti dead-features : init du decoder
    sur des exemples de donnees + RESAMPLING periodique des unites mortes (on
    re-initialise leur direction sur un exemple a forte erreur de recon).
    """
    n, d = acts.shape
    rng = np.random.default_rng(seed)
    W_enc = rng.normal(0, 0.3, size=(d, n_sae))
    b_enc = np.zeros(n_sae)
    # Init decoder ALIGNEE sur des activations reelles (anti dead-feature) :
    # chaque colonne part dans une direction effectivement presente dans les
    # donnees, ce qui evite des unites nees 'a cote' qui ne s'activeront jamais.
    seed_rows = rng.choice(n, size=n_sae, replace=True)
    W_dec = acts[seed_rows].copy() + rng.normal(0, 0.05, size=(n_sae, d))
    b_dec = np.zeros(d)
    for it in range(n_iters):
        W_dec = normalize_dec_cols(W_dec)
        pre = acts @ W_enc + b_enc
        f_full = np.maximum(0, pre)
        f = topk_mask(f_full, k)                        # sparsite EXACTE par k
        recon = f @ W_dec + b_dec
        err = recon - acts
        # Backward : le gradient ne circule que par les entrees survivantes du
        # topk (f==0 pour les autres -> elles ne contribuent pas a la recon).
        grad_recon = 2 * err / n
        grad_b_dec = np.sum(grad_recon, axis=0)
        grad_W_dec = f.T @ grad_recon
        grad_f = grad_recon @ W_dec.T
        survived = (f > 0)                              # masque topk effectif
        grad_pre = grad_f * survived * (pre > 0)        # stoppe hors-topk + ReLU
        grad_W_enc = acts.T @ grad_pre
        grad_b_enc = np.sum(grad_pre, axis=0)
        W_enc -= lr * grad_W_enc
        b_enc -= lr * grad_b_enc
        W_dec -= lr * grad_W_dec
        b_dec -= lr * grad_b_dec
        # --- Resampling des dead features ---
        if resample_every and (it + 1) % resample_every == 0 and it < n_iters - 1:
            act_rate = np.mean(f > dead_thresh, axis=0)
            dead = np.where(act_rate < 1e-3)[0]
            if dead.size > 0:
                # Re-init chaque unite morte sur l'exemple a plus forte erreur,
                # direction normalisee : on la 'replante' la ou la recon manque.
                recon_err = np.sum((recon - acts) ** 2, axis=1)
                worst = np.argsort(-recon_err)[:dead.size]
                for j, u in enumerate(dead):
                    v = acts[worst[j % worst.size]]
                    nv = np.linalg.norm(v) + 1e-8
                    W_dec[u] = v / nv
                    W_enc[:, u] = (v / nv) * 0.2          # encodeur aligne
                    b_enc[u] = 0.0
    W_dec = normalize_dec_cols(W_dec)
    return W_enc, b_enc, W_dec, b_dec


def evaluate_sae(W_enc, b_enc, W_dec, b_dec, k=None, cos_thresh=0.5):
    """
    Evalue un SAE : matching 1-1 par cosine vers les ground-truth, comptage des
    features recuperees, des dead features, et magnitude reconstruite moyenne
    (proxy du shrinkage : plus c'est proche de 1, moins il y a sous-estimation).
    """
    pre = acts @ W_enc + b_enc
    f_full = np.maximum(0, pre)
    f = topk_mask(f_full, k) if k is not None else f_full
    recon = f @ W_dec + b_dec
    active_rate = np.mean(f > 1e-4, axis=0)             # (n_sae,)
    n_dead = int(np.sum(active_rate < 1e-3))

    # Matching 1-1 par cosine entre colonnes de dictionnaire (W_dec) et D_true.
    # On parcourt les paires par cosine decroissant et on assigne sans reuse.
    pairs = []
    for s in range(W_dec.shape[0]):
        sv = W_dec[s]
        sn = np.linalg.norm(sv) + 1e-9
        if active_rate[s] < 1e-3:
            continue                                    # ignore les mortes
        for g in range(M_FEAT):
            gv = D_true[g]
            cos = float(np.dot(sv, gv) / (sn * (np.linalg.norm(gv) + 1e-9)))
            pairs.append((cos, s, g))
    pairs.sort(reverse=True)                            # meilleur cosine d'abord
    used_sae, used_gt, recovered = set(), set(), set()
    for cos, s, g in pairs:
        if cos < cos_thresh:
            break
        if s in used_sae or g in used_gt:
            continue
        used_sae.add(s)
        used_gt.add(g)
        recovered.add(g)                                # ground-truth unique
    n_recovered = len(recovered)

    # Shrinkage proxy : ratio moyen ||recon|| / ||acts|| sur les exemples.
    num = np.linalg.norm(recon, axis=1)
    den = np.linalg.norm(acts, axis=1) + 1e-9
    mag_ratio = float(np.mean(num / den))
    recon_loss = float(np.mean(np.sum((recon - acts) ** 2, axis=1)))
    return n_recovered, n_dead, mag_ratio, recon_loss


# --- Entrainement des deux SAE ---
# L1 avec lambda assez fort pour exhiber clairement shrinkage + dead features
# (comme le 02-code, le L1 naif sous-performe par construction).
l1_params = train_l1_sae(acts, N_SAE, l1_lambda=0.1)
topk_params = train_topk_sae(acts, N_SAE, k=K_TRUE)

rec_l1, dead_l1, mag_l1, loss_l1 = evaluate_sae(*l1_params, k=None)
rec_tk, dead_tk, mag_tk, loss_tk = evaluate_sae(*topk_params, k=K_TRUE)

print(f"\n  {'SAE':<10} {'recovered':<12} {'dead':<8} {'mag_ratio':<12} {'recon_loss':<12}")
print("  " + "-" * 56)
print(f"  {'L1 naif':<10} {rec_l1:<12} {dead_l1:<8} {mag_l1:<12.3f} {loss_l1:<12.4f}")
print(f"  {'TopK':<10} {rec_tk:<12} {dead_tk:<8} {mag_tk:<12.3f} {loss_tk:<12.4f}")

print(f"\n  Features ground-truth recuperees : L1={rec_l1}/{M_FEAT}, "
      f"TopK={rec_tk}/{M_FEAT}")
print(f"  Dead features : L1={dead_l1}/{N_SAE}, TopK={dead_tk}/{N_SAE}")
print(f"  Magnitude reconstruite (1.0 = pas de shrinkage) : "
      f"L1={mag_l1:.3f}, TopK={mag_tk:.3f}")

# --- Assertions : l'inegalite doit tenir par construction ---
assert rec_tk >= rec_l1, "TopK doit recuperer AU MOINS autant de features que L1"
assert dead_tk <= dead_l1, "TopK doit avoir AU PLUS autant de dead features que L1"
# Shrinkage : L1 sous-estime les magnitudes (mag_ratio < 1 plus marque) ;
# TopK reconstruit des magnitudes plus proches de 1.
assert mag_tk >= mag_l1 - 1e-6, "TopK ne doit pas sous-estimer plus que L1 (shrinkage)"

print("\n  -> TopK recupere >= de features ET a <= de dead features que L1 (OK).")
print("     L1 souffre du shrinkage (magnitudes sous-estimees) + dead features ;")
print("     TopK impose la sparsite par K (pas de penalite qui retrecit) et le")
print("     resampling ressuscite les unites mortes (Gao/OpenAI 2024).")
print("\n  Caveat honnete (comme 02-code) : c'est un toy minimal. On REPRODUIT le")
print("  mecanisme (TopK > L1 sur shrinkage et dead features), pas un resultat a")
print("  l'echelle d'un vrai LLM. La mono-semanticite parfaite reste un ideal.")

print("\nDone (HARD).")
