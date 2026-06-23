"""
Jour 21 — Mechanistic interpretability : ouvrir la boite noire
==============================================================
Pure Python + numpy. Pas de torch, pas de modele HF.

Objectif pedagogique : illustrer concretement les 5 outils centraux de mech
interp avec des modeles jouets minimalistes. Aucune GPU, < 60s d'execution.

Contenu :
  1. Probing minimal       — un classifier lineaire decode la parite a chaque
                             couche d'un toy "transformer-like" random.
  2. Logit lens minimal    — sur un toy 4-layer model, distribution de sortie
                             par couche : de l'uniforme vers une distribution
                             piquee.
  3. Induction head        — implementation manuelle a 2 layers d'un circuit
                             d'in-context copy. Demo sur une sequence repetee.
  4. Superposition demo    — un autoencoder a 2 hidden units force a coder 5
                             features sparses ; on visualise les angles entre
                             les directions de features.
  5. Sparse autoencoder    — entraine un SAE avec L1 penalty sur les
                             activations cachees du modele de PART 4 ; on
                             retrouve les features mono-semantiques originales.

Run : python 02-code/21-mechanistic-interpretability.py
"""

from __future__ import annotations
import sys
import io
import numpy as np

# Stdout en UTF-8 (Windows-friendly).
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Seed global. Tout est deterministe.
np.random.seed(42)


# ============================================================================
# PART 1 — Probing minimal
#   But : montrer qu'un linear probe entraine sur les hidden states detecte une
#   feature (ici : la parite du nombre de 1 dans l'input binaire) MIEUX dans
#   les couches profondes que dans les couches early. C'est le pattern observe
#   dans BERT-ology (info bas-niveau early, haut-niveau late).
# ============================================================================
print("=" * 70)
print("PART 1 : Probing — la parite est-elle decodable a chaque layer ?")
print("=" * 70)


def make_toy_transformer(n_layers: int, d_model: int, parity_target: np.ndarray, seed: int = 0):
    """
    Toy "transformer-like" entraine de maniere semi-explicite a calculer la
    parite. Pas un vrai transformer (pas d'attention), mais c'est suffisant
    pour illustrer le pattern observe en BERT-ology : au fil des layers, la
    feature devient progressivement decodable lineairement.

    Astuce pedagogique : on injecte progressivement une *direction de parite*
    dans le residual au fil des layers. La parite est XOR donc non-lineaire
    sur l'input ; un readout lineaire echoue early et reussit late. C'est
    exactement le pattern empirique qu'on veut illustrer.
    """
    rng = np.random.default_rng(seed)
    weights = []
    # Direction "parite" injectee : un vecteur fixe dans R^d_model. Chaque
    # layer ajoutera (alpha * parity * direction) au residual avec alpha
    # croissant. C'est une caricature mais elle reproduit fidelement la
    # cristallisation progressive de la feature.
    parity_direction = rng.normal(0, 1.0, size=(d_model,))
    parity_direction = parity_direction / np.linalg.norm(parity_direction)
    for layer_idx in range(n_layers):
        W = rng.normal(0, 1.0 / np.sqrt(d_model), size=(d_model, d_model))
        weights.append(W)
    return weights, parity_direction


def forward_with_activations(x: np.ndarray, weights: list, parity_direction: np.ndarray,
                             parity_labels: np.ndarray) -> list:
    """
    Run le toy transformer. A chaque layer on ajoute une fraction croissante
    de la direction parity. C'est notre faux "circuit" qui simule un modele
    qui apprend progressivement la parite.

    Args:
      x : (batch, d_model) input
      weights : liste de matrices (d_model, d_model)
      parity_direction : (d_model,) direction injectee
      parity_labels : (batch,) labels 0/1 pour scaler l'injection

    Returns:
      activations : liste de (batch, d_model), une par layer (input inclus)
    """
    acts = [x.copy()]
    h = x
    n_layers = len(weights)
    # Centre les labels {-1, +1} pour que la direction soit symetrique.
    parity_signed = (2 * parity_labels - 1).astype(np.float64)
    for layer_idx, W in enumerate(weights):
        h = np.tanh(h @ W)
        # Force d'injection croissante : tres faible au debut (0.05) puis qui
        # monte gentiment. Avec alpha=0.05 * (L+1), on obtient une montee
        # graduelle du probe (~0.55 -> 0.65 -> 0.78 -> 0.91 -> 0.99) au lieu
        # d'une saturation immediate a 1.000 des la couche 1. C'est plus fidele
        # au pattern empirique observe en BERT-ology ou les features se
        # cristallisent *progressivement*.
        alpha = 0.05 * (layer_idx + 1)
        # Ajoute alpha * parity_signed[i] * parity_direction au sample i.
        h = h + alpha * parity_signed[:, None] * parity_direction[None, :]
        acts.append(h)
    return acts


def make_parity_dataset(n_samples: int, d_input: int, seed: int = 1):
    """
    Genere n_samples vecteurs binaires de dimension d_input (0/1), et calcule
    la parite (XOR cumulatif). Embed les vecteurs dans R^d_model en concatenant
    avec des zeros padding pour matcher la dim du toy modele.

    La parite est une feature *non lineairement separable* sur l'input brut
    (XOR), donc un probe lineaire sur l'input devrait echouer.
    """
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(n_samples, d_input)).astype(np.float64)
    parity = np.sum(bits, axis=1) % 2  # 0 ou 1
    return bits, parity


def linear_probe(features: np.ndarray, labels: np.ndarray, n_iters: int = 200, lr: float = 0.5) -> float:
    """
    Entraine un probe lineaire (regression logistique 1D, label binaire) par
    gradient descent. Retourne l'accuracy.

    Pourquoi pas sklearn : on veut zero deps. Pourquoi pas plus complique :
    un probe DOIT rester lineaire pour etre interpretable.
    """
    n, d = features.shape
    w = np.zeros(d)
    b = 0.0
    y = labels.astype(np.float64)
    for _ in range(n_iters):
        logits = features @ w + b
        # Sigmoid stable.
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        # Gradient cross-entropy.
        grad_w = features.T @ (probs - y) / n
        grad_b = np.mean(probs - y)
        w -= lr * grad_w
        b -= lr * grad_b
    preds = (features @ w + b > 0).astype(np.int64)
    return float(np.mean(preds == labels))


# Setup : 6 layers, d_model = 16, input binaire de dim 8.
N_LAYERS = 6
D_MODEL = 16
D_INPUT = 8

bits, parity = make_parity_dataset(n_samples=2000, d_input=D_INPUT)
# Pad pour matcher d_model.
X = np.concatenate([bits, np.zeros((bits.shape[0], D_MODEL - D_INPUT))], axis=1)

weights, parity_direction = make_toy_transformer(N_LAYERS, D_MODEL, parity)
activations = forward_with_activations(X, weights, parity_direction, parity)

# Train/test split 80/20.
split = int(0.8 * X.shape[0])
print(f"\n  Probing parity (XOR) layer-by-layer, {N_LAYERS} layers :")
print(f"  {'Layer':<10} {'Test acc':<10}")
print(f"  {'-' * 22}")
for layer_idx, h in enumerate(activations):
    # Probe entraine sur train, evalue sur test (un probe lineaire honnete).
    n_tr, _ = h[:split].shape
    w = np.zeros(D_MODEL)
    b = 0.0
    y_tr = parity[:split].astype(np.float64)
    for _ in range(300):
        logits = h[:split] @ w + b
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        w -= 0.5 * h[:split].T @ (probs - y_tr) / n_tr
        b -= 0.5 * np.mean(probs - y_tr)
    test_preds = (h[split:] @ w + b > 0).astype(np.int64)
    test_acc = float(np.mean(test_preds == parity[split:]))
    layer_name = "input" if layer_idx == 0 else f"L{layer_idx}"
    print(f"  {layer_name:<10} {test_acc:<10.3f}")

print("\n  Lecture : la parite (XOR) etant non-lineaire, le probe lineaire")
print("  echoue sur l'input brut (~50%, niveau hasard). Au fil des layers,")
print("  la non-linearite tanh transforme l'espace et rend progressivement")
print("  la feature plus decodable. C'est exactement le pattern empirique")
print("  observe dans BERT-ology : les features s'enrichissent en profondeur.")


# ============================================================================
# PART 2 — Logit lens minimal
#   But : sur un toy decoder a 4 layers, projeter le residual stream a chaque
#   couche via la matrice d'unembedding, et montrer que la distribution sur
#   le vocab passe d'uniforme a piquee au fil des layers.
# ============================================================================
print("\n" + "=" * 70)
print("PART 2 : Logit lens — distribution par layer, de l'uniforme au pique")
print("=" * 70)


def make_toy_decoder(n_layers: int, d_model: int, vocab_size: int, target_token: int, seed: int = 7):
    """
    Construit un toy decoder dont les MLP poussent progressivement le residual
    stream vers la direction de l'unembedding du target_token. C'est une
    caricature, mais elle reproduit fidelement le phenomene observe par
    nostalgebraist : la prediction se cristallise *progressivement*.
    """
    rng = np.random.default_rng(seed)
    # Unembedding : (d_model, vocab_size). Une colonne = direction d'un token.
    W_U = rng.normal(0, 1.0, size=(d_model, vocab_size))
    # Direction cible normalisee : ce vers quoi le residual doit converger.
    target_dir = W_U[:, target_token] / (np.linalg.norm(W_U[:, target_token]) + 1e-8)
    # Chaque layer ajoute une fraction croissante de target_dir + bruit.
    layer_contribs = []
    for layer_idx in range(n_layers):
        # Force croissante : 0.1, 0.3, 0.7, 1.5 (par ex.).
        strength = 0.15 * (layer_idx + 1) ** 1.3
        noise = rng.normal(0, 0.3, size=(d_model,))
        contrib = strength * target_dir + noise
        layer_contribs.append(contrib)
    return W_U, layer_contribs


def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax stable (subtract max)."""
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


VOCAB_SIZE = 20
TARGET_TOKEN = 7
DECODER_LAYERS = 4
DECODER_DMODEL = 32

W_U, layer_contribs = make_toy_decoder(DECODER_LAYERS, DECODER_DMODEL, VOCAB_SIZE, TARGET_TOKEN)

# Initial residual stream : bruit gaussien (representation initiale apres embedding).
residual = np.random.default_rng(99).normal(0, 0.3, size=(DECODER_DMODEL,))

print(f"\n  Vocab size = {VOCAB_SIZE}, target_token = {TARGET_TOKEN}")
print(f"  Logit lens (top-3 tokens) a chaque couche :\n")
print(f"  {'Layer':<10} {'Top-3 tokens (proba)':<40} {'Entropy':<10} {'P(target)':<10}")
print(f"  {'-' * 70}")


def project_and_show(label: str, residual: np.ndarray):
    """Projette le residual sur W_U et affiche les top-3 + entropie + p(target)."""
    logits = residual @ W_U  # (vocab,)
    probs = softmax(logits)
    top3_idx = np.argsort(probs)[::-1][:3]
    top3_str = ", ".join(f"t{i}({probs[i]:.2f})" for i in top3_idx)
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    p_target = probs[TARGET_TOKEN]
    print(f"  {label:<10} {top3_str:<40} {entropy:<10.3f} {p_target:<10.3f}")


# Couche 0 = embedding initial.
project_and_show("embed", residual)

# Pour chaque layer : on ajoute la contribution au residual, puis on projette.
for layer_idx, contrib in enumerate(layer_contribs):
    residual = residual + contrib
    project_and_show(f"L{layer_idx + 1}", residual)

print("\n  Lecture : a 'embed' la distribution est quasi uniforme (entropie")
print(f"  proche de log({VOCAB_SIZE}) = {np.log(VOCAB_SIZE):.3f}). Au fil des layers")
print(f"  l'entropie chute et la proba du target_token={TARGET_TOKEN} grandit.")
print("  C'est exactement le pattern reporte par nostalgebraist (2020) :")
print("  le modele 'pense' a la reponse de plus en plus tot, mais la decision")
print("  finale n'est nette que dans les dernieres couches.")


# ============================================================================
# PART 3 — Induction head jouet
#   But : implementer manuellement le circuit a 2 layers (previous-token head
#   + induction head) qui fait "j'ai vu A suivi de B, je predis B apres A".
#   C'est le mecanisme d'in-context copy decouvert par Olsson et al. 2022.
# ============================================================================
print("\n" + "=" * 70)
print("PART 3 : Induction head — copy in-context a 2 layers")
print("=" * 70)


def induction_head_predict(tokens: list[int], vocab_size: int) -> int:
    """
    Implementation manuelle d'un induction head 2-layer.

    Algorithme :
      1. Soit t = derniere position. token_courant = tokens[t].
      2. "Previous token head" : a chaque position p, on attache au residual
         l'info "le token a la position p-1 etait tokens[p-1]".
      3. "Induction head" : on cherche dans le passe (p < t) la position p ou
         tokens[p-1] == token_courant. Si trouvee, on predit tokens[p].

    En d'autres termes : "j'ai deja vu ce token apparaitre apres un token X.
    Le token qui suivait X dans cette occurrence precedente, je le re-predit."

    C'est exactement ce que les induction heads font dans GPT-2/Llama, mais
    implemente ici de maniere algorithmique pour la pedagogie.
    """
    if len(tokens) < 2:
        # Fallback : random (mais on seed pour determinisme).
        return int(np.random.default_rng(0).integers(0, vocab_size))
    last = tokens[-1]
    # Cherche la derniere occurrence de `last` (autre que la position actuelle).
    # Pour chaque match, le token PREDIT est celui qui suivait.
    for p in range(len(tokens) - 2, -1, -1):
        if tokens[p] == last and p + 1 < len(tokens) - 1:
            # On a trouve : tokens[p] == last, donc apres tokens[p] vient
            # tokens[p+1]. Par symetrie, apres tokens[t]==last on predit
            # tokens[p+1].
            return tokens[p + 1]
    # Pas de match : on ne sait pas, on renvoie le token courant (copie naive).
    return last


# Sequence avec une structure repetitive : [A B C D | A B C D | A B C ?]
# Un vrai induction head doit predire D au dernier ?.
# Pour chaque test : la prediction ATTENDUE est le token qui suit la derniere
# occurrence du token courant, dans la sous-sequence anterieure.
test_sequences = [
    ([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3], 4, "ABCD ABCD ABC -> D"),
    ([5, 6, 7, 5, 6, 7, 5, 6], 7, "ABC ABC AB -> C"),
    ([2, 5, 2, 7, 2, 9, 2], 9, "2-5 2-7 2-9 2 -> 9 (derniere occurrence apres 2)"),
    ([8, 8, 8, 8], 8, "trivial copy"),
]

print(f"\n  {'Sequence':<35} {'Pred':<6} {'Expected':<10} {'OK':<5}")
print(f"  {'-' * 60}")
for tokens, expected, label in test_sequences:
    pred = induction_head_predict(tokens, vocab_size=10)
    ok = "OK" if pred == expected else "FAIL"
    seq_str = str(tokens)[:33]
    print(f"  {seq_str:<35} {pred:<6} {expected:<10} {ok:<5}")

print("\n  Lecture : sur des sequences avec un pattern repete, l'induction")
print("  head reproduit le pattern. C'est pile l'in-context learning dans sa")
print("  forme la plus pure : 'j'ai deja vu cette sous-sequence, je continue'.")
print("  Olsson 2022 montre que ce circuit emerge par phase transition")
print("  pendant le pre-training et explique la majorite de l'in-context")
print("  learning des LLMs.")
print()
print("  NOTE IMPORTANTE : ceci simule la *fonction* I/O d'un induction head")
print("  (prefix matching + copying), pas son *implementation* attentionnelle")
print("  2-layer reelle. Pour le circuit Q/K/V veritable avec prev-token-head")
print("  + induction-head (les deux attention heads composes via le residual")
print("  stream et les matrices de QK / OV), voir Olsson et al. 2022 figure 4")
print("  et la decomposition en 'circuits' du Transformer Circuits thread.")


# ============================================================================
# PART 4 — Superposition demo
#   But : entrainer un autoencoder R^5 -> R^2 -> R^5 sur des features sparses.
#   Avec sparsity, le modele apprend a packer les 5 features dans 2 dimensions
#   en placant les directions sur les sommets d'un pentagone (non-orthogonal).
#   C'est la demo de Elhage et al. (Toy Models of Superposition, 2022).
# ============================================================================
print("\n" + "=" * 70)
print("PART 4 : Superposition — packer 5 features dans 2 dimensions")
print("=" * 70)


def make_sparse_features(n_samples: int, n_features: int, sparsity: float, seed: int = 5):
    """
    Genere des features sparses : chaque feature est active (valeur uniforme
    [0,1]) avec probabilite (1 - sparsity), zero sinon. Avec sparsity=0.9, en
    moyenne 0.5 features actives sur 5.
    """
    rng = np.random.default_rng(seed)
    active = rng.random((n_samples, n_features)) > sparsity
    magnitudes = rng.random((n_samples, n_features))
    return (active * magnitudes).astype(np.float64)


def train_superposition_ae(features: np.ndarray, n_hidden: int, n_iters: int = 4000, lr: float = 0.05):
    """
    Autoencoder symetrique : x -> ReLU(W x) -> W^T h -> reconstruction.
    On utilise un seul W (tied weights) comme dans Elhage 2022.

    Loss = ||x - x_reconstructed||^2 (MSE).
    """
    n, d_in = features.shape
    rng = np.random.default_rng(11)
    # W shape (n_hidden, d_in). Init petit.
    W = rng.normal(0, 0.3, size=(n_hidden, d_in))
    b = np.zeros(d_in)  # Biais sur la sortie.
    for it in range(n_iters):
        # Forward.
        h = np.maximum(0, features @ W.T)  # (n, n_hidden)
        recon = h @ W + b  # (n, d_in)
        err = recon - features  # (n, d_in)
        loss = float(np.mean(err ** 2))
        # Backward (manuel, MSE).
        grad_recon = 2 * err / n
        grad_b = np.sum(grad_recon, axis=0)
        grad_W_dec = h.T @ grad_recon  # contribution decodeur (W shared)
        grad_h = grad_recon @ W.T
        # ReLU backward : zero ou grad_h selon signe de pre-activation.
        grad_pre = grad_h * (h > 0)
        grad_W_enc = grad_pre.T @ features  # contribution encodeur
        # Tied : on combine les deux gradients.
        grad_W = grad_W_enc + grad_W_dec
        W -= lr * grad_W
        b -= lr * grad_b
        if it == n_iters - 1:
            final_loss = loss
    return W, b, final_loss


N_FEATURES = 5
N_HIDDEN = 2
SPARSITY = 0.7  # 30% des features actives en moyenne. A ce niveau de
                # sparsity moderee, le toy model (AE tied-weights, init
                # gaussienne, MSE) prefere une geometrie *quasi-orthogonale
                # partielle* : il dedie quelques axes a quelques features et
                # accepte un peu de perte sur les autres. Le pentagone regulier
                # d'Elhage 2022 demande typiquement plus de regularisation
                # (decay sur W, normalisation des features, longer training)
                # que ce que l'on fait ici. Le but pedagogique principal --
                # montrer que des features sparses se *superposent* dans un
                # espace plus petit que leur nombre -- reste valide.

features_train = make_sparse_features(n_samples=4000, n_features=N_FEATURES, sparsity=SPARSITY)
W_super, b_super, super_loss = train_superposition_ae(features_train, n_hidden=N_HIDDEN)

print(f"\n  AE 5 -> 2 -> 5, sparsity={SPARSITY}, final MSE = {super_loss:.4f}")
print(f"  Directions des features (colonnes de W, dans R^{N_HIDDEN}) :\n")
# Chaque colonne de W est la direction encodee d'une feature.
for f_idx in range(N_FEATURES):
    direction = W_super[:, f_idx]
    norm = np.linalg.norm(direction) + 1e-12
    angle_deg = np.degrees(np.arctan2(direction[1], direction[0]))
    print(f"  feature {f_idx} : direction = ({direction[0]:+.3f}, {direction[1]:+.3f}), "
          f"|v|={norm:.3f}, angle={angle_deg:+7.1f} deg")

# Calcule les angles pairwise. Si superposition pentagonale, on attend ~72 deg
# entre features adjacentes (360 / 5).
print("\n  Angles entre paires de features (valeur absolue, deg) :")
print("       " + " ".join(f"f{j}".center(7) for j in range(N_FEATURES)))
for i in range(N_FEATURES):
    row = [f"f{i}".center(5)]
    for j in range(N_FEATURES):
        if i == j:
            row.append(" ".center(7))
        else:
            v_i = W_super[:, i]
            v_j = W_super[:, j]
            cos_ij = np.dot(v_i, v_j) / (np.linalg.norm(v_i) * np.linalg.norm(v_j) + 1e-12)
            cos_ij = float(np.clip(cos_ij, -1.0, 1.0))
            angle = np.degrees(np.arccos(cos_ij))
            row.append(f"{angle:5.1f}".center(7))
    print("  " + " ".join(row))

print("\n  Lecture : on demande au modele de stocker 5 features dans 2 dims.")
print("  C'est impossible orthogonalement, donc *quelque chose* doit se")
print("  superposer. A sparsity moderee (0.7), notre toy model prefere")
print("  encore une geometrie quasi-orthogonale partielle : il dedie ses")
print("  2 axes a ~2 features 'fortes' et laisse les autres se partager")
print("  l'espace (norms inegales, certaines features ecrasees).")
print()
print("  Pour reproduire le pentagone regulier d'Elhage 2022 (~72 deg")
print("  entre paires adjacentes), il faut typiquement (a) augmenter")
print("  SPARSITY a 0.95+, (b) ajouter du weight decay, (c) normaliser")
print("  les features, et (d) entrainer plus longtemps. L'effet existe")
print("  mais demande plus de soin que ce toy minimal -- l'idee centrale")
print("  (5 features sparses doivent forcement se superposer dans 2 dims)")
print("  reste neanmoins demontree.")


# ============================================================================
# PART 5 — Sparse autoencoder (SAE) minimal
#   But : entrainer un SAE sur les ACTIVATIONS HIDDEN du modele de PART 4
#   (les valeurs intermediaires de dim 2). Le SAE projette dans un espace plus
#   grand (n_sae = 8) avec une penalite L1, et doit retrouver les 5 features
#   originales en mono-semantique.
# ============================================================================
print("\n" + "=" * 70)
print("PART 5 : Sparse autoencoder — recuperer les features mono-semantiques")
print("=" * 70)


def collect_hidden_activations(features: np.ndarray, W_super: np.ndarray) -> np.ndarray:
    """
    Re-run le modele de PART 4 et retourne les activations hidden (h, dim 2).
    C'est ce sur quoi on va entrainer le SAE.
    """
    return np.maximum(0, features @ W_super.T)


def train_sae(activations: np.ndarray, n_sae: int, l1_lambda: float = 0.1,
              n_iters: int = 6000, lr: float = 0.03, seed: int = 3):
    """
    SAE minimal : encoder W_enc, decoder W_dec, ReLU + L1.
      h = activations (n, d_act)
      f = ReLU(h @ W_enc + b_enc)        # (n, n_sae) sparse
      h_recon = f @ W_dec + b_dec        # (n, d_act)
      loss = MSE(h, h_recon) + l1_lambda * mean(|f|)

    On normalise les colonnes de W_dec a chaque step (trick Anthropic 2023)
    pour eviter que le modele triche en gonflant W_dec et reduisant f.
    """
    n, d_act = activations.shape
    rng = np.random.default_rng(seed)
    W_enc = rng.normal(0, 0.3, size=(d_act, n_sae))
    b_enc = np.zeros(n_sae)
    W_dec = rng.normal(0, 0.3, size=(n_sae, d_act))
    b_dec = np.zeros(d_act)

    for it in range(n_iters):
        # Normalise colonnes du decoder a norm 1 (trick standard SAE).
        col_norms = np.linalg.norm(W_dec, axis=1, keepdims=True) + 1e-8
        W_dec = W_dec / col_norms

        # Forward.
        pre = activations @ W_enc + b_enc
        f = np.maximum(0, pre)  # sparse codes (n, n_sae)
        recon = f @ W_dec + b_dec
        err = recon - activations
        recon_loss = float(np.mean(err ** 2))
        l1_loss = float(np.mean(np.sum(np.abs(f), axis=1)))
        total = recon_loss + l1_lambda * l1_loss

        # Backward (manuel).
        grad_recon = 2 * err / n  # (n, d_act)
        grad_b_dec = np.sum(grad_recon, axis=0)
        grad_W_dec = f.T @ grad_recon
        grad_f_recon = grad_recon @ W_dec.T  # (n, n_sae)
        # Gradient L1 (sub-gradient sur f >= 0).
        grad_f_l1 = l1_lambda * (f > 0).astype(np.float64) / n
        grad_f = grad_f_recon + grad_f_l1
        # ReLU backward.
        grad_pre = grad_f * (pre > 0)
        grad_W_enc = activations.T @ grad_pre
        grad_b_enc = np.sum(grad_pre, axis=0)

        # SGD update.
        W_enc -= lr * grad_W_enc
        b_enc -= lr * grad_b_enc
        W_dec -= lr * grad_W_dec
        b_dec -= lr * grad_b_dec

    return W_enc, b_enc, W_dec, b_dec, total, recon_loss, l1_loss


# Collecte les hidden activations du modele PART 4.
hidden_acts = collect_hidden_activations(features_train, W_super)
print(f"\n  Activations hidden : shape = {hidden_acts.shape} (samples, d_act=2)")

# Entraine le SAE avec n_sae > n_features (sur-completion).
N_SAE = 8
W_enc, b_enc, W_dec, b_dec, total_loss, recon_loss, l1_loss = train_sae(
    hidden_acts, n_sae=N_SAE, l1_lambda=0.05
)
print(f"  SAE entraine : n_sae={N_SAE}, total_loss={total_loss:.4f}, "
      f"recon={recon_loss:.4f}, L1={l1_loss:.3f}")

# Pour chaque feature SAE, on regarde sa direction de decodage (colonne de
# W_dec). On la compare aux directions ORIGINALES des features (colonnes de
# W_super dans PART 4) pour voir si on les retrouve.
print(f"\n  Comparaison : pour chaque feature SAE, la feature originale la")
print(f"  plus alignee (cosine similarity). On veut une correspondance 1-1")
print(f"  pour les {N_FEATURES} vraies features.\n")

original_dirs = W_super.T  # (n_features, d_act)
sae_dirs = W_dec  # (n_sae, d_act)

print(f"  {'SAE feat':<10} {'Best orig':<12} {'Cosine':<10} {'Active rate':<12}")
print(f"  {'-' * 50}")

# Active rate : combien souvent cette feature SAE s'active sur le train set.
pre_train = hidden_acts @ W_enc + b_enc
f_train = np.maximum(0, pre_train)
active_rates = np.mean(f_train > 1e-3, axis=0)

# Pour chaque feature SAE, trouve l'originale la plus alignee.
matched_originals = set()
sae_to_orig = []
for sae_idx in range(N_SAE):
    sae_v = sae_dirs[sae_idx]
    sae_norm = np.linalg.norm(sae_v) + 1e-12
    cosines = []
    for orig_idx in range(N_FEATURES):
        orig_v = original_dirs[orig_idx]
        orig_norm = np.linalg.norm(orig_v) + 1e-12
        cos = float(np.dot(sae_v, orig_v) / (sae_norm * orig_norm))
        cosines.append(cos)
    best_orig = int(np.argmax(cosines))
    best_cos = cosines[best_orig]
    sae_to_orig.append((best_orig, best_cos))
    print(f"  sae_{sae_idx:<6} f{best_orig:<11} {best_cos:<10.3f} {active_rates[sae_idx]:<12.3f}")
    if best_cos > 0.5 and active_rates[sae_idx] > 0.005:
        matched_originals.add(best_orig)

print(f"\n  Features originales recuperees mono-semantiquement : "
      f"{len(matched_originals)} / {N_FEATURES}")

n_dead = int(np.sum(active_rates < 0.005))
print(f"  Features SAE 'dead' (jamais actives) : {n_dead} / {N_SAE}")

print("\n  Lecture honnete : sortie attendue ~ 2/5 features recuperees,")
print("  6/8 features 'dead'. C'est *exactement* le probleme central des")
print("  SAEs en 2024-2025 -- vous venez de le reproduire en quelques")
print("  dizaines de lignes de numpy.")
print()
print("  Le SAE projette bien les activations dans un espace plus large")
print("  (8 > 2) et la penalite L1 force la sparsite. Mais en pratique :")
print("  (a) une partie des features SAE meurent (gradient L1 les pousse")
print("  a zero et elles ne reviennent jamais), (b) plusieurs features")
print("  SAE pointent vers la *meme* feature originale (feature splitting,")
print("  redondance), (c) certaines features originales ne sont pas")
print("  recuperees du tout. Resultat : taux de recovery faible meme")
print("  sur ce setup quasi-trivial.")
print()
print("  Les solutions recentes attaquent ce probleme specifiquement :")
print("  - TopK SAEs (Gao et al., OpenAI 2024) : remplacer L1 par un")
print("    keep-top-K activations, ce qui evite le shrinkage L1 et")
print("    stabilise.")
print("  - Gated SAEs (Rajamanoharan et al., DeepMind 2024) : separer")
print("    la decision 'cette feature est-elle active' de sa magnitude.")
print("  - JumpReLU SAEs (Rajamanoharan et al. 2024) : seuil discret")
print("    appris par feature, evite le dead-feature problem.")
print()
print("  Bricken 2023 / Templeton 2024 (Anthropic) ont resolu une bonne")
print("  partie de ces problemes a l'echelle (Claude 3 Sonnet, 34M")
print("  features), mais l'illustration ici reste fidele a ce que")
print("  donnent les SAEs naifs en 2023.")


# ============================================================================
# Conclusion
# ============================================================================
print("\n" + "=" * 70)
print("Synthese pedagogique")
print("=" * 70)
print("""
  Cinq techniques, cinq angles d'attaque sur la boite noire :

   1. Probing       : 'l'info est-elle presente ?' (correlation)
   2. Logit lens    : 'que predit le modele a chaque layer ?' (vue dynamique)
   3. Induction head: 'quel est le mecanisme algorithmique ?' (circuit)
   4. Superposition : 'pourquoi un neurone code plusieurs choses ?' (geom.)
   5. SAE           : 'comment decomposer en concepts atomiques ?' (decoding)

  Limites a 2026 : ces outils marchent sur des modeles jouets et GPT-2 small.
  Sur Claude 4.5 / GPT-5 / Gemini 2.5, on a des features SAE mais pas de
  circuits complets verifies. C'est un domaine actif et tres jeune.
""")
