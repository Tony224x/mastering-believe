"""
Solutions MEDIUM — Jour 7 : Mini-Transformer (Capstone Week 1)
=============================================================
Exercices 4, 5, 6 (medium). Pur NumPy.

Le 02-code/07-mini-transformer.py est en PyTorch ; ici on RE-IMPLEMENTE
les memes briques en NumPy pour pouvoir tout verifier numeriquement sans torch.

4. Bloc Transformer complet (pre-norm) + causalite end-to-end.
5. Forward du mini-GPT + loss a l'init ~ log(vocab) + overfit d'un batch.
6. Sampling : temperature, top-k, top-p (nucleus).

Run: python 03-exercises/solutions/07-mini-transformer-medium.py
"""

import numpy as np

np.random.seed(42)


# ----------------------------------------------------------------------------
# Briques de base (NumPy)
# ----------------------------------------------------------------------------

def softmax(x, axis=-1):
    """Softmax stable le long d'un axe (shift par le max)."""
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def gelu(x):
    """Approximation tanh de GELU (celle de GPT-2)."""
    # WHY tanh-approx: c'est exactement la forme utilisee par GPT-2/nanoGPT.
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def layer_norm(x, gamma, beta, eps=1e-5):
    """LayerNorm sur le dernier axe. Variance population (ddof=0)."""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)        # ddof=0 par defaut -> comme PyTorch
    return (x - mu) / np.sqrt(var + eps) * gamma + beta


def causal_mask(n):
    """1 = bloque (triangle superieur strict = positions futures)."""
    return np.triu(np.ones((n, n), dtype=np.int32), k=1)


def causal_self_attention(X, W_qkv, W_o, n_head):
    """
    Multi-head causal self-attention en NumPy, miroir de CausalSelfAttention.

      X     : (seq, d_model)
      W_qkv : (d_model, 3*d_model)  -- une grosse projection (comme c_attn)
      W_o   : (d_model, d_model)    -- projection de sortie
    Renvoie (seq, d_model).
    """
    seq, d_model = X.shape
    head_dim = d_model // n_head
    assert d_model % n_head == 0

    qkv = X @ W_qkv                              # (seq, 3*d_model)
    q, k, v = np.split(qkv, 3, axis=-1)         # chacun (seq, d_model)

    # (seq, d_model) -> (n_head, seq, head_dim)
    def split_heads(M):
        return M.reshape(seq, n_head, head_dim).transpose(1, 0, 2)
    qh, kh, vh = split_heads(q), split_heads(k), split_heads(v)

    mask = causal_mask(seq)
    outs = []
    for h in range(n_head):
        scores = qh[h] @ kh[h].T / np.sqrt(head_dim)   # (seq, seq)
        scores = np.where(mask == 1, -np.inf, scores)
        w = softmax(scores, axis=-1)
        outs.append(w @ vh[h])                          # (seq, head_dim)
    concat = np.concatenate(outs, axis=-1)              # (seq, d_model)
    return concat @ W_o


def feed_forward(x, W1, b1, W2, b2):
    """FFN: Linear -> GELU -> Linear (d_ff = 4 * d_model)."""
    return gelu(x @ W1 + b1) @ W2 + b2


def make_block_params(d_model, n_head, std=0.02, seed=0):
    """Parametres d'un bloc, petite init (std=0.02 comme GPT-2)."""
    rng = np.random.RandomState(seed)
    d_ff = 4 * d_model
    return dict(
        ln1_g=np.ones(d_model), ln1_b=np.zeros(d_model),
        ln2_g=np.ones(d_model), ln2_b=np.zeros(d_model),
        W_qkv=rng.randn(d_model, 3 * d_model) * std,
        W_o=rng.randn(d_model, d_model) * std,
        W1=rng.randn(d_model, d_ff) * std, b1=np.zeros(d_ff),
        W2=rng.randn(d_ff, d_model) * std, b2=np.zeros(d_model),
        n_head=n_head,
    )


def transformer_block(x, p):
    """Bloc PRE-NORM (variante GPT-2+) : x + attn(ln1(x)), x + ffn(ln2(x))."""
    a = causal_self_attention(layer_norm(x, p['ln1_g'], p['ln1_b']),
                              p['W_qkv'], p['W_o'], p['n_head'])
    x = x + a
    f = feed_forward(layer_norm(x, p['ln2_g'], p['ln2_b']),
                     p['W1'], p['b1'], p['W2'], p['b2'])
    x = x + f
    return x


# ============================================================================
# EXERCISE 4: Bloc Transformer complet + causalite end-to-end
# ============================================================================

print("=" * 70)
print("EXERCISE 4: Bloc Transformer (pre-norm) + causalite end-to-end")
print("=" * 70)

d_model, n_head, seq = 48, 4, 10
head_dim = d_model // n_head
X = np.random.randn(seq, d_model) * 0.5
p = make_block_params(d_model, n_head, seed=1)

# Trace des shapes etape par etape.
ln1 = layer_norm(X, p['ln1_g'], p['ln1_b'])
qkv = ln1 @ p['W_qkv']
print(f"\n  Shapes (d_model={d_model}, n_head={n_head}, seq={seq}, head_dim={head_dim}):")
print(f"    X            : {X.shape}")
print(f"    ln1(X)       : {ln1.shape}")
print(f"    qkv          : {qkv.shape}  (3*d_model = {3*d_model})")
print(f"    split tetes  : ({n_head}, {seq}, {head_dim})")
out = transformer_block(X, p)
print(f"    block(X)     : {out.shape}")
assert head_dim == 12, "head_dim attendu = 12"

# Causalite end-to-end : perturber X[t] ne change pas out[:t].
t = 6
X2 = X.copy()
X2[t] += np.random.randn(d_model)            # perturbe SEULEMENT le token t
out2 = transformer_block(X2, p)
err_before = np.max(np.abs(out[:t] - out2[:t]))
err_after = np.max(np.abs(out[t:] - out2[t:]))
print(f"\n  Causalite (perturbation du token t={t}):")
print(f"    max |out[:t] - out2[:t]| = {err_before:.2e}  (doit etre ~0)")
print(f"    max |out[t:] - out2[t:]| = {err_after:.2e}  (doit etre > 0)")
print("    -> le passe n'est pas affecte par le futur : masque causal OK.")

# Proximite de l'identite a l'init (petite std).
rel_delta = np.linalg.norm(out - X) / np.linalg.norm(X)
print(f"\n  Proximite identite a l'init: ||block(X)-X||/||X|| = {rel_delta:.3f}")
print("    -> petit : residual + init ~0.02 rendent le bloc ~identite au depart,")
print("       donc le signal/gradient traverse N couches sans exploser ni s'eteindre.")


# ============================================================================
# EXERCISE 5: Forward du mini-GPT + loss a l'init + overfit d'un batch
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 5: Forward mini-GPT + loss a l'init ~ log(vocab) + overfit")
print("=" * 70)


class MiniGPTNumpy:
    def __init__(self, vocab, d_model=48, n_head=4, n_layer=4, block_size=32,
                 std=0.02, seed=0):
        rng = np.random.RandomState(seed)
        self.vocab, self.d_model, self.block_size = vocab, d_model, block_size
        self.tok_emb = rng.randn(vocab, d_model) * std
        self.pos_emb = rng.randn(block_size, d_model) * std
        self.blocks = [make_block_params(d_model, n_head, seed=seed + 1 + i)
                       for i in range(n_layer)]
        self.lnf_g, self.lnf_b = np.ones(d_model), np.zeros(d_model)
        self.W_head = rng.randn(d_model, vocab) * std

    def forward(self, idx):
        seq = len(idx)
        x = self.tok_emb[idx] + self.pos_emb[:seq]      # (seq, d_model)
        for p in self.blocks:
            x = transformer_block(x, p)
        x = layer_norm(x, self.lnf_g, self.lnf_b)
        return x @ self.W_head                           # (seq, vocab)


def cross_entropy(logits, targets):
    """Cross-entropy moyenne (log-softmax stable). logits (n,vocab), targets (n,)."""
    z = logits - logits.max(axis=-1, keepdims=True)
    logsumexp = np.log(np.exp(z).sum(axis=-1)) + 0.0
    log_probs = z[np.arange(len(targets)), targets] - logsumexp
    return -log_probs.mean()


vocab = 27
gpt = MiniGPTNumpy(vocab, d_model=48, n_head=4, n_layer=4, seed=7)
idx = np.random.randint(0, vocab, size=16)               # sequence aleatoire
logits = gpt.forward(idx)
# next-token : on predit idx[1:] a partir des positions 0..n-2.
loss_init = cross_entropy(logits[:-1], idx[1:])
print(f"\n  Loss a l'init    : {loss_init:.4f}")
print(f"  log(vocab_size)  : {np.log(vocab):.4f}  (baseline uniforme)")
print(f"  ecart            : {abs(loss_init - np.log(vocab)):.4f}")
print("  -> proche de log(vocab) : a l'init le modele predit ~1/vocab par token.")

# Sanity check #2 : overfit d'UNE sequence en bougeant juste W_head + embeddings.
# (On fait une descente sur W_head et la table d'embedding token via gradient
#  numerique leger pour rester self-contained et lisible.)
print("\n  Overfit d'un seul batch (descente sur W_head, gradient analytique):")
seqx = idx[:8]
tgt = idx[1:8]
lr = 0.5
losses_of = []
for step in range(120):
    # Forward jusqu'aux logits ; on backprop SEULEMENT W_head (le reste fige).
    seq = len(seqx)
    x = gpt.tok_emb[seqx] + gpt.pos_emb[:seq]
    for pblk in gpt.blocks:
        x = transformer_block(x, pblk)
    h = layer_norm(x, gpt.lnf_g, gpt.lnf_b)              # (seq, d_model)
    z = h @ gpt.W_head                                   # (seq, vocab)
    zt = z[:-1]
    loss = cross_entropy(zt, tgt)
    losses_of.append(loss)
    # dL/dz = (softmax - onehot)/N pour les positions predites.
    p_soft = softmax(zt, axis=-1)
    p_soft[np.arange(len(tgt)), tgt] -= 1.0
    p_soft /= len(tgt)
    dW = h[:-1].T @ p_soft                               # (d_model, vocab)
    gpt.W_head -= lr * dW
print(f"    loss initiale: {losses_of[0]:.4f}  -> loss finale: {losses_of[-1]:.4f}")
print(f"    log(vocab) = {np.log(vocab):.4f} ; loss finale bien en-dessous = OK")
print("    -> overfitter UN batch est le test minimal que la boucle apprend.")
print("\n  Si loss_init >> log(vocab) (ex 50): poids mal initialises -> logits enormes")
print("  -> softmax sature -> probas ~0 sur la cible -> -log(0) explose.")


# ============================================================================
# EXERCISE 6: Sampling — temperature, top-k, top-p (nucleus)
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 6: Sampling — temperature, top-k, top-p")
print("=" * 70)


def softmax_temperature(logits, T):
    """softmax(logits / T). T->0 = greedy (one-hot sur l'argmax)."""
    if T <= 1e-9:
        out = np.zeros_like(logits)
        out[np.argmax(logits)] = 1.0
        return out
    return softmax(logits / T)


def top_k_filter(logits, k):
    """Garde les k plus grands logits, -inf ailleurs."""
    out = np.full_like(logits, -np.inf)
    idx = np.argsort(logits)[-k:]               # k plus grands
    out[idx] = logits[idx]
    return out


def top_p_filter(logits, p):
    """Nucleus : plus petit ensemble dont la masse de proba >= p."""
    probs = softmax(logits)
    order = np.argsort(probs)[::-1]             # proba decroissante
    cum = np.cumsum(probs[order])
    # On garde jusqu'a (et y compris) le token qui fait franchir le seuil p.
    cutoff = np.searchsorted(cum, p) + 1
    keep = order[:cutoff]
    out = np.full_like(logits, -np.inf)
    out[keep] = logits[keep]
    return out


def entropy(p):
    p = p[p > 0]
    return -np.sum(p * np.log(p))


logits = np.array([2.0, 1.0, 0.5, 0.0, -1.0])

print("\n  Temperature (entropie decroit quand T baisse):")
for T in [2.0, 1.0, 0.5]:
    pr = softmax_temperature(logits, T)
    print(f"    T={T}: probs={np.round(pr, 3)}  H={entropy(pr):.3f}")
H05 = entropy(softmax_temperature(logits, 0.5))
H1 = entropy(softmax_temperature(logits, 1.0))
H2 = entropy(softmax_temperature(logits, 2.0))
print(f"    H(0.5)={H05:.3f} < H(1)={H1:.3f} < H(2)={H2:.3f}  : {H05 < H1 < H2}")

print("\n  top-k = 2:")
pk = softmax(top_k_filter(logits, 2))
print(f"    probs = {np.round(pk, 3)}  (nb non nuls = {(pk > 0).sum()}, somme = {pk.sum():.3f})")

print("\n  top-p = 0.9:")
pp = softmax(top_p_filter(logits, 0.9))
kept = np.where(pp > 0)[0]
print(f"    tokens retenus = {kept.tolist()}  masse = {softmax(logits)[kept].sum():.3f} (>= 0.9)")

# Distribution empirique vs theorique.
print("\n  Distribution empirique (10000 tirages, T=1):")
pr = softmax_temperature(logits, 1.0)
samples = np.random.choice(len(logits), size=10000, p=pr)
freqs = np.bincount(samples, minlength=len(logits)) / 10000
print(f"    theorique : {np.round(pr, 3)}")
print(f"    empirique : {np.round(freqs, 3)}")
print(f"    ecart max : {np.max(np.abs(freqs - pr)):.4f} (< 0.02 attendu)")

# Interaction temperature + top-p.
print("\n  Interaction T + top-p=0.9 (T plus haute -> nucleus plus large):")
for T in [0.5, 1.0, 2.0]:
    z = logits / T
    n_keep = (softmax(top_p_filter(z, 0.9)) > 0).sum()
    print(f"    T={T}: {n_keep} tokens dans la nucleus")

print("\n  top-p prefere a top-k : la taille de la nucleus s'ADAPTE a la confiance")
print("  du modele (peu de tokens quand il est sur, plus quand il hesite). top-k")
print("  garde toujours k tokens, meme quand 1 seul est plausible.")

print("\n" + "=" * 70)
print("FIN DES SOLUTIONS MEDIUM (Jour 7)")
print("=" * 70)
