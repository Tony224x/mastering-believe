"""
Solutions MEDIUM — Jour 14 : Capstone (extensions mini-LLaMA)
=============================================================
Exercices 4, 5, 6 (medium).

Pur NumPy : on implemente un mini-LM char-level ENTRAINABLE (forward + backward),
plus l'attention RoPE+KV-cache du capstone pour prouver l'equivalence cache/no-cache.
Chaque etape non triviale est commentee avec le POURQUOI.

(Le code de reference 02-code/14-capstone.py est en PyTorch ; ici tout tourne en
NumPy pour etre lance sans framework.)

Run: python 03-exercises/solutions/14-capstone-medium.py
"""

import sys
import io
import math
import numpy as np

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

np.random.seed(42)


# ============================================================================
# Tokenizer char-level (comme le capstone)
# ============================================================================

class CharTokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


# ============================================================================
# Mini-LM char-level ENTRAINABLE (NumPy)
# Architecture volontairement simple pour un backward tractable :
#   embedding -> attention causale mono-head (avec masque) -> lm_head (tie).
# Suffit pour montrer : training + courbe de loss + perplexite + weight tying.
# ============================================================================

class MiniLM:
    def __init__(self, vocab, d=32, tie=True, seed=0):
        rng = np.random.default_rng(seed)
        self.vocab = vocab
        self.d = d
        self.tie = tie
        self.E = rng.standard_normal((vocab, d)) * 0.1        # token embedding
        # attention mono-head
        self.Wq = rng.standard_normal((d, d)) * (1/np.sqrt(d))
        self.Wk = rng.standard_normal((d, d)) * (1/np.sqrt(d))
        self.Wv = rng.standard_normal((d, d)) * (1/np.sqrt(d))
        self.Wo = rng.standard_normal((d, d)) * (1/np.sqrt(d))
        if not tie:
            self.Wlm = rng.standard_normal((d, vocab)) * (1/np.sqrt(d))

    def lm_weight(self):
        # tie: lm_head partage la table d'embedding (E.T) -> economise vocab*d params
        return self.E.T if self.tie else self.Wlm

    def forward(self, ids):
        """
        ids: (T,) sequence d'entiers. Renvoie logits (T, vocab) + cache pour backward.
        """
        T = len(ids)
        x = self.E[ids]                       # (T, d)
        Q = x @ self.Wq                       # (T, d)
        K = x @ self.Wk
        V = x @ self.Wv
        scores = Q @ K.T / math.sqrt(self.d)  # (T, T)
        mask = np.triu(np.full((T, T), -np.inf), k=1)
        scores = scores + mask                # causal
        A = softmax(scores, axis=-1)          # (T, T)
        ctx = A @ V                            # (T, d)
        attn_out = ctx @ self.Wo              # (T, d)
        h = x + attn_out                       # residual
        logits = h @ self.lm_weight()         # (T, vocab)
        cache = (ids, x, Q, K, V, scores, A, ctx, attn_out, h)
        return logits, cache

    def loss_and_grad(self, ids, targets):
        """Cross-entropy next-token + tous les gradients (backprop a la main)."""
        logits, (ids_, x, Q, K, V, scores, A, ctx, attn_out, h) = self.forward(ids)
        T = len(ids)
        P = softmax(logits, axis=-1)           # (T, vocab)
        loss = -np.mean(np.log(P[np.arange(T), targets] + 1e-12))

        # dL/dlogits
        dlogits = P.copy()
        dlogits[np.arange(T), targets] -= 1.0
        dlogits /= T                            # (T, vocab)

        Wlm = self.lm_weight()
        grads = {}
        # logits = h @ Wlm
        dh = dlogits @ Wlm.T                     # (T, d)
        dWlm = h.T @ dlogits                     # (d, vocab)
        # h = x + attn_out
        dx = dh.copy()
        dattn_out = dh.copy()
        # attn_out = ctx @ Wo
        grads['Wo'] = ctx.T @ dattn_out
        dctx = dattn_out @ self.Wo.T            # (T, d)
        # ctx = A @ V
        dA = dctx @ V.T                          # (T, T)
        dV = A.T @ dctx                          # (T, d)
        # A = softmax(scores) (par ligne)
        # dscores = A * (dA - sum(dA*A, axis=-1, keepdims))
        dscores = A * (dA - (dA * A).sum(axis=-1, keepdims=True))
        dscores = dscores / math.sqrt(self.d)
        # scores = Q @ K.T
        dQ = dscores @ K                         # (T, d)
        dK = dscores.T @ Q                       # (T, d)
        # Q,K,V = x @ Wq/Wk/Wv
        grads['Wq'] = x.T @ dQ
        grads['Wk'] = x.T @ dK
        grads['Wv'] = x.T @ dV
        dx = dx + dQ @ self.Wq.T + dK @ self.Wk.T + dV @ self.Wv.T
        # x = E[ids] ; et si tie, Wlm = E.T -> E recoit AUSSI un gradient via lm_head
        dE = np.zeros_like(self.E)
        np.add.at(dE, ids, dx)                   # gradient via l'embedding lookup
        if self.tie:
            dE += dWlm.T                          # gradient via lm_head partage
        else:
            grads['Wlm'] = dWlm
        grads['E'] = dE
        return loss, grads

    def step(self, grads, lr):
        for name, g in grads.items():
            setattr(self, name, getattr(self, name) - lr * g)

    def n_params(self):
        n = self.E.size + self.Wq.size + self.Wk.size + self.Wv.size + self.Wo.size
        if not self.tie:
            n += self.Wlm.size
        return n


# ============================================================================
# EXERCICE 4 : Entrainer le mini-LM + courbe de loss
# ============================================================================

print("=" * 70)
print("EXERCICE 4 : Entrainer le mini-LM + courbe de loss")
print("=" * 70)

corpus = "le chat dort sur le tapis. le chien dort sur le canape. " * 6
tok = CharTokenizer(corpus)
data = tok.encode(corpus)
print(f"\nVocab size: {tok.vocab_size}, corpus len: {len(data)} chars")

model = MiniLM(tok.vocab_size, d=32, tie=True, seed=1)

# Generation AVANT entrainement
def generate(model, tok, prompt, n=40, seed=0):
    rng = np.random.default_rng(seed)
    ids = tok.encode(prompt)
    for _ in range(n):
        logits, _ = model.forward(ids[-32:])  # contexte limite
        p = softmax(logits[-1])
        ids.append(int(rng.choice(len(p), p=p)))
    return tok.decode(ids)


print(f"\nAVANT entrainement : {generate(model, tok, 'le ', 30)!r}")

# Boucle d'entrainement
seq_len = 32
losses = []
lr = 0.3
for epoch in range(400):
    # un batch = une fenetre aleatoire
    i = np.random.randint(0, len(data) - seq_len - 1)
    ids = data[i:i + seq_len]
    targets = data[i + 1:i + seq_len + 1]
    loss, grads = model.loss_and_grad(ids, targets)
    model.step(grads, lr)
    losses.append(loss)
    if epoch % 80 == 0 or epoch == 399:
        print(f"  epoch {epoch:>3d} : loss = {loss:.4f}")

# Courbe de loss ASCII (loss moyenne par tranche)
print("\nCourbe de loss (moyenne par tranche de 40 epochs):")
for s in range(0, 400, 40):
    avg = np.mean(losses[s:s + 40])
    bar = "#" * int(avg * 12)
    print(f"  ep {s:>3d}-{s+39:>3d} : {avg:.3f} {bar}")

print(f"\nAPRES entrainement : {generate(model, tok, 'le ', 30, seed=1)!r}")
print("""
Analyse: la loss descend nettement -> le modele apprend les motifs du corpus.
Loss d'un modele parfait ~ entropie du corpus (ici tres basse, corpus repetitif).
Corpus minuscule -> overfitting attendu : le modele memorise (loss train tres basse,
generaliserait mal sur du texte nouveau).
""")


# ============================================================================
# EXERCICE 6 : Perplexite avant/apres + weight tying
# (place avant l'Ex5 car elle reutilise `model` ; l'Ex5 cree sa propre attention)
# ============================================================================

print("=" * 70)
print("EXERCICE 6 : Perplexite + weight tying")
print("=" * 70)


def perplexity(model, ids):
    """PPL = exp( -(1/N) sum log p(token_i | <i) ). Exp de la cross-entropy moyenne."""
    nll = 0.0
    N = len(ids) - 1
    logits, _ = model.forward(ids)
    P = softmax(logits, axis=-1)
    for t in range(N):
        nll += -math.log(P[t, ids[t + 1]] + 1e-12)
    return math.exp(nll / N)


# Borne : modele uniforme -> PPL = vocab_size
uniform_logits = np.zeros((1, tok.vocab_size))
ppl_uniform = math.exp(-math.log(softmax(uniform_logits)[0, 0] + 1e-12))
print(f"\nPPL d'un modele uniforme = {ppl_uniform:.2f} (== vocab_size = {tok.vocab_size})")

# Avant/apres : on compare un modele frais vs le modele entraine
fresh = MiniLM(tok.vocab_size, d=32, tie=True, seed=99)
test_ids = tok.encode("le chat dort sur le tapis.")
print(f"PPL modele frais (aleatoire) : {perplexity(fresh, data[:64]):.2f} (~ vocab_size)")
print(f"PPL modele entraine (train)  : {perplexity(model, data[:64]):.2f}")
print(f"PPL modele entraine (test)   : {perplexity(model, test_ids):.2f}")

# Weight tying : economie de params
tied = MiniLM(tok.vocab_size, d=32, tie=True, seed=2)
untied = MiniLM(tok.vocab_size, d=32, tie=False, seed=2)
saving = untied.n_params() - tied.n_params()
print(f"\nParams sans tying : {untied.n_params():,}")
print(f"Params avec tying : {tied.n_params():,}")
print(f"Economie          : {saving:,} = vocab*d = {tok.vocab_size * 32} "
      f"({100*saving/untied.n_params():.1f}% du total)")
print("""
Analyse:
- PPL est plus interpretable que la loss : c'est le nombre moyen de choix 'equivalents'
  que le modele hesite par token. PPL=vocab -> il devine au hasard ; PPL faible -> il sait.
- PPL compare des modeles independamment de la base du log et du vocab (via exp(CE)).
- Weight tying economise vocab*d params quasi gratuitement : input et output embeddings
  vivent dans le meme espace, les partager regularise et reduit la taille (standard en LLM).
""")


# ============================================================================
# EXERCICE 5 : KV cache vs forward complet (attention RoPE+GQA du capstone)
# ============================================================================

print("=" * 70)
print("EXERCICE 5 : KV cache == forward complet (RoPE + GQA)")
print("=" * 70)


def precompute_rope(head_dim, max_seq, base=10000.0):
    freqs = 1.0 / (base ** (np.arange(0, head_dim, 2) / head_dim))
    pos = np.arange(max_seq)
    angles = np.outer(pos, freqs)             # (max_seq, head_dim/2)
    return np.cos(angles), np.sin(angles)


def apply_rope(x, cos, sin):
    """x: (seq, head_dim). Convention interleaved (paires 0-1, 2-3, ...)."""
    x1 = x[:, 0::2]
    x2 = x[:, 1::2]
    out = np.empty_like(x)
    out[:, 0::2] = x1 * cos - x2 * sin
    out[:, 1::2] = x1 * sin + x2 * cos
    return out


class RoPEAttention:
    """Attention mono-head causale avec RoPE. Deux modes: full et cached."""

    def __init__(self, d, max_seq=64, seed=0):
        rng = np.random.default_rng(seed)
        self.d = d
        self.Wq = rng.standard_normal((d, d)) * 0.1
        self.Wk = rng.standard_normal((d, d)) * 0.1
        self.Wv = rng.standard_normal((d, d)) * 0.1
        self.Wo = rng.standard_normal((d, d)) * 0.1
        self.cos, self.sin = precompute_rope(d, max_seq)

    def full(self, X):
        """Forward complet sur (T, d). Renvoie sortie (T, d)."""
        T = X.shape[0]
        Q = apply_rope(X @ self.Wq, self.cos[:T], self.sin[:T])
        K = apply_rope(X @ self.Wk, self.cos[:T], self.sin[:T])
        V = X @ self.Wv
        scores = Q @ K.T / math.sqrt(self.d) + np.triu(np.full((T, T), -np.inf), 1)
        out = softmax(scores) @ V
        return out @ self.Wo

    def step(self, x_new, start_pos, cache):
        """Decode un token a la position absolue start_pos, avec cache."""
        q = apply_rope((x_new @ self.Wq)[None, :],
                       self.cos[start_pos:start_pos+1], self.sin[start_pos:start_pos+1])[0]
        k = apply_rope((x_new @ self.Wk)[None, :],
                       self.cos[start_pos:start_pos+1], self.sin[start_pos:start_pos+1])[0]
        v = x_new @ self.Wv
        cache['k'].append(k)
        cache['v'].append(v)
        K = np.stack(cache['k'])
        V = np.stack(cache['v'])
        scores = (K @ q) / math.sqrt(self.d)
        w = softmax(scores)
        out = w @ V
        return out @ self.Wo


d = 16
attn = RoPEAttention(d, max_seq=64, seed=3)
X = np.random.randn(20, d)

# Mode A : pour chaque t, full sur [0..t], prendre la derniere sortie
out_full = np.stack([attn.full(X[:t + 1])[-1] for t in range(X.shape[0])])

# Mode B : cache, token par token avec start_pos correct
cache = {'k': [], 'v': []}
out_cache = np.stack([attn.step(X[t], t, cache) for t in range(X.shape[0])])

print(f"\nEquivalence cache vs full : ecart max = "
      f"{np.max(np.abs(out_full - out_cache)):.2e}")

# Piege : oublier d'incrementer start_pos (toujours 0) -> RoPE faux -> divergence
cache_bug = {'k': [], 'v': []}
out_bug = np.stack([attn.step(X[t], 0, cache_bug) for t in range(X.shape[0])])
print(f"Avec start_pos FIGE a 0 (bug) : ecart max = "
      f"{np.max(np.abs(out_full - out_bug)):.2e}  -> diverge !")
print("""
Analyse:
- Le cache donne EXACTEMENT les memes logits/sorties que le forward complet (ecart
  machine) : il ne fait que reutiliser les K,V deja calcules.
- start_pos doit etre la position ABSOLUE : RoPE encode la position dans la rotation.
  Si on fige start_pos=0 au decode, tous les tokens recoivent l'angle de la position 0
  -> les positions relatives sont fausses -> divergence (preuve par l'absurde).
- Speedup: le full recalcule O(t) cles a chaque step (O(n^2) total) ; le cache n'ajoute
  qu'une cle par step (O(n) de calcul d'attention par token).
""")

print("=" * 70)
print("Fin solutions medium Jour 14.")
print("=" * 70)
