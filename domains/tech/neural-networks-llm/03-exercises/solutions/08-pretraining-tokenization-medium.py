"""
Solutions MEDIUM — Jour 8 : Pretraining & Tokenization
======================================================
Exercices 4, 5, 6 (medium). NumPy + stdlib (comme 02-code/08).

4. BPE complet : train + encode + decode + invariant round-trip.
5. Cross-entropy, bits-per-token, perplexite (bigramme vs uniforme).
6. Scaling laws : optimum Chinchilla + frontiere iso-compute.

Run: python 03-exercises/solutions/08-pretraining-tokenization-medium.py
"""

import numpy as np
from collections import Counter

np.random.seed(42)

END_OF_WORD = "</w>"


# ----------------------------------------------------------------------------
# Briques BPE (reprises de 02-code/08)
# ----------------------------------------------------------------------------

def pre_tokenize(word):
    return list(word) + [END_OF_WORD]


def get_pair_stats(word_freqs):
    pairs = Counter()
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += freq
    return pairs


def merge_pair(pair, word_freqs):
    a, b = pair
    merged = a + b
    new = {}
    for word, freq in word_freqs.items():
        nw, i = [], 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                nw.append(merged); i += 2
            else:
                nw.append(word[i]); i += 1
        new[tuple(nw)] = freq
    return new


def train_bpe(corpus_freqs, num_merges):
    merges = []
    cur = dict(corpus_freqs)
    for _ in range(num_merges):
        pairs = get_pair_stats(cur)
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        cur = merge_pair(best, cur)
        merges.append(best)
    return merges


def encode_word(word, merges):
    tokens = pre_tokenize(word)
    for pair in merges:
        merged = pair[0] + pair[1]
        nt, i = [], 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                nt.append(merged); i += 2
            else:
                nt.append(tokens[i]); i += 1
        tokens = nt
    return tokens


# ============================================================================
# EXERCISE 4: BPE complet + decode + round-trip
# ============================================================================

print("=" * 70)
print("EXERCISE 4: BPE complet — train, encode, decode, round-trip")
print("=" * 70)


def decode_tokens(tokens):
    """
    Reconstruit le mot. Les tokens BPE sont des fragments ; le dernier porte
    le marqueur </w>. On concatene tout et on retire le marqueur final.
    """
    s = "".join(tokens)
    # Le </w> ne doit apparaitre qu'a la fin (fin de mot).
    assert s.endswith(END_OF_WORD), "le dernier token doit porter </w>"
    return s[:-len(END_OF_WORD)]


corpus = ["low"] * 5 + ["lower"] * 2 + ["newest"] * 6 + ["widest"] * 3
word_freqs = Counter(tuple(pre_tokenize(w)) for w in corpus)
merges = train_bpe(word_freqs, num_merges=10)

# Round-trip sur mots vus ET non vus.
test_words = ["low", "lowest", "newer", "wider", "slowest", "widest", "newestwide"]
print("\n  Round-trip decode(encode(w)) == w:")
all_ok = True
for w in test_words:
    toks = encode_word(w, merges)
    back = decode_tokens(toks)
    ok = (back == w)
    all_ok = all_ok and ok
    disp = " | ".join(t.replace(END_OF_WORD, "_") for t in toks)
    print(f"    {w:<12s} -> [{disp}] -> {back!r}  [{'OK' if ok else 'FAIL'}]")
print(f"  Tous les round-trips OK : {all_ok}")

# Couverture vs taille de vocab.
print("\n  Longueur moyenne de sequence selon le nombre de merges:")
eval_words = ["lowest", "newer", "widest", "slowest", "lower", "newest"]
for nm in [0, 5, 10, 20, 40]:
    mg = train_bpe(word_freqs, num_merges=nm)
    avg_len = np.mean([len(encode_word(w, mg)) for w in eval_words])
    vocab_extra = nm  # chaque merge ajoute 1 token au vocab
    print(f"    merges={nm:>2}: longueur moy = {avg_len:.2f} tokens/mot  (+{vocab_extra} tokens vocab)")
print("  -> plus de merges = sequences plus courtes mais vocab plus gros (le tradeoff).")

# Determinisme : l'ordre des merges compte.
print("\n  Ordre des merges:")
correct = encode_word("newest", merges)
shuffled = list(reversed(merges))
wrong = encode_word("newest", shuffled)
print(f"    ordre correct  : {[t.replace(END_OF_WORD, '_') for t in correct]}")
print(f"    ordre inverse  : {[t.replace(END_OF_WORD, '_') for t in wrong]}")
print("    -> les merges tardifs dependent des precoces ; l'encodeur DOIT suivre l'ordre.")


# ============================================================================
# EXERCISE 5: Cross-entropy, bits-per-token, perplexite
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 5: Cross-entropy, bits-per-token, perplexite")
print("=" * 70)

text = ("le chat dort sur le tapis le chien aboie le chat mange "
        "le chat dort le chien dort le chat court ") * 6
chars = sorted(set(text))
V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
ids = np.array([stoi[c] for c in text])

# Split train/test.
n_train = int(len(ids) * 0.8)
train_ids, test_ids = ids[:n_train], ids[n_train:]

# Bigramme avec lissage de Laplace (+1).
counts = np.ones((V, V))                       # +1 = Laplace
for a, b in zip(train_ids[:-1], train_ids[1:]):
    counts[a, b] += 1
P = counts / counts.sum(axis=1, keepdims=True)  # lignes normalisees
assert np.allclose(P.sum(axis=1), 1.0)


def metrics(P, seq):
    """Cross-entropy (nats), bits/token, perplexite sur une suite next-token."""
    logps = np.log(P[seq[:-1], seq[1:]])
    ce_nats = -logps.mean()
    bpt = ce_nats / np.log(2)
    ppl = np.exp(ce_nats)
    return ce_nats, bpt, ppl


ce, bpt, ppl = metrics(P, test_ids)
print(f"\n  Bigramme (test): CE = {ce:.4f} nats | bits/token = {bpt:.4f} | perplexite = {ppl:.4f}")

# Relation perplexite = 2 ** bits_per_token.
print(f"  Verif perplexite == 2**bits/token : {ppl:.6f} vs {2 ** bpt:.6f} "
      f"(ecart {abs(ppl - 2 ** bpt):.2e})")

# Modele uniforme : perplexite == V, bits/token == log2(V).
P_unif = np.full((V, V), 1.0 / V)
ce_u, bpt_u, ppl_u = metrics(P_unif, test_ids)
print(f"\n  Uniforme: perplexite = {ppl_u:.4f} (== V = {V}) | bits/token = {bpt_u:.4f} (== log2(V) = {np.log2(V):.4f})")
print(f"  Verif: perplexite uniforme == V : {np.isclose(ppl_u, V)}")

gain = (1 - ppl / ppl_u) * 100
print(f"\n  Le bigramme reduit la perplexite de {gain:.1f}% vs uniforme ({ppl:.2f} vs {ppl_u:.2f}).")

# Effet du lissage.
print("\n  Sans lissage: une transition inconnue -> P=0 -> log(0) = -inf -> perplexite infinie.")
counts0 = np.zeros((V, V))
for a, b in zip(train_ids[:-1], train_ids[1:]):
    counts0[a, b] += 1
rowsum = counts0.sum(axis=1, keepdims=True)
P0 = np.divide(counts0, rowsum, out=np.zeros_like(counts0), where=rowsum > 0)
probs0 = P0[test_ids[:-1], test_ids[1:]]
print(f"    nb de transitions test avec P=0 (jamais vues): {(probs0 == 0).sum()}")
print("    -> ces zeros font diverger la perplexite. Le lissage est indispensable.")


# ============================================================================
# EXERCISE 6: Scaling laws — Chinchilla + frontiere iso-compute
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 6: Scaling laws — Chinchilla")
print("=" * 70)


def compute_flops(N, D):
    """Compute d'entrainement ~ 6 * N * D FLOPs (N params, D tokens)."""
    return 6.0 * N * D


def chinchilla_optimal(C, tokens_per_param=20.0):
    """
    Resout C = 6*N*D avec D = 20*N -> N = sqrt(C / (6*20)).
    Renvoie (N_opt, D_opt).
    """
    N = np.sqrt(C / (6.0 * tokens_per_param))
    D = tokens_per_param * N
    return N, D


# Cas reels.
print("\n  Cas reels (ratio D/N = tokens par parametre):")
for name, N, D in [("GPT-3", 175e9, 300e9), ("Chinchilla", 70e9, 1.4e12),
                   ("LLaMA-3-8B", 8e9, 15e12)]:
    C = compute_flops(N, D)
    print(f"    {name:<12s} N={N:.2e} D={D:.2e} | D/N = {D / N:5.1f} | C = {C:.2e} FLOPs")
print("  -> GPT-3 (1.7) sous-entraine ; Chinchilla (20) optimal ; LLaMA-3 (1875) inference-aware.")

# Chinchilla a compute de GPT-3.
C_gpt3 = compute_flops(175e9, 300e9)
N_opt, D_opt = chinchilla_optimal(C_gpt3)
print(f"\n  A compute = compute(GPT-3) = {C_gpt3:.2e}:")
print(f"    optimum Chinchilla : N = {N_opt:.2e} params, D = {D_opt:.2e} tokens (D/N = {D_opt/N_opt:.0f})")
print(f"    -> modele PLUS PETIT ({N_opt/175e9:.2f}x GPT-3) mais PLUS de donnees.")

# Frontiere iso-compute : minimiser L(N,D) = E + A/N^a + B/D^b sous 6*N*D = C.
print("\n  Frontiere iso-compute (loss modelisee Chinchilla):")
A, B, alpha, beta, E = 406.0, 410.0, 0.34, 0.28, 1.69
C = 1e21
best = None
for ratio in [1, 5, 10, 20, 40, 100, 500]:        # D/N candidats
    # 6*N*D = C et D = ratio*N -> N = sqrt(C / (6*ratio))
    N = np.sqrt(C / (6.0 * ratio))
    D = ratio * N
    L = E + A / N ** alpha + B / D ** beta
    marker = ""
    if best is None or L < best[0]:
        best = (L, ratio)
    print(f"    D/N={ratio:>4}: N={N:.2e} D={D:.2e} L={L:.4f}")
print(f"  -> loss minimale autour de D/N = {best[1]} (proche de l'optimum theorique ~20).")

# Inference-aware.
print("\n  Inference-aware : le compute d'INFERENCE ~ 2*N par token genere.")
N_chin = 70e9
N_llama = 8e9
print(f"    Servir 1e12 tokens : Chinchilla-70B coute ~{2 * N_chin * 1e12:.2e} FLOPs,")
print(f"                          LLaMA-8B coute ~{2 * N_llama * 1e12:.2e} FLOPs ({N_chin/N_llama:.1f}x moins).")
print("    -> on sur-entraine un petit modele : training cher UNE fois, inference moins chere MILLIARDS de fois.")

print("\n" + "=" * 70)
print("FIN DES SOLUTIONS MEDIUM (Jour 8)")
print("=" * 70)
