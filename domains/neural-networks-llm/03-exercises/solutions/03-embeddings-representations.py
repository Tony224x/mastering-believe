"""
Solutions — Jour 3 : Embeddings & Representations
===================================================
Solutions pour les 8 exercices (easy, medium, hard).

Run: python 03-exercises/solutions/03-embeddings-representations.py
"""

import sys
import io
import numpy as np
import time

# Force UTF-8 output to handle special characters in comments/docstrings
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

np.random.seed(42)


# ============================================================================
# HELPERS: Reused across exercises
# ============================================================================

def cosine_similarity(a, b):
    """cos(a, b) = (a . b) / (||a|| * ||b||)"""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def sigmoid(z):
    """Numerically stable sigmoid."""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


# Shared corpus — same semantic structure as the main code
CORPUS = [
    "le roi gouverne le royaume avec sagesse",
    "la reine gouverne le royaume avec bonte",
    "le prince herite du royaume",
    "la princesse herite du royaume",
    "le roi et la reine vivent au chateau",
    "le prince et la princesse vivent au chateau",
    "le chat dort sur le tapis",
    "le chien dort sur le canape",
    "le chat mange du poisson",
    "le chien mange de la viande",
    "le chat et le chien jouent ensemble",
    "homme et femme vivent ensemble",
    "le garcon et la fille jouent ensemble",
    "le roi est un homme puissant",
    "la reine est une femme puissante",
    "le prince est un jeune homme",
    "la princesse est une jeune femme",
    "le garcon mange du poisson",
    "la fille mange de la viande",
    "le chat est un animal fidele",
    "le chien est un animal fidele",
    "le roi aime le chateau",
    "la reine aime le chateau",
    "homme et femme gouvernent le royaume",
]


def build_vocab(corpus):
    """Build vocabulary from corpus, return word2idx, idx2word, word_counts."""
    all_words = []
    for sentence in corpus:
        all_words.extend(sentence.split())
    word_counts = {}
    for w in all_words:
        word_counts[w] = word_counts.get(w, 0) + 1
    sorted_words = sorted(word_counts.keys(), key=lambda w: -word_counts[w])
    word2idx = {w: i for i, w in enumerate(sorted_words)}
    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word, word_counts, all_words


# ============================================================================
# EXERCISE 1: One-hot vs embeddings — calcul a la main
# ============================================================================

print("=" * 70)
print("EXERCISE 1: One-hot vs Embeddings")
print("=" * 70)

vocab_ex1 = ["paris", "lyon", "france", "chat", "chien"]
V_ex1 = len(vocab_ex1)

# Part a: One-hot vectors
print("\n--- Part a: One-hot encoding ---")
onehot = np.eye(V_ex1)
for w, vec in zip(vocab_ex1, onehot):
    print(f"  {w:>8s} = {vec.astype(int).tolist()}")

print("\n  Similarite cosinus (toutes les paires):")
for i in range(V_ex1):
    for j in range(i + 1, V_ex1):
        sim = cosine_similarity(onehot[i], onehot[j])
        print(f"    cos({vocab_ex1[i]}, {vocab_ex1[j]}) = {sim:.4f}")

print("\n  CONSTAT: Toutes les paires ont similarite = 0.00")

# Part b: Dense embeddings
print("\n--- Part b: Embeddings denses (inventes) ---")
# Dimensions interpretees comme: [ville, animal, france]
dense_embeddings = {
    "paris":  np.array([0.9, 0.1, 0.8]),
    "lyon":   np.array([0.8, 0.1, 0.7]),
    "france": np.array([0.5, 0.0, 1.0]),
    "chat":   np.array([0.0, 0.9, 0.1]),
    "chien":  np.array([0.1, 0.8, 0.1]),
}

# Part c: Cosine similarities with dense embeddings
print("\n--- Part c: Similarites avec embeddings denses ---")
pairs_sorted = []
for i in range(V_ex1):
    for j in range(i + 1, V_ex1):
        w1, w2 = vocab_ex1[i], vocab_ex1[j]
        sim = cosine_similarity(dense_embeddings[w1], dense_embeddings[w2])
        pairs_sorted.append((w1, w2, sim))
        print(f"    cos({w1}, {w2}) = {sim:.4f}")

pairs_sorted.sort(key=lambda x: -x[2])
print("\n  Classement (plus similaire en premier):")
for w1, w2, sim in pairs_sorted:
    print(f"    ({w1}, {w2}) = {sim:.4f}")

# Part d: Parameter count
print("\n--- Part d: Nombre de parametres ---")
V_large = 50000
d_embed = 256
print(f"  Vocabulaire: {V_large} mots")
print(f"  One-hot: chaque vecteur = {V_large} dims → matrice premiere couche = {V_large} x hidden")
print(f"  Embedding (d={d_embed}): matrice = {V_large} x {d_embed} = {V_large * d_embed:,} parametres")
print(f"  Ratio de compression: {V_large}/{d_embed} = {V_large/d_embed:.0f}x")


# ============================================================================
# EXERCISE 2: Construire les paires Skip-gram
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 2: Paires Skip-gram a la main")
print("=" * 70)

phrase = "le chat mange du poisson frais"
words = phrase.split()
print(f"\n  Phrase: '{phrase}'")
print(f"  Mots: {words}")

for window_size in [1, 2, 3]:
    pairs = []
    for i, center in enumerate(words):
        for j in range(max(0, i - window_size), min(len(words), i + window_size + 1)):
            if j != i:
                pairs.append((center, words[j]))

    print(f"\n--- Window size = {window_size} ---")
    print(f"  Nombre de paires: {len(pairs)}")
    for center, ctx in pairs:
        print(f"    ({center}, {ctx})")

# Analysis: which words appear most
print("\n--- Analyse des mots aux bords ---")
print("  Les mots au debut ('le') et a la fin ('frais') ont MOINS de paires")
print("  car leur fenetre est tronquee par les limites de la phrase.")
print("  Les mots au milieu ('mange', 'du') ont le maximum de paires.")

print("\n--- Window petite vs grande ---")
print("  Window petite (1): capture les relations SYNTAXIQUES (mots adjacents)")
print("    Ex: (chat, le), (chat, mange) — relations grammaticales")
print("  Window grande (3+): capture les relations SEMANTIQUES (mots dans le meme contexte)")
print("    Ex: (chat, poisson) — relation thematique")

# Corpus comparison
print("\n--- Corpus A vs Corpus B (window=1) ---")
corpus_a = ["le chat dort", "le chien dort"]
corpus_b = ["le chat mange", "le chien joue"]

for name, corp in [("A", corpus_a), ("B", corpus_b)]:
    all_pairs = []
    for sent in corp:
        ws = sent.split()
        for i, center in enumerate(ws):
            for j in range(max(0, i - 1), min(len(ws), i + 2)):
                if j != i:
                    all_pairs.append((center, ws[j]))
    print(f"\n  Corpus {name}: {corp}")
    print(f"  Paires: {all_pairs}")

print("\n  Corpus A: 'chat' et 'chien' partagent les memes contextes ('le', 'dort')")
print("  Corpus B: 'chat' a le contexte ('mange'), 'chien' a ('joue') — differents")
print("  → Dans A, chat/chien auront des embeddings PLUS PROCHES car memes contextes.")


# ============================================================================
# EXERCISE 3: Cosine vs Euclidean
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 3: Cosine vs Euclidean")
print("=" * 70)

A = np.array([1, 2, 3], dtype=float)
B = np.array([2, 4, 6], dtype=float)
C = np.array([3, 2, 1], dtype=float)
D = np.array([0.1, 0.2, 0.3], dtype=float)

vectors = {"A": A, "B": B, "C": C, "D": D}
names = list(vectors.keys())

# Part 1 & 2: All pairwise similarities and distances
print("\n--- Cosine Similarity ---")
cos_pairs = []
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        n1, n2 = names[i], names[j]
        sim = cosine_similarity(vectors[n1], vectors[n2])
        cos_pairs.append((n1, n2, sim))
        print(f"  cos({n1}, {n2}) = {sim:.4f}")

print("\n--- Euclidean Distance ---")
euc_pairs = []
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        n1, n2 = names[i], names[j]
        dist = np.linalg.norm(vectors[n1] - vectors[n2])
        euc_pairs.append((n1, n2, dist))
        print(f"  dist({n1}, {n2}) = {dist:.4f}")

# Part 3: Rankings
print("\n--- Classement par cosinus (plus similaire en premier) ---")
cos_pairs.sort(key=lambda x: -x[2])
for n1, n2, sim in cos_pairs:
    print(f"  ({n1}, {n2}) = {sim:.4f}")

print("\n--- Classement par distance (plus proche en premier) ---")
euc_pairs.sort(key=lambda x: x[2])
for n1, n2, dist in euc_pairs:
    print(f"  ({n1}, {n2}) = {dist:.4f}")

# Part 4: Analysis
print("\n--- Analyse ---")
print("  Cosinus: A, B et D sont IDENTIQUES (cos = 1.0) car meme direction")
print("  Euclidienne: A et D sont tres proches (0.94) mais B est loin de D (6.55)")
print("  → Les classements DIVERGENT !")
print()
print("  En NLP: la direction d'un embedding = le SENS du mot")
print("  La magnitude = la frequence ou l'intensite")
print("  Un mot rare et un mot frequent avec le meme sens doivent etre 'similaires'")
print("  → Le cosinus capture ca, la distance euclidienne non.")

# Part 6 - Bonus: Normalized vectors
print("\n--- Bonus: Apres normalisation L2 ---")
for name, vec in vectors.items():
    normed = vec / np.linalg.norm(vec)
    print(f"  {name}_norm = [{', '.join(f'{x:.4f}' for x in normed)}]")

normed_vectors = {k: v / np.linalg.norm(v) for k, v in vectors.items()}

print("\n  Distances apres normalisation:")
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        n1, n2 = names[i], names[j]
        dist = np.linalg.norm(normed_vectors[n1] - normed_vectors[n2])
        cos = cosine_similarity(normed_vectors[n1], normed_vectors[n2])
        print(f"  ({n1}, {n2}): dist = {dist:.4f}, cos = {cos:.4f}")

print("\n  CONSTAT: apres normalisation, dist^2 = 2 - 2*cos")
print("  Les deux mesures donnent le MEME classement. CQFD.")


# ============================================================================
# EXERCISE 4: Word2Vec CBOW from scratch
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 4: Word2Vec CBOW from Scratch")
print("=" * 70)

word2idx, idx2word, word_counts, all_words = build_vocab(CORPUS)
V = len(word2idx)
EMBED_DIM = 20
WINDOW_SIZE = 2
K_NEG = 5

# Negative sampling distribution (freq^0.75)
word_freqs = np.array([word_counts[idx2word[i]] for i in range(V)], dtype=float)
neg_dist = word_freqs ** 0.75
neg_dist /= neg_dist.sum()

# CBOW training data: for each center word, collect all context words
print("\n--- Building CBOW training data ---")
cbow_data = []  # Each entry: (center_idx, [context_idx_1, context_idx_2, ...])
for sentence in CORPUS:
    words = sentence.split()
    for i, center_word in enumerate(words):
        context_indices = []
        for j in range(max(0, i - WINDOW_SIZE), min(len(words), i + WINDOW_SIZE + 1)):
            if j != i:
                context_indices.append(word2idx[words[j]])
        if context_indices:
            cbow_data.append((word2idx[center_word], context_indices))

print(f"  Training samples: {len(cbow_data)}")

# Initialize embeddings
W_in_cbow = np.random.randn(V, EMBED_DIM) * 0.1
W_out_cbow = np.random.randn(V, EMBED_DIM) * 0.1

# Train CBOW
LR = 0.025
EPOCHS = 100

print(f"\n--- Training CBOW ---")
t_start = time.time()
cbow_losses = []

for epoch in range(EPOCHS):
    np.random.shuffle(cbow_data)
    epoch_loss = 0.0
    lr = LR * (1.0 - epoch / EPOCHS)
    lr = max(lr, LR * 0.01)

    for center_idx, context_indices in cbow_data:
        # Forward: h = mean of context embeddings
        # WHY mean? CBOW's core idea: the average context predicts the center word.
        # Each context word contributes equally to the prediction.
        h = np.mean(W_in_cbow[context_indices], axis=0)  # (d,)

        # Positive pair: predict center word
        v_center = W_out_cbow[center_idx]
        dot_pos = np.dot(h, v_center)
        sig_pos = sigmoid(dot_pos)
        grad_pos = sig_pos - 1.0
        epoch_loss += -np.log(max(sig_pos, 1e-10))

        # Gradient for h (accumulated from positive + negatives)
        grad_h = grad_pos * v_center

        # Update center word output embedding
        W_out_cbow[center_idx] -= lr * grad_pos * h

        # Negative samples
        neg_indices = np.random.choice(V, size=K_NEG, p=neg_dist)
        for neg_idx in neg_indices:
            if neg_idx == center_idx:
                continue
            v_neg = W_out_cbow[neg_idx]
            dot_neg = np.dot(h, v_neg)
            sig_neg = sigmoid(dot_neg)
            grad_neg = sig_neg
            epoch_loss += -np.log(max(1 - sig_neg, 1e-10))
            grad_h += grad_neg * v_neg
            W_out_cbow[neg_idx] -= lr * grad_neg * h

        # Update context word embeddings
        # WHY divide by len? The gradient flows through the mean operation.
        # d(mean)/d(each_input) = 1/n, so each context word gets 1/n of the gradient.
        grad_per_ctx = grad_h / len(context_indices)
        for ctx_idx in context_indices:
            W_in_cbow[ctx_idx] -= lr * grad_per_ctx

    cbow_losses.append(epoch_loss / len(cbow_data))
    if (epoch + 1) % 25 == 0:
        print(f"  Epoch {epoch+1:3d}/{EPOCHS} | Loss: {cbow_losses[-1]:.4f}")

t_cbow = time.time() - t_start
print(f"\n  CBOW training time: {t_cbow:.2f}s")

# Compare CBOW with Skip-gram (train Skip-gram quickly)
print("\n--- Training Skip-gram for comparison ---")
W_in_sg = np.random.randn(V, EMBED_DIM) * 0.1
W_out_sg = np.random.randn(V, EMBED_DIM) * 0.1

# Build skip-gram pairs
sg_pairs = []
for sentence in CORPUS:
    words = sentence.split()
    for i, center in enumerate(words):
        for j in range(max(0, i - WINDOW_SIZE), min(len(words), i + WINDOW_SIZE + 1)):
            if j != i:
                sg_pairs.append((word2idx[center], word2idx[words[j]]))

t_start = time.time()
for epoch in range(EPOCHS):
    np.random.shuffle(sg_pairs)
    lr = LR * (1.0 - epoch / EPOCHS)
    lr = max(lr, LR * 0.01)

    for center_idx, context_idx in sg_pairs:
        v_center = W_in_sg[center_idx]
        v_context = W_out_sg[context_idx]
        dot_pos = np.dot(v_center, v_context)
        sig_pos = sigmoid(dot_pos)
        grad_pos = sig_pos - 1.0
        grad_center = grad_pos * v_context
        W_out_sg[context_idx] -= lr * grad_pos * v_center

        neg_indices = np.random.choice(V, size=K_NEG, p=neg_dist)
        for neg_idx in neg_indices:
            if neg_idx == context_idx:
                continue
            v_neg = W_out_sg[neg_idx]
            dot_neg = np.dot(v_center, v_neg)
            sig_neg = sigmoid(dot_neg)
            grad_center += sig_neg * v_neg
            W_out_sg[neg_idx] -= lr * sig_neg * v_center

        W_in_sg[center_idx] -= lr * grad_center

t_sg = time.time() - t_start
print(f"  Skip-gram training time: {t_sg:.2f}s")

# Compare key pairs
print("\n--- Comparaison CBOW vs Skip-gram ---")
test_pairs = [("roi", "reine"), ("chat", "chien"), ("homme", "femme")]
for w1, w2 in test_pairs:
    if w1 in word2idx and w2 in word2idx:
        sim_cbow = cosine_similarity(W_in_cbow[word2idx[w1]], W_in_cbow[word2idx[w2]])
        sim_sg = cosine_similarity(W_in_sg[word2idx[w1]], W_in_sg[word2idx[w2]])
        print(f"  ({w1}, {w2}): CBOW={sim_cbow:.4f}, Skip-gram={sim_sg:.4f}")

print(f"\n  Temps: CBOW={t_cbow:.2f}s, Skip-gram={t_sg:.2f}s")
print(f"  CBOW est generalement plus rapide car moins de paires par mot central.")


# ============================================================================
# EXERCISE 5: Negative Sampling — comparing strategies
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 5: Negative Sampling Strategies")
print("=" * 70)


def train_skipgram(neg_distribution, epochs=80, k_neg=5, label=""):
    """Train skip-gram with a given negative sampling distribution."""
    W_in = np.random.randn(V, EMBED_DIM) * 0.1
    W_out = np.random.randn(V, EMBED_DIM) * 0.1
    lr_init = 0.025

    t0 = time.time()
    final_loss = 0.0

    for epoch in range(epochs):
        np.random.shuffle(sg_pairs)
        epoch_loss = 0.0
        lr = lr_init * (1.0 - epoch / epochs)
        lr = max(lr, lr_init * 0.01)

        for center_idx, context_idx in sg_pairs:
            v_c = W_in[center_idx]
            v_ctx = W_out[context_idx]
            dot_p = np.dot(v_c, v_ctx)
            sig_p = sigmoid(dot_p)
            g_p = sig_p - 1.0
            epoch_loss += -np.log(max(sig_p, 1e-10))
            gc = g_p * v_ctx
            W_out[context_idx] -= lr * g_p * v_c

            negs = np.random.choice(V, size=k_neg, p=neg_distribution)
            for ni in negs:
                if ni == context_idx:
                    continue
                v_n = W_out[ni]
                dot_n = np.dot(v_c, v_n)
                sig_n = sigmoid(dot_n)
                epoch_loss += -np.log(max(1 - sig_n, 1e-10))
                gc += sig_n * v_n
                W_out[ni] -= lr * sig_n * v_c

            W_in[center_idx] -= lr * gc

        final_loss = epoch_loss / len(sg_pairs)

    elapsed = time.time() - t0
    return W_in, final_loss, elapsed


# Three strategies
# 1. Uniform
dist_uniform = np.ones(V) / V

# 2. Proportional to frequency
dist_freq = word_freqs / word_freqs.sum()

# 3. Standard Word2Vec (freq^0.75)
dist_w2v = word_freqs ** 0.75
dist_w2v /= dist_w2v.sum()

strategies = [
    ("Uniforme", dist_uniform),
    ("Proportionnel", dist_freq),
    ("freq^0.75 (W2V)", dist_w2v),
]

results = {}
for name, dist in strategies:
    emb, loss, t = train_skipgram(dist, epochs=80)
    results[name] = (emb, loss, t)
    print(f"\n  {name:>20s}: loss={loss:.4f}, time={t:.2f}s")

    # Intra-group similarity
    groups = {
        "Royaute": ["roi", "reine", "prince", "princesse"],
        "Animaux": ["chat", "chien"],
        "Genre": ["homme", "femme", "garcon", "fille"],
    }
    for gname, gwords in groups.items():
        indices = [word2idx[w] for w in gwords if w in word2idx]
        if len(indices) >= 2:
            sims = []
            for ii in range(len(indices)):
                for jj in range(ii + 1, len(indices)):
                    sims.append(cosine_similarity(emb[indices[ii]], emb[indices[jj]]))
            print(f"    Intra-{gname}: {np.mean(sims):.4f}")

print("\n  ANALYSE:")
print("  - Uniforme: les mots frequents ('le', 'la') sont sous-echantillonnes comme negatifs")
print("    → le modele n'apprend pas assez a les repousser → embeddings bruites")
print("  - Proportionnel: les mots rares sont presque jamais echantillonnes")
print("    → le modele ne les distingue pas bien des positifs")
print("  - freq^0.75: compromis — booste les rares, penalise les frequents → meilleur")


# ============================================================================
# EXERCISE 6: Semantic Search mini-engine
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 6: Mini Semantic Search Engine")
print("=" * 70)

# Use embeddings from the standard W2V training
emb_search = results["freq^0.75 (W2V)"][0]

# Documents to index
documents = [
    "le roi gouverne le royaume avec sagesse",
    "la reine gouverne le royaume avec bonte",
    "le chat dort sur le tapis",
    "le chien mange de la viande",
    "le prince et la princesse vivent au chateau",
    "homme et femme jouent ensemble",
    "le garcon mange du poisson",
    "le chat et le chien sont des animaux fideles",
]


class SemanticIndex:
    """
    Mini semantic search index.
    Documents are encoded as TF-IDF weighted average of word embeddings.
    """

    def __init__(self, documents, embeddings, word2idx):
        self.documents = documents
        self.embeddings = embeddings
        self.word2idx = word2idx

        # Compute IDF: log(N / df) where df = number of docs containing the word
        N = len(documents)
        doc_freq = {}
        for doc in documents:
            seen = set()
            for w in doc.split():
                if w not in seen:
                    doc_freq[w] = doc_freq.get(w, 0) + 1
                    seen.add(w)
        self.idf = {w: np.log(N / (df + 1)) for w, df in doc_freq.items()}

        # Encode each document
        self.doc_vectors = []
        for doc in documents:
            vec = self._encode(doc)
            self.doc_vectors.append(vec)
        self.doc_vectors = np.array(self.doc_vectors)  # (n_docs, d)

    def _encode(self, text):
        """Encode text as TF-IDF weighted average of word embeddings."""
        words = text.split()
        # Term frequency for this document
        tf = {}
        for w in words:
            tf[w] = tf.get(w, 0) + 1
        for w in tf:
            tf[w] /= len(words)

        vecs = []
        weights = []
        for w in words:
            if w in self.word2idx:
                tfidf = tf.get(w, 0) * self.idf.get(w, 1.0)
                vecs.append(self.embeddings[self.word2idx[w]])
                weights.append(tfidf)

        if not vecs:
            return np.zeros(self.embeddings.shape[1])

        vecs = np.array(vecs)
        weights = np.array(weights)
        weights /= weights.sum() + 1e-10
        return np.average(vecs, axis=0, weights=weights)

    def search(self, query, top_k=3):
        """Find top_k most similar documents to query."""
        q_vec = self._encode(query)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []

        # Cosine similarity with all documents
        sims = []
        for i, dv in enumerate(self.doc_vectors):
            d_norm = np.linalg.norm(dv)
            if d_norm == 0:
                sims.append(0.0)
            else:
                sims.append(np.dot(q_vec, dv) / (q_norm * d_norm))

        top_indices = np.argsort(-np.array(sims))[:top_k]
        return [(self.documents[i], sims[i]) for i in top_indices]


index = SemanticIndex(documents, emb_search, word2idx)

# Test queries
queries = [
    "le roi et la reine",
    "animal domestique",
    "nourriture",
    "chateau royal",
    "homme et femme",
]

for query in queries:
    print(f"\n  Requete: '{query}'")
    results_list = index.search(query, top_k=3)
    for doc, sim in results_list:
        print(f"    [{sim:.4f}] {doc}")

# Show advantage over keyword search
print("\n--- Avantage vs recherche par mots-cles ---")
print("  Requete: 'animal domestique'")
print("  Mots-cles exacts: AUCUN document contient 'domestique' → 0 resultats")
print("  Semantique: trouve les documents sur chat/chien (animaux) → pertinent !")


# ============================================================================
# EXERCISE 7: GloVe from scratch (simplified)
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 7: GloVe from Scratch")
print("=" * 70)

# Step 1: Build co-occurrence matrix
print("\n--- Building co-occurrence matrix ---")
cooc_matrix = np.zeros((V, V))

for sentence in CORPUS:
    words = sentence.split()
    for i, w1 in enumerate(words):
        idx1 = word2idx[w1]
        for j in range(max(0, i - WINDOW_SIZE), min(len(words), i + WINDOW_SIZE + 1)):
            if j != i:
                idx2 = word2idx[words[j]]
                distance = abs(i - j)
                # Weight by 1/distance: closer words co-occur "more strongly"
                cooc_matrix[idx1][idx2] += 1.0 / distance

# Show top words' co-occurrence
top_words_glove = ["roi", "reine", "chat", "chien", "homme", "femme"]
top_indices_glove = [word2idx[w] for w in top_words_glove if w in word2idx]

print(f"\n  Matrice de co-occurrence (mots selectionnes):")
print(f"  {'':>8s}", end="")
for w in top_words_glove:
    print(f"  {w:>6s}", end="")
print()
for i, w1 in enumerate(top_words_glove):
    idx1 = word2idx[w1]
    print(f"  {w1:>8s}", end="")
    for w2 in top_words_glove:
        idx2 = word2idx[w2]
        print(f"  {cooc_matrix[idx1][idx2]:6.2f}", end="")
    print()

# Step 2: Implement GloVe
print("\n--- Training GloVe ---")

# GloVe parameters
X_MAX = 100.0
ALPHA = 0.75


def f_weight(x, x_max=X_MAX, alpha=ALPHA):
    """Weighting function: (x/x_max)^alpha if x < x_max, else 1."""
    return np.minimum((x / x_max) ** alpha, 1.0)


# Initialize
W_glove = np.random.randn(V, EMBED_DIM) * 0.1        # Word vectors
W_context_glove = np.random.randn(V, EMBED_DIM) * 0.1 # Context vectors
b_w = np.zeros(V)                                      # Word biases
b_c = np.zeros(V)                                      # Context biases

# Adam optimizer state for GloVe
m_W = np.zeros_like(W_glove)
v_W = np.zeros_like(W_glove)
m_Wc = np.zeros_like(W_context_glove)
v_Wc = np.zeros_like(W_context_glove)
m_bw = np.zeros_like(b_w)
v_bw = np.zeros_like(b_w)
m_bc = np.zeros_like(b_c)
v_bc = np.zeros_like(b_c)

# Find all non-zero pairs for training
nonzero_pairs = []
for i in range(V):
    for j in range(V):
        if cooc_matrix[i][j] > 0:
            nonzero_pairs.append((i, j, cooc_matrix[i][j]))

print(f"  Non-zero pairs: {len(nonzero_pairs)}")

LR_GLOVE = 0.05
EPOCHS_GLOVE = 200
BETA1, BETA2 = 0.9, 0.999
EPS = 1e-8

glove_losses = []

for epoch in range(EPOCHS_GLOVE):
    np.random.shuffle(nonzero_pairs)
    epoch_loss = 0.0
    t_adam = epoch + 1  # Adam time step

    for i, j, x_ij in nonzero_pairs:
        # Forward: predict log(x_ij) from w_i . w_j + b_i + b_j
        diff = np.dot(W_glove[i], W_context_glove[j]) + b_w[i] + b_c[j] - np.log(x_ij)
        weight = f_weight(x_ij)
        loss = weight * diff ** 2
        epoch_loss += loss

        # Gradient: d(loss)/d(params)
        grad_common = 2.0 * weight * diff  # scalar

        grad_wi = grad_common * W_context_glove[j]
        grad_wj = grad_common * W_glove[i]
        grad_bi = grad_common
        grad_bj = grad_common

        # Simple SGD update (Adam on the full matrices would be cleaner
        # but this keeps it readable for learning purposes)
        W_glove[i] -= LR_GLOVE * grad_wi
        W_context_glove[j] -= LR_GLOVE * grad_wj
        b_w[i] -= LR_GLOVE * grad_bi
        b_c[j] -= LR_GLOVE * grad_bj

    glove_losses.append(epoch_loss / len(nonzero_pairs))
    if (epoch + 1) % 50 == 0:
        print(f"  Epoch {epoch+1:3d}/{EPOCHS_GLOVE} | Loss: {glove_losses[-1]:.6f}")

# GloVe final embeddings: average of word and context vectors (standard practice)
E_glove = (W_glove + W_context_glove) / 2.0

# Compare with Word2Vec
print("\n--- Comparaison GloVe vs Word2Vec ---")
emb_w2v = results["freq^0.75 (W2V)"][0]

for w1, w2 in [("roi", "reine"), ("chat", "chien"), ("homme", "femme")]:
    if w1 in word2idx and w2 in word2idx:
        sim_glove = cosine_similarity(E_glove[word2idx[w1]], E_glove[word2idx[w2]])
        sim_w2v = cosine_similarity(emb_w2v[word2idx[w1]], emb_w2v[word2idx[w2]])
        print(f"  ({w1}, {w2}): GloVe={sim_glove:.4f}, Word2Vec={sim_w2v:.4f}")

print("\n  ANALYSE:")
print("  GloVe utilise les stats GLOBALES (toute la matrice de co-occurrence)")
print("  Word2Vec utilise des fenetres LOCALES (paires individuelles)")
print("  Sur un PETIT corpus, GloVe peut etre plus stable car il voit tout d'un coup")
print("  Sur un GROS corpus, Word2Vec scale mieux (SGD sur des paires)")


# ============================================================================
# EXERCISE 8: FastText with subword n-grams (simplified)
# ============================================================================

print("\n" + "=" * 70)
print("EXERCISE 8: FastText Subword Embeddings")
print("=" * 70)


def get_ngrams(word, min_n=3, max_n=6):
    """
    Get character n-grams for a word, with boundary markers < and >.

    Example: "chat" -> ["<ch", "cha", "hat", "at>", "<cha", "chat", "hat>",
                         "<chat", "chat>", "<chat>"]
    Plus the word itself as a special token.
    """
    # Add boundary markers
    padded = f"<{word}>"
    ngrams = []

    for n in range(min_n, min(max_n + 1, len(padded) + 1)):
        for i in range(len(padded) - n + 1):
            ngrams.append(padded[i:i + n])

    return ngrams


# Demo n-grams
print("\n--- N-grams demonstration ---")
demo_words = ["chat", "chien", "chateau", "chatte"]
for w in demo_words:
    ngrams = get_ngrams(w)
    print(f"  '{w}' -> {ngrams}")

# Build n-gram vocabulary
print("\n--- Building n-gram vocabulary ---")
ngram2idx = {}
word_ngrams = {}  # word -> list of ngram indices

for w in word2idx:
    ngrams = get_ngrams(w)
    word_ngrams[w] = []
    for ng in ngrams:
        if ng not in ngram2idx:
            ngram2idx[ng] = len(ngram2idx)
        word_ngrams[w].append(ngram2idx[ng])

# Also add the word itself as a special "n-gram"
for w in word2idx:
    special = f"__WORD__{w}"
    if special not in ngram2idx:
        ngram2idx[special] = len(ngram2idx)
    word_ngrams[w].append(ngram2idx[special])

N_NGRAMS = len(ngram2idx)
print(f"  Vocabulary: {V} words")
print(f"  Total unique n-grams: {N_NGRAMS}")
print(f"  Ratio n-grams/words: {N_NGRAMS/V:.1f}x")

# Initialize FastText embeddings
W_ft = np.random.randn(N_NGRAMS, EMBED_DIM) * 0.1  # One embedding per n-gram
W_ft_out = np.random.randn(V, EMBED_DIM) * 0.1       # Output embeddings (per word)


def get_word_embedding_ft(word):
    """FastText word embedding = sum of its n-gram embeddings."""
    if word in word_ngrams:
        indices = word_ngrams[word]
    else:
        # OOV: use whatever n-grams we know
        ngrams = get_ngrams(word)
        indices = [ngram2idx[ng] for ng in ngrams if ng in ngram2idx]
    if not indices:
        return np.zeros(EMBED_DIM)
    return np.sum(W_ft[indices], axis=0)


# Train FastText Skip-gram
print("\n--- Training FastText Skip-gram ---")

LR_FT = 0.01  # Smaller LR because sum of many n-grams amplifies gradients
EPOCHS_FT = 80

ft_losses = []

for epoch in range(EPOCHS_FT):
    np.random.shuffle(sg_pairs)
    epoch_loss = 0.0
    lr = LR_FT * (1.0 - epoch / EPOCHS_FT)
    lr = max(lr, LR_FT * 0.01)

    for center_idx, context_idx in sg_pairs:
        center_word = idx2word[center_idx]
        center_ngram_ids = word_ngrams[center_word]

        # Forward: word embedding = sum of n-gram embeddings
        v_center = np.sum(W_ft[center_ngram_ids], axis=0)  # (d,)

        # Positive pair
        v_ctx = W_ft_out[context_idx]
        dot_p = np.dot(v_center, v_ctx)
        sig_p = sigmoid(dot_p)
        g_p = sig_p - 1.0
        epoch_loss += -np.log(max(sig_p, 1e-10))

        grad_center = g_p * v_ctx
        W_ft_out[context_idx] -= lr * g_p * v_center

        # Negative samples
        neg_idx_list = np.random.choice(V, size=K_NEG, p=neg_dist)
        for ni in neg_idx_list:
            if ni == context_idx:
                continue
            v_neg = W_ft_out[ni]
            dot_n = np.dot(v_center, v_neg)
            sig_n = sigmoid(dot_n)
            epoch_loss += -np.log(max(1 - sig_n, 1e-10))
            grad_center += sig_n * v_neg
            W_ft_out[ni] -= lr * sig_n * v_center

        # Update ALL n-gram embeddings of the center word
        # Each n-gram gets the same gradient because d(sum)/d(each) = 1
        for ng_id in center_ngram_ids:
            W_ft[ng_id] -= lr * grad_center

    ft_losses.append(epoch_loss / len(sg_pairs))
    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1:3d}/{EPOCHS_FT} | Loss: {ft_losses[-1]:.4f}")

# Test OOV words
print("\n--- Testing OOV (out-of-vocabulary) words ---")
oov_words = ["royaute", "chatte", "chiens", "gouvernement", "animaux"]

for oov in oov_words:
    # Get the OOV embedding from shared n-grams
    ngrams = get_ngrams(oov)
    known_ngrams = [ng for ng in ngrams if ng in ngram2idx]
    known_indices = [ngram2idx[ng] for ng in known_ngrams]

    if not known_indices:
        print(f"\n  '{oov}': No known n-grams → no embedding")
        continue

    oov_vec = np.sum(W_ft[known_indices], axis=0)
    oov_norm = np.linalg.norm(oov_vec)
    if oov_norm == 0:
        print(f"\n  '{oov}': Zero embedding")
        continue

    oov_vec_normed = oov_vec / oov_norm

    # Find most similar known words
    print(f"\n  '{oov}' ({len(known_ngrams)}/{len(ngrams)} n-grams connus)")
    print(f"    N-grams connus: {known_ngrams[:8]}...")

    sims = []
    for w in word2idx:
        w_vec = get_word_embedding_ft(w)
        w_norm = np.linalg.norm(w_vec)
        if w_norm > 0:
            sim = np.dot(oov_vec, w_vec) / (oov_norm * w_norm)
            sims.append((w, sim))

    sims.sort(key=lambda x: -x[1])
    print(f"    Mots les plus proches:")
    for w, s in sims[:5]:
        print(f"      {w:>12s}: {s:.4f}")

# Morphological analysis
print("\n--- Analyse morphologique ---")
morpho_pairs = [
    ("roi", "reine"),
    ("homme", "femme"),
    ("prince", "princesse"),
]

print("  FastText vs Word2Vec standard — mots derives et morphologie:")
for w1, w2 in morpho_pairs:
    if w1 in word2idx and w2 in word2idx:
        # FastText similarity
        v1_ft = get_word_embedding_ft(w1)
        v2_ft = get_word_embedding_ft(w2)
        n1, n2 = np.linalg.norm(v1_ft), np.linalg.norm(v2_ft)
        sim_ft = np.dot(v1_ft, v2_ft) / (n1 * n2) if n1 > 0 and n2 > 0 else 0

        # Standard Word2Vec similarity
        sim_w2v = cosine_similarity(emb_w2v[word2idx[w1]], emb_w2v[word2idx[w2]])

        # Shared n-grams (shows morphological overlap)
        ng1 = set(get_ngrams(w1))
        ng2 = set(get_ngrams(w2))
        shared = ng1 & ng2

        print(f"  ({w1}, {w2}): FastText={sim_ft:.4f}, W2V={sim_w2v:.4f}, "
              f"n-grams partages: {len(shared)} ({shared if len(shared) < 5 else '...'})")

# Memory benchmark
print("\n--- Benchmark memoire ---")
params_w2v = V * EMBED_DIM * 2  # W_in + W_out
params_ft = N_NGRAMS * EMBED_DIM + V * EMBED_DIM  # W_ft + W_ft_out

print(f"  Word2Vec: {params_w2v:,} parametres ({V} mots x {EMBED_DIM}d x 2 matrices)")
print(f"  FastText: {params_ft:,} parametres ({N_NGRAMS} n-grams x {EMBED_DIM}d + {V} x {EMBED_DIM}d)")
print(f"  Ratio FastText/Word2Vec: {params_ft/params_w2v:.2f}x")
print(f"  Trade-off: {params_ft/params_w2v:.1f}x plus de parametres, mais gere les mots inconnus.")


# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("RESUME DES SOLUTIONS")
print("=" * 70)

print("""
  Ex 1-3 (Easy):
    - One-hot: similarite = 0 partout, embeddings denses capturent la semantique
    - Skip-gram: les paires (centre, contexte) sont la matiere premiere
    - Cosinus mesure le SENS (direction), euclidienne mesure la MAGNITUDE

  Ex 4-6 (Medium):
    - CBOW: la moyenne des contextes predit le centre, plus rapide, moins bon sur mots rares
    - Negative sampling: freq^0.75 est le meilleur compromis, k=5-15 suffit
    - Recherche semantique: TF-IDF ponderation + cosinus = mini RAG

  Ex 7-8 (Hard):
    - GloVe: factorisation de matrice de co-occurrence, stats globales, complementaire a W2V
    - FastText: n-grams de caracteres, gere les OOV, capture la morphologie
    - Trade-off memoire vs robustesse: FastText a ~2-3x plus de params mais zero OOV
""")
