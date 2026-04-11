"""
Jour 3 — Embeddings & Representations FROM SCRATCH
====================================================
Pure Python + NumPy. No gensim, no torch.
Every line is commented with WHY, not just what.

Covers:
  1. One-hot encoding and its limits (cosine sim = 0 everywhere)
  2. Word2Vec Skip-gram with negative sampling from scratch
  3. Cosine similarity and finding similar words
  4. Vector analogies (king - man + woman ≈ queen)
  5. PCA projection to 2D + ASCII scatter plot
  6. Semantic clustering demonstration

Run: python 02-code/03-embeddings-representations.py
"""

import sys
import io
import numpy as np

# Force UTF-8 output to handle special characters in comments/docstrings
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

np.random.seed(42)


# ============================================================================
# PART 0: Helper — ASCII scatter plot (no matplotlib needed)
# ============================================================================

def ascii_scatter(points_2d, labels, width=70, height=25, title=""):
    """
    Plot labeled 2D points as ASCII art.
    points_2d: (n, 2) array
    labels: list of strings, one per point
    """
    # We need to map continuous coordinates to a discrete grid
    xs = points_2d[:, 0]
    ys = points_2d[:, 1]

    # Add padding so edge points aren't clipped
    x_min, x_max = xs.min() - 0.5, xs.max() + 0.5
    y_min, y_max = ys.min() - 0.5, ys.max() + 0.5

    # Create empty grid — each cell is a string
    grid = [[' ' for _ in range(width)] for _ in range(height)]

    # Place each label on the grid at its mapped position
    for i, (x, y) in enumerate(zip(xs, ys)):
        # Map continuous coords to grid indices
        col = int((x - x_min) / (x_max - x_min) * (width - 1))
        # Invert y axis because row 0 is top of the screen
        row = int((1 - (y - y_min) / (y_max - y_min)) * (height - 1))
        col = max(0, min(width - 1, col))
        row = max(0, min(height - 1, row))

        # Write the label (truncate if it would overflow the grid)
        label = labels[i][:8]  # max 8 chars to avoid overlap
        for j, ch in enumerate(label):
            if col + j < width:
                grid[row][col + j] = ch

    # Render the grid
    print(f"\n  {title}")
    print(f"  +{'-' * width}+")
    for row in grid:
        print(f"  |{''.join(row)}|")
    print(f"  +{'-' * width}+")


def ascii_bar(values, labels, width=40, title=""):
    """Simple horizontal bar chart."""
    print(f"\n  {title}")
    print(f"  {'-' * (width + 20)}")
    max_val = max(abs(v) for v in values) if values else 1
    for label, val in zip(labels, values):
        bar_len = int(abs(val) / max_val * width)
        bar = '#' * bar_len
        print(f"  {label:>12s} | {bar} {val:.4f}")


# ============================================================================
# PART 1: One-Hot Encoding and Its Limits
# ============================================================================

print("=" * 70)
print("PART 1: One-Hot Encoding — Why It Fails")
print("=" * 70)

# Define a small vocabulary to demonstrate the problem
vocab_onehot = ["chat", "chien", "roi", "reine", "homme", "femme"]
V_demo = len(vocab_onehot)

# Build one-hot vectors: each word gets a vector of size V with a single 1
# This is the simplest possible representation — and the worst
onehot_vectors = np.eye(V_demo)  # Identity matrix = perfect one-hot

print(f"\nVocabulaire ({V_demo} mots): {vocab_onehot}")
print(f"\nVecteurs one-hot:")
for word, vec in zip(vocab_onehot, onehot_vectors):
    print(f"  {word:>6s} = {vec.astype(int).tolist()}")


def cosine_similarity(a, b):
    """
    cos(a, b) = (a . b) / (||a|| * ||b||)

    WHY cosine and not euclidean? Because cosine measures DIRECTION (meaning),
    not magnitude (frequency/intensity). Two words with same meaning but
    different frequencies should still be "similar".
    """
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    # Guard against zero vectors (would cause division by zero)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# Show that ALL pairs of one-hot vectors have cosine similarity = 0
# This is the fundamental problem: "chat" is as different from "chien" as from "roi"
print(f"\nSimilarite cosinus entre toutes les paires one-hot:")
print(f"  {'':>8s}", end="")
for w in vocab_onehot:
    print(f"  {w:>6s}", end="")
print()

for i, w1 in enumerate(vocab_onehot):
    print(f"  {w1:>8s}", end="")
    for j, w2 in enumerate(vocab_onehot):
        sim = cosine_similarity(onehot_vectors[i], onehot_vectors[j])
        print(f"  {sim:6.2f}", end="")
    print()

print(f"\n  CONSTAT: Toutes les paires non-diagonales ont sim = 0.00")
print(f"  'chat' est aussi different de 'chien' que de 'roi' — absurde !")
print(f"  En plus, chaque vecteur a {V_demo} dimensions pour {V_demo} mots.")
print(f"  Avec V=50000 mots, ca ferait des vecteurs de 50000 dims — 99.998% de zeros.")


# ============================================================================
# PART 2: Word2Vec Skip-gram with Negative Sampling — FROM SCRATCH
# ============================================================================

print("\n" + "=" * 70)
print("PART 2: Word2Vec Skip-gram — Training from Scratch")
print("=" * 70)

# --- Step 2a: Build a corpus with clear semantic structure ---
# WHY this corpus? It has deliberate semantic groups (royalty, animals, food, family)
# so we can verify that the embeddings capture meaning after training.
# Repeated patterns teach the model that words in similar contexts are similar.
corpus = [
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

# --- Step 2b: Build vocabulary ---
# WHY: we need a mapping from word -> integer index for the embedding matrix lookup
print("\n--- Building vocabulary ---")

# Tokenize: split each sentence into words
all_words = []
for sentence in corpus:
    all_words.extend(sentence.split())

# Count word frequencies — needed for negative sampling probability distribution
word_counts = {}
for w in all_words:
    word_counts[w] = word_counts.get(w, 0) + 1

# Create word-to-index and index-to-word mappings
# Sort by frequency (descending) so the most common words have low indices
sorted_words = sorted(word_counts.keys(), key=lambda w: -word_counts[w])
word2idx = {w: i for i, w in enumerate(sorted_words)}
idx2word = {i: w for w, i in word2idx.items()}
V = len(word2idx)  # vocabulary size

print(f"  Taille du vocabulaire: {V} mots")
print(f"  Mots les plus frequents: {sorted_words[:10]}")
print(f"  Corpus: {len(corpus)} phrases, {len(all_words)} tokens")

# --- Step 2c: Create training pairs (center_word, context_word) ---
# WHY: Skip-gram learns by predicting context words from a center word.
# For each word in a sentence, we pair it with every word within a window.
WINDOW_SIZE = 2  # How many words to look at on each side

training_pairs = []
for sentence in corpus:
    words = sentence.split()
    for i, center_word in enumerate(words):
        center_idx = word2idx[center_word]
        # Look at words within the window on both sides
        for j in range(max(0, i - WINDOW_SIZE), min(len(words), i + WINDOW_SIZE + 1)):
            if j != i:  # Skip the center word itself
                context_idx = word2idx[words[j]]
                training_pairs.append((center_idx, context_idx))

print(f"  Paires d'entrainement: {len(training_pairs)}")
print(f"  Exemples:")
for pair in training_pairs[:5]:
    print(f"    ({idx2word[pair[0]]}, {idx2word[pair[1]]})")

# --- Step 2d: Build negative sampling distribution ---
# WHY: Instead of computing softmax over ALL V words (O(V) = expensive),
# we sample k negative examples and do binary classification (O(k) = cheap).
# The distribution is freq^(3/4) — this boosts rare words relative to their raw freq
# while still oversampling frequent words vs uniform.

# freq^(3/4) distribution
word_freqs = np.array([word_counts[idx2word[i]] for i in range(V)], dtype=np.float64)
neg_sampling_dist = word_freqs ** 0.75  # The 3/4 power trick from the paper
neg_sampling_dist /= neg_sampling_dist.sum()  # Normalize to probability distribution

print(f"\n--- Negative sampling distribution ---")
print(f"  Top-5 sampling probabilities:")
top_5 = np.argsort(-neg_sampling_dist)[:5]
for idx in top_5:
    print(f"    {idx2word[idx]:>10s}: raw freq={word_freqs[idx]:.0f}, "
          f"sample prob={neg_sampling_dist[idx]:.4f}")

# --- Step 2e: Initialize embedding matrices ---
# WHY two matrices? W_in holds the "real" embeddings (what we'll use after training).
# W_out is the context matrix used during training for prediction.
# Both are learned, but we only keep W_in.

EMBED_DIM = 20  # Small dimension for our tiny corpus (real models use 100-300)
# WHY 20? Our corpus has ~35 unique words. The embedding dimension should be
# much smaller than V to force compression, but big enough to encode relationships.

# Xavier/Glorot initialization: scale by 1/sqrt(fan_in) to keep variance stable
# across layers during forward pass — same principle as in MLP (Jour 2)
W_in = np.random.randn(V, EMBED_DIM) * 0.1   # (V, d) — input embeddings
W_out = np.random.randn(V, EMBED_DIM) * 0.1  # (V, d) — context embeddings

# --- Step 2f: Training loop with negative sampling ---
# WHY negative sampling loss?
# For a positive pair (center, context):
#   L = -log(sigma(v_context . v_center)) - sum_neg log(sigma(-v_neg . v_center))
# We want: high dot product for real pairs, low dot product for fake pairs.

def sigmoid(z):
    """Numerically stable sigmoid — clip to avoid overflow in exp."""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


K_NEG = 5         # Number of negative samples per positive pair (paper: 5-20)
LEARNING_RATE = 0.025  # Initial learning rate (will decay linearly)
EPOCHS = 100      # Number of full passes through all training pairs

print(f"\n--- Training Word2Vec Skip-gram ---")
print(f"  Embedding dim: {EMBED_DIM}")
print(f"  Negative samples: {K_NEG}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Epochs: {EPOCHS}")

total_loss_history = []

for epoch in range(EPOCHS):
    # Shuffle training pairs each epoch for SGD stochasticity
    np.random.shuffle(training_pairs)

    epoch_loss = 0.0

    # Linear LR decay — WHY? Gradually reducing the step size helps converge
    # to a better solution. Same principle as LR scheduling in Jour 2.
    lr = LEARNING_RATE * (1.0 - epoch / EPOCHS)
    lr = max(lr, LEARNING_RATE * 0.01)  # Floor to avoid zero LR

    for center_idx, context_idx in training_pairs:
        # --- Get the center embedding ---
        # WHY: This is the "lookup" — equivalent to one-hot @ W_in, but much faster
        v_center = W_in[center_idx]  # (d,) — the embedding we're learning

        # --- Positive pair: (center, true_context) ---
        # WHY: We want v_center . v_context to be HIGH (similar = good)
        v_context = W_out[context_idx]  # (d,)
        dot_pos = np.dot(v_center, v_context)  # scalar — how "similar" they are
        sig_pos = sigmoid(dot_pos)

        # Gradient for positive pair:
        # d/d(dot) [-log(sigma(dot))] = sigma(dot) - 1
        # WHY: when sigma(dot) is close to 1 (confident correct), gradient ≈ 0
        # when sigma(dot) is close to 0 (wrong), gradient ≈ -1 (strong push)
        grad_pos = sig_pos - 1.0  # scalar

        # Loss for positive pair: -log(sigma(dot_pos))
        # Clip to avoid log(0)
        epoch_loss += -np.log(max(sig_pos, 1e-10))

        # --- Accumulate gradients for center word ---
        # WHY accumulate? We need gradients from both positive and all negative pairs
        # before updating. grad_center = grad_pos * v_context + sum(grad_neg_k * v_neg_k)
        grad_center = grad_pos * v_context

        # Update context word: W_out[context] -= lr * grad_pos * v_center
        W_out[context_idx] -= lr * grad_pos * v_center

        # --- Negative samples ---
        # WHY: We sample K words that are NOT the true context, and train the model
        # to output LOW dot product for them. This is the "negative" part.
        neg_indices = np.random.choice(V, size=K_NEG, p=neg_sampling_dist)

        for neg_idx in neg_indices:
            # Skip if we accidentally sampled the true context
            if neg_idx == context_idx:
                continue

            v_neg = W_out[neg_idx]  # (d,)
            dot_neg = np.dot(v_center, v_neg)
            sig_neg = sigmoid(dot_neg)

            # Gradient for negative pair:
            # d/d(dot) [-log(sigma(-dot))] = sigma(dot) (note: no -1 here)
            # WHY: we want sigma(dot_neg) close to 0, so gradient pushes dot down
            grad_neg = sig_neg  # scalar

            # Loss for negative pair: -log(sigma(-dot_neg))
            epoch_loss += -np.log(max(1 - sig_neg, 1e-10))

            # Accumulate gradient for center word
            grad_center += grad_neg * v_neg

            # Update negative word embedding
            W_out[neg_idx] -= lr * grad_neg * v_center

        # --- Update center word embedding ---
        W_in[center_idx] -= lr * grad_center

    total_loss_history.append(epoch_loss / len(training_pairs))

    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1:3d}/{EPOCHS} | Loss: {total_loss_history[-1]:.4f} | LR: {lr:.5f}")

print(f"\n  Training complete! Final loss: {total_loss_history[-1]:.4f}")

# --- Display loss curve ---
print("\n  Loss curve:")
if len(total_loss_history) > 1:
    # Simple ASCII loss plot
    w_plot = 50
    h_plot = 8
    sampled = total_loss_history
    if len(sampled) > w_plot:
        step = len(sampled) // w_plot
        sampled = [sampled[i] for i in range(0, len(sampled), step)][:w_plot]
    max_l = max(sampled)
    min_l = min(sampled)
    range_l = max_l - min_l if max_l != min_l else 1.0
    print(f"  {max_l:.4f} |", end="")
    for row in range(h_plot):
        if row > 0:
            print(f"         |", end="")
        threshold = max_l - (row / (h_plot - 1)) * range_l
        for val in sampled:
            print("#" if val >= threshold else " ", end="")
        print("|")
    print(f"  {min_l:.4f} +{'-' * len(sampled)}+")
    print(f"           Epoch 1{' ' * (len(sampled) - 12)}Epoch {EPOCHS}")


# ============================================================================
# PART 3: Cosine Similarity — Finding Similar Words
# ============================================================================

print("\n" + "=" * 70)
print("PART 3: Cosine Similarity — Finding Similar Words")
print("=" * 70)

# WHY cosine and not euclidean? Because cosine measures direction (meaning),
# not magnitude. Two embeddings pointing in the same direction are semantically
# similar regardless of their length.

# Normalize embeddings (optional but makes cosine = dot product, faster)
# WHY normalize? After normalization, ||v|| = 1 for all vectors, so
# cos(a,b) = a.b which is just a dot product — computationally cheaper.
embeddings = W_in.copy()  # Use the learned input embeddings
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
embeddings_normed = embeddings / norms


def most_similar(word, top_k=5):
    """
    Find the top_k most similar words to the given word using cosine similarity.
    WHY: This is the core operation behind semantic search, RAG retrieval, etc.
    """
    if word not in word2idx:
        print(f"  '{word}' not in vocabulary")
        return []

    word_vec = embeddings_normed[word2idx[word]]  # (d,) normalized

    # Compute cosine similarity with ALL words in one matrix multiply
    # WHY matrix multiply? embeddings_normed @ word_vec gives dot products
    # (= cosine similarity when both are normalized) for all words at once.
    # This is O(V*d) but vectorized = fast.
    similarities = embeddings_normed @ word_vec  # (V,)

    # Get top_k+1 indices (excluding the word itself)
    # WHY argsort? It sorts by similarity; we take the highest (most similar)
    top_indices = np.argsort(-similarities)

    results = []
    for idx in top_indices:
        if idx2word[idx] != word:
            results.append((idx2word[idx], similarities[idx]))
            if len(results) >= top_k:
                break

    return results


# Test with key words from each semantic group
test_words = ["roi", "chat", "homme", "gouverne", "chateau"]

for word in test_words:
    similar = most_similar(word, top_k=5)
    print(f"\n  Mots les plus similaires a '{word}':")
    for w, sim in similar:
        print(f"    {w:>12s}  cos_sim = {sim:.4f}")


# ============================================================================
# PART 4: Vector Analogies
# ============================================================================

print("\n" + "=" * 70)
print("PART 4: Vector Analogies (a - b + c = ?)")
print("=" * 70)

# WHY analogies work: Word2Vec learns to encode relationships as DIRECTIONS.
# "roi - homme" captures the direction of "royalty/power".
# Adding that direction to "femme" should give "reine".
# This only works if the embedding space has captured these regularities.


def analogy(a, b, c, top_k=3):
    """
    Solve: a - b + c = ?
    Example: roi - homme + femme = reine

    WHY this formula? If the relationship (a->b) is parallel to (c->?),
    then vec(?) = vec(a) - vec(b) + vec(c).
    The "difference vector" (a-b) captures the relationship, and we
    translate it from c.
    """
    if a not in word2idx or b not in word2idx or c not in word2idx:
        print(f"  Word not in vocabulary!")
        return []

    # Compute the target vector
    vec_target = embeddings[word2idx[a]] - embeddings[word2idx[b]] + embeddings[word2idx[c]]

    # Normalize for cosine similarity
    vec_target_norm = vec_target / (np.linalg.norm(vec_target) + 1e-10)

    # Find most similar words to the target vector
    similarities = embeddings_normed @ vec_target_norm

    # Exclude the input words from results
    exclude = {word2idx[a], word2idx[b], word2idx[c]}
    top_indices = np.argsort(-similarities)

    results = []
    for idx in top_indices:
        if idx not in exclude:
            results.append((idx2word[idx], similarities[idx]))
            if len(results) >= top_k:
                break

    return results


# Test analogies — these may not be perfect on a tiny corpus,
# but they should approximate the right direction.
analogies_to_test = [
    ("roi", "homme", "femme", "reine ?"),       # royalty + gender
    ("prince", "homme", "femme", "princesse ?"), # royalty + gender
    ("roi", "royaume", "chat", "???"),           # domain transfer
    ("chat", "tapis", "chien", "canape ?"),      # animal + resting place
]

for a, b, c, expected in analogies_to_test:
    results = analogy(a, b, c)
    print(f"\n  {a} - {b} + {c} = {expected}")
    for word, sim in results:
        marker = " <--" if sim > 0.3 else ""
        print(f"    {word:>12s}  cos_sim = {sim:.4f}{marker}")

print(f"\n  NOTE: Sur un corpus de {len(corpus)} phrases, les analogies sont approximatives.")
print(f"  Avec un corpus de millions de phrases, elles deviennent tres precises.")


# ============================================================================
# PART 5: PCA Projection + ASCII Scatter Plot
# ============================================================================

print("\n" + "=" * 70)
print("PART 5: PCA Projection to 2D — Visualizing the Embedding Space")
print("=" * 70)

# WHY PCA? Our embeddings live in 20D space. Humans can only see 2D.
# PCA finds the 2 directions of maximum variance — the "best" 2D projection
# that preserves as much information as possible.


def pca_2d(X):
    """
    Principal Component Analysis — project X to 2 dimensions.
    Pure numpy implementation.

    Steps:
    1. Center the data (subtract mean)
    2. Compute covariance matrix
    3. Eigen-decompose to find principal directions
    4. Project onto top-2 eigenvectors

    WHY center first? PCA finds directions of maximum VARIANCE.
    If the data is not centered, the first PC would just point toward the mean,
    which is not informative.
    """
    # Step 1: Center — each feature has mean 0
    X_centered = X - X.mean(axis=0)

    # Step 2: Covariance matrix — (d, d) matrix where C[i,j] = correlation between dims i and j
    # WHY covariance? It tells us which dimensions vary together.
    # High covariance = these dims are redundant, PCA will merge them.
    cov_matrix = (X_centered.T @ X_centered) / (X_centered.shape[0] - 1)

    # Step 3: Eigendecomposition — find the directions (eigenvectors) along which
    # data varies the most (largest eigenvalues)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # eigh returns in ascending order, we want descending (largest variance first)
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Step 4: Project — multiply centered data by top-2 eigenvectors
    # WHY top-2? We want a 2D visualization. These 2 directions capture
    # the most variance (information) in the data.
    projected = X_centered @ eigenvectors[:, :2]  # (n, 2)

    # Report how much variance is captured
    total_var = eigenvalues.sum()
    explained = (eigenvalues[0] + eigenvalues[1]) / total_var * 100
    print(f"  Variance expliquee par les 2 premieres composantes: {explained:.1f}%")

    return projected


# Select interesting words to visualize (skip function words like "le", "la", "du")
interesting_words = [
    "roi", "reine", "prince", "princesse",
    "homme", "femme", "garcon", "fille",
    "chat", "chien",
    "gouverne", "chateau", "royaume",
    "mange", "dort", "vivent",
]

# Filter to words that exist in our vocabulary
viz_words = [w for w in interesting_words if w in word2idx]
viz_indices = [word2idx[w] for w in viz_words]
viz_embeddings = embeddings[viz_indices]

print(f"\n  Projecting {len(viz_words)} words from {EMBED_DIM}D to 2D...")

# Apply PCA
projected_2d = pca_2d(viz_embeddings)

# Display the scatter plot
ascii_scatter(projected_2d, viz_words, width=70, height=25,
              title="Embedding Space (PCA 2D Projection)")

# Also show the semantic groups with color-coded labels
print("\n  Legende des groupes semantiques:")
groups = {
    "Royaute": ["roi", "reine", "prince", "princesse", "royaume", "chateau", "gouverne"],
    "Genre":   ["homme", "femme", "garcon", "fille"],
    "Animaux": ["chat", "chien"],
    "Actions": ["mange", "dort", "vivent"],
}
for group_name, group_words in groups.items():
    present = [w for w in group_words if w in viz_words]
    print(f"    {group_name:>10s}: {', '.join(present)}")


# ============================================================================
# PART 6: Semantic Clustering — Proving Embeddings Capture Meaning
# ============================================================================

print("\n" + "=" * 70)
print("PART 6: Semantic Clustering Verification")
print("=" * 70)

# WHY verify clustering? If embeddings truly capture meaning, then the average
# intra-group similarity should be MUCH higher than inter-group similarity.
# This is a quantitative proof that the model learned semantics.

semantic_groups = {
    "Royaute":  ["roi", "reine", "prince", "princesse"],
    "Humains":  ["homme", "femme", "garcon", "fille"],
    "Animaux":  ["chat", "chien"],
}


def group_avg_similarity(words):
    """Average pairwise cosine similarity within a group."""
    indices = [word2idx[w] for w in words if w in word2idx]
    if len(indices) < 2:
        return 0.0

    sims = []
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            sim = cosine_similarity(embeddings[indices[i]], embeddings[indices[j]])
            sims.append(sim)
    return np.mean(sims)


def cross_group_similarity(words1, words2):
    """Average cosine similarity between two groups."""
    indices1 = [word2idx[w] for w in words1 if w in word2idx]
    indices2 = [word2idx[w] for w in words2 if w in word2idx]
    if not indices1 or not indices2:
        return 0.0

    sims = []
    for i in indices1:
        for j in indices2:
            sim = cosine_similarity(embeddings[i], embeddings[j])
            sims.append(sim)
    return np.mean(sims)


# Intra-group similarities (should be HIGH)
print("\n  Similarite INTRA-groupe (devrait etre elevee):")
for group_name, words in semantic_groups.items():
    sim = group_avg_similarity(words)
    bar = '#' * int(max(0, sim) * 40)
    print(f"    {group_name:>10s}: {sim:.4f}  |{bar}|")

# Inter-group similarities (should be LOWER)
print("\n  Similarite INTER-groupe (devrait etre plus basse):")
group_names = list(semantic_groups.keys())
for i in range(len(group_names)):
    for j in range(i + 1, len(group_names)):
        g1, g2 = group_names[i], group_names[j]
        sim = cross_group_similarity(semantic_groups[g1], semantic_groups[g2])
        bar = '#' * int(max(0, sim) * 40)
        print(f"    {g1:>10s} vs {g2:<10s}: {sim:.4f}  |{bar}|")


# --- Comparison with one-hot: similarity is ALWAYS 0 ---
print("\n  Comparaison avec one-hot encoding:")
print("    One-hot: TOUTES les similarites intra-groupe = 0.00")
print("    One-hot: TOUTES les similarites inter-groupe = 0.00")
print("    Embeddings: les groupes semantiques ont des similarites intra > inter")
print("    => Les embeddings capturent le SENS, one-hot ne capture RIEN.")


# ============================================================================
# PART 7: Similarity Matrix — Full Picture
# ============================================================================

print("\n" + "=" * 70)
print("PART 7: Full Similarity Matrix (selected words)")
print("=" * 70)

# Show a heatmap-style similarity matrix for key words
key_words = ["roi", "reine", "homme", "femme", "chat", "chien", "royaume", "chateau"]
key_words = [w for w in key_words if w in word2idx]

print(f"\n  Matrice de similarite cosinus:")
print(f"  {'':>10s}", end="")
for w in key_words:
    print(f" {w:>8s}", end="")
print()
print(f"  {'':>10s} " + "-" * (9 * len(key_words)))

for w1 in key_words:
    print(f"  {w1:>10s}", end="")
    for w2 in key_words:
        sim = cosine_similarity(embeddings[word2idx[w1]], embeddings[word2idx[w2]])
        # Use symbols to make patterns visible
        if w1 == w2:
            symbol = " 1.0000"
        elif sim > 0.5:
            symbol = f" {sim:.4f}"
        elif sim > 0.2:
            symbol = f" {sim:.4f}"
        else:
            symbol = f" {sim:.4f}"
        print(f" {symbol}", end="")
    print()

print(f"\n  Observations attendues:")
print(f"    - (roi, reine) devraient etre proches (meme contexte)")
print(f"    - (homme, femme) devraient etre proches")
print(f"    - (chat, chien) devraient etre proches")
print(f"    - (roi, chat) devraient etre plus eloignes")


# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("RESUME — Ce qu'on a appris")
print("=" * 70)

print(f"""
  1. ONE-HOT ENCODING est une representation naive:
     - Tous les mots sont orthogonaux (similarite = 0)
     - Dimension = taille du vocabulaire (sparse, inefficace)
     - Aucune information semantique

  2. WORD2VEC SKIP-GRAM apprend des embeddings denses:
     - Tache: predire les mots de contexte a partir du mot central
     - Negative sampling: O(k) au lieu de O(V) — rend le training possible
     - Les mots qui apparaissent dans les memes contextes → embeddings proches

  3. Les EMBEDDINGS CAPTURES LE SENS:
     - Similarite cosinus reflète la proximite semantique
     - Les analogies vectorielles emergent (roi - homme + femme ≈ reine)
     - Les clusters semantiques se forment naturellement

  4. APPLICATIONS: recherche semantique (RAG), clustering, classification,
     recommandation. Les embeddings sont la BASE de tout systeme NLP moderne.

  Prochain: Jour 4 — Sequences & Attention (RNN, LSTM, et les premices du Transformer)
""")
