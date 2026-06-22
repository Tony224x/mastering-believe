"""
Jour 8 — BPE Tokenizer FROM SCRATCH
====================================
Pure Python. No tokenizers library.
Every line commented with WHY, not just what.

Covers:
  1. BPE training: initial vocab + merge loop
  2. Step-by-step display of merges
  3. Encoding a new word with learned merges
  4. Comparison with word-level and char-level
  5. Demonstration of morphology capture

Run: python 02-code/08-pretraining-tokenization.py
"""

import sys
import io
from collections import Counter, defaultdict

# Force UTF-8 output on Windows to avoid encoding errors on special characters
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


# ============================================================================
# PART 1: Corpus and pre-tokenization
# ============================================================================

print("=" * 70)
print("PART 1: BPE Training — Corpus Preparation")
print("=" * 70)

# Tiny corpus — just enough to see interesting merges emerge
# Repetition is critical: BPE relies on frequency, so we repeat common words
corpus = [
    "low", "low", "low", "low", "low",
    "lower", "lower",
    "newest", "newest", "newest", "newest", "newest", "newest",
    "widest", "widest", "widest",
]

print(f"\nCorpus: {corpus}")
print(f"Nombre de mots: {len(corpus)}")

# WHY we add an end-of-word marker "</w>":
# Without it, BPE cannot distinguish "low" at the end of a word from
# "low" inside "lowest". The marker tells the algorithm where a word ends,
# so it can learn different merges for prefixes vs suffixes.
END_OF_WORD = "</w>"


def pre_tokenize(word):
    """
    Split a word into a list of characters, with </w> at the end.
    Example: "low" -> ["l", "o", "w", "</w>"]

    This is the INITIAL state before any merges: each character is its own token.
    """
    return list(word) + [END_OF_WORD]


# Build the word frequency dictionary: maps pre-tokenized word -> count
# We use a tuple as the key because lists are not hashable.
word_freqs = Counter()
for word in corpus:
    tokens = tuple(pre_tokenize(word))
    word_freqs[tokens] += 1

print("\nFrequences initiales (mots -> tokens -> count):")
for tokens, count in word_freqs.items():
    display_tokens = " ".join(tokens)
    print(f"  {display_tokens:<30s} x {count}")


# ============================================================================
# PART 2: Counting adjacent pairs
# ============================================================================

def get_pair_stats(word_freqs):
    """
    Count all adjacent pairs across all words, weighted by word frequency.

    WHY weighted: if "low" appears 5 times, the pair (l, o) should count
    5 times, not 1. This makes sure that merges reflect the corpus distribution.

    Returns: Counter mapping (token_a, token_b) -> total count
    """
    pairs = Counter()
    for word, freq in word_freqs.items():
        # Look at every consecutive pair inside the word
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pairs[pair] += freq  # weight by word frequency
    return pairs


# Sanity check: show the initial pair distribution
initial_pairs = get_pair_stats(word_freqs)
print("\nPaires initiales (top 8):")
for pair, count in initial_pairs.most_common(8):
    print(f"  {pair[0]:<5s} + {pair[1]:<5s} -> {count}")


# ============================================================================
# PART 3: Merging a pair across the corpus
# ============================================================================

def merge_pair(pair, word_freqs):
    """
    Apply a merge: replace every occurrence of (a, b) by the single token "ab".

    WHY we create a new dict: we cannot mutate keys of a dict we iterate over.
    WHY tuple: hashability again.

    Returns: new word_freqs after the merge
    """
    new_word_freqs = {}
    a, b = pair
    merged_token = a + b  # the new symbol, e.g. "lo"

    for word, freq in word_freqs.items():
        new_word = []
        i = 0
        # Walk through the tokens, replacing (a, b) with merged_token
        while i < len(word):
            # Check if the current position starts with the pair to merge
            if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                new_word.append(merged_token)
                i += 2  # skip both tokens of the merged pair
            else:
                new_word.append(word[i])
                i += 1
        new_word_freqs[tuple(new_word)] = freq
    return new_word_freqs


# ============================================================================
# PART 4: The main BPE training loop
# ============================================================================

def train_bpe(corpus_freqs, num_merges, verbose=True):
    """
    Run BPE training for `num_merges` iterations.

    Returns:
      - merges: list of pairs in order (needed for encoding new words)
      - vocab: final set of tokens
    """
    # Start with the initial vocabulary = all characters seen + </w>
    vocab = set()
    for word in corpus_freqs:
        for token in word:
            vocab.add(token)

    merges = []  # ordered list: merges[i] was applied at step i
    current_freqs = dict(corpus_freqs)

    if verbose:
        print(f"\nVocabulaire initial ({len(vocab)} tokens): {sorted(vocab)}")
        print(f"\n{'=' * 70}")
        print(f"Debut de l'entrainement BPE ({num_merges} merges cibles)")
        print(f"{'=' * 70}")

    for step in range(num_merges):
        pairs = get_pair_stats(current_freqs)
        if not pairs:
            # No more pairs to merge -> corpus is fully merged
            if verbose:
                print(f"\nPlus de paires a merger apres {step} etapes.")
            break

        # Select the most frequent pair — this is the core BPE heuristic
        best_pair, best_count = pairs.most_common(1)[0]

        # Apply the merge across the whole corpus
        current_freqs = merge_pair(best_pair, current_freqs)

        # Add the new token to the vocabulary
        merged_token = best_pair[0] + best_pair[1]
        vocab.add(merged_token)
        merges.append(best_pair)

        if verbose:
            print(f"\nMerge #{step + 1}: ({best_pair[0]!r}, {best_pair[1]!r}) "
                  f"-> {merged_token!r}  (count={best_count})")
            # Show a sample of the current tokenization
            sample = list(current_freqs.keys())[:3]
            for s in sample:
                print(f"   sample: {' '.join(s)}")

    return merges, vocab


# Run training with 10 merges — enough to see morphology emerge
merges, vocab = train_bpe(word_freqs, num_merges=10, verbose=True)

print(f"\n{'=' * 70}")
print(f"Entrainement termine: {len(merges)} merges, vocabulaire = {len(vocab)}")
print(f"{'=' * 70}")
print(f"Vocabulaire final ({len(vocab)} tokens):")
for t in sorted(vocab, key=lambda x: (len(x), x)):
    print(f"  {t!r}")


# ============================================================================
# PART 5: Encoding a new word with the learned merges
# ============================================================================

print("\n" + "=" * 70)
print("PART 5: Encoding new words")
print("=" * 70)


def encode_word(word, merges):
    """
    Tokenize a new word by applying the learned merges in order.

    WHY "in order": BPE is deterministic. Later merges depend on earlier ones.
    If we skipped the order, we would produce different tokens.

    Greedy: we apply each merge to ALL remaining positions in one pass,
    even if the pair appears multiple times.
    """
    # Initial pre-tokenization: list of single characters + </w>
    tokens = pre_tokenize(word)

    # Apply each merge in the order it was learned
    for pair in merges:
        merged = pair[0] + pair[1]
        new_tokens = []
        i = 0
        while i < len(tokens):
            # Look for the pair at the current position
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                new_tokens.append(merged)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens


# Test on words seen during training AND unseen words
test_words = ["low", "lowest", "newer", "wider", "slowest", "widestest"]

print("\nTokenization des mots test:")
print(f"{'Mot':<15s} {'Tokens':<40s}")
print("-" * 55)
for w in test_words:
    result = encode_word(w, merges)
    # Format the tokens for display, hiding the </w> marker for clarity
    display = [t.replace(END_OF_WORD, "_") for t in result]
    print(f"{w:<15s} {' | '.join(display):<40s}")


# ============================================================================
# PART 6: Char-level vs Word-level vs BPE — sequence length comparison
# ============================================================================

print("\n" + "=" * 70)
print("PART 6: Pourquoi BPE ? Comparaison de longueur de sequences")
print("=" * 70)

sample_text = "lowest newer widest"

# Approach A: char-level (each char is a token)
char_tokens = list(sample_text.replace(" ", "_"))
print(f"\nChar-level  ({len(char_tokens)} tokens): {char_tokens}")

# Approach B: word-level (each word is a token)
word_tokens = sample_text.split()
print(f"Word-level  ({len(word_tokens)} tokens): {word_tokens}")

# Approach C: BPE
bpe_tokens = []
for w in sample_text.split():
    bpe_tokens.extend(encode_word(w, merges))
bpe_display = [t.replace(END_OF_WORD, "_") for t in bpe_tokens]
print(f"BPE         ({len(bpe_tokens)} tokens): {bpe_display}")

print("""
Observation:
  - Char-level:  sequence tres longue, vocab minuscule, le modele apprend tout
  - Word-level:  sequence courte, vocab explose, probleme OOV
  - BPE:         compromis optimal, morphologie capturee
""")


# ============================================================================
# PART 7: Bonus — morphology demonstration
# ============================================================================

print("=" * 70)
print("PART 7: La morphologie emerge des merges")
print("=" * 70)

# Train on a slightly larger corpus that has clear morphology
morph_corpus = [
    "run", "run", "run",
    "running", "running", "running",
    "runner", "runner",
    "jump", "jump",
    "jumping", "jumping",
    "jumper",
    "walk", "walking", "walker",
]

morph_freqs = Counter()
for w in morph_corpus:
    morph_freqs[tuple(pre_tokenize(w))] += 1

# Train silently
morph_merges, morph_vocab = train_bpe(morph_freqs, num_merges=15, verbose=False)

print(f"\nTotal merges: {len(morph_merges)}")
print(f"Vocab size: {len(morph_vocab)}")
print("\nMerges appris (ordre):")
for i, pair in enumerate(morph_merges):
    print(f"  {i + 1:2d}. {pair[0]!r} + {pair[1]!r} -> {(pair[0] + pair[1])!r}")

print("\nEncoding:")
for w in ["running", "jumping", "walking", "swimmer"]:
    tokens = encode_word(w, morph_merges)
    display = [t.replace(END_OF_WORD, "_") for t in tokens]
    print(f"  {w:<12s} -> {' | '.join(display)}")

print("""
Remarque: "swimmer" est UNSEEN au training, mais BPE le decompose quand meme
en sous-tokens connus. C'est ca la force du subword tokenization: pas d'OOV,
meme les mots nouveaux ont une representation raisonnable.
""")

print("=" * 70)
print("Fin — tu as maintenant implemente BPE from scratch.")
print("=" * 70)
