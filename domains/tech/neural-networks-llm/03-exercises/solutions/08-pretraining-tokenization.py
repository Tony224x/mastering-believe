"""
Solutions — Jour 8 : Pre-training & Tokenization
=================================================
Full worked solutions for the 3 easy exercises.

Run: python 03-exercises/solutions/08-pretraining-tokenization.py
"""

import sys
import io
from collections import Counter

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


# ============================================================================
# Exercice 1 — BPE a la main sur "hug hug hug pug pug bun bun bunny bunny bunny"
# ============================================================================

print("=" * 70)
print("Exercice 1: BPE a la main")
print("=" * 70)

corpus = ["hug", "hug", "hug", "pug", "pug", "bun", "bun",
          "bunny", "bunny", "bunny"]

# Step 1: pre-tokenization
EOW = "</w>"
word_freqs = Counter()
for w in corpus:
    word_freqs[tuple(list(w) + [EOW])] += 1

print("\n1) Frequences apres pre-tokenization:")
for w, f in word_freqs.items():
    print(f"   {' '.join(w):<25s} x {f}")

# Step 2: initial vocabulary
initial_vocab = set()
for w in word_freqs:
    for t in w:
        initial_vocab.add(t)
print(f"\n2) Vocabulaire initial ({len(initial_vocab)} tokens): "
      f"{sorted(initial_vocab)}")


def count_pairs(freqs):
    """Count weighted adjacent pairs across all words."""
    pairs = Counter()
    for word, f in freqs.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += f
    return pairs


def apply_merge(pair, freqs):
    """Replace every (a, b) with the merged token a+b."""
    new_freqs = {}
    a, b = pair
    merged = a + b
    for word, f in freqs.items():
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_freqs[tuple(new_word)] = f
    return new_freqs


# Iterations 1 to 3
merges = []
current = dict(word_freqs)
for it in range(3):
    pairs = count_pairs(current)
    best_pair, best_count = pairs.most_common(1)[0]
    print(f"\n3/4/5) Iteration {it + 1}:")
    print(f"   Top 3 pairs: {pairs.most_common(3)}")
    print(f"   -> merge ({best_pair[0]!r}, {best_pair[1]!r}) "
          f"count={best_count}")
    current = apply_merge(best_pair, current)
    merges.append(best_pair)
    print(f"   Corpus apres merge:")
    for w, f in current.items():
        print(f"      {' | '.join(w):<30s} x {f}")

# Final vocab
final_vocab = set(initial_vocab)
for pair in merges:
    final_vocab.add(pair[0] + pair[1])
print(f"\n6) Vocabulaire final ({len(final_vocab)} tokens):")
for t in sorted(final_vocab, key=lambda x: (len(x), x)):
    print(f"   {t!r}")


def encode(word, merges):
    tokens = list(word) + [EOW]
    for pair in merges:
        merged = pair[0] + pair[1]
        new_tokens = []
        i = 0
        while i < len(tokens):
            if (i < len(tokens) - 1 and tokens[i] == pair[0]
                    and tokens[i + 1] == pair[1]):
                new_tokens.append(merged)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens


print("\n7) Encoding new words:")
print(f"   'bug'    -> {encode('bug', merges)}")
print(f"   'hungry' -> {encode('hungry', merges)}")
print("""
Observation: 'bug' reutilise les merges qui impliquent 'g</w>' si appris.
'hungry' contient 'hun' et 'gry' — si ces sous-parties n'ont pas ete
merges pendant le training, on retombe sur des caracteres individuels.
C'est la force de BPE: pas d'OOV, meme pour les mots totalement inconnus.
""")


# ============================================================================
# Exercice 2 — CLM vs MLM loss structure
# ============================================================================

print("=" * 70)
print("Exercice 2: CLM vs MLM loss")
print("=" * 70)

sentence = ["The", "cat", "sits", "on", "the", "mat"]
N = len(sentence)
print(f"\nPhrase: {sentence}  (N={N} tokens)")

# 1) CLM: predict token at position i+1 given tokens <= i
print("\n1) CLM (GPT-style):")
print("   Input:  [The, cat, sits, on, the]")
print("   Target: [cat, sits, on, the, mat]")
print("   Loss = -sum_{i=0}^{N-2} log p_{i+1}[target_i+1]")
print(f"   Nombre de termes: N-1 = {N - 1}")
print("   Signal: 100% des tokens contribuent (sauf le tout premier)")

# 2) MLM: only masked positions
print("\n2) MLM (BERT-style), positions 2 et 4 masquees:")
print("   Input:  [The, cat, [MASK], on, [MASK], mat]")
print("   Loss = -log p_2[sits] - log p_4[the]")
print("   Nombre de termes: 2 (seulement les masques)")
print(f"   Proportion: 2/{N} = {2 / N:.0%} des tokens contribuent")

# 3) Span corruption
print("\n3) Span corruption (T5-style), span [sits, on] masque avec <X>:")
print("   Encoder input:  [The, cat, <X>, the, mat]")
print("   Decoder target: [<X>, sits, on, <Y>]")
print("   <Y> est le end-of-span sentinel — utile quand plusieurs spans")

# 4) Efficiency comparison
print("\n4) Sur 1000 tokens:")
print("   CLM:  ~1000 predictions utiles")
print("   MLM:  ~150 predictions utiles (15% de masking)")
print("   -> CLM fournit environ 6-7x plus de signal par token.")
print("   C'est une des raisons pour lesquelles CLM a gagne en 2024.")


# ============================================================================
# Exercice 3 — Scaling laws Chinchilla
# ============================================================================

print("\n" + "=" * 70)
print("Exercice 3: Scaling laws Chinchilla")
print("=" * 70)


def compute_flops(N, D):
    """Approximate training FLOPs for a transformer: C = 6 * N * D."""
    return 6 * N * D


# Cas 1: GPT-3
N_gpt3 = 175e9
D_gpt3 = 300e9
C_gpt3 = compute_flops(N_gpt3, D_gpt3)
ratio_gpt3 = D_gpt3 / N_gpt3
print(f"\n1) GPT-3 (175B params, 300B tokens):")
print(f"   C = 6 * {N_gpt3:.0e} * {D_gpt3:.0e} = {C_gpt3:.2e} FLOPs")
print(f"   Ratio D/N = {ratio_gpt3:.2f}  (optimal Chinchilla: 20)")
print(f"   -> SOUS-ENTRAINE (ratio 1.71 << 20)")

# Cas 2: Chinchilla
N_chi = 70e9
D_chi = 1.4e12
C_chi = compute_flops(N_chi, D_chi)
ratio_chi = D_chi / N_chi
print(f"\n2) Chinchilla (70B params, 1.4T tokens):")
print(f"   C = {C_chi:.2e} FLOPs")
print(f"   Ratio D/N = {ratio_chi:.2f}  (= optimal)")
print(f"   Chinchilla utilise {C_chi / C_gpt3:.2f}x plus de FLOPs que GPT-3")
print(f"   mais bat GPT-3 sur la plupart des benchmarks avec 2.5x moins "
      f"de params.")

# Cas 3: budget C = 1e22 FLOPs
# Equation: C = 6 * N * D, avec D = 20 * N
# -> C = 6 * N * 20N = 120 * N^2
# -> N = sqrt(C / 120)
C_budget = 1e22
N_opt = (C_budget / 120) ** 0.5
D_opt = 20 * N_opt
print(f"\n3) Budget C = 1e22 FLOPs:")
print(f"   N_opt = sqrt(C / 120) = {N_opt:.2e} params (~{N_opt / 1e9:.1f}B)")
print(f"   D_opt = 20 * N_opt = {D_opt:.2e} tokens (~{D_opt / 1e9:.0f}B)")
print("   Verification: C = 6 * N * D = "
      f"{compute_flops(N_opt, D_opt):.2e}  (doit egaler 1e22)")

# Cas 4: LLaMA 3 8B
N_l3 = 8e9
D_l3 = 15e12
ratio_l3 = D_l3 / N_l3
print(f"\n4) LLaMA 3 8B (8B params, 15T tokens):")
print(f"   Ratio D/N = {ratio_l3:.0f}  ({ratio_l3 / 20:.1f}x l'optimum)")
print("   Pourquoi ? Parce que l'optimum Chinchilla minimise le TRAINING COST")
print("   mais pas l'INFERENCE COST. Meta sur-entraine un petit modele pour")
print("   reduire le cout en production (servir des milliards de requetes).")

# Bonus: scaling
print("\n5) Bonus — si compute est double (C -> 2C):")
print("   N_opt ∝ sqrt(C) -> N doit etre multiplie par sqrt(2) ≈ 1.41")
print("   D_opt ∝ sqrt(C) -> D doit etre multiplie par sqrt(2) ≈ 1.41")
print("   (contrairement a Kaplan qui disait de privilegier N)")
