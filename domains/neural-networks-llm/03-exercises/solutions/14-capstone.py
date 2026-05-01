"""
Solutions — Jour 14 : Capstone

Run: python 03-exercises/solutions/14-capstone.py
"""

import sys
import io

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


# ============================================================================
# Exercice 1 — Attention explained in 5 lines
# ============================================================================

print("=" * 70)
print("Exercice 1: Attention en 5 lignes")
print("=" * 70)

print("""
Solution de reference (il y en a plusieurs possibles):

1. Dans une phrase, chaque mot a besoin de regarder d'autres mots pour
   comprendre son sens (ex: 'il' pointe vers un nom plus tot).
2. L'attention donne a chaque mot trois roles: une question (query),
   une etiquette identifiante (key), et un contenu informatif (value).
3. Pour chaque mot, on compare sa query a toutes les keys et on prend
   une moyenne ponderee des values selon la ressemblance.
4. Geometriquement: chaque mot tire de l'information des autres
   en fonction de leur proximite dans un espace vectoriel appris.
5. Contrairement aux RNN qui traitent les mots un par un, tous les mots
   peuvent communiquer en parallele et capturer des dependances lointaines.

Points cles qui doivent apparaitre:
  - Query / Key / Value explicites
  - Moyenne ponderee (weighted sum)
  - Parallelisme (vs RNN sequentiel)
  - Dependances longues (vs LSTM qui les perd)
""")


# ============================================================================
# Exercice 2 — Parameter counting
# ============================================================================

print("=" * 70)
print("Exercice 2: Comptage des parametres mini-LLaMA")
print("=" * 70)

# Config
vocab_size = 32
d_model = 64
n_layers = 2
n_heads = 4
n_kv_heads = 2
head_dim = d_model // n_heads  # = 16
d_ff = 256

# 1) Token embedding
tok_emb = vocab_size * d_model
print(f"\n1) Token embedding: {vocab_size} * {d_model} = {tok_emb}")

# 2) Attention per layer (GQA)
w_q = d_model * (n_heads * head_dim)
w_k = d_model * (n_kv_heads * head_dim)
w_v = d_model * (n_kv_heads * head_dim)
w_o = (n_heads * head_dim) * d_model
attn_per_layer = w_q + w_k + w_v + w_o
print(f"\n2) Attention per layer:")
print(f"   W_q = {d_model} * {n_heads * head_dim} = {w_q}")
print(f"   W_k = {d_model} * {n_kv_heads * head_dim} = {w_k}")
print(f"   W_v = {d_model} * {n_kv_heads * head_dim} = {w_v}")
print(f"   W_o = {n_heads * head_dim} * {d_model} = {w_o}")
print(f"   Total attention = {attn_per_layer}")

# 3) FFN (SwiGLU) per layer
w_gate = d_model * d_ff
w_up = d_model * d_ff
w_down = d_ff * d_model
ffn_per_layer = w_gate + w_up + w_down
print(f"\n3) FFN SwiGLU per layer:")
print(f"   W_gate = {d_model} * {d_ff} = {w_gate}")
print(f"   W_up   = {d_model} * {d_ff} = {w_up}")
print(f"   W_down = {d_ff} * {d_model} = {w_down}")
print(f"   Total FFN = {ffn_per_layer}")

# 4) RMSNorms per layer
norm_per_layer = 2 * d_model
print(f"\n4) 2 RMSNorms per layer: 2 * {d_model} = {norm_per_layer}")

# 5) Total per block
total_per_layer = attn_per_layer + ffn_per_layer + norm_per_layer
print(f"\n5) Total per TransformerBlock: {total_per_layer}")

# 6) Stack
stack = n_layers * total_per_layer
print(f"\n6) Stack of {n_layers} layers: {stack}")

# 7) Final norm
final_norm = d_model
print(f"\n7) Final RMSNorm: {final_norm}")

# 8) lm_head
lm_head = d_model * vocab_size
print(f"\n8) lm_head: {d_model} * {vocab_size} = {lm_head}")

# 9) Grand total
total = tok_emb + stack + final_norm + lm_head
print(f"\n9) GRAND TOTAL: "
      f"{tok_emb} + {stack} + {final_norm} + {lm_head} = {total}")

# 10) Weight tying bonus
saved = vocab_size * d_model
tied_total = total - saved
print(f"\n10) Weight tying (lm_head = token_emb^T):")
print(f"    Saved: {saved} ({saved / total * 100:.2f}% du modele)")
print(f"    New total: {tied_total}")


# ============================================================================
# Exercice 3 — Modifying GQA config
# ============================================================================

print("\n" + "=" * 70)
print("Exercice 3: Modifier mini-LLaMA (8 KV heads)")
print("=" * 70)

print("""
1) Contrainte: n_heads doit etre divisible par n_kv_heads, et n_heads >= n_kv_heads.
   Config originale: n_heads=4, n_kv_heads=2 -> OK (group size 2)
   Nouvelle demande: n_kv_heads=8.
   Impossible avec n_heads=4: on ne peut pas avoir plus de KV heads que de Q heads.

2) Nouvelles configs valides:
   Option A: n_heads=8, n_kv_heads=8 (MHA, pas de GQA)
   Option B: n_heads=16, n_kv_heads=8 (GQA group size 2)
   Option C: n_heads=8, n_kv_heads=2 (GQA group size 4, original config mais plus de heads)

   Pour passer simplement a 8 KV heads en gardant une philosophie MHA,
   on prend Option A: n_heads=8, n_kv_heads=8.
""")

# Recalculate with new config
new_n_heads = 8
new_n_kv_heads = 8
new_head_dim = d_model // new_n_heads  # = 8 (smaller!)

new_w_q = d_model * (new_n_heads * new_head_dim)
new_w_k = d_model * (new_n_kv_heads * new_head_dim)
new_w_v = d_model * (new_n_kv_heads * new_head_dim)
new_w_o = (new_n_heads * new_head_dim) * d_model
new_attn = new_w_q + new_w_k + new_w_v + new_w_o

print(f"3) Attention per layer (n_heads=8, n_kv_heads=8, head_dim={new_head_dim}):")
print(f"   W_q = {new_w_q}")
print(f"   W_k = {new_w_k}")
print(f"   W_v = {new_w_v}")
print(f"   W_o = {new_w_o}")
print(f"   Total = {new_attn}")
print(f"\n   Original attn: {attn_per_layer}")
print(f"   New attn:      {new_attn}")
print(f"   Diff: {new_attn - attn_per_layer} "
      f"({(new_attn - attn_per_layer) / attn_per_layer * 100:+.1f}%)")
print("   Remarque: le total est le meme car (n_heads*head_dim) = d_model dans les deux cas")
print("   La difference est sur K et V (n_kv_heads a change).")

# 4) KV cache size
seq_len = 128
batch = 1
bytes_per_elem = 4  # fp32

old_cache = 2 * n_layers * n_kv_heads * head_dim * seq_len * batch * bytes_per_elem
new_cache = 2 * n_layers * new_n_kv_heads * new_head_dim * seq_len * batch * bytes_per_elem

print(f"\n4) KV cache (seq={seq_len}, batch={batch}, fp32):")
print(f"   Original (n_kv_heads=2, head_dim=16): {old_cache} bytes = {old_cache / 1024:.1f} KB")
print(f"   New      (n_kv_heads=8, head_dim=8):  {new_cache} bytes = {new_cache / 1024:.1f} KB")
print(f"   Ratio: {new_cache / old_cache:.1f}x")
# Note: here the ratio is 4 because n_kv_heads*head_dim doubled (2*16=32 -> 8*8=64)

print("""
5) Trade-off:
   Plus de KV heads:
     + qualite (chaque query peut avoir une K/V dediee, plus de diversite)
     - plus de memoire (KV cache plus gros)
     - plus lent en inference (memory-bound)

   Moins de KV heads (MQA/GQA):
     + cache plus petit, inference plus rapide
     + peu de perte de qualite si bien choisi (GQA ~= MHA a 0.3% pres)
     - un peu moins de qualite (surtout au MQA extreme)

   Quand prefere-t-on MHA:
     - training (pas de contrainte d'inference)
     - quand la qualite est critique (modeles de recherche)
     - quand le batch est petit (le cache reste petit)

   Quand prefere-t-on GQA:
     - inference a grande echelle
     - gros modeles (70B+) ou le cache explose
     - production avec gros batch

6) Pour modifier le code et tester:
   - Ouvrir 02-code/14-capstone.py
   - Ligne dans MiniLLaMA constructeur: n_kv_heads=2 -> n_kv_heads=8
   - Aussi changer n_heads=4 -> n_heads=8 (sinon ValueError)
   - Lancer: python 02-code/14-capstone.py
   - Observer les nouveaux params affiches
""")
