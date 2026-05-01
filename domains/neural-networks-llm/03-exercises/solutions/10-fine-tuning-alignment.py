"""
Solutions — Jour 10 : Fine-tuning & Alignment

Run: python 03-exercises/solutions/10-fine-tuning-alignment.py
"""

import sys
import io
import math

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


# ============================================================================
# Exercice 1 — SFT format
# ============================================================================

print("=" * 70)
print("Exercice 1: SFT format and loss masking")
print("=" * 70)

examples = [
    ("Explique la photosynthese simplement",
     "La photosynthese est le processus par lequel les plantes utilisent la "
     "lumiere du soleil pour transformer l'eau et le CO2 en sucre et oxygene."),
    ("Traduis 'hello' en francais",
     "Bonjour."),
    ("Liste 3 pays d'Europe",
     "France, Allemagne, Italie."),
]

for i, (user, assistant) in enumerate(examples, 1):
    print(f"\nExemple {i}:")
    formatted = f"<|user|> {user} <|assistant|> {assistant} <|end|>"
    print(f"  Formatted: {formatted}")
    # Mark which tokens contribute to the loss
    prompt_part = f"<|user|> {user} <|assistant|>"
    response_part = f" {assistant} <|end|>"
    print(f"  Loss masked over: ONLY the response + <|end|>")
    print(f"    PROMPT (LOSS=0): {prompt_part!r}")
    print(f"    REPONSE (LOSS=1): {response_part!r}")

print("""
4) Pourquoi pas de loss sur le prompt ?
   Si on calculait la loss sur le prompt, le modele apprendrait a GENERER
   les prompts, pas a y REPONDRE. On veut qu'il soit expert en "reponses
   assistant", pas en "instructions utilisateur".

5) En PyTorch:
   labels = input_ids.clone()
   labels[:prompt_end] = -100  # ignore these positions in the loss
   loss = F.cross_entropy(logits, labels, ignore_index=-100)
""")


# ============================================================================
# Exercice 2 — DPO margin interpretation
# ============================================================================

print("=" * 70)
print("Exercice 2: DPO margin")
print("=" * 70)

beta = 0.1
scenarios = [
    ("A", -1.0, -4.0, -2.0, -2.0),
    ("B", -2.0, -2.0, -2.0, -2.0),
    ("C", -3.0, -1.0, -2.0, -2.0),
    ("D", -1.0, -5.0, -1.5, -4.0),
]

print(f"\n{'Scenario':>8s} {'margin':>10s} {'loss':>10s}  interpretation")
print("-" * 70)

for name, lp_t_w, lp_t_l, lp_r_w, lp_r_l in scenarios:
    margin = beta * ((lp_t_w - lp_r_w) - (lp_t_l - lp_r_l))
    # Numerically stable log(1 + exp(-margin))
    if margin >= 0:
        loss = math.log(1 + math.exp(-margin))
    else:
        loss = -margin + math.log(1 + math.exp(margin))

    if margin > 0.1:
        note = "policy alignee, loss faible"
    elif margin < -0.1:
        note = "policy opposee, loss forte, gradient pousse vers y_w"
    else:
        note = "margin ~= 0, loss = log(2) ~= 0.693"
    print(f"{name:>8s} {margin:>+10.4f} {loss:>10.4f}  {note}")

print("""
Details:
- A: margin = 0.1 * (1.0 - (-2.0)) = 0.3 -> sigmoid(0.3)=0.574, loss=0.554
- B: margin = 0 -> sigmoid(0)=0.5, loss=log(2)=0.693 (baseline)
- C: margin = 0.1 * (-1.0 - 1.0) = -0.2 -> sigmoid(-0.2)=0.45, loss=0.798
- D: margin = 0.1 * (0.5 - (-1.0)) = 0.15 -> loss=0.619

Bonus beta=1.0: multiplie le margin par 10, les loss sont beaucoup plus
polarisees. La policy converge plus vite mais aussi de maniere plus
instable (risque d'overfit aux preferences).
""")


# ============================================================================
# Exercice 3 — LoRA parameter counting
# ============================================================================

print("=" * 70)
print("Exercice 3: LoRA parameter counting")
print("=" * 70)

# 1) Linear 4096x4096
d = 4096
full = d * d
print(f"\n1) Linear {d}x{d}:")
print(f"   Full FT: {full:,} params")
for r in [1, 4, 8, 16, 64]:
    lora_params = 2 * d * r
    ratio = full / lora_params
    print(f"   LoRA r={r:<3d}: {lora_params:>10,d}  ratio {ratio:>6.0f}x")

# 2) LLaMA 7B attention
n_layers = 32
n_proj = 4  # Q, K, V, O
full_attn = n_layers * n_proj * d * d
lora_attn_r16 = n_layers * n_proj * 2 * d * 16
print(f"\n2) LLaMA 7B attention layers (Q, K, V, O sur 32 couches):")
print(f"   Full FT: {full_attn:,} ({full_attn / 1e9:.2f}B)")
print(f"   LoRA r=16: {lora_attn_r16:,} ({lora_attn_r16 / 1e6:.1f}M)")
print(f"   Ratio: {full_attn / lora_attn_r16:.0f}x")

# 3) Memory for gradients + Adam
print("\n3) Memoire pour une couche d'attention (Q, K, V, O = 4 linears):")
bytes_per_fp32 = 4
params_per_layer = 4 * d * d  # Q, K, V, O
params_mem = params_per_layer * bytes_per_fp32
grad_mem = params_mem  # gradients same size
adam_mem = 2 * params_mem  # moment 1 + moment 2
total_full = params_mem + grad_mem + adam_mem
print(f"   Full FT:")
print(f"     params (fp32):    {params_mem / 1e6:.1f} MB")
print(f"     gradients (fp32): {grad_mem / 1e6:.1f} MB")
print(f"     Adam states:      {adam_mem / 1e6:.1f} MB")
print(f"     TOTAL:            {total_full / 1e6:.1f} MB per layer")
print(f"     Stack 32 couches: {32 * total_full / 1e9:.1f} GB")

r = 16
lora_params_layer = 4 * 2 * d * r
lora_mem = lora_params_layer * bytes_per_fp32
lora_total = lora_mem + lora_mem + 2 * lora_mem  # same formula
print(f"   LoRA r=16:")
print(f"     Trainable params:  {lora_params_layer:,}")
print(f"     TOTAL (params+grad+adam): {lora_total / 1e6:.2f} MB per layer")
print(f"     Stack 32 couches: {32 * lora_total / 1e6:.1f} MB")

# 4) Deployment
print("\n4) Deploiement de 50 modeles fine-tunes sur LLaMA 7B:")
base_7b_fp16 = 14  # GB
print(f"   Base model (fp16): {base_7b_fp16} GB")
full_ft_total = 50 * base_7b_fp16
print(f"   Full FT: 50 * {base_7b_fp16} = {full_ft_total} GB")
lora_adapter_mb = 20  # typical size
lora_total_gb = base_7b_fp16 + (50 * lora_adapter_mb / 1000)
print(f"   LoRA:    {base_7b_fp16} GB base + 50 * {lora_adapter_mb} MB "
      f"= {lora_total_gb:.1f} GB")
print(f"   Ratio: {full_ft_total / lora_total_gb:.1f}x")

print("""
5) Bonus — pourquoi init B=0 et pas A?
   Avec B=0 et A random:
     LoRA_output = x @ A^T @ B^T = x @ A^T @ 0 = 0
     Donc au step 0, le modele LoRA = modele base (aucun impact).

   Avec A=0 et B random:
     LoRA_output = x @ 0 @ B^T = 0
     Aussi nul, donc ca marcherait en principe. Mais:
     - Le gradient de A est 0 (car la chain rule passe par B qui multiplie 0)
       -> A n'apprendrait jamais rien! B ne peut pas apprendre a A = 0 fige.
     En fait: avec B=0, A recoit des gradients non nuls (via le chemin W.T).
     C'est plus stable d'init B=0 + A random que l'inverse.

   Avec A et B TOUS random:
     Le modele part deja biaise. Les premiers steps font exploser la loss
     car l'adapter introduit du bruit avant meme d'avoir appris.
""")
