# Exercice J20 — niveau hard

## Objectif

Faire converger un mini-OpenVLA + LoRA sur une **tache jouet de regression d'action** et mesurer le **gap LoRA vs full finetuning**, comme dans la section 5 du papier OpenVLA (REFERENCES.md #13).

Tu dois experimentalement reproduire le constat du papier : **LoRA r=32 atteint quasi la performance du full finetuning, avec environ 1-2% des params**.

Reference : `domains/robotics-ai/01-theory/20-openvla-architecture.md` section 4 et 5 + repo OpenVLA `vla-scripts/finetune.py`.

## Consigne

1. **Construis un dataset jouet d'action conditionnees image** :
   - Genere `N=2000` couples `(image, action)` ou `image = torch.randn(3, 32, 32)` et `action = sin_target_function(image_mean)` (par exemple une tache `action_dim=4` deterministe). L'objectif est que le modele apprenne une fonction simple pour mesurer la convergence.
   - Split 80/20 train/val.

2. **Construis un mini-VLA** (tu peux importer `OpenVLAMini` depuis `02-code/20-openvla-architecture.py`, mais reduis `image_size=32, patch_size=4, max_text_len=4` pour aller vite, et change `action_dim=4`).
   - Adapte la sortie pour faire de la **regression continue** (au lieu de tokens discrets) : ajoute une `nn.Linear(llm_hidden, action_dim)` apres le dernier hidden state du dernier token. C'est legitime pedagogiquement parce qu'on isole le mecanisme LoRA, pas la tokenization.

3. **Trois experiences** a comparer sur 200 epochs avec AdamW lr=3e-4 :
   - **A. Full finetuning** : tous les params trainables.
   - **B. LoRA r=8** : freeze tout sauf LoRA sur `q_proj, v_proj` de toutes les couches LLM, plus le `nn.Linear` de regression final (qui doit rester trainable pour que la tete sache mapper les hidden states a l'action).
   - **C. LoRA r=32** : meme chose avec rang 32.

4. **Reporte un tableau** :

| Run | Trainable params | Val MSE finale | Temps/epoch |
|---|---|---|---|
| Full FT | ... | ... | ... |
| LoRA r=8 | ... | ... | ... |
| LoRA r=32 | ... | ... | ... |

5. **Discussion (markdown commentaire en tete de ton script)** :
   - Le ratio trainable LoRA r=32 / Full FT est-il coherent avec ce que tu attendais (theoriquement ~1-2% ?).
   - Le gap de Val MSE est-il < 10% ? Si non, pourquoi ? Hypotheses : tete de regression a re-initialiser, lr inadequat, dataset trop simple.

6. **Probe adversariale** : essaie LoRA r=1. Compare le Val MSE final. Documente le crash ou la degradation : c'est la **limite basse** ou le rang devient insuffisant pour exprimer la tache.

## Criteres de reussite

- [ ] Les 3 experiences tournent sans erreur et convergent (Val MSE descend monotonement).
- [ ] Le tableau comparatif est rempli avec des valeurs reelles obtenues sur ta machine.
- [ ] Le ratio params trainables LoRA r=32 / Full FT est entre 0.5% et 5% (cas de figure du mini-modele).
- [ ] Le gap Val MSE LoRA r=32 vs Full FT est analyse explicitement (peut etre positif ou negatif sur dataset jouet, l'important est de discuter).
- [ ] La probe r=1 est reportee avec une phrase de conclusion (typiquement : "r=1 est insuffisant des que la tache demande > 1 direction d'update significative").
- [ ] Bonus : tu testes aussi LoRA sur `mlp.gate, mlp.up, mlp.down` en plus des `q,v` et observes l'effet.

## Indices

- Si la loss explose des le step 0, ta `LoRALinear` n'a probablement **pas** initialise `B` a zero, ou ton scaling est faux.
- Si LoRA stagne tres haut, augmente `alpha` (jusqu'a `alpha = 4*r` peut aider) ou ton lr.
- Pour mesurer les params trainables : `sum(p.numel() for p in model.parameters() if p.requires_grad)`.
