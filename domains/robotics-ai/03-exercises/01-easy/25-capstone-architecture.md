# Exercice easy J25 — Compter les parametres et verifier le forward

## Objectif

Prendre en main l'architecture Diffusion Policy en l'**instanciant**, en **comptant** ses parametres, et en verifiant que le **forward** produit les bonnes shapes.

## Consigne

1. Importer la `DiffusionPolicy` et `DPConfig` depuis `domains/robotics-ai/02-code/25-capstone-architecture.py`.
2. Instancier la policy avec la config par defaut (PushT-like).
3. Afficher :
   - Le nombre total de parametres et le nombre de parametres entrainables.
   - La repartition par sous-module : `vision_encoder`, `state_encoder`, `denoiser`.
4. Generer un batch synthetique de taille `B=3` avec `make_fake_batch` et :
   - Appeler `policy.compute_loss(batch)` -> verifier que c'est un scalaire `>= 0`.
   - Appeler `policy.predict_action(batch)` (avec `num_diffusion_steps=4` pour ne pas exploser le temps) -> verifier la shape `(3, T_act, action_dim)`.
5. Repondre par ecrit (3 phrases max) : **quel est le sous-module le plus lourd, et pourquoi ?**

## Criteres de reussite

- Le code tourne sans erreur, en CPU pur, en moins de 30 secondes.
- Le print expose un decompte clair `vision/state/denoiser/total`.
- L'assertion sur la shape de `predict_action` passe.
- La reponse ecrite identifie correctement le bloc dominant (indice : c'est celui qui a les channels les plus larges, jusqu'a 1024).

## Bonus

- Refaire le compte avec `cfg.unet_down_channels = (128, 256, 512)` : combien de parametres economises ?
