# J18 - Exercice HARD - JEPA video conditionnee + planning latent goal-image

## Objectif

Etendre la mini-JEPA `02-code/18-jepa-cosmos.py` en une version **action-conditionnee video** sur sequences toy, puis l'utiliser pour faire du **planning dans l'espace latent** a la maniere de V-JEPA 2 (REFERENCES.md #21) : etant donne `(o_t, o_goal)`, choisir l'action qui rapproche le mieux le latent predit `z_{t+1}` du latent goal `z_goal`.

## Consigne

### Partie 1 - Toy video env

1. Construis un env synthetique 1D : un point se deplace sur une grille `8x8`. Observation = image binaire 8x8 avec 1 pixel actif a la position du point. Actions discretes : `{up, down, left, right, stay}`. La transition est deterministe et bornee (pas de sortie de grille).

2. Genere un dataset de `N=5000` triplets `(o_t, a_t, o_{t+1})`.

### Partie 2 - JEPA action-conditionnee

3. Modifie l'architecture mini-JEPA pour conditionner le predictor sur l'action :
   ```
   z_ctx = encoder_ctx(o_t)
   a_emb = action_embedding(a_t)        # nn.Embedding(5, action_emb_dim)
   z_hat = predictor(concat(z_ctx, a_emb))
   z_tgt = encoder_tgt(o_{t+1}).detach()
   loss = MSE(z_hat, z_tgt)
   ```

4. Conserve l'EMA de `encoder_tgt` (decay=0.99).

5. Entraine sur les 5000 triplets pendant 200 epochs. Verifie la convergence : la loss latente doit descendre puis se stabiliser.

### Partie 3 - Planning latent goal-image

6. Implemente `plan_action(model, o_t, o_goal)` :
   ```
   z_t = model.encoder_ctx(o_t)
   z_goal = model.encoder_tgt(o_goal)         # encoder cible (figee)
   for a in [up, down, left, right, stay]:
       z_hat_a = model.predictor(concat(z_t, embed(a)))
       score_a = -||z_hat_a - z_goal||^2
   return argmax_a score_a
   ```

7. Evalue sur 200 episodes : pour chaque episode, tirer `o_0` aleatoire, tirer `o_goal` aleatoire (a 1-3 cases de distance), executer le planner pendant max 10 steps, mesurer si on atteint `o_goal`.

8. Compare avec **deux baselines** :
   - **Random policy** : action uniforme.
   - **Pixel planner** : meme algo mais en utilisant `||decoder(z_hat_a) - o_goal||^2` (pixel space) — il faut donc entrainer en parallele un PixelAE conditionne par action.

### Partie 4 - Multimodal stress test

9. Modifie l'env : avec proba 30%, l'action `up` echoue silencieusement (le point reste sur place). Le futur n'est plus deterministe.

10. Re-entraine les deux modeles. Mesure la **qualite des reconstructions pixel** vs **qualite du plan** :
    - Le PixelAE produit-il des images floues ? (calculer la variance pixel-wise des reconstructions sur des batches identiques).
    - La JEPA est-elle plus robuste a cette stochasticite ?

## Criteres de reussite

- L'env toy fonctionne, dataset de 5000 transitions genere et split train/eval.
- La JEPA conditionnee converge (loss latente strictement decroissante puis plateau).
- Le **JEPA-planner atteint le goal** dans > 70% des episodes en ≤ 10 steps, contre < 20% pour random.
- Le **pixel-planner** est competitif au moins en deterministe — sinon, expliquer pourquoi (sur cette tache simple, les deux peuvent marcher).
- Sur la version stochastique (action `up` echoue 30% du temps) :
  - Les reconstructions PixelAE deviennent floues (variance pixel-wise reduit, le decoder moyenne les modes).
  - La JEPA conserve un latent informatif (probe / planner reussite plus stable).
- L'analyse cite explicitement V-JEPA 2 (REFERENCES.md #21) et identifie que ce TP est une **version miniature** du zero-shot pick-and-place demontre par Meta : meme pattern (encoder context, encoder goal, predictor latent action-conditionne, argmax sur actions discretes ou planning continu via gradient).
- Bonus : remplacer l'argmax discret par une **MPC dans l'espace latent** (CEM ou random shooting sur sequences d'actions de horizon `H=3`). Comparer le succes single-step vs multi-step planning.
- Bonus 2 : esquisser comment cette pipeline se brancherait sur un foundation model **Cosmos** (REFERENCES.md #22) — typiquement remplacer `encoder_ctx` et `encoder_tgt` par un Cosmos-Tokenizer pre-entraine, ne reapprendre que le predictor (LoRA-style).
