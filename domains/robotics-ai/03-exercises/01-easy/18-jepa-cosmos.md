# J18 - Exercice EASY - Categoriser 10 scenarios entre Dreamer, JEPA, Cosmos

## Objectif

Verifier que tu sais distinguer les trois paradigmes (Dreamer = generer pixels du futur + imagination RL, JEPA = predire dans l'espace latent, Cosmos = foundation video pretrain a scale) et choisir le bon outil par scenario.

## Consigne

Pour chacun des **10 scenarios** ci-dessous, indique :

1. Quel paradigme est le plus adapte (**Dreamer** | **JEPA / V-JEPA 2** | **Cosmos** | **aucun des trois** — propose alors une alternative).
2. Une justification en **1 phrase max** s'appuyant sur la propriete cle du paradigme retenu (loss pixel vs latent vs pretrain scale, online vs offline, etc.).

### Scenarios

1. Tu as un quadrupede en simulation MuJoCo et 30 minutes de wallclock GPU. Tu veux apprendre une policy locomotion sans collecter 100M steps reels.
2. Tu veux pretrainer un encodeur visuel sur 1M heures de video YouTube, puis le brancher en backbone d'un VLA pour ton robot industriel.
3. Tu travailles sur un humanoide et tu n'as que 200 demos teleoperees. Tu veux **augmenter** ton dataset en generant 50k trajectoires videos plausibles.
4. Tu veux que ton robot saisisse une tasse zero-shot a partir d'une **goal image** ("voici a quoi ca doit ressembler quand c'est fait"), sans demos sur cette tache.
5. Tu fais un Dreamer-like mais ton decoder produit toujours des images floues. Quelqu'un te suggere d'enlever le decoder. C'est qui ?
6. Tu veux **tokenizer** tes propres videos d'usine en tokens compacts pour entrainer un transformer policy. Tu n'as ni le temps ni le GPU pour entrainer un tokenizer from scratch.
7. Tu fais du Atari avec 100k interactions seulement. Tu veux apprendre un actor-critic dans une "imagination" du jeu pour ne pas exploser le budget d'interactions.
8. Tu lis le blog AI Meta de juin 2025 qui annonce qu'un robot fait pick-and-place sur des objets jamais vus, **uniquement** depuis du pretraining self-supervised sur 1M h de video sans label. Quel paradigme ?
9. Tu prepares un capstone Diffusion Policy (cf J16 et J24-J28). Tu hesites a "ajouter un world model" pour generer le futur du PushT. Question piege : c'est quel paradigme et est-ce une bonne idee ?
10. Tu veux faire **du planning** dans l'espace latent (encoder l'observation actuelle et l'objectif, chercher l'action qui rapproche `z_t` de `z_goal`).

## Criteres de reussite

- Les 10 scenarios sont classifies. Pas plus d'une erreur.
- Chaque justification mentionne explicitement une **propriete distinctive** du paradigme retenu (ex : "Cosmos parce que pretrain scale 20M h", "JEPA parce que MSE latent + EMA target", "Dreamer parce que imagination pour data-efficiency RL").
- Le scenario 9 (capstone Diffusion Policy) est correctement traite : un world model n'est **pas necessaire** pour Diffusion Policy. Diffusion Policy est une **action diffusion** (predit l'action) ; un world model predit le futur du monde — confusion classique a eviter.
- Le scenario 10 doit identifier **JEPA** : V-JEPA 2 a explicitement demontre du goal-image planning dans l'espace latent (REFERENCES.md #21, blog Meta 2025).
- Bonus : pour le scenario 5, citer explicitement l'argument LeCun (99% des pixels sont du bruit pour la decision, generer la moyenne d'une distribution multimodale donne du flou).
