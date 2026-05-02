# Exercice (easy) — Ecrire le README portfolio-grade du capstone

## Objectif

Produire un `README.md` prets-a-shipper pour le capstone Diffusion Policy. Ce README doit tenir sur **un seul ecran** et permettre a un recruteur ou un staff engineer de comprendre, lancer, et juger le projet en moins de **3 minutes**.

## Consigne

Tu disposes du code du capstone (J24 -> J28 du parcours) qui implemente une Diffusion Policy entrainee sur PushT (et un mock d'integration LogiSim). Ecris le `README.md` en respectant **strictement** les 5 sections demandees, dans cet ordre.

Sections obligatoires :

1. **Title + 1-line pitch + GIF** (placeholder accepte si tu n'as pas de GIF reel : reference a un fichier `figures/demo.gif` non commite). Le pitch doit contenir au moins **un nombre concret** (ex : `78% success rate`, `36 ms/step`).

2. **What it does** — 3 bullets, dont un avec un chiffre concret. Aucun jargon non explicite (si tu ecris "DDPM", explique en 5 mots a cote).

3. **Quickstart** — 3 commandes maximum (typiquement `pip install -e .`, `python demo.py`, optionnel `make eval`). Chaque commande sur une ligne separee dans un bloc `bash`.

4. **Architecture** — un mini-schema ASCII ou Mermaid (4-7 lignes) decrivant le flux : `image -> ResNet18 -> conditioning -> UNet 1D -> denoising DDPM -> action chunk`. Suivi de **4 a 6 lignes de prose** maximum.

5. **Results** — un tableau Markdown avec au minimum 2 lignes (Diffusion Policy vs baseline BC), 3 colonnes (`method`, `success rate`, `mean episode length`). Plus un lien vers `eval/results.json`.

## Contraintes

- Pas de section "About the author", pas de section "Why this matters", pas de section "License" (l'apparenter a un repo open-source pas a un blog post).
- Au plus **80 lignes** au total (titres compris). Si tu depasses, tu enleves.
- Un GIF de demo est attendu en haut, sous le titre. Place le tag `![demo](figures/demo.gif)` meme sans fichier reel — ca compte comme contrat de packaging.
- Le tableau Results doit refleter des **chiffres realistes** (pas 99.9%, pas 12%). Pour une Diffusion Policy correctement entrainee sur PushT, on est typiquement entre 75% et 90% de success rate.

## Criteres de reussite

- 5 sections presentes, dans l'ordre, exactement.
- Le quickstart tient en 3 commandes ou moins.
- Au moins un chiffre concret dans le pitch + un dans le tableau Results.
- Le schema architecture est lisible en 5 secondes, pas un pave de 30 lignes.
- README sous 80 lignes total.
- Aucun texte de remplissage type "this project demonstrates the power of...".

## Pour aller plus loin

- Ajouter un badge CI (`![tests](...)`) en haut.
- Ajouter une ligne `Acknowledgements` minimaliste creditant Diffusion Policy (Chi et al. 2023).
- Tester le README en simulant la lecture en 30 secondes : qu'est-ce qu'on retient ? Si tu ne sais pas dire ce que fait le projet apres 30s de scroll, refactore.
