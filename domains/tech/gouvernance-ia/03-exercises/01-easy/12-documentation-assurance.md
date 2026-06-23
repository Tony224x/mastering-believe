# Exercice (easy) — Rédiger une agent card

## Objectif

Savoir produire la **preuve statique** minimale d'un agent : une *agent card* qui répond, sans lire le code, à « qui, quoi, sur quelles données, sous quelle responsabilité, avec quelles limites ». C'est la forme documentaire des 4 piliers de gouvernance.

## Consigne

On vous confie un agent réel à documenter : **`triage-bot`**, qui lit les emails entrants du support et les classe par priorité (P1/P2/P3). Il n'a **aucune** action externe (il ne fait que tagger). Il lit l'adresse email et le corps du message de l'expéditeur. Son owner est *Léa Martin (Support Lead)*. Il a été évalué à 88 % de bon tri sur 300 emails en avril 2026. Limite connue : il sous-classe les emails très courts.

1. Écrivez une fonction `make_card(fields: dict) -> str` qui rend une **agent card en Markdown** à partir d'un dictionnaire de champs.
2. La card DOIT contenir au minimum ces sections : `Identity & ownership` (agent_id, owner, status), `Intended use` (purpose), `Permissions`, `Personal data` (avec base légale si applicable), `Evaluation`, `Known limitations`.
3. Si un champ obligatoire (`owner`, `permissions`) est vide, la card doit l'indiquer visiblement (ex. marqueur `⚠️`) au lieu de l'omettre silencieusement.
4. Remplissez le dictionnaire avec les informations de `triage-bot` ci-dessus et imprimez la card.
5. `triage-bot` ne traite que des données personnelles (email, contenu) : déclarez-les avec une base légale plausible (ex. *intérêt légitime*).

## Critères de réussite

- [ ] `make_card` renvoie une chaîne Markdown contenant les 6 sections listées.
- [ ] Un champ obligatoire manquant est rendu visible (`⚠️`), jamais omis en silence.
- [ ] La card de `triage-bot` est imprimée et chacune de ses sections est remplie (owner, permissions, données perso + base légale, éval, limites).
- [ ] La section `Personal data` affiche bien la base légale choisie.
- [ ] Le script tourne en **stdlib pure** (`python <fichier>`), sans dépendance externe.
