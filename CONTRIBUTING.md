# Contribuer

Ce dépôt est public et ouvert : utilisation, fork et contributions sont bienvenus. Ce fichier décrit
ce qui est attendu pour qu'une contribution soit mergeable — pas pour filtrer, pour éviter les
allers-retours.

## Deux garde-fous automatisés, à passer en local avant la PR

```bash
python shared/tools/build_catalog.py --check   # CATALOG.md + bloc README synchronisés avec les meta.toml
python shared/tools/check_links.py             # aucun lien relatif cassé dans les .md suivis par git
```

La CI rejoue exactement ces deux commandes (`.github/workflows/catalog-and-links.yml`). Si l'un des
deux échoue en local, il échouera aussi en CI : corrige avant d'ouvrir la PR.

- **Catalogue** : `domains/CATALOG.md` et le bloc `<!-- CATALOG:START -->` du README sont **générés**.
  Ne les édite jamais à la main — lance `python shared/tools/build_catalog.py` et commite le résultat.
- **Liens** : un lien relatif doit résoudre depuis le dossier du fichier qui le contient. Les liens
  dans les blocs de code sont ignorés (exemples illustratifs), pas les liens réels.

## Corriger un contenu existant (le cas le plus utile)

Fautes, imprécisions, exercice dont la solution ne tient pas debout, source discutable : la PR est
bienvenue et n'a pas besoin d'être grosse.

1. Un commit = une intention (`fix(domaine): …`, `docs(theorie): …`).
2. Dis **ce qui était faux et pourquoi** dans le message de commit ou la PR : c'est ce qui permet de
   vérifier sans refaire tout le raisonnement.
3. Si tu changes une affirmation factuelle, cite la source (dans `REFERENCES.md` du domaine quand il
   existe, sinon dans la PR).

## Ajouter un domaine

Le plus court chemin est le skill `mastering-domain-creator` (`.claude/skills/mastering-domain-creator/`) :
il déroule le pipeline complet (interview, recherche sourcée, plan challengé, création module par
module, deux passes de vérification, capstone). La procédure manuelle est décrite dans `CLAUDE.md`
§ *Creating a New Domain*.

Le squelette minimal d'un domaine :

```
domains/<track>/<domaine>/
├── README.md         # périmètre, prérequis, planning (durée libre), critères de réussite
├── meta.toml         # slug, title, track, status, level, duration, stack, focus, pillar, guardrail, prerequisites, tags
├── 01-theory/        # NN-slug.md, un module = 30-60 min d'étude
├── 02-code/          # quand pertinent — exemples exécutables en standalone
├── 03-exercises/     # 01-easy/ 02-medium/ 03-hard/ + solutions/ séparées
└── 04-projects/      # mini-projets libres ou capstones supplémentaires
```

Règles non négociables :

- `slug` = nom du dossier, `track` = dossier parent (le générateur le vérifie et échoue sinon)
- numérotation `01-`, `02-`… partout : c'est l'ordre d'apprentissage qui est encodé, pas une décoration
- exercices : 3 faciles, 3 moyens, 2 difficiles et un capstone au minimum
- solutions dans un dossier séparé, jamais mélangées aux énoncés
- `03-exercises/workspace/` est gitignoré : c'est l'espace personnel de l'apprenant, n'y commite rien
- français pour la théorie, anglais pour le code quand le domaine est technique
- aucun contenu personnel ou sensible dans un dépôt public

## Méthode pédagogique attendue

Le dépôt tient une ligne qui n'est pas négociable, dans `CLAUDE.md` § *Learning Methodology Rules* :

1. **Pareto first** — les 20 % qui donnent 80 % des résultats, en premier
2. **Concret avant abstrait** — un exemple, puis le principe
3. **Ancrages de répétition espacée** — 3 à 5 questions type flash-card à la fin de chaque module
4. **Pratique délibérée** — l'exercice cible une faiblesse, il ne répète pas ce qui est déjà acquis
5. **Surcharge progressive** — chaque niveau doit être légèrement au-delà du confort actuel
6. **Capstone réel** — chaque domaine se termine par un projet présentable

Un contenu qui explique beaucoup mais ne fait travailler personne n'est pas dans la ligne du dépôt.

## Licence des contributions

MIT, comme le reste du dépôt (voir [LICENSE](LICENSE)).
