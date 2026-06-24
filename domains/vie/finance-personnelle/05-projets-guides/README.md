# 05 — Projets guides (Finance Personnelle)

> **Disclaimer (vaut pour tous les projets de ce dossier).** Contenu
> **purement éducatif**. Rien ici n'est un conseil financier, fiscal ou en
> investissement personnalisé. Les taux et rendements sont **illustratifs et
> non garantis** ; tout investissement comporte un risque de perte en capital.
> Pour des décisions réelles, consultez un conseiller agréé.
>
> Contexte métier partagé : [`shared/logistics-context.md`](../../../../shared/logistics-context.md).

## Le pont avec le contexte LogiSim

Ce domaine porte sur la **finance personnelle** — celle d'un individu, pas la
comptabilité d'entreprise (le `README.md` du domaine exclut explicitement le
stock-picking, le conseil fiscal et la finance d'entreprise du périmètre). Pour
rester dans ce périmètre **tout en partageant le contexte LogiSim** commun aux
`05-projets-guides/` du repo, les protagonistes de ces projets sont des
**personnes qui travaillent dans l'écosystème LogiSim** — opérateurs de l'OCC,
technicienne terrain, chef de mission — confrontées à des décisions d'argent
ordinaires. LogiSim sert de **décor narratif** crédible, pas de sujet : on
n'analyse jamais les finances de l'entreprise, seulement celles de ses gens.

C'est volontaire et honnête : forcer de la « finance perso » sur un bilan
d'entreprise aurait trahi le périmètre du domaine.

## Particularité « Track Vie » : code léger, mais code réel

Comme les autres domaines de la *Track Vie*, le code est **léger** (simulateurs
et scripts d'analyse, stdlib uniquement) — mais il est **réel et exécutable** :
chaque corrigé tourne sans dépendance, de façon déterministe, et a été vérifié
(`python …` + `ruff check` + probes adversariales). Le livrable d'apprentissage
combine **un script qui calcule** et **une `analyse.md` qui interprète** — parce
qu'en finance personnelle, un nombre nu n'aide pas à décider ; c'est la lecture
qui fait la compétence.

## Projets

| # | Projet | Protagoniste LogiSim | Modules mobilisés | Niveau |
|---|---|---|---|---|
| 01 | **Le plan d'épargne d'une équipe OCC** | 4 opérateurs de la salle de contrôle | 01 intérêts composés · 02 épargne · 05 frais | facile → moyen |
| 02 | **Faut-il s'endetter pour la voiture ?** | une technicienne terrain | 03 dette & crédit · 01 coût d'opportunité | moyen |
| 03 | **Allocation & indépendance financière** | un chef de mission | 04 investir · 05 frais · 06 indépendance · 01 | difficile (intégrateur) |

Chaque dossier suit le même gabarit : un `README.md` en 7 sections (contexte,
objectif, consigne, étapes guidées, critères de réussite, corrigé, pour aller
plus loin) + un `solution/` contenant le script commenté **et** une `analyse.md`.

## Méthodologie

Pour chaque projet :
1. **Lis le contexte** et la consigne — repère la décision financière réelle en jeu.
2. **Écris ton script** d'abord, sans regarder le corrigé. Vise un résultat *déterministe* et *interprétable*.
3. **Exécute-le** sur plusieurs jeux de paramètres (c'est tout l'intérêt d'un simulateur : tester *tes* hypothèses).
4. **Confronte** ta sortie et ta lecture à `solution/` + `analyse.md`.
5. **Lis les limites honnêtes** de l'analyse : ce que le modèle *ne dit pas* (inflation, fiscalité, risque) est aussi important que ce qu'il dit.

## Lancer les corrigés

```bash
python 01-epargne-equipe-occ/solution/epargne_simulator.py
python 02-credit-vehicule-technicien/solution/credit_analyzer.py
python 03-allocation-independance-chef-mission/solution/tableau_bord.py
```

stdlib uniquement (Python 3.11+ recommandé) — aucune installation requise.
