# J21 — Architecture des coding agents : ACI, boucle edit/search/run et SWE-bench

> **Temps estime** : 3h | **Prerequis** : J1-J20
> **Objectif** : comprendre comment un coding agent perçoit et modifie un depot de code — l'Agent-Computer Interface (ACI), la boucle edit/search/run, et comment SWE-bench evalue ces systemes en conditions reelles.

---

## 1. Un LLM est un nouvel "utilisateur" du systeme de fichiers

Quand tu lances Claude Code, Cursor ou Aider, le LLM n'interagit pas directement avec le systeme de fichiers comme un programme classique. Il **communique en langage naturel** avec une couche intermediaire — l'**Agent-Computer Interface (ACI)** — qui traduit ses intentions en appels bas niveau.

Le papier SWE-agent (Yang et al., 2024, NeurIPS) formalise ce concept :

> "Just as humans use GUIs or CLIs adapted to their cognitive interface, LLMs need interfaces tailored to their strengths — long context, language understanding — and weaknesses — hallucination, imprecision."

Un LLM a des besoins specifiques quand il travaille sur un repo :
- **Lire des fichiers** sans etre noye dans 10 000 lignes d'un coup
- **Naviguer** un depot sans connaître son arborescence par coeur
- **Editer** du code avec precision, sans regenerer tout un fichier
- **Executer** des tests et interpreter les sorties
- **Chercher** un symbole ou un pattern dans des dizaines de fichiers

Ces besoins sont fondamentalement **differents** de ceux d'un IDE humain (GUI, autocompletion, debugger visuel). L'ACI est la reponse : un jeu d'outils conçu *pour le LLM*, pas pour l'humain.

> **Analogie** : un chirurgien et un bricoleur du dimanche peuvent tous les deux "couper" quelque chose, mais leurs instruments sont radicalement differents. Un LLM a besoin d'instruments adaptes a sa nature — textuels, contextualises, deterministes — pas d'une souris et d'un ecran.

---

## 2. L'Agent-Computer Interface (ACI) — anatomie

### 2.1 Les 4 categories d'outils

Une ACI minimaliste pour un coding agent couvre 4 domaines :

| Categorie | Exemples d'outils | Pourquoi c'est necessaire |
|-----------|-------------------|--------------------------|
| **Viewer** | `open_file(path, line_start, line_end)`, `list_dir(path)` | Limiter la fenetre de contexte, eviter de noyer le LLM |
| **Search** | `search(pattern, path)`, `grep(regex)`, `find_definition(symbol)` | Localiser rapidement un symbole dans un grand depot |
| **Edit** | `edit(file, old, new)`, `insert(file, line, content)` | Modifications chirurgicales sans regenerer tout le fichier |
| **Run** | `run_tests()`, `run_command(cmd)`, `lint(file)` | Obtenir du feedback immediat sur un changement |

### 2.2 Ce que SWE-agent apporte en plus

SWE-agent introduit des outils penses specifiquement pour les taches de deboggage :

- **`scroll_up` / `scroll_down`** : le fichier est affiche par blocs de N lignes, comme un pager. Le LLM navigue sans charger tout le fichier.
- **`goto(line)`** : sauter directement a une ligne connue (apres un search).
- **`search_file(pattern, file)`** : grep dans un seul fichier.
- **`find_file(name)`** : localiser un fichier par nom dans tout le repo.
- **`edit(start_line, end_line, new_content)`** : remplacement de plage de lignes — plus robuste que "chercher/remplacer" une chaine qui peut aparaître plusieurs fois.

### 2.3 Pourquoi les interfaces humaines sont mauvaises pour les LLM

Les LLM se comportent mal quand on leur donne des outils naifs :

- **`cat fichier.py`** : si le fichier a 3 000 lignes, le contexte explose et le LLM perd la notion du debut
- **Regeneration complete** : demander au LLM d'ecrire tout le fichier corrige genere des hallucinations et perd des changements
- **Shell brut** : les erreurs de sous-process sont souvent cryptiques et mal interprétees par le LLM

Les resultats du papier SWE-agent montrent qu'une bonne ACI multiplie par **3 a 5** le taux de resolution sur SWE-bench comparé a un shell brut.

---

## 3. La boucle edit/search/run

La boucle fondamentale d'un coding agent est :

```
[OBSERVATION] -> [REFLEXION] -> [ACTION] -> [OBSERVATION] -> ...
```

En pratique pour corriger un bug :

```
1. search("ImportError")         -> localise le fichier et la ligne
2. open_file("src/utils.py", 40) -> lit le contexte autour du bug
3. edit(old="from foo import X", new="from bar import X")
4. run_tests()                   -> voit si les tests passent
5. Si echec -> retour a l'etape 1 avec nouvelle info
6. Si succes -> commit / fin
```

### 3.1 Localisation (Search)

Avant de toucher du code, l'agent doit **trouver** le bug. Strategies :

- **Grep de l'erreur** : chercher le message d'erreur dans les fichiers source
- **Stack trace parsing** : extraire le fichier et la ligne du traceback
- **Repo-map** (Aider) : une carte miniature du depot (toutes les classes, fonctions, leurs signatures) tient dans le contexte. Le LLM decide quels fichiers lire en entier.

### 3.2 Edition (Edit)

Les bonnes ACI utilisent des **edit blocks** plutot que la regeneration :

```
<<<< ORIGINAL
def add(a, b):
    return a - b   # bug : soustraction
====
def add(a, b):
    return a + b   # fix
>>>> UPDATED
```

Aider appelle cela des "edit blocks" (format SEARCH/REPLACE). Claude Code utilise un format similaire de diff. L'avantage :
- **Moins de tokens** : on ne retransmet pas les parties non modifiees
- **Moins d'hallucinations** : l'edit cible est precis
- **Verificable** : on peut rejeter un edit si le pattern ORIGINAL n'est pas trouve

### 3.3 Execution (Run)

Apres chaque edit, l'agent lance les tests :

```python
result = run_command("python -m pytest tests/ -x --tb=short")
# -x : stoppe au premier echec
# --tb=short : sortie plus courte = moins de tokens
```

L'agent interprete le resultat :
- **PASSED** → succes, si c'est l'objectif final, s'arreter
- **FAILED** → analyser le nouveau message d'erreur, relancer la boucle
- **ERROR** (syntax error, import error) → l'edit a casse quelque chose, revenir en arriere

### 3.4 Iteration et convergence

Un bon coding agent est parametre avec :
- **Max iterations** : eviter les boucles infinies (typiquement 30-50 steps)
- **Budget de tokens** : couper si le contexte explose
- **Backtracking** : certains agents (Claude Code) peuvent annuler un edit si les tests empirent

---

## 4. SWE-bench — evaluer les coding agents sur des vrais bugs

### 4.1 Le probleme de l'evaluation

Comment mesurer qu'un coding agent est bon ? Les benchmarks classiques (HumanEval, MBPP) testent des problemes algorithmiques isoles. Mais corriger un bug dans un vrai depot logiciel, c'est beaucoup plus difficile :
- Le depot peut avoir 100 000 lignes
- Le bug peut impliquer 5 fichiers differents
- Les tests de regression sont nombreux et lents
- L'historique git est une source d'information

### 4.2 SWE-bench (Jimenez et al., 2023, ICLR 2024)

SWE-bench est constitue de **2 294 issues GitHub** reelles provenant de 12 depots Python populaires (Django, Flask, scikit-learn, pytest, etc.). Pour chaque issue :

1. Le repo est clone a l'etat **avant** le fix
2. L'agent reçoit l'issue (texte) + le code du repo
3. L'agent produit un patch
4. Le patch est applique, les tests de validation sont lances
5. **Succes = tous les tests de validation passent**

```
Issue #42 : "ModelForm raises ValueError when field has no default"
 -> L'agent doit retrouver le fichier, comprendre le bug,
    editer le fix, verifier que les tests Django passent.
```

### 4.3 SWE-bench Verified

La variante **Verified** (500 instances) est annotee par des humains pour s'assurer que les issues sont solubles et que les tests de validation sont pertinents. C'est le benchmark le plus difficile et le plus citable.

Scores indicatifs (2024-2025) :
- Agents naifs (LLM + shell brut) : ~5%
- SWE-agent (GPT-4) : ~12%
- Claude 3.5 Sonnet + SWE-agent : ~23%
- Meilleurs agents (Devin v2, agents Claude Code-like) : ~50-60%

### 4.4 Ce que SWE-bench revele

- **La localisation est le goulot d'etranglement** : les agents qui echouent ne trouvent souvent pas le bon fichier
- **La regression est facile** : corriger un bug en en cassant 3 autres est frequent
- **Le contexte long aide peu** : mettre tout le repo dans le contexte ne marche pas mieux que la recherche incrementale

---

## 5. Outils de coding agents en production

### 5.1 Aider — repo-map et edit blocks

Aider (Paul Gauthier) est open-source et a introduit deux innovations majeures :

**Repo-map** : un arbre miniature du depot, genere par `ctags` ou `tree-sitter`, qui liste toutes les classes, fonctions et leurs emplacements. Exemple :

```
src/utils.py
  def add_numbers(a, b)
  class Calculator
    def multiply(self, x, y)
src/main.py
  def main()
```

Cette "carte" tient dans le contexte et permet au LLM de decider *quels fichiers* lire en entier, sans charger le depot complet.

**Auto-commit** : apres chaque edit valide, Aider cree automatiquement un commit git avec un message genere. Cela permet de revenir en arriere proprement.

### 5.2 Claude Code — hierarchie et CLAUDE.md

Claude Code (Anthropic) ajoute :
- **CLAUDE.md** : fichier de contexte projet lu systematiquement, qui donne les conventions, commandes, architecture
- **Permissions incrementales** : l'utilisateur approuve les actions dangereuses (ex: `rm`, execution de scripts)
- **Sous-agents** : pour les taches longues, Claude Code peut spawner des sous-agents paralleles

### 5.3 Cursor — shadow workspace

Cursor (Anysphere) maintient un **shadow workspace** : une copie du code sur laquelle l'agent travaille en parallele de l'editeur humain. Les suggestions n'apparaissent qu'apres validation, ce qui evite de perturber le flux de travail.

---

## 6. Relation avec J2 (tool use)

Les outils de l'ACI sont exactement les tools de J2 : des fonctions avec schema JSON que le LLM appelle de maniere structuree. La difference est que ces tools sont optimises pour la **navigation et edition de code** plutot que pour des APIs tierces.

---

## 7. Flash Cards — Test de comprehension

**Q1 : Qu'est-ce qu'une Agent-Computer Interface (ACI) et pourquoi les interfaces humaines (shell brut, GUI) sont-elles mauvaises pour les LLM ?**
> R : L'ACI est une couche d'outils conçue specifiquement pour les besoins cognitifs des LLM (contexte long, langage, precision limitee). Les interfaces humaines sont mauvaises parce que : un `cat` d'un fichier de 3 000 lignes noie le contexte, la regeneration complete d'un fichier genere des hallucinations, et les erreurs de shell brut sont cryptiques. L'ACI offre un file viewer par blocs, des edits chirurgicaux et des commandes de test avec sortie condensee.

**Q2 : Decris les 3 etapes de la boucle edit/search/run d'un coding agent face a un bug.**
> R : (1) **Search** : localiser le bug via grep du message d'erreur, parsing de la stack trace ou repo-map ; (2) **Edit** : modifier le code de maniere chirurgicale avec des edit blocks (ORIGINAL/UPDATED) plutot que de regenerer tout le fichier ; (3) **Run** : lancer les tests (`pytest -x --tb=short`) et interpreter le resultat — PASSED → fin, FAILED → nouvelle iteration, ERROR → revert de l'edit.

**Q3 : Qu'est-ce que SWE-bench et pourquoi est-il plus difficile que HumanEval ?**
> R : SWE-bench est un benchmark de 2 294 issues GitHub reelles sur de vrais depots Python (Django, Flask, etc.). L'agent doit retrouver le bug dans le code existant, editer le fix, et passer des tests de regression. C'est plus difficile que HumanEval parce que : le depot peut faire 100 000 lignes, le bug peut impliquer plusieurs fichiers, et les tests de validation sont les vrais tests du projet (pas des tests crees pour le benchmark).

**Q4 : Qu'est-ce qu'un repo-map (Aider) et quel probleme resout-il ?**
> R : Le repo-map est un arbre miniature du depot listant toutes les classes, fonctions et leurs fichiers, genere par `ctags` ou `tree-sitter`. Il resout le probleme de la localisation : sans carte, le LLM ne sait pas quels fichiers lire dans un depot de plusieurs centaines de fichiers. Le repo-map tient dans le contexte et permet au LLM de decider quels fichiers charger en entier, rendant la navigation incrementale plutot que exhaustive.

**Q5 : Pourquoi les edit blocks (format SEARCH/REPLACE) sont-ils preferes a la regeneration complete d'un fichier ?**
> R : Les edit blocks sont preferes pour 3 raisons : (1) **Moins de tokens** — on ne retransmet pas les parties non modifiees du fichier ; (2) **Moins d'hallucinations** — le LLM ne doit pas se souvenir de toutes les lignes non modifiees, seulement de la portion a changer ; (3) **Verificabilite** — si le pattern ORIGINAL n'est pas trouve dans le fichier, l'edit est rejete, ce qui evite les modifications silencieuses au mauvais endroit.

---

## Points cles a retenir

- Un **coding agent = LLM + ACI** : la qualite de l'interface conditionne la performance, pas seulement la taille du modele
- L'**ACI** fournit 4 categories d'outils : viewer (fichier par blocs), search (grep/repo-map), edit (chirurgical), run (tests)
- La **boucle edit/search/run** est le pattern de base : search → edit block → run tests → iterer
- Les **edit blocks** (SEARCH/REPLACE) evitent la regeneration complete et reduisent les hallucinations
- **SWE-bench** : 2 294 issues GitHub reelles, mesure si le patch passe les tests de regression du vrai projet
- **SWE-bench Verified** : sous-ensemble de 500 instances annotees, la reference la plus citee
- La **localisation** (trouver le bon fichier/la bonne ligne) est le veritable goulot d'etranglement
- **Aider** : repo-map + edit blocks + auto-commit — une ACI open-source tres influente
- **Claude Code** : CLAUDE.md pour le contexte projet + permissions incrementales + sous-agents paralleles
- Les outils ACI sont des **tools J2** optimises pour la navigation/edition de code, pas pour des APIs

---

## Pour aller plus loin

- **Yang, Jimenez, Wettig et al. — "SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering" (NeurIPS 2024)** : https://arxiv.org/abs/2405.15793 — Le papier fondateur sur l'ACI. Lire en particulier la Section 3 (interface design) et la Table 1 (ablations).
- **SWE-agent repo** : https://github.com/SWE-agent/SWE-agent — Code source complet, outils ACI, scripts d'evaluation.
- **Jimenez et al. — "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" (ICLR 2024)** : https://arxiv.org/abs/2310.06770 — Le papier du benchmark. Voir aussi le leaderboard : https://www.swebench.com/verified.html
- **Aider** : https://aider.chat — Documentation de l'ACI, repo-map, edit blocks. Code source : https://github.com/Aider-AI/aider
