# Exercices — Architecture des coding agents (J21)

> Prerequis : avoir lu `01-theory/21-coding-agents-architecture.md` et execute `02-code/21-coding-agents-architecture.py`.

---

## Exercice 1 : Implémenter un outil `list_dir` pour l'ACI

### Objectif
Comprendre que la **navigation du depot** est un outil ACI a part entiere : sans lui, l'agent ne sait pas quels fichiers existent et commence a halluciner des chemins.

### Consigne
En partant de `02-code/21-coding-agents-architecture.py` :

1. Ajoute une methode `list_dir(self, rel_path: str = ".", max_depth: int = 2) -> str` a la classe `ACI`.
   - Elle doit parcourir recursivement l'arborescence jusqu'a `max_depth` niveaux.
   - Le retour doit etre un arbre indenté lisible, par exemple :
     ```
     [list_dir] .
       mathlib/
         __init__.py
         operations.py
       tests/
         __init__.py
         test_operations.py
     ```
   - Si `rel_path` n'existe pas, retourner `[ERROR] Path not found: <rel_path>`.
   - Les fichiers et dossiers sont tries par nom (alphabetique).
2. Instancie un `ACI` sur le repo jouet (`create_toy_repo` dans un tempdir) et appelle `list_dir()` sur la racine.
3. Appelle aussi `list_dir("mathlib")` et `list_dir("missing_folder")`.
4. Affiche les 3 resultats.

### Criteres de reussite
- [ ] `list_dir(".")` affiche une arborescence indentee a 2 niveaux max
- [ ] `list_dir("mathlib")` affiche uniquement le contenu de `mathlib/`
- [ ] `list_dir("missing_folder")` retourne une chaine commencant par `[ERROR]`
- [ ] L'arborescence est triee alphabetiquement
- [ ] Le code est stdlib pur (pas de dependance externe)

---

## Exercice 2 : Ajouter un mecanisme de backtracking a l'agent

### Objectif
Comprendre pourquoi un coding agent doit pouvoir **annuler un edit** quand les tests empirent apres une modification.

### Consigne
1. Ajoute a la classe `ACI` une methode `snapshot(self, rel_path: str) -> str` qui sauvegarde le contenu actuel du fichier et retourne une cle opaque (par exemple l'index de la sauvegarde).
2. Ajoute une methode `restore(self, key: str) -> str` qui restaure le fichier a l'etat sauvegarde pour la cle donnee.
3. Cree un scenario de test dans `if __name__ == "__main__":` :
   a. Cree le repo jouet dans un tempdir.
   b. Prends un snapshot de `mathlib/operations.py`.
   c. Applique un edit **incorrect** (remplace `return a - b` par `return a ** b` — toujours faux).
   d. Lance les tests : confirme qu'ils echouent encore.
   e. Restaure le snapshot.
   f. Verifie que le fichier est revenu a son etat original (compare le contenu).
   g. Affiche "Backtrack successful" ou "Backtrack failed".

### Criteres de reussite
- [ ] `snapshot` retourne une cle non-vide
- [ ] `restore` reecrit exactement le contenu original
- [ ] Le test confirme que l'edit incorrect est annule
- [ ] La comparaison de contenu avant/apres restore est explicite
- [ ] Le scenario complet s'execute sans erreur avec `python <fichier>.py`

---

## Exercice 3 : Mini repo-map par analyse syntaxique

### Objectif
Reproduire le principe du **repo-map** d'Aider : generer une carte compacte du depot qui liste les noms de fonctions et classes, permettant a l'agent de naviguer un grand depot sans lire chaque fichier en entier.

### Consigne
1. Cree une fonction `build_repo_map(root: Path) -> str` (stdlib pur, utilise le module `ast`).
   - Parcourt tous les fichiers `.py` du repo (recursif).
   - Pour chaque fichier, extrait avec `ast.parse` la liste des :
     - **Classes** : `class Foo:`
     - **Fonctions de module** (niveau 0 de l'AST) : `def bar():`
     - **Methodes** (niveau 1, enfants directs de ClassDef) : `  def baz(self):`
   - Format de sortie (une ligne par symbole) :
     ```
     mathlib/operations.py
       def add(a, b)
       def multiply(a, b)
     tests/test_operations.py
       def test_add()
       def test_multiply()
     ```
   - Ignore les fichiers avec des erreurs de syntaxe (`SyntaxError`) — afficher `  [SyntaxError]` et continuer.
2. Applique `build_repo_map` au repo jouet cree dans un tempdir.
3. Affiche le resultat.
4. Compte le nombre total de symboles trouves et affiche-le.

### Criteres de reussite
- [ ] `build_repo_map` utilise `ast.parse` (pas de regex)
- [ ] Les fonctions de module et les methodes sont distingues par indentation
- [ ] Les classes sont listees avec leur nom
- [ ] Les fichiers avec `SyntaxError` n'interrompent pas le parcours
- [ ] Le compte final est exact (6 symboles dans le repo jouet : 2 fonctions dans operations.py + 2 fonctions dans test_operations.py + les __init__)
