# Solution — Prompts qui marchent

> Module `02-prompts-qui-marchent` · Badge : **Prompt clair** · Lisible **sans coder**.

Les fichiers `.py` du même nom sont des **clés techniques optionnelles** (smoke tests).

## Easy (chemin principal)

### Structure RCCFC (modèle)
```
Rôle : …
Contexte : …
Contraintes : …
Format : …
Critères de succès : …
```

### Prompt 3 blocs minimal (exemple)
```
Rôle : coach Excel pour débutant non-tech.
Contexte : tableau fictif Date | Libellé | Montant | Type (entrée/sortie).
Tâche : propose une formule SOMME pour les entrées et explique-la en 3 phrases.
Format : formule + explication + 1 cas de test.
Contraintes : français Excel ; pas de VBA ; données fictives seulement.
```


## Medium (bonus)

### 3 prompts modèles à garder
1. **Réflexion** — rôle tuteur socratique, 5 questions max, pas de réponse toute faite
2. **Excel** — colonnes nommées + formule FR + test
3. **Slides** — outline 8 titres, une idée par slide, ton pitch PME fictif

Teste chacun **une fois** et note ce que tu as dû corriger à la main.


## Hard (bonus)

### Bibliothèque 6 prompts
Classe : 2 réflexion · 2 Excel · 2 slides. Chaque entrée a : titre, usage, texte RCCFC, exemple de sortie attendue, piège.


## Clés structurées (rappel)

### easy

- **fuzzy** : Ameliore mon Excel.
- **structured_example** : Role: formateur Excel Microsoft 365 pour non-developpeurs.
Contexte: tableau A1:E20 colonnes Date|Libelle|Categorie|Montant|Type(entree/sortie), ligne 1 en-tetes.
Tache: propose 3 ameliorations concretes (formules ou structure) pour obtenir solde mensuel.
Format: liste numerotee | action | formule FR si besoin | benefice en 1 phrase.
Contraintes: pas de VBA ; donnees fictives ; Excel FR-CA.

- **why_better** : Le modele recoit le schema + format de sortie + limites techniques.

### medium

- **three_templates_head** :
  - coach carriere socratique
  - SOMME.SI entrees/sorties
  - outline 10 slides pitch
- **iteration_examples** :
  - Raccourcis de 30 %
  - Supprime le jargon
  - Donne un cas de test numerique

### hard

- **library_min** : 6
- **few_shot_min** : 2
- **anti_patterns** :
  - Prompt d'un seul mot
  - Coller des donnees reelles
  - Demander un devoir entier sans reecriture
  - Accepter stats sans source ouvrable
