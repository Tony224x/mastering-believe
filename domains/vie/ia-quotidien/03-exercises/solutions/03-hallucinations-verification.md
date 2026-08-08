# Solution — Hallucinations & vérification

> Module `03-hallucinations-verification` · Badge : **Vérificateur** · Lisible **sans coder**.

Les fichiers `.py` du même nom sont des **clés techniques optionnelles** (smoke tests).

## Easy (chemin principal)

### Checklist V-A-I-R
| Lettre | Question |
|--------|----------|
| **V**érifiable | Puis-je ouvrir la source en 2 clics ? |
| **A**ncrée | Le chiffre est-il lié à *mon* contexte ? |
| **I**nvention | Y a-t-il un % trop rond / étude inventée ? |
| **R**isque | Données sensibles ou usage scolaire/pro non vérifié ? |

### Décision type
Réponse plausible + source non ouvrable → **ne pas utiliser** tant que non vérifié.


## Medium (bonus)

### Nettoyage d'un paragraphe contaminé
1. Liste chaque fait claimable
2. Marque [A_VERIFIER] ou supprime
3. Réécris le paragraphe **sans** stats inventées
4. Ajoute une note « sources à trouver : … »


## Hard (bonus)

### Checklist 1 page opérationnelle
Sections : interdits de collage · V-A-I-R · règles école/formation · règles travail · 2 exemples.
Fais critiquer par ChatGPT en « avocat du diable », intègre 2 critiques.


## Clés structurées (rappel)

### easy

- **likely_outcome** : Numero d'article invente ou loi confondue — decision typique: jeter ou reformuler sans citation.
- **vair_example** :
  - **V** : non (source non ouverte)
  - **A** : non (pas fournie par l'utilisateur)
  - **I** : oui (precision excessive)
  - **R** : eleve si utilise en devoir/travail

### medium

- **cleaning_moves** :
  - retirer % non sources
  - remplacer par 'souvent'/'parfois'
  - marquer [A_VERIFIER]
  - garder le raisonnement qualitatif
- **before_after_required** : True

### hard

- **sections** :
  - Interdits collage
  - V-A-I-R
  - formation
  - Travail employeur
  - Exemples
- **case_actions** :
  - **stats_pitch** : exiger source ouvrable ou retirer le chiffre
  - **formule** : tester sur 3 lignes connues dans Excel
  - **donnee_sensible** : anonymiser / jeu fictif / ne pas coller
