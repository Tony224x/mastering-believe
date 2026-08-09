# Solution — Hallucinations & vérification

> Module `03-hallucinations-verification` · Badge : **Vérificateur** · Lisible **sans coder**.

Les fichiers `.py` du même nom sont des **clés techniques optionnelles** (smoke tests).

## Easy (chemin principal)

Voici un exemple qui marche quand une réponse « sonne juste » mais ne tient pas la route.

### Checklist V-A-I-R

| Lettre | Question |
|--------|----------|
| **V**érifiable | Puis-je ouvrir la source en 2 clics ? |
| **A**ncrée | Le chiffre est-il lié à *mon* contexte ? |
| **I**nvention | Y a-t-il un % trop rond / une étude inventée ? |
| **R**isque | Données sensibles ou usage scolaire/pro non vérifié ? |

### Décision type

Réponse plausible + source non ouvrable → **ne pas utiliser** tant que non vérifié.  
Reformule sans citation, ou marque `[A_VERIFIER]`.

### Mini-cas (fictif)

> « L’article 12 de la Loi-IA 2024 impose 48 h de délai… »

| V | A | I | R |
|---|---|---|---|
| non (source non ouverte) | non (pas fournie par toi) | oui (précision excessive) | élevé si collé dans un devoir |


## Medium (bonus)

### Nettoyage d’un paragraphe contaminé (ordre)

1. Liste chaque fait claimable (chiffre, nom d’étude, date)
2. Marque `[A_VERIFIER]` ou **supprime**
3. Réécris le paragraphe **sans** stats inventées
4. Ajoute une note : « sources à trouver : … »

Astuce : remplacer un % inventé par « souvent » / « parfois » tant que tu n’as pas la preuve.


## Hard (bonus)

### Checklist 1 page opérationnelle

Sections à garder sur une feuille :

1. Interdits de collage  
2. V-A-I-R  
3. Règles école / formation  
4. Règles travail  
5. 2 exemples (stats pitch · formule Excel · donnée sensible)

Fais critiquer par ChatGPT en « avocat du diable », puis intègre **2** critiques réelles.

| Cas | Action |
|-----|--------|
| Stats dans un pitch | exiger source ouvrable **ou** retirer le chiffre |
| Formule proposée | tester sur 3 lignes connues dans Excel |
| Donnée sensible | anonymiser / jeu fictif / ne pas coller |


## Clés structurées (rappel)

### easy

- **likely_outcome** : Numero d’article invente ou loi confondue — decision typique: jeter ou reformuler sans citation.
- **vair_example** :
  - **V** : non (source non ouverte)
  - **A** : non (pas fournie par l’utilisateur)
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
