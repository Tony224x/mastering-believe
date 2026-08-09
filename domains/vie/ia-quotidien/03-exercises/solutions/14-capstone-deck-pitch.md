# Solution — Capstone final deck pitch

> Module `14-capstone-deck-pitch` · Badge : **Deck présentable** · Lisible **sans coder**.

Les fichiers `.py` du même nom sont des **clés techniques optionnelles** (smoke tests).

## Easy (chemin principal)

Voici un exemple qui marche pour valider le parcours **sans** viser le portfolio parfait du premier coup.

### Contrat minimum vs bonus

| Élément | Minimum (validé) | Bonus |
|---------|------------------|-------|
| Slides | **8** | 10–12 |
| Puces / slide | ≤ **3** | idem, plus soigné |
| Notes orateur | **3** slides | 4 |
| Journal IA | **5 puces** | ½–1 page |
| Oral | ~6 min | + questions pièges |
| Excel | optionnel | annexe `Budget-PME-Demo.xlsx` |

### Polish easy (1–2 soirs, pas un rush)

1. 5 slides les plus chargées → **3 puces max** chacune (cible −20 % de texte)
2. Titres uniformes (style conclusion)
3. Avant / après documenté × **2** slides

**Rythme :** 1–2 soirs, pas une seule nuit blanche.


## Medium (bonus)

### Vérification faits + journal

| Contrôle | Fait ? |
|----------|--------|
| Chaque chiffre a une source **ou** est retiré | ☐ |
| Notes orateur sur **4** slides | ☐ |
| Journal : généré / réécrit / refusé / vérifié | ☐ |
| Auto-éval /20 (grille simple) | ☐ |

Rubrique max : **20** points.


## Hard (bonus)

### Portfolio présentable

Livrables :

- `Pitch-PME-Final.pptx`
- notes orateur
- `journal-ia.md`
- `Budget-PME-Demo.xlsx` (optionnel)

Oral **6–8 min** · score cible ≥ **14/20** · outil principal : **ChatGPT**.

Règles outline : 8–12 slides · ≤ 3 puces.

Validateur optionnel : `02-code/14-capstone-deck-pitch.py`.


## Clés structurées (rappel)

### easy

- **cut_target_pct** : 20
- **slides_to_trim** : 5
- **before_after_min** : 2

### medium

- **audit** : chaque chiffre a une source ou est retire
- **notes_slides** : 4
- **rubric_max** : 20

### hard

- **deliverables** :
  - Pitch-PME-Final.pptx
  - notes orateur
  - journal-ia.md
  - Budget-PME-Demo.xlsx (optionnel)
- **oral_min** : 6
- **oral_max** : 8
- **score_min** : 14
- **primary_tool** : ChatGPT
- **validator** : 02-code/14-capstone-deck-pitch.py
- **outline_rules** :
  - **min_slides** : 8
  - **max_slides** : 12
  - **max_bullets** : 3
