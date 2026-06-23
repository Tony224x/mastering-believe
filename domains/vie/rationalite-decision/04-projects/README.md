# Portfolio — Boîte à Outils de Jugement

> **Usage** : copiez ce fichier dans votre espace personnel (hors du repo, ou dans `03-exercises/workspace/`), remplissez chaque section, et faites-le évoluer au fil du temps. C'est un document vivant, pas un examen.
>
> **Objectif** : disposer d'un système de jugement personnel opérationnel, traçable et améliorable après les 14 modules du domaine `rationalite-decision`.
>
> **Principe du *latticework*** (Munger, *Poor Charlie's Almanack*, 2005/2023) : un réseau de modèles mentaux issus de plusieurs disciplines, qu'on active selon le contexte. Ce gabarit vous aide à construire le vôtre — non pas en accumulant des outils, mais en sachant **quand activer lequel**.

---

## Section 1 — Ma Checklist de Pré-Décision

> Personnalisez cette checklist pour vos décisions récurrentes. Gardez-la à portée de main (notes téléphone, carnet, post-it).

```
PRÉ-DÉCISION
Date : __________ | Décision : __________

□ 1. CLARIFIER
     Décision exacte : ____________________
     Options réelles (y compris « ne rien faire ») : ____________________
     Horizon de temps : ____________________

□ 2. BIAIS
     Ancrage : y a-t-il un premier chiffre ou une première option qui me biaise ?
     → Estimation indépendante : ____________________
     Disponibilité : quel événement récent ou saillant domine ma pensée ?
     → Taux de base historique : ____________________
     Cadrage : ma préférence change-t-elle si je reformule en gains puis en pertes ?
     → Reformulation inverse : ____________________
     Confirmation : argument le plus fort CONTRE mon option favorite :
     → ____________________

□ 3. PROBABILITÉS
     Scénario principal : _____%
     Classe de référence utilisée : ____________________
     Ajustements depuis la base : ____________________

□ 4. CONSÉQUENCES
     Optimiste (___%) : ____________________
     Central   (___%) : ____________________
     Pessimiste(___%) : ____________________
     Scénario ruineux à éviter à tout prix : ____________________

□ 5. VÉRIFICATION
     Fait clé 1 vérifié via : ____________________  (→ utiliser SIFT — Section 3)
     Fait clé 2 vérifié via : ____________________

□ 6. DÉCISION & SUIVI
     Décision retenue : ____________________
     Probabilité de succès : _____%
     Date de revue : __________
```

---

## Section 2 — Mon Mini-Arbre de Décision

> À utiliser dès qu'une décision comporte plusieurs options et au moins un événement incertain. Dessiner ou écrire en texte.

```
Décision : __________  |  Date : __________

Option A — ____________________
  ├─ [Événement favorable, p = ___] → Résultat A+ (valeur : ___)
  └─ [Événement défavorable, p = ___] → Résultat A- (valeur : ___)
  Espérance(A) = ___ × ___ + ___ × ___ = ___

Option B — ____________________
  ├─ [Événement favorable, p = ___] → Résultat B+ (valeur : ___)
  └─ [Événement défavorable, p = ___] → Résultat B- (valeur : ___)
  Espérance(B) = ___ × ___ + ___ × ___ = ___

Option C (optionnelle) — ____________________
  ├─ [p = ___] → Résultat C+ (valeur : ___)
  └─ [p = ___] → Résultat C- (valeur : ___)
  Espérance(C) = ___

Option retenue d'après l'arbre : ____________________
Nuance (aversion au risque, scénario ruineux, coûts implicites) :
  ____________________
```

---

## Section 3 — Mon Protocole de Vérification SIFT

> Fiche de référence rapide. À appliquer sur chaque information externe qui fonde une décision importante.

```
PROTOCOLE SIFT — FICHE RAPIDE

S — STOP
    Pause avant de partager ou d'intégrer l'information.
    L'urgence est souvent fabriquée.

I — INVESTIGATE THE SOURCE
    Qui publie ? Quel historique ? Quel intérêt ?
    → Lecture LATÉRALE : ouvrir 2-3 onglets sur la source,
      ne pas lire le document lui-même verticalement.

F — FIND BETTER COVERAGE
    D'autres sources indépendantes confirment-elles ?
    → Chercher : [sujet] site:reuters.com
    → Chercher : [sujet] "meta-analysis" OR "systematic review"

T — TRACE TO ORIGINAL
    Remonter à la source primaire.
    → Citation  : Google Scholar + doi.org
    → Image     : Google Images (clic droit) ou TinEye.com
    → Vérifier  : date, auteur, contexte original

LLM SPÉCIFIQUE
    → Titre entre guillemets sur Google Scholar.
    → DOI sur doi.org.
    → Auteur + revue + année correspondent ?
    → Si introuvable en 3 étapes → probablement halluciné.
```

---

## Section 4 — Mon Journal de Prévisions Calibré (score de Brier)

> Visez ≥ 1 prédiction par semaine. Copiez ce tableau dans un tableur pour calculer automatiquement le score de Brier, ou utilisez `02-code/08-calibration-forecasting.py`.

### Format

| # | Date | Question (binaire + date résolution) | p (%) | Classe de référence | Outcome | (p−o)² | Note |
|---|------|--------------------------------------|--------|---------------------|---------|--------|------|
| 1 | | | | | | | |
| 2 | | | | | | | |
| … | | | | | | | |

### Règles de bonne formulation

- **Question binaire** : « X se produira-t-il avant [date] ? » → réponse OUI (1) ou NON (0).
- **Date de résolution** : fixée à l'avance, non modifiable.
- **Probabilité** : un chiffre entre 1 % et 99 % (éviter 0 % et 100 % sauf certitude absolue).
- **Classe de référence** : noter la base utilisée (ex. : « il pleut 35 % des lundis en juin dans cette ville »).

### Score de Brier

**Formule** : Brier = (1/N) × Σ (pᵢ − oᵢ)²

- pᵢ = probabilité annoncée (0 à 1), oᵢ = outcome (0 ou 1)
- Baseline aléatoire ≈ 0,25 | Superforecasters ≈ 0,10–0,15

### Suivi mensuel

| Mois | N prédictions | Score Brier | Commentaire |
|------|--------------|-------------|-------------|
| | | | |

### Objectifs de progression

| Horizon | Score de Brier cible |
|---------|---------------------|
| Mois 1 | < 0,25 (battre le hasard) |
| Mois 3 | < 0,20 |
| Mois 6 | < 0,18 |
| Mois 12 | < 0,15 (bon forecaster amateur) |

---

## Section 5 — Mon Analyse de Second Ordre

> À remplir pour chaque décision importante. Identifie les effets qui apparaissent via des boucles de rétroaction (Module 12 — Pensée systémique).

```
ANALYSE DE SECOND ORDRE — [Décision]  |  Date : __________

Effets de 1er ordre (< 1 semaine) :
  →

Effets de 2e ordre (1 semaine – 3 mois) :
  →

Effets de 3e ordre (> 3 mois, boucles de rétroaction) :
  →

Boucle renforçante principale (ce qui s'amplifie si ça marche) :
  →

Boucle équilibrante principale (ce qui va freiner ou corriger) :
  →

Point de levier identifié (où intervenir le plus efficacement) :
  →
```

---

## Section 6 — Mon Latticework Personnel

> Listez les modèles mentaux que vous activez le plus souvent, avec une note sur le contexte d'usage. Complétez au fil des modules.

| Modèle | Module source | Quand l'activer | Exemple personnel |
|--------|--------------|-----------------|------------------|
| Classe de référence (base rate) | 03, 08 | Avant toute estimation ou prédiction | |
| Mise à jour bayésienne | 04 | Quand une nouvelle information change la donne | |
| Heuristiques & biais (ancrage, disponibilité, cadrage, confirmation) | 05 | Avant une décision importante | |
| Heuristiques rapides (recognize-and-act) | 06 | Décisions rapides sous contrainte de temps | |
| Espérance + arbre de décision | 07 | Choix sous risque, plusieurs branches | |
| Score de Brier + journal | 08 | Prédictions à scorer dans le temps | |
| Contrefactuel + RCT | 09 | Quand je veux établir une causalité | |
| Lecture critique d'une étude | 10 | Avant de citer une statistique | |
| SIFT | 11 | Toute information externe ou sortie de LLM | |
| Stocks, flux, boucles (pensée systémique) | 12 | Effets à long terme, systèmes complexes | |
| Pre-mortem + checklist anti-biais | 13 | Avant un projet, une décision irréversible | |
| ___(à compléter)___ | | | |

---

## Section 7 — Revue Trimestrielle

> À compléter tous les 3 mois pour garder le système vivant et en amélioration continue.

### Revue [Trimestre / Date]

**Score de Brier moyen ce trimestre** : _____ (objectif : < 0,20)

**Prédiction la mieux calibrée** : _____________________

**Zone de sur-confiance identifiée** (prévu X %, réalisé Y %) : _____________________

**Zone de sous-confiance identifiée** : _____________________

**Décision revue (issue de la checklist)** : _____________________
- Prédiction initiale : _____%
- Outcome réel : 0 / 1
- Score de Brier : _____
- Leçon : _____________________

**Effet de second ordre non anticipé ce trimestre** : _____________________

**Un biais que j'ai surpris en moi ce trimestre** : _____________________

**Ajustement au système (checklist, arbre, SIFT, latticework)** : _____________________

---

*Ce gabarit est issu du Module 14 — Capstone du domaine `rationalite-decision` de Mastering Believe.*
*Repo public : https://github.com/alexandrebeleanu/mastering-believe*
