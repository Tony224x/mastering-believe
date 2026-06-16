# Template Portfolio — Boîte à Outils du Jugement

> **Usage** : copiez ce fichier dans votre espace personnel (hors du repo, ou dans `03-exercises/workspace/`), remplissez chaque section, et faites-le évoluer au fil du temps. C'est un document vivant, pas un examen.
>
> **Objectif** : avoir un système de jugement personnel opérationnel et traçable après avoir complété les 7 modules.

---

## Section 1 — Ma Checklist de Pré-Décision

> Personnalisez cette checklist pour vos décisions récurrentes. Imprimez-la ou gardez-la dans un outil de notes rapides.

```
PRÉ-DÉCISION
Date : __________ | Décision : __________

□ 1. CLARIFIER
     Décision exacte : ____________________
     Options réelles (pas seulement A vs statu quo) : ____________________
     Horizon de temps : ____________________

□ 2. BIAIS
     Ancrage : y a-t-il un premier chiffre qui me biaise ?
     → Estimation indépendante : ____________________
     Disponibilité : quel exemple récent domine ma pensée ?
     → Taux de base historique : ____________________
     Cadrage : ma préférence change-t-elle en gains/pertes ?
     → Reformulation inverse : ____________________
     Confirmation : argument le plus fort CONTRE mon option favorite :
     → ____________________

□ 3. PROBABILITÉS
     Scénario principal : _____%
     Classe de référence : ____________________
     Ajustements : ____________________

□ 4. CONSÉQUENCES
     Optimiste (___%) : ____________________
     Central   (___%) : ____________________
     Pessimiste(___%) : ____________________
     Scénario ruineux à éviter : ____________________

□ 5. VÉRIFICATION
     Fait clé 1 vérifié via : ____________________
     Fait clé 2 vérifié via : ____________________

□ 6. DÉCISION & SUIVI
     Décision : ____________________
     Probabilité de succès : _____%
     Date de revue : __________
```

---

## Section 2 — Mon Journal de Prévisions

> Visez au moins 1 prédiction par semaine. Copiez ce tableau dans un tableur pour calculer automatiquement le score de Brier, ou utilisez le script `02-code/06-calibration-verification.py`.

### Format

| Date | Question (binaire + date résolution) | p (%) | Outcome | (p−o)² | Note |
|------|--------------------------------------|--------|---------|--------|------|
| | | | | | |
| | | | | | |

### Règles de bonne formulation

- **Question binaire** : "X se produira-t-il avant [date] ?" → réponse OUI (1) ou NON (0).
- **Date de résolution** : fixée à l'avance, non modifiable.
- **Probabilité** : un chiffre entre 1 % et 99 % (éviter 0 % et 100 % sauf certitude absolue).
- **Classe de référence** : noter la base utilisée (ex. : "il pleut 40 % des matins en juin ici").

### Suivi mensuel

| Mois | N prédictions | Score Brier | Commentaire |
|------|--------------|-------------|-------------|
| | | | |

**Objectifs de progression** :
- Mois 1 : < 0,25 (battre le hasard)
- Mois 3 : < 0,20
- Mois 6 : < 0,18
- Mois 12 : < 0,15 (niveau bon forecaster amateur)

---

## Section 3 — Mon Protocole de Vérification

> Fiche de référence rapide à garder à portée de main (bureau, notes téléphone).

```
PROTOCOLE DE VÉRIFICATION — FICHE RAPIDE

S — STOP
    Avant de partager ou de décider : pause obligatoire.
    L'urgence est souvent fabriquée.

I — INVESTIGATE THE SOURCE
    Qui publie ? Quel historique ? Quel intérêt ?
    → Ouvrir 2-3 onglets sur la source (lecture LATÉRALE).
    → Ne pas lire verticalement le document lui-même.

F — FIND BETTER COVERAGE
    D'autres sources indépendantes confirment-elles ?
    → Chercher [sujet] + site:reuters.com / scholar.google.com
    → Chercher [sujet] + "meta-analysis" ou "systematic review"

T — TRACE TO ORIGINAL
    Remonter à la source primaire.
    → Citation : Google Scholar + doi.org
    → Image : Google Images (clic droit) ou TinEye.com
    → Vérifier : date, auteur, contexte original.

LLM SPÉCIFIQUE
    → Titre entre guillemets sur Google Scholar.
    → DOI sur doi.org.
    → Auteur + revue + année correspondent ?
    → Si introuvable en 3 étapes → probablement halluciné.
```

---

## Section 4 — Mon Latticework Personnel

> Listez les modèles mentaux que vous activez le plus souvent, avec une note sur le contexte d'usage.

| Modèle | Module source | Quand l'activer | Exemple personnel |
|--------|--------------|-----------------|------------------|
| Taux de base (base rate) | 02, 04 | Avant toute estimation | |
| Mise à jour bayésienne | 03 | Quand une info change la donne | |
| Espérance + utilité | 05 | Décision sous risque financier | |
| Arbre de décision | 05 | Problème à plusieurs étapes | |
| Score de Brier | 06 | Après résolution d'une prédiction | |
| SIFT | 06 | Information douteuse ou LLM | |
| Classe de référence | 02, 06, 07 | Avant de donner une probabilité | |
| ___(à compléter)___ | | | |

---

## Section 5 — Revue Trimestrielle

> À compléter tous les 3 mois pour garder le système vivant.

### Revue [Trimestre / Date]

**Score de Brier moyen ce trimestre** : _____ (objectif : < 0,20)

**Prédiction la mieux calibrée** : _____________________

**Zone de sur-confiance identifiée** (prédit X %, réalité Y %) : _____________________

**Zone de sous-confiance identifiée** : _____________________

**Décision revue (issue de la checklist)** : _____________________
- Prédiction initiale : _____%
- Outcome réel : 0 / 1
- Brier : _____
- Leçon : _____________________

**Un biais que j'ai surpris en moi ce trimestre** : _____________________

**Ajustement au système** : _____________________

---

*Ce template est issu du Module 07 — Capstone du domaine `rationalite-decision` de Mastering Believe.*
*Repo public : https://github.com/[votre-fork]*
