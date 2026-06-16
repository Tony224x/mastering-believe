# Module 14 — Capstone : La Boîte à Outils de Jugement

> **Temps estimé** : 60 min | **Prérequis** : Modules 01–13
>
> **Objectif** : Assembler un *latticework* personnel de jugement en cinq outils intégrés — checklist de pré-décision, mini-arbre de décision, analyse de second ordre, journal de prévisions calibré (score de Brier), protocole SIFT — et les appliquer sur une seule décision neutre et concrète pour constituer un livrable de portfolio.

---

## 1. Le latticework comme cadre organisateur

Charlie Munger (*Poor Charlie's Almanack*, 2005/2023) a popularisé l'idée d'un *latticework of mental models* : un réseau de grilles de lecture issues de plusieurs disciplines, qu'on **active selon le contexte**, pas qu'on récite en liste.

Ce capstone vous demande de construire **votre** version opérationnelle de ce réseau. Non pas en accumulant des modèles, mais en sachant **quand activer lequel** :

| Situation | Outil à activer |
|-----------|----------------|
| Incertitude sur un événement futur | Module 08 — Journal Brier + calibration |
| Plusieurs options avec branches | Module 07 — Arbre de décision |
| Effets dans le temps, chaînes causales | Module 12 — Analyse de second ordre |
| Information externe douteuse | Module 11 — Protocole SIFT |
| Toute décision importante | Checklist de pré-décision |
| Décision d'équipe / risque de groupthink | Module 13 (pre-mortem + checklist anti-biais) |
| Résultat d'étude ou statistique invoqué | Module 10 (grille de lecture) |

---

## 2. Pièce 1 — Checklist de pré-décision

La checklist sert à **ralentir** avant d'agir, pour vérifier qu'aucun biais évident ne pollue le raisonnement.

```
PRÉ-DÉCISION — [Date] — [Décision envisagée]

1. CLARIFIER
   □ Quelle est exactement la décision à prendre ?
   □ Quelles sont toutes les options réelles (y compris « ne rien faire ») ?
   □ Quel est l'horizon de temps ?

2. BIAIS À VÉRIFIER
   □ Ancrage : suis-je accroché à un premier chiffre ou une première option ?
     → Estimer indépendamment, puis comparer.
   □ Disponibilité : un événement récent ou saillant domine-t-il mon attention ?
     → Chercher le taux de base historique (classe de référence).
   □ Cadrage : ma préférence changerait-elle si je reformulais en gains puis en pertes ?
   □ Confirmation : quel est l'argument le plus solide CONTRE mon option favorite ?

3. PROBABILITÉS
   □ Probabilité estimée du scénario principal : _____%
   □ Classe de référence utilisée : ____________________
   □ Ajustements depuis la base : ____________________

4. CONSÉQUENCES (3 scénarios)
   □ Optimiste (___%) : ____________________
   □ Central   (___%) : ____________________
   □ Pessimiste(___%) : ____________________
   □ Scénario ruineux à éviter à tout prix : ____________________

5. VÉRIFICATION DES INFORMATIONS CLÉS
   □ Fait clé 1 vérifié via : ____________________
   □ Fait clé 2 vérifié via : ____________________
   (→ utiliser le protocole SIFT — Pièce 5)

6. DÉCISION ET SUIVI
   □ Décision retenue : ____________________
   □ Probabilité subjective de succès : _____%
   □ Date de revue : ____________________
```

---

## 3. Pièce 2 — Mini-arbre de décision (réinvestit Module 07)

L'arbre de décision externalise les branches d'un choix pour calculer l'**espérance** de chaque option. Il est particulièrement utile quand une décision a plusieurs étapes ou dépend d'un événement incertain.

### Structure minimale

```
Option A ──┬── [Événement p1] ── Résultat A+ (valeur V1)
           └── [Événement 1-p1] ── Résultat A- (valeur V2)
             Espérance(A) = p1 × V1 + (1-p1) × V2

Option B ──┬── [Événement p2] ── Résultat B+ (valeur V3)
           └── [Événement 1-p2] ── Résultat B- (valeur V4)
             Espérance(B) = p2 × V3 + (1-p2) × V4
```

### Exemple neutre — choisir entre deux itinéraires de livraison

- **Option A** (autoroute) : p = 0,80 d'arriver à temps (+10 pts satisfaction), 0,20 de retard (−5 pts).
  - Espérance(A) = 0,80 × 10 + 0,20 × (−5) = **7,0**
- **Option B** (nationale) : p = 0,60 d'arriver à temps (+10 pts), 0,40 de retard (−2 pts).
  - Espérance(B) = 0,60 × 10 + 0,40 × (−2) = **5,2**

Décision rationnelle : Option A, sauf si l'aversion au risque de retard lourd justifie de préférer B.

> **À retenir** : l'arbre de décision ne remplace pas le jugement — il rend les probabilités et les valeurs *explicites*, ce qui facilite la discussion et la révision.

---

## 4. Pièce 3 — Analyse de second ordre (réinvestit Module 12)

La pensée systémique (Meadows, *Thinking in Systems*, 2008) distingue les **effets immédiats** (premier ordre) des **effets en retour** qui se manifestent plus tard via des boucles de rétroaction.

### Questions à se poser systématiquement

1. **Stocks et flux** : qu'est-ce qui s'accumule ou se vide si je prends cette décision ?
2. **Boucles renforçantes** : quel effet s'amplifie avec le temps si la décision est bonne ? Si elle est mauvaise ?
3. **Boucles équilibrantes** : quelles forces naturelles vont s'opposer à la décision ou la corriger ?
4. **Délais** : à quel horizon les effets de second ordre apparaissent-ils ?
5. **Points de levier** : sur quel paramètre du système puis-je agir pour amplifier un bon effet ou amortir un mauvais ?

### Template à remplir

```
ANALYSE DE SECOND ORDRE — [Décision]

Effets de 1er ordre (< 1 semaine) :
  →

Effets de 2e ordre (1 semaine – 3 mois) :
  →

Effets de 3e ordre (> 3 mois, boucles de rétroaction) :
  →

Boucle renforçante principale identifiée :
  →

Boucle équilibrante principale (ce qui va freiner l'effet) :
  →

Point de levier (où puis-je intervenir le plus efficacement ?) :
  →
```

---

## 5. Pièce 4 — Journal de prévisions calibré + score de Brier (réinvestit Module 08)

Le **score de Brier** mesure la calibration : plus il est bas, meilleure est la prévision.

**Formule** : Brier = (1/N) × Σ (pᵢ − oᵢ)²

- pᵢ = probabilité annoncée (entre 0 et 1)
- oᵢ = outcome réel (0 ou 1)
- Baseline aléatoire ≈ 0,25 ; superforecasters ≈ 0,10–0,15

### Format du journal

| Date | Question binaire + date de résolution | p (%) | Classe de référence | Outcome | (p−o)² | Note |
|------|---------------------------------------|--------|---------------------|---------|--------|------|
| | | | | | | |

### Règles

1. **Question binaire et datée** : réponse observable à une date fixée à l'avance.
2. **Classe de référence** : noter la base utilisée (ex. « il pleut 35 % des lundis en juin dans cette ville »).
3. **Pas de modification** après l'événement : la probabilité et la question sont gelées à l'inscription.
4. **Minimum ≥ 10 prédictions** pour calculer un score significatif.
5. **Revue mensuelle** : identifier les zones de sur-confiance (prévu 80 %, réalisé 50 %) et sous-confiance.

### Objectifs de progression

| Horizon | Score de Brier cible |
|---------|---------------------|
| Mois 1 | < 0,25 (battre le hasard) |
| Mois 3 | < 0,20 |
| Mois 6 | < 0,18 |
| Mois 12 | < 0,15 (bon forecaster amateur) |

---

## 6. Pièce 5 — Protocole de vérification SIFT (réinvestit Module 11)

Le protocole SIFT (Caulfield, *Web Literacy for Student Fact-Checkers*, 2017) s'applique chaque fois qu'une information externe entre dans le raisonnement.

```
S — STOP
    Pause avant de partager ou d'intégrer l'information.
    L'urgence est souvent fabriquée.

I — INVESTIGATE THE SOURCE
    Qui publie ? Quel historique ? Quel intérêt ?
    → Lecture LATÉRALE : ouvrir 2-3 onglets sur la source,
      pas lire le document lui-même verticalement.

F — FIND BETTER COVERAGE
    D'autres sources indépendantes confirment-elles ?
    → Chercher : [sujet] site:reuters.com
    → Chercher : [sujet] "meta-analysis" OR "systematic review"

T — TRACE TO ORIGINAL
    Remonter à la source primaire.
    → Citation : Google Scholar + doi.org
    → Image   : Google Images (clic droit) ou TinEye.com
    → Vérifier : date, auteur, contexte original

LLM SPÉCIFIQUE
    → Titre entre guillemets sur Google Scholar.
    → DOI sur doi.org.
    → Auteur + revue + année correspondent ?
    → Si introuvable en 3 étapes → probablement halluciné.
```

---

## 7. Intégration : la boîte en action sur une décision

Les cinq pièces ne fonctionnent pas en silo — elles se déclenchent en séquence logique :

```
DÉCISION IDENTIFIÉE
        │
        ▼
[Pièce 1] Checklist de pré-décision
  ─ clarifier, débiaiser, estimer la probabilité
        │
        ▼
[Pièce 2] Mini-arbre de décision
  ─ si plusieurs branches, calculer l'espérance
        │
        ▼
[Pièce 3] Analyse de second ordre
  ─ quels effets à 1, 2, 3 ordres ? boucles ?
        │
        ▼
[Pièce 5] Protocole SIFT (sur les infos clés)
  ─ vérifier les faits qui fondent la décision
        │
        ▼
DÉCISION PRISE → enregistrer une prédiction
        │
        ▼
[Pièce 4] Journal Brier
  ─ suivre, scorer à la résolution, recalibrer
```

---

## 8. Grille d'auto-évaluation du capstone

| Critère | Non commencé | Partiel | Complet |
|---------|:---:|:---:|:---:|
| Checklist remplie sur une décision réelle et neutre | ☐ | ☐ | ☐ |
| Mini-arbre de décision avec espérances calculées | ☐ | ☐ | ☐ |
| Analyse de second ordre (3 ordres + boucle) | ☐ | ☐ | ☐ |
| Journal ≥ 10 prédictions avec score de Brier calculé | ☐ | ☐ | ☐ |
| Protocole SIFT appliqué sur ≥ 1 information clé | ☐ | ☐ | ☐ |
| Latticework personnel mis à jour (Section 4 du portfolio) | ☐ | ☐ | ☐ |
| Score de Brier < 0,25 (battre la baseline) | ☐ | ☐ | ☐ |

**Niveau "complet" = 6/7 critères cochés "Complet".**

---

> **À retenir** :
> - Le *latticework* est un réseau d'outils activés selon le contexte, pas une liste à réciter.
> - Les cinq pièces se complètent : la checklist débiaiser, l'arbre quantifie, le second ordre anticipe, SIFT vérifie, le journal calibre.
> - La calibration s'améliore par la pratique et le feedback — 30 prédictions scorées valent plus que 10 heures de lecture sur les biais.
> - Score de Brier < 0,20 à 3 mois = objectif atteignable dès la première série de 10 prédictions.

---

## Flash-cards (Module 14)

**Q1 : Qu'est-ce qu'un *latticework of mental models* et pourquoi l'utiliser en capstone ?**
> R : Un réseau de modèles mentaux issus de plusieurs disciplines (Munger), qu'on active selon le contexte. En capstone, il sert de cadre pour choisir lequel des cinq outils déclencher en premier face à une décision.

**Q2 : À quoi sert le mini-arbre de décision, et quand l'activer ?**
> R : À rendre explicites les probabilités et les valeurs de chaque branche pour calculer l'espérance. À activer dès qu'une décision comporte plusieurs options et au moins un événement incertain intermédiaire.

**Q3 : Quelle est la différence entre un effet de premier ordre et un effet de second ordre ?**
> R : Un effet de premier ordre est immédiat et direct (< 1 semaine). Un effet de second ordre apparaît via une boucle de rétroaction (semaines à mois) — souvent plus important que l'effet direct.

**Q4 : Quel est le score de Brier d'une prévision aléatoire, et que visent les superforecasters ?**
> R : Baseline aléatoire ≈ 0,25. Superforecasters atteignent 0,10–0,15. Objectif à 3 mois pour un débutant : < 0,20.

**Q5 : Quelle est la règle numéro un du protocole SIFT face à un contenu LLM ?**
> R : Chercher le titre entre guillemets sur Google Scholar et vérifier le DOI sur doi.org. Si introuvable en 3 étapes, la source est probablement hallucinée.

---

## Points clés à retenir

1. La checklist de pré-décision ralentit le Système 1 et force l'examen des biais avant d'agir.
2. L'arbre de décision rend visibles les probabilités et les valeurs — il facilite la révision collective.
3. L'analyse de second ordre révèle les boucles de rétroaction que l'intuition ignore.
4. Le journal de prévisions + score de Brier est la seule façon de mesurer objectivement sa calibration.
5. Le protocole SIFT s'applique à chaque information qui fonde une décision, y compris les sorties de LLM.
6. Un *latticework* n'est complet que si l'on sait **quand ne pas l'utiliser** — certaines décisions sont trop simples pour mériter un arbre.

---

## Pour aller plus loin

- **Latticework** : Munger, C. T. (2005/2023). *Poor Charlie's Almanack.* Stripe Press. https://www.stripe.press/poor-charlies-almanack/
- **Superforecasters** : Tetlock, P. E. & Gardner, D. (2015). *Superforecasting.* Crown Publishers. https://www.goodjudgment.com/
- **Pensée systémique** : Meadows, D. H. (2008). *Thinking in Systems.* Chelsea Green Publishing.
- **SIFT** : Caulfield, M. (2017). *Web Literacy for Student Fact-Checkers.* CC BY 4.0. https://pressbooks.pub/webliteracy/
- **Calibration** : Good Judgment Open (tournoi en ligne) — https://www.gjopen.com/
- **Arbres de décision** : Peterson, M. (2017). *An Introduction to Decision Theory*, 2e éd. Cambridge University Press.
