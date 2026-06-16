# Module 07 — Capstone : La Boîte à Outils du Jugement

> **Temps estimé** : 60 min | **Prérequis** : Modules 01-06

> **Objectif** : Assembler un *latticework* personnel de jugement — checklist de pré-décision, journal de prévisions calibré, protocole de vérification — et l'appliquer à une décision neutre et concrète de votre vie.

---

## 1. Pourquoi un latticework ?

Charlie Munger (Poor Charlie's Almanack, 2005) a popularisé l'idée de construire un *latticework of mental models* : un réseau de modèles issus de plusieurs disciplines, qu'on active selon le contexte. Un seul marteau mental transforme tous les problèmes en clous.

L'idée n'est pas d'accumuler des modèles mais de savoir **quand en activer quel** :
- Problème probabiliste → Bayes + base rates.
- Choix sous risque → espérance + utilité.
- Information douteuse → SIFT + lecture latérale.
- Estimation → ancrage + classe de référence.
- Prédiction à valider dans le temps → score de Brier + journal.

Ce module vous guide pour assembler votre boîte à outils en 3 pièces opérationnelles.

---

## 2. Pièce 1 — Checklist de pré-décision

Une checklist n'est pas une procédure bureaucratique : c'est un filet contre les erreurs de pilote automatique (Système 1). Elle prend 3 à 5 minutes et force à activer Système 2 avant de décider.

### Checklist de pré-décision (modèle à personnaliser)

```
PRÉ-DÉCISION — [Date] — [Décision envisagée]

1. CLARIFIER
   □ Quelle est exactement la décision à prendre ?
   □ Quelles sont les options réelles (pas seulement l'option A vs statu quo) ?
   □ Quel est l'horizon de temps ?

2. BIAIS À VÉRIFIER
   □ Ancrage : suis-je ancré sur un premier chiffre/idée ? → Estimer indépendamment d'abord.
   □ Disponibilité : est-ce qu'un exemple récent/dramatique domine ma pensée ?
     → Chercher le taux de base historique.
   □ Cadrage : ma préférence changerait-elle si je reformulais en gains/pertes ?
   □ Confirmation : ai-je cherché les arguments *contre* ma position favorite ?

3. PROBABILITÉS
   □ Quelle est ma probabilité estimée pour le scénario principal ? (ex. 65 %)
   □ Quelle est ma classe de référence ? (combien de cas similaires ont eu ce résultat ?)
   □ Ajustements à partir du taux de base ?

4. CONSÉQUENCES
   □ Gain/perte espéré dans les 3 scénarios : optimiste / central / pessimiste.
   □ Quel scénario est "ruineux" (à éviter même à faible probabilité) ?
   □ Suis-je neutre au risque ici, ou averse ? Pourquoi ?

5. VÉRIFICATION DES INFORMATIONS
   □ Les faits clés ont-ils des sources vérifiables ?
   □ Ai-je fait une lecture latérale sur les sources critiques ?

6. DÉCISION ET SUIVI
   □ Décision prise : ___________
   □ Probabilité subjective du succès : _____ %
   □ Date de revue pour scorer le résultat : ___________
```

---

## 3. Pièce 2 — Journal de prévisions calibré

Le journal de prévisions est l'outil le plus sous-estimé de la calibration. Les superforecasters l'utilisent systématiquement.

### Format minimal d'une entrée

```
[Date de prédiction] | [Question précise] | [Ma prédiction : X %] | [Date de résolution] | [Outcome : 0/1] | [Brier : (p-o)²]
```

### Règles du journal

1. **Question binaire et datée** : "L'équipe X gagnera-t-elle son prochain match le [date] ?" — pas "l'équipe X va bien s'en sortir".
2. **Résolution claire** : l'outcome doit être observable et non ambigu.
3. **Score systématique** : calculer le score de Brier après chaque résolution.
4. **Revue mensuelle** : regarder son score moyen et identifier les zones de sur- ou sous-confiance.

### Tableau d'exemple

| Date | Question | p (%) | Résultat | (p−o)² | Notes |
|------|----------|--------|---------|--------|-------|
| 2026-06-16 | Pluie demain matin ? | 75 | 1 | 0,0625 | OK |
| 2026-06-17 | Livraison arrivera dans les délais ? | 60 | 0 | 0,36 | Sur-confiant |
| 2026-06-18 | Je finirai cette tâche avant 18h ? | 50 | 1 | 0,25 | Baseline |

**Score moyen** : (0,0625 + 0,36 + 0,25) / 3 = **0,224** — proche de la baseline (0,25), à améliorer.

---

## 4. Pièce 3 — Protocole de vérification en 5 étapes

À activer chaque fois qu'une information est utilisée pour une décision importante.

```
PROTOCOLE DE VÉRIFICATION

1. STOP : s'arrêter avant de partager ou de décider.

2. SOURCE : Qui publie ? Quel historique ?
   → Ouvrir 2-3 onglets sur la source (lecture latérale).

3. COUVERTURE : D'autres sources indépendantes confirment-elles ?
   → Chercher "[sujet] + site:reuters.com" ou "[sujet] + meta-analyse".

4. REMONTÉE : Si c'est une citation ou une image :
   → Citation : Google Scholar + doi.org
   → Image : Google Images (clic droit) ou TinEye
   → Vérifier date, auteur, contexte original.

5. LLM SPÉCIFIQUE : Si l'info vient d'un LLM :
   → Vérifier le titre entre guillemets sur Google Scholar.
   → Valider le DOI sur doi.org.
   → Si introuvable → probablement halluciné, ne pas utiliser.
```

---

## 5. Application à une décision concrète et neutre

### Exemple guidé : choisir entre deux formations en ligne

**Contexte** : vous hésitez entre deux formations en ligne (coûts, durées, contenus différents). Décision à 150 €.

**Étape 1 — Checklist** :
- *Ancrage* : ne pas se laisser influencer par la formation vue en premier (prix ou "popularité").
- *Disponibilité* : le témoignage enthousiaste d'un ami n'est qu'un échantillon de 1.
- *Classe de référence* : quel % de personnes finissent réellement les formations en ligne de ce type ? (Réponse courante : 5-15 % selon les MOOCs, source : MIT ResearchGate 2019). Ajuster ses attentes.
- *Cadrage* : "cette formation me coûte 150 €" vs "si je ne la fais pas, je manque une compétence valorisée X €/mois sur le marché". Reformuler les deux.

**Étape 2 — Prédiction** :
- Probabilité de finir la formation A dans les 3 mois : 40 %.
- Probabilité que la compétence acquise soit utile dans mon travail dans 6 mois : 60 %.
- Enregistrer dans le journal, résoudre dans 3 et 6 mois.

**Étape 3 — Vérification des informations** :
- Le taux de complétion cité par la plateforme (ex. "85 % de finissants") : lire latéralement. Les études indépendantes donnent des chiffres très différents.
- Les avis sur la plateforme : vérifier sur une source externe (ex. Course Report, Switchup) plutôt que sur le site vendeur.

**Résultat** : décision prise avec les biais explicités, une prédiction quantifiée à valider, et les informations clés vérifiées indépendamment.

---

## 6. Votre boîte à outils personnalisée

À construire dans `04-projects/README.md` (template fourni). Incluez :

1. **Votre checklist** : adapter le modèle ci-dessus à vos décisions récurrentes (professionnelles, personnelles, financières).
2. **Votre journal** : utiliser un tableur ou un carnet dédié. Viser au moins 1 prédiction par semaine.
3. **Votre protocole de vérification** : réduire à une fiche de 5 lignes tenue à portée de main.

**Indicateur de progrès** : votre score de Brier moyen sur 30 prédictions. Objectif à 6 mois : < 0,20.

---

> **À retenir** :
> - Un latticework de jugement n'est pas une liste de concepts à mémoriser : c'est un ensemble d'habitudes activées selon le contexte.
> - Les 3 outils opérationnels : checklist de pré-décision, journal calibré, protocole de vérification.
> - La calibration s'améliore par la pratique et le feedback — tenir un journal et se scorer est l'investissement le plus rentable.
> - Le score de Brier est votre boussole : objectif < 0,20 sur des questions ouvertes à 3 mois.

---

## Flash-cards (Module 07)

**Q1** : Qu'est-ce qu'un "latticework of mental models" et pourquoi est-il utile ?
**R1** : Un réseau de modèles mentaux issus de plusieurs disciplines, qu'on active selon le contexte. Utile car un seul cadre d'analyse ("marteau") transforme tous les problèmes en clous — le latticework permet de choisir l'outil adapté au problème.

**Q2** : Quels 3 outils constituent la boîte à outils du jugement présentée dans ce module ?
**R2** : 1) Checklist de pré-décision (activer Système 2, vérifier les biais). 2) Journal de prévisions calibré (prédictions binaires datées + score de Brier). 3) Protocole de vérification (SIFT + lecture latérale + vérification LLM).

**Q3** : Quel est l'objectif de score de Brier à viser après 6 mois de pratique sur des questions ouvertes ?
**R3** : < 0,20 (baseline du hasard = 0,25 ; les superforecasters atteignent ~ 0,10-0,15 sur des questions géopolitiques ouvertes).

**Q4** : Pourquoi la "classe de référence" est-elle le premier réflexe avant toute estimation ?
**R4** : Pour ancrer sur un taux de base historique plutôt que sur l'optimisme ou le cas individuel. Sans classe de référence, on invente une probabilité de toutes pièces, souvent biaisée par l'espoir ou la disponibilité.

**Q5** : Quand activer le protocole de vérification plutôt que la checklist de décision ?
**R5** : Le protocole de vérification s'active chaque fois qu'une information externe est utilisée comme entrée d'une décision (citation, statistique, image, texte de LLM). La checklist s'active sur le processus de décision lui-même. Les deux peuvent se combiner.

---

## Points clés à retenir

1. Un latticework de jugement = ensemble d'outils activés selon le contexte, pas une liste à réciter.
2. La checklist de pré-décision force à expliciter les biais et les probabilités avant d'agir.
3. Le journal de prévisions calibré + score de Brier = feedback loop essentielle pour progresser.
4. Le protocole de vérification (SIFT + lecture latérale + vérif LLM) est non partisan et universel.
5. La calibration s'apprend : 30 prédictions scorées = plus de progrès que 10 heures de lecture sur les biais.

---

## Pour aller plus loin

- **Latticework** : Munger, C. T. (2005/2023). *Poor Charlie's Almanack.* Stripe Press. https://www.stripe.press/poor-charlies-almanack/
- **Superforecasters** : Tetlock, P. E. & Gardner, D. (2015). *Superforecasting.* Crown Publishers.
- **Rationalité et intelligence** : Stanovich, K. E. (2011). *Rationality and the Reflective Mind.* Oxford University Press. https://global.oup.com/academic/product/rationality-and-the-reflective-mind-9780195341140
- **Pratiquer** : Good Judgment Open — https://www.gjopen.com/ (questions de prévision ouvertes, scorées en Brier)
- **SIFT** : Caulfield, M. (2019). https://hapgood.us/2019/06/19/sift-the-four-moves/
