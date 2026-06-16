# Module 07 — Capstone : La Boîte à Outils du Jugement

> **Temps estimé** : 60 min | **Prérequis** : Modules 01-06

> **Objectif** : Assembler un *latticework* personnel de jugement — checklist de pré-décision, journal de prévisions calibré, protocole de vérification — et l'appliquer à une décision neutre et concrète.

---

## 1. Pourquoi un latticework ?

Charlie Munger (Poor Charlie's Almanack, 2005) a popularisé l'idée de construire un *latticework of mental models* : un réseau de modèles issus de plusieurs disciplines, qu'on active selon le contexte.

L'idée n'est pas d'accumuler des modèles mais de savoir **quand en activer quel** :
- Problème probabiliste → Bayes + base rates.
- Choix sous risque → espérance + utilité.
- Information douteuse → SIFT + lecture latérale.
- Estimation → ancrage + classe de référence.
- Prédiction à valider → score de Brier + journal.

---

## 2. Pièce 1 — Checklist de pré-décision

```
PRÉ-DÉCISION — [Date] — [Décision envisagée]

1. CLARIFIER
   □ Quelle est exactement la décision à prendre ?
   □ Quelles sont les options réelles ?
   □ Quel est l'horizon de temps ?

2. BIAIS À VÉRIFIER
   □ Ancrage : suis-je ancré sur un premier chiffre/idée ?
     → Estimer indépendamment d'abord.
   □ Disponibilité : un exemple récent/dramatique domine-t-il ma pensée ?
     → Chercher le taux de base historique.
   □ Cadrage : ma préférence changerait-elle en reformulant en gains/pertes ?
   □ Confirmation : ai-je cherché les arguments *contre* ma position favorite ?

3. PROBABILITÉS
   □ Probabilité estimée pour le scénario principal : _____%
   □ Classe de référence : ____________________
   □ Ajustements depuis le taux de base : ____________________

4. CONSÉQUENCES
   □ Scénario optimiste / central / pessimiste avec probabilités.
   □ Quel scénario est "ruineux" (à éviter même à faible probabilité) ?

5. VÉRIFICATION DES INFORMATIONS
   □ Les faits clés ont-ils des sources vérifiables ?
   □ Ai-je fait une lecture latérale sur les sources critiques ?

6. DÉCISION ET SUIVI
   □ Décision prise : ___________
   □ Probabilité subjective du succès : _____%
   □ Date de revue : ___________
```

---

## 3. Pièce 2 — Journal de prévisions calibré

### Format minimal

```
[Date] | [Question précise] | [Ma prédiction : X %] | [Date résolution] | [Outcome : 0/1] | [Brier : (p-o)²]
```

### Règles

1. **Question binaire et datée** : réponse observable et non ambiguë.
2. **Score systématique** : calculer le score de Brier après chaque résolution.
3. **Revue mensuelle** : score moyen + zones de sur-/sous-confiance.

### Exemple de tableau

| Date | Question | p (%) | Résultat | (p−o)² | Notes |
|------|----------|--------|---------|--------|-------|
| 2026-06-16 | Pluie demain matin ? | 75 | 1 | 0,0625 | OK |
| 2026-06-17 | Livraison dans les délais ? | 60 | 0 | 0,36 | Sur-confiant |
| 2026-06-18 | Finirai cette tâche avant 18h ? | 50 | 1 | 0,25 | Baseline |

**Score moyen** : (0,0625 + 0,36 + 0,25) / 3 = **0,224**

---

## 4. Pièce 3 — Protocole de vérification en 5 étapes

```
1. STOP : s'arrêter avant de partager ou de décider.

2. SOURCE : Qui publie ? Quel historique ?
   → Lecture latérale (2-3 onglets sur la source).

3. COUVERTURE : D'autres sources indépendantes confirment-elles ?
   → Chercher "[sujet] site:reuters.com" ou "[sujet] meta-analyse".

4. REMONTÉE :
   → Citation : Google Scholar + doi.org
   → Image : Google Images (clic droit) ou TinEye

5. LLM SPÉCIFIQUE :
   → Titre entre guillemets sur Google Scholar.
   → DOI sur doi.org.
   → Si introuvable → probablement halluciné.
```

---

## 5. Application à une décision concrète et neutre

### Exemple guidé : choisir entre deux formations en ligne

**Checklist** :
- *Ancrage* : ne pas se laisser influencer par la formation vue en premier.
- *Disponibilité* : le témoignage d'un ami n'est qu'un échantillon de 1.
- *Classe de référence* : quel % de personnes finissent réellement les MOOCs ? (5-15 % selon MIT ResearchGate 2019).
- *Cadrage* : "cette formation me coûte 150 €" vs "compétence valorisée X €/mois sur le marché".

**Prédiction** :
- Probabilité de finir la formation A dans les 3 mois : 40 %.
- Enregistrer dans le journal, résoudre dans 3 mois.

**Vérification** :
- Le taux de complétion cité par la plateforme (ex. "85 % de finissants") : lire latéralement. Les études indépendantes donnent des chiffres très différents.

---

## 6. Votre boîte à outils personnalisée

À construire dans `04-projects/README.md` (template fourni). Incluez :

1. **Votre checklist** : adaptée à vos décisions récurrentes.
2. **Votre journal** : au moins 1 prédiction par semaine. Viser < 0,20 de Brier à 6 mois.
3. **Votre protocole de vérification** : fiche de 5 lignes tenue à portée de main.

**Indicateur de progrès** : votre score de Brier moyen sur 30 prédictions. Objectif à 6 mois : < 0,20.

---

> **À retenir** :
> - Un latticework de jugement = ensemble d'habitudes activées selon le contexte.
> - Les 3 outils : checklist de pré-décision, journal calibré, protocole de vérification.
> - La calibration s'améliore par la pratique et le feedback.
> - Score de Brier < 0,20 à 6 mois = objectif atteignable.

---

## Flash-cards (Module 07)

**Q1** : Qu'est-ce qu'un "latticework of mental models" ?
**R1** : Un réseau de modèles mentaux issus de plusieurs disciplines, activés selon le contexte. Utile car un seul cadre transforme tous les problèmes en clous.

**Q2** : Quels 3 outils constituent la boîte à outils du jugement ?
**R2** : 1) Checklist de pré-décision. 2) Journal de prévisions calibré (score de Brier). 3) Protocole de vérification (SIFT + lecture latérale + vérification LLM).

**Q3** : Quel est l'objectif de score de Brier à viser après 6 mois ?
**R3** : < 0,20 (baseline = 0,25 ; superforecasters ≈ 0,10-0,15).

**Q4** : Pourquoi la "classe de référence" est-elle le premier réflexe avant toute estimation ?
**R4** : Pour ancrer sur un taux de base historique plutôt que sur l'optimisme ou le cas individuel.

**Q5** : Quand activer le protocole de vérification plutôt que la checklist ?
**R5** : Le protocole s'active chaque fois qu'une information externe est utilisée (citation, stat, image, LLM). La checklist s'active sur le processus de décision lui-même.

---

## Points clés à retenir

1. Un latticework = outils activés selon le contexte, pas une liste à réciter.
2. La checklist force à expliciter les biais et les probabilités avant d'agir.
3. Le journal + score de Brier = feedback loop essentielle pour progresser.
4. Le protocole SIFT + lecture latérale est non partisan et universel.
5. La calibration s'apprend : 30 prédictions scorées > 10 heures de lecture sur les biais.

---

## Pour aller plus loin

- **Latticework** : Munger, C. T. (2005/2023). *Poor Charlie's Almanack.* Stripe Press. https://www.stripe.press/poor-charlies-almanack/
- **Superforecasters** : Tetlock, P. E. & Gardner, D. (2015). *Superforecasting.* Crown Publishers.
- **Rationalité et intelligence** : Stanovich, K. E. (2011). *Rationality and the Reflective Mind.* Oxford University Press.
- **Pratiquer** : Good Judgment Open — https://www.gjopen.com/
- **SIFT** : Caulfield, M. (2019). https://hapgood.us/2019/06/19/sift-the-four-moves/
