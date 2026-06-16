# Module 06 — Calibration & Vérification de l'Information

> **Temps estimé** : 55 min | **Prérequis** : Modules 01-05

> **Objectif** : Apprendre à penser en probabilités plutôt qu'en certitudes binaires, mesurer la qualité de ses prédictions avec le score de Brier, et appliquer un protocole de vérification robuste à l'ère de l'IA.

---

## Partie A — Calibration et prévision

### 1. Penser en probabilités

Un expert bien calibré ne dit pas "ça va arriver" ou "ça n'arrivera pas" — il dit "70 % de chances" et il a raison 70 % du temps sur l'ensemble de ses prédictions à 70 %.

**Exemple concret** : un prévisionniste météo dit "90 % de pluie" 100 jours de suite. S'il pleut environ 90 fois sur ces 100 jours, il est *bien calibré*. S'il pleut 50 fois, il est *sur-confiant*. S'il pleut 98 fois, il est *sous-confiant*.

**Pourquoi c'est difficile** : le cerveau pense naturellement en catégories ("probable" / "improbable") plutôt qu'en intervalles numériques. Transformer une intuition en pourcentage oblige à préciser une estimation vague.

**Technique : classes de référence** (Kahneman, inspiré par Flyvbjerg). Pour estimer la probabilité d'un événement :
1. Trouver la *classe de référence* : "combien de projets similaires ont réussi ?"
2. Ancrer sur ce taux de base.
3. Ajuster pour les particularités du cas présent.

Exemple : vous estimez la probabilité de finir un projet en 3 semaines. Classe de référence : 40 % des projets similaires respectent le calendrier prévu. Point de départ : 40 %. Ajustements selon les spécificités du projet.

---

### 2. Le score de Brier : mesurer la qualité d'une prédiction

**Définition** : le score de Brier mesure l'erreur quadratique entre la probabilité prédite et l'outcome réel (0 ou 1).

```
Brier = (1/N) × Σ (pᵢ − oᵢ)²
```

où pᵢ est la probabilité prédite (entre 0 et 1) et oᵢ est l'outcome (0 = non arrivé, 1 = arrivé).

**Interprétation** :
- Score de 0,00 → parfait (chaque prédiction = outcome exact).
- Score de 0,25 → équivalent à dire "50 % partout" sans réfléchir (baseline non informative).
- Score de 1,00 → parfaitement tort sur tout.

**Un meilleur score est un score plus bas.**

**Exemple chiffré** : vous faites 4 prédictions :

| Événement | p prédit | Outcome | (p − o)² |
|-----------|----------|---------|----------|
| Pluie demain | 0,80 | 1 (pluie) | (0,80−1)² = 0,04 |
| Match nul | 0,30 | 0 (pas nul) | (0,30−0)² = 0,09 |
| Retard train | 0,60 | 1 (retard) | (0,60−1)² = 0,16 |
| Record battu | 0,10 | 0 | (0,10−0)² = 0,01 |

```
Brier = (0,04 + 0,09 + 0,16 + 0,01) / 4 = 0,30 / 4 = 0,075
```

Score de 0,075 — bien meilleur que le hasard (0,25). Plus le score est bas, plus les prédictions sont précises *et* bien calibrées.

---

### 3. Les leçons des superforecasters

Le **Good Judgment Project** (GJP, IARPA 2011-2015) a organisé un tournoi international de prévision. Les superforecasters — un groupe de ~3 % des participants — surpassaient les analystes de renseignement (avec accès à des données classifiées) de 30 % sur le score de Brier.

**Ce qui les distingue (Tetlock & Gardner, 2015)** :

1. **Raisonnement probabiliste** : ils donnent des chiffres (pas "probable" ou "certain").
2. **Décomposition** : ils fractionnent les questions complexes (méthode Fermi).
3. **Mise à jour incrémentale** : ils ajustent leurs probabilités au fil des nouvelles informations — petit à petit, pas de volte-face dramatiques.
4. **Classes de référence** : ils commencent par le taux de base historique.
5. **Recherche active du désaccord** (cf. biais de confirmation du module 04).

**Résultat clé** : la calibration se *pratique* et s'*améliore*. Tenir un journal de prédictions et se scorer régulièrement est l'entraînement le plus efficace (Mellers et al., 2014).

---

## Partie B — Vérification de l'information à l'ère de l'IA

### 4. Pourquoi vérifier est plus urgent que jamais

Les modèles de langage (LLMs) génèrent des textes fluides et confiants, y compris lorsqu'ils *hallucinent* — inventent des faits, des citations, des URLs. Une étude de 2023 (Stanford HAI) a montré que GPT-4 cite des sources inexistantes dans environ 3 à 10 % des cas selon le domaine.

Par ailleurs, les techniques de manipulation de l'information se sont industrialisées : images sorties de leur contexte, citations tronquées, "chiffres réels mais comparaison trompeuse". Ce ne sont pas des phénomènes partisans — ils traversent tous les sujets.

**Résultat de Pennycook & Rand (2019)** : la susceptibilité aux fausses informations est mieux expliquée par un *manque de réflexion analytique* que par des motivations partisanes. Autrement dit, le levier est cognitif et universel — pas idéologique.

---

### 5. La méthode SIFT (Caulfield, 2019)

**SIFT** est une méthode de vérification non partisane enseignée dans des centaines d'universités. Elle repose sur 4 mouvements :

**S — Stop** : avant de partager ou de croire, faites une pause. L'urgence est souvent fabriquée.

**I — Investigate the source** : qui publie cette information ? Quel est son modèle économique, son historique, ses biais déclarés ? *Ne lisez pas verticalement* (en cherchant des preuves dans le document lui-même) mais *latéralement*.

**F — Find better coverage** : existe-t-il d'autres sources indépendantes qui confirment (ou infirment) ? Une information importante est couverte par plusieurs sources de référence.

**T — Trace claims, quotes, and media to their original context** : remonter à la source primaire. La citation est-elle complète ? L'image date-t-elle vraiment de cet événement ?

---

### 6. Lecture latérale vs lecture verticale

**Lecture verticale** (l'habitude par défaut) : lire le document de bout en bout pour évaluer sa crédibilité à l'intérieur.

**Lecture latérale** (la méthode des fact-checkers professionnels) : quitter immédiatement la page et ouvrir plusieurs onglets pour chercher ce que d'autres disent *de* cette source.

**Exemple pratique** : vous lisez un article affirmant que "le café réduit le risque d'Alzheimer de 50 %". Lecture latérale : ouvrir Google Scholar et chercher "[auteur de l'étude] + retraction" ou "café Alzheimer meta-analyse". En 2 minutes, vous trouvez que les méta-analyses récentes donnent un effet bien plus modeste (OR ~0,85) avec une hétérogénéité importante entre études.

---

### 7. Hallucinations de LLM et fausses citations

**Signes d'alerte pour une citation générée par un LLM** :

1. L'URL n'existe pas ou renvoie une erreur 404.
2. Le DOI est invalide (vérifiable sur doi.org).
3. L'auteur existe mais n'a pas écrit ce titre (rechercher sur Google Scholar).
4. La date ou la revue ne correspond pas à la réalité.
5. La citation est trop parfaitement formulée pour soutenir l'argument — c'est suspect.

**Protocole de vérification d'une citation** :
1. Copier le titre exact entre guillemets dans Google Scholar.
2. Si trouvé : vérifier auteur, revue, année, et lire le résumé.
3. Si non trouvé : chercher le DOI sur doi.org.
4. Si toujours introuvable : la citation est probablement hallucinée.

---

### 8. Images sorties de leur contexte

Les outils de vérification d'images :
- **Google Images** (recherche inversée) : clic droit → "Rechercher l'image".
- **TinEye** : https://tineye.com — historique des apparitions d'une image.
- **InVID / WeVerify** : https://weverify.eu — extension navigateur pour les vidéos et images, donnant la date de première apparition.

**Méthode** : une image "récente" a-t-elle des apparitions datées d'années antérieures ? La recherche inversée révèle le vrai contexte original.

---

> **À retenir** :
> - La calibration se mesure : score de Brier = erreur quadratique entre probabilité prédite et outcome. Plus bas = meilleur.
> - Les superforecasters ne sont pas des génies : ils appliquent des habitudes (chiffrer, décomposer, mettre à jour, rechercher le désaccord).
> - SIFT + lecture latérale = protocole universel de vérification, indépendant des croyances politiques.
> - Les LLMs hallucinent : vérifier toute citation avec Google Scholar + doi.org.

---

## Flash-cards (Module 06)

**Q1** : Calculez le score de Brier pour une prédiction unique : p = 0,7, outcome = 1.
**R1** : (0,7 − 1)² = (−0,3)² = **0,09**. (Score parfait = 0, baseline = 0,25, votre score est meilleur que le hasard.)

**Q2** : Que signifie être "bien calibré" ?
**R2** : Que vos prédictions à X % se réalisent X % du temps sur l'ensemble du jeu. Ex. : vos prédictions à 80 % se réalisent environ 80 fois sur 100.

**Q3** : Quels sont les 3 leviers principaux identifiés par Mellers et al. (2014) pour améliorer la calibration ?
**R3** : Training (entraînement probabiliste), Teaming (travail en équipe avec confrontation des prédictions), Tracking (suivi et scoring régulier de ses prédictions passées).

**Q4** : Quelle est la différence entre lecture verticale et lecture latérale ?
**R4** : Verticale = lire le document lui-même pour évaluer sa crédibilité. Latérale = quitter immédiatement la page et chercher ce que d'autres sources disent *de* cette source. Les fact-checkers professionnels utilisent la lecture latérale.

**Q5** : Comment vérifier qu'une citation générée par un LLM est authentique ?
**R5** : 1) Chercher le titre exact entre guillemets sur Google Scholar. 2) Vérifier le DOI sur doi.org. 3) Confirmer auteur + revue + année. 4) Si introuvable après ces 3 étapes, la citation est probablement hallucinée.

---

## Points clés à retenir

1. Calibrer = donner des probabilités numériques, pas des catégories vagues, et se scorer ensuite.
2. Score de Brier : (p − o)² moyenné sur N prédictions. Objectif : descendre sous 0,20 sur des questions ouvertes.
3. Les superforecasters s'améliorent par la pratique : tenir un journal de prédictions est l'entraînement le plus efficace.
4. SIFT (Stop, Investigate, Find, Trace) = protocole universel, non partisan, de vérification.
5. Lecture latérale : quitter la page pour voir ce que d'autres en disent — plus efficace que lire verticalement.
6. LLMs hallucinent des citations : vérification obligatoire par Google Scholar + doi.org.

---

## Pour aller plus loin

- **Superforecasting** : Tetlock, P. E. & Gardner, D. (2015). *Superforecasting: The Art and Science of Prediction.* Crown Publishers. https://www.goodjudgment.com/
- **Article empirique** : Mellers, B., Ungar, L., Baron, J., … Tetlock, P. (2014). Psychological Strategies for Winning a Geopolitical Forecasting Tournament. *Psychological Science*, 25(5), 1106-1115. https://journals.sagepub.com/doi/10.1177/0956797614524255
- **SIFT** : Caulfield, M. (2017). *Web Literacy for Student Fact-Checkers.* CC BY 4.0. https://pressbooks.pub/webliteracy/
- **Accuracy nudge** : Pennycook, G. et al. (2021). Shifting attention to accuracy can reduce misinformation online. *Nature*, 592, 590-595. https://www.nature.com/articles/s41586-021-03344-2
- **Désinformation et réflexion analytique** : Pennycook, G. & Rand, D. G. (2019). *Cognition*, 188, 39-50. https://www.sciencedirect.com/science/article/abs/pii/S001002771830163X
- **Tournoi de prévision** : Good Judgment Open. https://www.gjopen.com/
