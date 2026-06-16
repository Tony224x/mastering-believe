# Module 06 — Calibration & Vérification de l'Information

> **Temps estimé** : 55 min | **Prérequis** : Modules 01-05

> **Objectif** : Apprendre à penser en probabilités, mesurer la qualité de ses prédictions avec le score de Brier, et appliquer un protocole de vérification robuste à l'ère de l'IA.

---

## Partie A — Calibration et prévision

### 1. Penser en probabilités

Un expert bien calibré ne dit pas "ça va arriver" — il dit "70 % de chances" et a raison 70 % du temps sur l'ensemble de ses prédictions à 70 %.

**Exemple** : un prévisionniste météo dit "90 % de pluie" 100 jours de suite. S'il pleut environ 90 fois sur ces 100 jours, il est *bien calibré*.

**Technique : classes de référence** (Kahneman, inspiré par Flyvbjerg) :
1. Trouver la *classe de référence* : "combien de projets similaires ont réussi ?"
2. Ancrer sur ce taux de base.
3. Ajuster pour les particularités du cas présent.

---

### 2. Le score de Brier : mesurer la qualité d'une prédiction

**Formule** :

```
Brier = (1/N) × Σ (pᵢ − oᵢ)²
```

où pᵢ est la probabilité prédite (0-1) et oᵢ est l'outcome (0 ou 1).

**Interprétation** :
- Score de 0,00 → parfait.
- Score de 0,25 → baseline (dire "50 % partout").
- Score de 1,00 → parfaitement tort.

**Un meilleur score est un score plus bas.**

**Exemple chiffré** :

| Événement | p prédit | Outcome | (p − o)² |
|-----------|----------|---------|----------|
| Pluie demain | 0,80 | 1 | 0,04 |
| Match nul | 0,30 | 0 | 0,09 |
| Retard train | 0,60 | 1 | 0,16 |
| Record battu | 0,10 | 0 | 0,01 |

```
Brier = (0,04 + 0,09 + 0,16 + 0,01) / 4 = 0,075
```

Score de 0,075 — bien meilleur que le hasard (0,25).

---

### 3. Les leçons des superforecasters

Le **Good Judgment Project** (GJP, IARPA 2011-2015) : les superforecasters surpassaient les analystes de renseignement de 30 % sur le score de Brier.

**Ce qui les distingue (Tetlock & Gardner, 2015)** :
1. **Raisonnement probabiliste** : ils donnent des chiffres.
2. **Décomposition** : ils fractionnent les questions complexes.
3. **Mise à jour incrémentale** : ils ajustent au fil des nouvelles informations.
4. **Classes de référence** : ils commencent par le taux de base.
5. **Recherche active du désaccord**.

**Résultat clé** : la calibration se *pratique* et s'*améliore*. Tenir un journal de prédictions est l'entraînement le plus efficace (Mellers et al., 2014).

---

## Partie B — Vérification de l'information à l'ère de l'IA

### 4. Pourquoi vérifier est plus urgent que jamais

Les LLMs génèrent des textes fluides et confiants, y compris lorsqu'ils *hallucinent* — inventent des faits, des citations, des URLs. Résultat Pennycook & Rand (2019) : la susceptibilité aux fausses informations s'explique mieux par un *manque de réflexion analytique* que par des motivations partisanes — le levier est cognitif et universel.

---

### 5. La méthode SIFT (Caulfield, 2019)

**SIFT** repose sur 4 mouvements :

**S — Stop** : avant de partager ou de croire, faites une pause.

**I — Investigate the source** : qui publie ? Quel historique ? *Lecture latérale* (sortir de la page) et non verticale.

**F — Find better coverage** : d'autres sources indépendantes confirment-elles ?

**T — Trace claims, quotes, and media to their original context** : remonter à la source primaire. La citation est-elle complète ? L'image date-t-elle de cet événement ?

---

### 6. Lecture latérale vs lecture verticale

**Lecture verticale** (habitude par défaut) : lire le document de bout en bout.

**Lecture latérale** (méthode des fact-checkers) : quitter la page et ouvrir plusieurs onglets pour chercher ce que d'autres disent *de* cette source.

**Exemple pratique** : article affirmant "le café réduit le risque d'Alzheimer de 50 %". Lecture latérale : chercher "[auteur] + retraction" et "café Alzheimer méta-analyse". En 2 minutes : les méta-analyses récentes donnent un effet bien plus modeste (OR ~0,85) avec hétérogénéité importante.

---

### 7. Hallucinations de LLM et fausses citations

**Signes d'alerte** :
1. L'URL n'existe pas ou renvoie 404.
2. Le DOI est invalide (vérifiable sur doi.org).
3. L'auteur existe mais n'a pas écrit ce titre.
4. La date ou la revue ne correspondent pas.
5. La citation est trop parfaitement formulée.

**Protocole de vérification d'une citation** :
1. Copier le titre exact entre guillemets dans Google Scholar.
2. Si trouvé : vérifier auteur, revue, année.
3. Si non trouvé : vérifier le DOI sur doi.org.
4. Si toujours introuvable : probablement hallucinée.

---

### 8. Images sorties de leur contexte

**Outils** :
- **Google Images** : clic droit → "Rechercher l'image".
- **TinEye** : https://tineye.com
- **InVID / WeVerify** : https://weverify.eu

**Méthode** : une "image récente" a-t-elle des apparitions datées d'années antérieures ?

---

> **À retenir** :
> - Score de Brier : (p − o)² moyenné sur N prédictions. Objectif : < 0,20.
> - Les superforecasters s'améliorent par la pratique : tenir un journal est l'entraînement le plus efficace.
> - SIFT + lecture latérale = protocole universel non partisan.
> - LLMs hallucinent : vérifier toute citation avec Google Scholar + doi.org.

---

## Flash-cards (Module 06)

**Q1** : Calculez le score de Brier pour une prédiction unique : p = 0,7, outcome = 1.
**R1** : (0,7 − 1)² = (−0,3)² = **0,09**.

**Q2** : Que signifie être "bien calibré" ?
**R2** : Vos prédictions à X % se réalisent X % du temps sur l'ensemble du jeu.

**Q3** : Quels sont les 3 leviers principaux identifiés par Mellers et al. (2014) ?
**R3** : Training (entraînement probabiliste), Teaming (travail en équipe), Tracking (suivi et scoring régulier).

**Q4** : Quelle est la différence entre lecture verticale et lecture latérale ?
**R4** : Verticale = lire le document lui-même. Latérale = quitter la page et chercher ce que d'autres sources disent *de* cette source.

**Q5** : Comment vérifier qu'une citation générée par un LLM est authentique ?
**R5** : 1) Titre exact sur Google Scholar. 2) DOI sur doi.org. 3) Confirmer auteur + revue + année. Si introuvable → probablement hallucinée.

---

## Points clés à retenir

1. Calibrer = donner des probabilités numériques et se scorer ensuite.
2. Score de Brier : (p − o)² moyenné sur N prédictions. Objectif : < 0,20.
3. Les superforecasters s'améliorent par la pratique : journal de prédictions = entraînement clé.
4. SIFT = protocole universel, non partisan, de vérification.
5. Lecture latérale : quitter la page pour voir ce que d'autres en disent.
6. LLMs hallucinent : vérification obligatoire par Google Scholar + doi.org.

---

## Pour aller plus loin

- **Superforecasting** : Tetlock, P. E. & Gardner, D. (2015). *Superforecasting.* Crown Publishers. https://www.goodjudgment.com/
- **Article empirique** : Mellers et al. (2014). *Psychological Science*, 25(5). https://journals.sagepub.com/doi/10.1177/0956797614524255
- **SIFT** : Caulfield, M. (2017). *Web Literacy for Student Fact-Checkers.* https://pressbooks.pub/webliteracy/
- **Accuracy nudge** : Pennycook et al. (2021). *Nature*, 592. https://www.nature.com/articles/s41586-021-03344-2
- **Désinformation** : Pennycook & Rand (2019). *Cognition*, 188. https://www.sciencedirect.com/science/article/abs/pii/S001002771830163X
- **Tournoi de prévision** : Good Judgment Open. https://www.gjopen.com/
