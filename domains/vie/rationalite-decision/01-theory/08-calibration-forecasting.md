# Module 08 — Calibration & Forecasting

> **Temps estimé** : 45 min | **Prérequis** : Modules 01-07
>
> **Objectif** : Savoir penser en probabilités, calculer et interpréter le score de Brier, appliquer les leçons des superforecasters, et tenir un journal de prévisions pour s'améliorer durablement.

---

## 1. Penser en probabilités — sortir du binaire

Notre cerveau préfère les certitudes : "ça va marcher" ou "ça va rater". Mais le monde est probabiliste. Un bon raisonneur ne dit pas "il pleuvra demain" — il dit "70 % de chances de pluie".

**Exemple concret — météo** :

Un prévisionniste annonce "80 % de pluie" 100 jours de suite. Si la pluie tombe environ 80 fois sur ces 100 jours, il est *bien calibré*. Si la pluie tombe seulement 50 fois, il est *sur-confiant* (il annonçait 80 % mais n'avait pas tort huit fois sur dix).

**Pourquoi c'est difficile ?**

Deux obstacles cognitifs principaux :
1. **La pensée binaire** : "soit ça arrive, soit non" — on ignore les nuances de probabilité.
2. **Le biais de confiance excessive** : dans les études, des experts surestiment systématiquement la précision de leurs jugements (Kahneman, 2011).

**Technique : la classe de référence**

Avant d'estimer une probabilité, demandez-vous : *"Pour des événements similaires dans le passé, combien se sont produits ?"* Cette ancre empirique réduit les biais d'optimisme.

> Exemple sportif : votre équipe favorite affronte une équipe de niveau équivalent. Intuition : "on va gagner !". Classe de référence : les matchs entre équipes de niveau similaire se soldent à ~33 % par victoire chaque camp, 33 % match nul. Votre estimation de départ devrait donc être autour de 35 %, à ajuster selon le lieu du match, la forme récente, etc.

---

## 2. Le score de Brier — mesurer la qualité de ses prédictions

Penser en probabilités ne sert à rien si on ne se score pas. Le **score de Brier** est l'outil standard.

**Formule :**

```
Brier = (1/N) × Σ (pᵢ − oᵢ)²
```

- `pᵢ` : probabilité annoncée pour l'événement `i` (entre 0 et 1)
- `oᵢ` : résultat observé — **1** si l'événement s'est produit, **0** sinon
- N : nombre de prédictions

**Repères essentiels :**

| Score | Signification |
|-------|---------------|
| 0,00 | Parfait (impossible en pratique) |
| < 0,15 | Excellent |
| 0,15 – 0,20 | Bon |
| 0,25 | Baseline "toujours 50 %" |
| > 0,25 | Moins bon que de n'avoir aucune opinion |
| 1,00 | Parfaitement et confiamment faux |

**Un score plus bas est un meilleur score.**

**Exemple chiffré — météo + sport :**

| Prédiction | p annoncée | Résultat | (p − o)² |
|------------|-----------|---------|----------|
| Pluie demain | 0,80 | 1 (pluie) | (0,80−1)² = 0,04 |
| Match nul ce soir | 0,30 | 0 (pas nul) | (0,30−0)² = 0,09 |
| Température > 28°C | 0,60 | 1 (oui) | (0,60−1)² = 0,16 |
| Record du 100 m battu | 0,10 | 0 (non) | (0,10−0)² = 0,01 |

```
Brier = (0,04 + 0,09 + 0,16 + 0,01) / 4 = 0,075
```

Score de 0,075 — nettement meilleur que le hasard (0,25).

**Pourquoi le carré ?**

Le terme au carré pénalise *davantage* les erreurs confiantes. Dire "90 % de pluie" et se tromper coûte `(0,90)² = 0,81` sur cette prédiction. Dire "55 %" et se tromper ne coûte que `(0,55)² ≈ 0,30`. La formule encourage la prudence et l'honnêteté sur son incertitude.

---

## 3. La courbe de calibration — visualiser ses biais systématiques

Le score de Brier est un score global. La **courbe de calibration** va plus loin : elle révèle si vous êtes sur-confiant, sous-confiant, ou mal calibré sur une plage spécifique.

**Construction :**

1. Regrouper les prédictions par tranche de probabilité (ex. [0-20 %), [20-40 %), ...).
2. Dans chaque tranche, calculer la fréquence réelle d'occurrence.
3. Comparer : probabilité moyenne annoncée vs fréquence réelle.

**Interprétation :**

- **Calibration parfaite** : tous les points sur la diagonale — si vous annoncez 70 %, ça arrive 70 % du temps.
- **Sur-confiance** : la courbe est *en-dessous* de la diagonale — vous annoncez 70 % mais ça arrive seulement 50 % du temps.
- **Sous-confiance** : la courbe est *au-dessus* — vous annoncez 40 % mais ça arrive 60 % du temps.

> Exemple météo : un prévisionniste novice remarque que quand il annonce "70 % de pluie", la pluie tombe en réalité 85 % du temps. Il est sous-confiant sur cette tranche — il devrait annoncer 85 %, pas 70 %.

---

## 4. Les superforecasters — ce que la recherche a découvert

Le **Good Judgment Project** (GJP), mené sous financement IARPA de 2011 à 2015, a opposé des milliers de prévisionnistes amateurs à des analystes de renseignement professionnels sur des questions géopolitiques et économiques. Résultat : les meilleurs amateurs (*superforecasters*) surpassaient les experts de 30 % en score de Brier.

> Référence : Tetlock, P. E. & Gardner, D. (2015). *Superforecasting : The Art and Science of Prediction.* Crown Publishers.

**Les 5 pratiques distinctives (Tetlock & Gardner, 2015) :**

1. **Raisonnement probabiliste** — ils donnent des chiffres précis, pas des formules vagues ("probable" = quel pourcentage ?).
2. **Décomposition (technique de Fermi)** — ils fractionnent les questions complexes en sous-questions plus faciles à estimer.
3. **Classe de référence** — ils ancrent leur estimation sur des taux de base historiques.
4. **Mise à jour incrémentale** — ils révisent leurs prédictions quand de nouvelles informations arrivent, sans sur-réagir.
5. **Recherche active du désaccord** — ils cherchent les contre-arguments à leurs propres prédictions.

**Ce que les superforecasters ne font PAS :**

- Ils n'ont pas de QI exceptionnel.
- Ils n'ont pas accès à des informations privilégiées.
- Ils ne sont pas des experts du domaine concerné.

Ce qui les distingue : **des habitudes de raisonnement**, entraînées délibérément.

**Résultat empirique de Mellers et al. (2014) :**

L'entraînement probabiliste (training), le travail en groupe (teaming) et le suivi systématique de ses scores (tracking) améliorent significativement la calibration. La pratique régulière compte plus que l'expertise initiale.

---

## 5. Tenir un journal de prévisions — le protocole pratique

Le journal de prévisions est l'équivalent du journal d'entraînement d'un sportif. Sans trace écrite, pas d'amélioration mesurable.

**Format minimal pour chaque entrée :**

```
Date : [YYYY-MM-DD]
Prédiction : [énoncé précis et vérifiable]
Probabilité : [XX %]
Date de vérification : [quand saura-t-on ?]
Résultat : [à remplir après]
```

**Règles d'or :**

1. **Précision de l'énoncé** : "il pleuvra demain matin avant 12 h à Paris" est vérifiable. "Il fera mauvais temps" ne l'est pas.
2. **Probabilité obligatoire** : pas de "je pense que oui" — forcer un chiffre (ex. 65 %).
3. **Révisions autorisées** : noter la date et la raison de chaque révision.
4. **Scoring régulier** : calculer le score de Brier tous les mois sur les prédictions closes.

**Exemple d'entrée complète (thème sport) :**

```
Date : 2026-06-16
Prédiction : L'équipe de France termine dans le top 3 du tournoi.
Probabilité : 55 %
Date de vérification : 2026-07-15
Résultat : [à remplir]
Score Brier contribution : [(0.55 - résultat)²]
```

**Fréquence conseillée** : 2 à 5 prédictions par semaine sur des événements vérifiables à court terme (météo, résultats sportifs, durée d'une tâche). La taille d'échantillon est clé : 50 prédictions minimum pour que le score soit interprétable.

---

> **À retenir** :
> - Penser en probabilités, c'est quitter le binaire "oui/non" pour donner un chiffre entre 0 et 100 %.
> - Le score de Brier = moyenne de (p − o)² sur N prédictions. Objectif : < 0,20.
> - Sur-confiance (le défaut le plus courant) : on annonce 80 % mais l'événement n'arrive que 55 % du temps.
> - Les superforecasters s'améliorent par la pratique : journal + scoring régulier = entraînement clé.
> - Classe de référence : ancrer toute estimation sur des données historiques similaires.

---

## Flash-cards (Module 08)

**Q1 : Que signifie être "bien calibré" ?**
> R : Vos prédictions à X % se réalisent X % du temps sur l'ensemble du jeu. Quand vous dites "70 %", l'événement arrive environ 70 fois sur 100.

**Q2 : Calculez le score de Brier pour p = 0,80, outcome = 1.**
> R : (0,80 − 1)² = (−0,20)² = **0,04**. Plus près de 0 = meilleur.

**Q3 : Quel est le score de Brier d'un prévisionniste qui dit toujours "50 %" ?**
> R : (0,5 − 0)² = (0,5 − 1)² = 0,25 dans tous les cas. Score de référence = **0,25**.

**Q4 : Citez 3 pratiques parmi les 5 identifiées par Tetlock & Gardner.**
> R : (1) Donner des chiffres précis. (2) Décomposer les questions complexes. (3) Partir de la classe de référence (taux de base). (4) Mettre à jour au fil des nouvelles infos. (5) Chercher activement le désaccord.

**Q5 : Pourquoi le score de Brier utilise-t-il le carré ?**
> R : Pour pénaliser davantage les erreurs commises avec forte confiance. Se tromper en disant "90 %" coûte bien plus que se tromper en disant "55 %".

---

## Points clés à retenir

1. Penser en probabilités remplace le binaire "oui/non" par un chiffre entre 0 et 1.
2. Le score de Brier mesure la qualité des prédictions : (p − o)² moyenné sur N. Objectif : < 0,20.
3. La sur-confiance est le biais le plus fréquent : on annonce plus que ce que les faits confirment.
4. La classe de référence ancre les estimations sur des données historiques, réduisant les biais d'optimisme.
5. Les superforecasters s'améliorent par la pratique, pas par l'expertise innée : journal de prédictions + scoring régulier.
6. Décomposer (technique de Fermi) et mettre à jour incrementalement sont les deux leviers techniques les plus puissants.

---

## Pour aller plus loin

- **Superforecasting** : Tetlock, P. E. & Gardner, D. (2015). *Superforecasting : The Art and Science of Prediction.* Crown Publishers. https://www.goodjudgment.com/
- **Article empirique** : Mellers, B., Ungar, L., Baron, J., Ramos, J., Gurcay, B., Fincher, K., Scott, S. E., Moore, D., Atanasov, P., Swift, S. A., Murray, T., Stone, E. & Tetlock, P. E. (2014). Psychological Strategies for Winning a Geopolitical Forecasting Tournament. *Psychological Science*, 25(5), 1106-1115. https://journals.sagepub.com/doi/10.1177/0956797614524255
- **Tournoi de prévision** : Good Judgment Open (pratique en ligne). https://www.gjopen.com/
- **Thinking, Fast and Slow** : Kahneman, D. (2011). Farrar, Straus and Giroux. https://us.macmillan.com/books/9780374533557/thinkingfastandslow
