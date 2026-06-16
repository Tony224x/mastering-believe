# Module 03 — Probabilités en fréquences naturelles

> **Temps estimé** : 45 min | **Prérequis** : Modules 01–02
> **Objectif** : maîtriser la représentation en fréquences naturelles (Gigerenzer) pour raisonner correctement sur les taux de base, les faux positifs et les valeurs prédictives — sans formule de Bayes.

---

## 1. Pourquoi les pourcentages nous trompent

> Un médecin informe sa patiente : « Ce test de dépistage a une sensibilité de 90 % et une spécificité de 95 %. Vous êtes positive. La prévalence de cette maladie dans votre groupe d'âge est de 1 %. »
> La patiente demande : « Docteur, quelle est la probabilité que je sois vraiment malade ? »

Dans une étude classique (Gigerenzer & Hoffrage, 1995), la majorité des médecins interrogés ont répondu « environ 90 % ». La réponse correcte est **environ 15 %**.

Ce n'est pas un problème de mathématiques difficiles. C'est un problème de **format de représentation**. Penser en pourcentages simultanément (90 %, 95 %, 1 %) surcharge la mémoire de travail. Penser en effectifs concrets rend la solution immédiate.

---

## 2. La méthode des fréquences naturelles

**Principe de Gigerenzer** : remplacer les probabilités par des effectifs sur une population fictive. Votre cerveau a évolué pour dénombrer des événements, pas pour multiplier des probabilités.

**Étapes** :

1. Choisir une population de référence (ex. 1 000 personnes).
2. Dénombrer les malades et les sains d'après la prévalence.
3. Appliquer la sensibilité aux malades → vrais positifs et faux négatifs.
4. Appliquer la spécificité aux sains → vrais négatifs et faux positifs.
5. Lire la valeur prédictive directement dans le tableau.

**Application à l'exemple du médecin :**

Population : **1 000 personnes**

| Réalité → | Malade (1 %) | Sain (99 %) | Total |
|-----------|-------------|------------|-------|
| **Test positif** | **9** (VP) | **50** (FP) | **59** |
| **Test négatif** | 1 (FN) | 940 (VN) | 941 |
| **Total** | 10 | 990 | 1 000 |

*Calcul ligne par ligne :*
- Malades : 1 000 × 1 % = **10**
- Vrais positifs (VP) : 10 × 90 % = **9** (malades correctement détectés)
- Faux négatifs (FN) : 10 × 10 % = **1** (malades ratés)
- Sains : 1 000 × 99 % = **990**
- Vrais négatifs (VN) : 990 × 95 % = **940,5 ≈ 940** (sains correctement rejetés)
- Faux positifs (FP) : 990 × 5 % = **49,5 ≈ 50** (sains incorrectement détectés)

**Question :** Parmi les 59 personnes testées positives, combien sont vraiment malades ?

**Réponse :** 9 sur 59 = **15 %**

> **À retenir** : le tableau à 4 cases transforme un calcul mental difficile en simple lecture. Construisez toujours ce tableau avant d'interpréter un résultat de test.

---

## 3. Les quatre cases du tableau et ce qu'elles signifient

Tout système de détection (test médical, filtre anti-spam, capteur industriel) produit exactement quatre types de résultats :

| | Réalité : Positif | Réalité : Négatif |
|---|---|---|
| **Test : Positif** | **Vrai positif (VP)** — bonne détection | **Faux positif (FP)** — fausse alarme |
| **Test : Négatif** | **Faux négatif (FN)** — cas raté | **Vrai négatif (VN)** — bon rejet |

**Définitions opérationnelles :**

- **Sensibilité** = VP / (VP + FN) — proportion de malades détectés par le test
- **Spécificité** = VN / (VN + FP) — proportion de sains correctement rejetés
- **Valeur Prédictive Positive (VPP)** = VP / (VP + FP) — probabilité d'être malade sachant test positif
- **Valeur Prédictive Négative (VPN)** = VN / (VN + FN) — probabilité d'être sain sachant test négatif

> **À retenir** : sensibilité et VPP mesurent des choses différentes. La sensibilité est une propriété fixe du test. La VPP dépend aussi du taux de base — elle varie selon la population testée.

---

## 4. Le rôle décisif du taux de base

**Le taux de base** (ou prévalence) est la proportion naturelle d'un phénomène dans la population, avant tout test.

Avec le **même test** (sensibilité 90 %, spécificité 95 %), la VPP change radicalement selon la prévalence :

| Prévalence | VP | FP | VPP |
|------------|----|----|-----|
| 1 % (maladie rare) | 9 | 50 | **15 %** |
| 10 % (maladie modérée) | 90 | 45 | **67 %** |
| 50 % (population à risque élevé) | 450 | 25 | **95 %** |

*(Calculs sur 1 000 personnes dans chaque cas)*

**Interprétation** : tester la même personne dans trois contextes différents (dépistage général, consultation spécialisée, population à haut risque) donne trois probabilités post-test très différentes — avec exactement le même test.

> **À retenir** : le taux de base est le point de départ obligatoire de toute interprétation. Ignorer la prévalence, c'est comme lire une carte sans savoir où on se trouve.

---

## 5. L'erreur classique : confondre P(test+ | malade) et P(malade | test+)

La **sensibilité** répond à : *Si je suis malade, quelle est la probabilité que le test soit positif ?*
→ P(test+ | malade) = 90 %

La **VPP** répond à : *Si le test est positif, quelle est la probabilité que je sois malade ?*
→ P(malade | test+) = 15 % (avec prévalence 1 %)

Ces deux probabilités conditionnelles sont très différentes, mais nous avons naturellement tendance à les confondre. Cette confusion s'appelle l'**erreur de transposition** ou *prosecutor's fallacy* (erreur du procureur en contexte judiciaire).

**Exemple neutre pour ancrer l'intuition :**

> Dans une ville, 80 % des jours de pluie, le ciel est couvert le matin. Est-ce que cela signifie que si le ciel est couvert ce matin, il va pleuvoir avec 80 % de probabilité ?

Non. Si le ciel est couvert 40 % des jours et qu'il pleut 10 % des jours, on calcule :

- P(couvert | pluie) = 80 % ← c'est ce qu'on nous dit
- P(pluie | couvert) = (10 % × 80 %) / 40 % = 20 % ← c'est ce qu'on veut savoir

> **À retenir** : P(A|B) ≠ P(B|A). Toujours identifier quelle question on pose avant d'interpréter une probabilité conditionnelle.

---

## 6. Applications pratiques des fréquences naturelles

**Domaine médical :**
Un test positif lors d'un dépistage de masse (population générale, prévalence faible) signifie souvent moins de 20 % de probabilité réelle. Ce résultat justifie un test de confirmation, pas une décision immédiate.

**Contrôle qualité industriel :**
Un capteur à 95 % de sensibilité et 90 % de spécificité sur une ligne où 2 % des pièces sont défectueuses → VPP ≈ 16 %. Avant d'écarter une pièce, vérifier manuellement — plus de 8 pièces sur 10 signalées sont en réalité conformes.

**Systèmes de détection de fraude / spam :**
Avec une prévalence de fraude de 0,1 %, même un excellent algorithme (99,9 % de spécificité) génère autant de fausses alertes que de vraies détections. Toujours regarder la VPP, pas seulement la précision du modèle.

**Comment améliorer la VPP ?**

1. **Augmenter la prévalence** : tester des populations à risque élevé plutôt que la population générale.
2. **Améliorer la spécificité** : réduire les faux positifs améliore la VPP surtout quand la prévalence est faible.
3. **Chaîner les tests** : un deuxième test confirmatoire utilise la VPP du premier comme nouveau taux de base.

---

## Flash-cards (5)

**Q1 : Qu'est-ce qu'une fréquence naturelle et pourquoi facilite-t-elle le raisonnement probabiliste ?**
> R : Une fréquence naturelle exprime une probabilité en effectifs concrets (« 9 sur 59 » plutôt que « 15 % »). Elle réduit la charge cognitive et évite les erreurs de calcul en permettant de lire directement les résultats dans un tableau.

**Q2 : Un test a une sensibilité de 95 %. Cela signifie-t-il que si vous êtes positif, vous avez 95 % de chances d'être malade ?**
> R : Non. La sensibilité mesure P(test+ | malade). Ce qu'on veut savoir est P(malade | test+), la VPP, qui dépend aussi du taux de base. Avec une prévalence faible, la VPP peut être bien inférieure à 95 %.

**Q3 : Comment construire le tableau des fréquences naturelles en 4 étapes ?**
> R : (1) Choisir une population (ex. 1 000). (2) Répartir malades/sains selon la prévalence. (3) Appliquer sensibilité aux malades → VP et FN. (4) Appliquer spécificité aux sains → VN et FP. VPP = VP / (VP + FP).

**Q4 : Un test de dépistage a une VPP de 20 %. Que signifie ce chiffre concrètement ?**
> R : Sur 100 personnes testées positives, seulement 20 sont réellement atteintes. 80 reçoivent un faux positif. Ce résultat justifie un test de confirmation avant toute décision.

**Q5 : Comment le taux de base influence-t-il la valeur prédictive positive ?**
> R : Plus la prévalence est faible, plus les faux positifs dominent parmi les tests positifs, et plus la VPP chute. Même un très bon test produit majoritairement des faux positifs si la maladie est rare dans la population testée.

---

## Points clés à retenir

- Les fréquences naturelles (effectifs concrets) réduisent les erreurs de raisonnement par rapport aux probabilités exprimées en pourcentage.
- Le taux de base est le point de départ obligatoire : l'ignorer fausse complètement l'interprétation d'un test.
- Sensibilité ≠ VPP : ce sont deux questions différentes. P(test+ | malade) ≠ P(malade | test+).
- Avec une prévalence faible, même un excellent test produit majoritairement des faux positifs — c'est mathématique, pas un défaut du test.
- Le script `02-code/03-probabilites-frequences.py` calcule automatiquement VPP et VPN et affiche le tableau pour tout scénario.

---

## Pour aller plus loin

- **Gigerenzer, G. & Hoffrage, U.** (1995). *How to improve Bayesian reasoning without instruction: Frequency formats.* Psychological Review 102(4):684-704. — Article fondateur sur les fréquences naturelles ; montre empiriquement que le format effectifs > pourcentages pour les médecins et les profanes.
- **Gigerenzer, G.** (2002). *Reckoning with Risk: Learning to Live with Uncertainty.* Penguin Books. — Version grand public ; chapitres sur les tests médicaux et la communication du risque.
- **Peterson, M.** (2017). *An Introduction to Decision Theory* (2e éd.). Cambridge University Press. https://www.cambridge.org/core/books/an-introduction-to-decision-theory/B9EEB3DCE5D0CAFFB6F3F30B1D0A06A6 — Chapitre 4 : probabilité conditionnelle et cadre bayésien.
- **Script interactif** : `02-code/03-probabilites-frequences.py` — calculateur VPP/VPN avec tableau et comparaison multi-scénarios.
