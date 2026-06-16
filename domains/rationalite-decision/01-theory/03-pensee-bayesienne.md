# Module 03 — Pensee bayesienne et mise a jour

> **Temps estime** : 45 min | **Prerequis** : Modules 01-02
> **Objectif** : comprendre et appliquer la mise a jour bayesienne — changer d'avis proportionnellement aux preuves, ni trop ni trop peu.

---

## 1. Commencer par une intuition

Imaginez que vous partez un matin sans regarder par la fenetre. Quelle est la probabilite qu'il pleuve aujourd'hui ? Si vous etes a Paris en novembre, peut-etre 40 %.

Vous ouvrez la fenetre : le ciel est gris et couvert. Quelle est maintenant la probabilite de pluie ? Peut-etre 70 %.

Vous entendez le son de la pluie sur le toit. Maintenant ? 95 %.

Ce processus — partir d'une croyance initiale, la mettre a jour avec chaque nouvelle observation, obtenir une croyance revisee — est exactement le **raisonnement bayesien**.

La pensee bayesienne repond a une question simple : **quand une nouvelle preuve arrive, de combien dois-je changer d'avis ?**

---

## 2. Les trois ingredients

### Prior (P(H)) : votre croyance avant la preuve

Le **prior** est votre estimation de probabilite pour une hypothese H *avant* d'observer une nouvelle donnee. Il encode ce que vous saviez deja.

Exemple : avant de voir le resultat d'un test medical, la probabilite d'etre malade = taux de base de la maladie dans la population = votre prior.

Le prior peut venir de :
- Statistiques historiques (taux de base)
- Expertise personnelle
- Raisonnement par classe de reference

**Piege frequent** : sous-estimer le prior (base rate neglect, vu au module 02) ou l'ignorer completement.

### Vraisemblance (P(E|H)) : la force de la preuve

La **vraisemblance** est la probabilite d'observer la preuve E si l'hypothese H est vraie.

Elle repond a la question : "Si H etait vraie, quelle serait la probabilite de voir cette evidence ?"

Exemple : si vous etes malade (H = malade), quelle est la probabilite que le test soit positif ? = sensibilite = P(test+ | malade).

### Posterior (P(H|E)) : votre croyance apres la preuve

Le **posterior** est la probabilite de l'hypothese H apres avoir observe la preuve E. C'est le resultat du raisonnement bayesien.

```
Posterior = Prior × Vraisemblance / Normalisation
P(H|E) = P(E|H) × P(H) / P(E)
```

Le prior du lendemain = le posterior d'aujourd'hui. C'est ainsi que la croyance evolue de facon continue avec l'accumulation des preuves.

---

## 3. La formule de Bayes : deux formes

### Forme probabiliste (formule complete)

```
P(H|E) = P(E|H) × P(H) / [P(E|H) × P(H) + P(E|non-H) × P(non-H)]
```

Que l'on peut ecrire de facon plus compacte :

```
P(H|E) = P(E|H) × P(H) / P(E)
```

Ou P(E) est la probabilite totale d'observer la preuve E (en sommant sur toutes les hypotheses possibles).

### Forme odds (souvent plus intuitive)

Les **odds** (cotes) d'une hypothese H sont definis comme :

```
Odds(H) = P(H) / P(non-H)
```

Par exemple, si P(H) = 0,25, alors Odds(H) = 0,25 / 0,75 = 1/3 (ou "1 contre 3").

La mise a jour bayesienne devient alors multiplicative :

```
Odds posterieur = Odds prieur × Rapport de vraisemblance

Rapport de vraisemblance (LR) = P(E|H) / P(E|non-H)
```

**Avantage** : on n'a qu'a multiplier. Si LR > 1, la preuve soutient H. Si LR < 1, la preuve affaiblit H.

---

## 4. Un exemple chiffre : les urnes

> Il y a deux urnes. Urne A contient 70 billes rouges et 30 billes bleues. Urne B contient 40 billes rouges et 60 billes bleues. On choisit une urne au hasard (probabilite 50/50) et on tire une bille rouge. Quelle est la probabilite que la bille vienne de l'urne A ?

**Prior** : P(Urne A) = 0,50

**Vraisemblances** :
- P(rouge | Urne A) = 0,70
- P(rouge | Urne B) = 0,40

**P(rouge)** = 0,70 × 0,50 + 0,40 × 0,50 = 0,35 + 0,20 = 0,55

**Posterior** :
```
P(Urne A | rouge) = P(rouge | Urne A) × P(Urne A) / P(rouge)
                  = 0,70 × 0,50 / 0,55
                  = 0,35 / 0,55
                  ≈ 0,636 ≈ 64 %
```

La bille rouge augmente notre confiance dans l'urne A de 50 % (prior) a 64 % (posterior). Si on tire une deuxieme bille rouge, on repart avec P(Urne A) = 64 % comme nouveau prior.

---

## 5. Mise a jour sequentielle

Une propriete puissante du raisonnement bayesien : **on peut mettre a jour en sequence**. Chaque posterior devient le prior de la mise a jour suivante.

```
Prior → [preuve 1] → Posterior 1 → [preuve 2] → Posterior 2 → ...
```

Exemple : test medical avec deux tests independants.

**Situation** : maladie a 1 % de prevalence. Sensibilite 90 %, specificite 95 %.

**Apres le premier test positif** (calcul du module 02) : P(malade) ≈ 15,4 %

**Apres un second test positif independant** (on reutilise 15,4 % comme nouveau prior) :
```
P(test+ | malade) = 0,90
P(test+ | sain) = 0,05
P(malade) = 0,154

P(test+) = 0,90 × 0,154 + 0,05 × 0,846 = 0,1386 + 0,0423 = 0,1809

P(malade | 2eme test+) = 0,90 × 0,154 / 0,1809 = 0,1386 / 0,1809 ≈ 76,6 %
```

Apres deux tests positifs independants, la probabilite passe de 1 % a ~77 %. Chaque preuve s'accumule de facon quantitative.

> **A retenir** : la mise a jour sequentielle est la methode formelle pour "changer d'avis proportionnellement aux preuves". Chaque nouvelle observation ne repart pas de zero — elle s'appuie sur toutes les preuves precedentes encodees dans le prior.

---

## 6. Rapport de vraisemblance : evaluer la force d'une preuve

Le **rapport de vraisemblance (LR)** mesure combien de fois une preuve est plus probable si H est vraie que si H est fausse :

```
LR = P(E|H) / P(E|non-H)
```

Interpretations :
- LR = 1 → la preuve n'informe pas (egalement probable dans les deux cas)
- LR = 10 → la preuve est 10× plus probable si H est vraie (preuve forte)
- LR = 0,1 → la preuve est 10× moins probable si H est vraie (preuve contre H)

Exemple (test medical) :
```
LR = sensibilite / (1 - specificite) = 0,90 / 0,05 = 18
```

Un test positif est 18 fois plus probable chez un malade que chez un sain. C'est un LR fort — mais pas suffisant pour dominer un taux de base tres faible.

---

## 7. Ce que la pensee bayesienne n'est pas

**Ce n'est pas du relativisme** : deux personnes partant du meme prior et observant la meme preuve arriveront au meme posterior. Le processus est objectif.

**Ce n'est pas du subjectivisme pur** : les priors doivent etre ancres dans des donnees (taux de base, statistiques historiques). Un prior arbitraire produit un posterior arbitraire.

**Ce n'est pas toujours optimal** : quand les preuves sont ambigues ou les categories mal definies, le cadre bayesien peut etre mal applique. Mais c'est generalement mieux que de ne pas quantifier du tout.

---

## Flash-cards (5)

**Q1** : Qu'est-ce qu'un "prior" dans le raisonnement bayesien ?
**R1** : La probabilite d'une hypothese H *avant* d'observer la nouvelle preuve. Il encode ce que l'on sait deja (souvent le taux de base ou une expertise anterieure).

**Q2** : Comment calcule-t-on le posterior avec la forme probabiliste de Bayes ?
**R2** : P(H|E) = P(E|H) × P(H) / P(E). Le posterior est proportionnel au produit du prior et de la vraisemblance.

**Q3** : Qu'est-ce que le rapport de vraisemblance (LR) et que signifie LR = 1 ?
**R3** : LR = P(E|H) / P(E|non-H). LR = 1 signifie que la preuve est egalement probable si H est vraie ou fausse : elle n'apporte aucune information nouvelle.

**Q4** : Comment fonctionne la mise a jour sequentielle ?
**R4** : Chaque posterior devient le prior de la mise a jour suivante. On n'a pas besoin de tout recalculer depuis le debut : l'historique des preuves est encode dans le prior courant.

**Q5** : Quelle est la forme odds de la mise a jour bayesienne ?
**R5** : Odds(H|E) = Odds(H) × LR. On multiplie les odds prieur par le rapport de vraisemblance pour obtenir les odds posterieur. C'est souvent plus rapide que la formule probabiliste.

---

## Points cles a retenir

- La pensee bayesienne repond a : "de combien changer d'avis quand une preuve arrive ?"
- Prior × Vraisemblance → Posterior. Le prior du lendemain est le posterior d'aujourd'hui.
- La forme odds (Odds posterieur = Odds prieur × LR) est souvent plus rapide a calculer.
- La mise a jour sequentielle permet d'accumuler les preuves de facon quantitative et coherente.
- Le raisonnement bayesien n'est pas du relativisme : a priors et preuves identiques, deux personnes arrivent au meme posterior.
- Le script `02-code/03-pensee-bayesienne.py` permet d'experimenter la mise a jour bayesienne interactivement.

---

## Pour aller plus loin

- **Peterson, M.** (2017). *An Introduction to Decision Theory* (2e ed.). Cambridge University Press. https://www.cambridge.org/core/books/an-introduction-to-decision-theory/B9EEB3DCE5D0CAFFB6F3F30B1D0A06A6 — Section bayesianisme, ch. 4.
- **Stanford Encyclopedia of Philosophy** — Normative Theories of Rational Choice. https://plato.stanford.edu/entries/rationality-normative-utility/ — Expose rigoureux et libre d'acces.
- **Script interactif** : `02-code/03-pensee-bayesienne.py` — mise a jour bayesienne (prior/likelihood → posterior) avec mode pas-a-pas.
