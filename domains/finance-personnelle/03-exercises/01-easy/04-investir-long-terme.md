# Exercices — Module 04 : Investir simplement et sur le long terme

> Ces exercices sont conçus pour ancrer les concepts du Module 04 par la pratique concrète. Progressifs : du calcul d'impact des frais à la construction d'une allocation simulée.

---

## Exercice 1 — L'impact des frais sur 20 ans

### Objectif
Visualiser et quantifier l'effet dévastateur des frais composés sur le long terme, sans outil externe.

### Consigne

Vous disposez de 5 000 € de capital initial et vous investissez **150 €/mois** pendant **20 ans**.  
Hypothèse de rendement brut : **7 % par an** (composé annuellement pour simplifier).

**Partie A** : Calculez (à la main ou avec une calculatrice simple) le capital final pour deux scénarios :
- Scénario 1 : frais annuels de **0,20 %** (fonds indiciel type)
- Scénario 2 : frais annuels de **1,80 %** (fonds actif courant)

*Méthode simplifiée* : utilisez le rendement net = rendement brut − frais annuels, puis appliquez la formule de capitalisation (ou le simulateur Python du domaine : `02-code/04-investir-long-terme.py`).

**Partie B** : Calculez la différence en euros entre les deux scénarios. En quel pourcentage le scénario à hauts frais réduit-il le capital final par rapport au scénario à bas frais ?

**Partie C** : En vous basant sur les données SPIVA (Module 04, §4), argumentez en 3-4 phrases pourquoi payer des frais plus élevés pour une gestion active ne se justifie généralement pas sur 20 ans.

### Critères de réussite
- [ ] Capital final calculé pour les deux scénarios avec la méthode choisie
- [ ] Différence en euros et en pourcentage correctement calculée
- [ ] Argumentation s'appuyant sur les données SPIVA et l'arithmétique de Sharpe
- [ ] Aucune source citée inventée — uniquement celles du module

---

## Exercice 2 — Lire et comparer des fonds

### Objectif
S'entraîner à identifier les informations clés d'un fonds (TER, indice répliqué, diversification) pour comparer deux options.

### Consigne

Imaginez que vous comparez deux fonds disponibles dans une enveloppe d'investissement hypothétique :

**Fonds A** — Fonds indiciel MSCI World
- Indice répliqué : MSCI World (~1 500 entreprises, 23 pays développés)
- TER : 0,12 %/an
- Mode de gestion : passif (réplication physique)
- Répartition : ~70 % Amérique du Nord, ~15 % Europe, ~8 % Asie-Pacifique, ~7 % autres

**Fonds B** — Fonds actions internationales géré activement
- Univers d'investissement : actions mondiales
- TER : 2,10 %/an
- Mode de gestion : actif (sélection de titres)
- Performance sur 5 ans : +8,2 %/an (vs. MSCI World : +9,1 %/an sur la même période)

**Questions :**
1. Calculez l'impact des frais sur 10 ans pour 10 000 € investis, sans versements complémentaires, en utilisant un rendement brut de 8 % pour les deux (pour comparer uniquement l'effet des frais).
2. Le Fonds B a fait +8,2 %/an sur 5 ans. Est-ce une bonne performance ? Que manque-t-il comme information pour juger ?
3. Vous souhaitez une exposition internationale diversifiée à moindre coût. Lequel choisissez-vous et pourquoi ? (3-4 phrases)

### Critères de réussite
- [ ] Calcul de l'impact des frais sur 10 ans pour les deux fonds
- [ ] Identification des informations manquantes (comparaison nette de frais, persistance, horizon)
- [ ] Choix argumenté sur des critères objectifs (frais, diversification, cohérence avec l'horizon)
- [ ] Pas de jugement de valeur sans données — posture factuelle

---

## Exercice 3 — Construire une allocation "3 fonds" simulée

### Objectif
Appliquer le concept d'allocation "3 fonds" à un profil fictif, et calculer la projection sur 25 ans.

### Consigne

**Profil fictif** : Camille, 35 ans, épargne 400 €/mois pour l'investissement long terme, capital déjà investi : 8 000 €. Horizon : 25 ans. Tolérance au risque : modérée.

**Partie A** : Proposez une allocation "3 fonds" pour Camille (en %). Justifiez en 2-3 phrases pourquoi vous avez choisi ces proportions compte tenu de l'horizon et de la tolérance au risque.

**Partie B** : Calculez la projection à 25 ans en utilisant :
- Rendement hypothétique actions : 7 %/an
- Rendement hypothétique obligations : 3 %/an
- Frais moyens des fonds : 0,20 %/an

*Méthode* : vous pouvez utiliser le simulateur `02-code/04-investir-long-terme.py` ou calculer manuellement avec le rendement pondéré de l'allocation.

**Partie C** : Si Camille réduit ses frais de 0,20 % à 0,10 %/an (en cherchant des fonds moins chers), quelle est la différence de capital final sur 25 ans ? En quel pourcentage ?

### Critères de réussite
- [ ] Allocation "3 fonds" proposée avec des pourcentages cohérents (total = 100 %)
- [ ] Justification de l'allocation par rapport à l'âge, l'horizon et la tolérance au risque
- [ ] Projection à 25 ans calculée avec les hypothèses données
- [ ] Comparaison de l'impact d'un dixième de point de frais en moins sur 25 ans
