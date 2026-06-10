# Exercices Hard — ML System Design Intro

---

## Exercice 1 : Plateforme ML interne multi-equipes (feature store + serving + gouvernance)

### Objectif
Concevoir une plateforme ML mutualisee qui sert plusieurs equipes avec des contraintes de latence et de coherence contradictoires.

### Consigne
Tu es staff engineer d'une scale-up (marketplace, 30M users). Trois equipes ML ont chacune bricole leur stack ; tu dois concevoir LA plateforme commune.

**Les 3 cas d'usage a servir :**
- **Fraude** : 3 000 predictions/sec, p99 < 30 ms, features fraiches < 1 min, cout d'un faux negatif tres eleve
- **Recommandations** : 15 000 predictions/sec en pic, p99 < 100 ms, features fraiches < 1 h acceptables
- **Pricing dynamique** : batch quotidien sur 50M d'annonces, fenetre de calcul de 4 h max

**Contraintes chiffrees :**
- 120 features partagees entre les equipes (ex : historique user) + features specifiques
- Budget infra plateforme : 60 000 $/mois
- 2 incidents de training-serving skew le trimestre dernier (perte estimee 400 K$)
- Equipe plateforme : 5 ingenieurs

**Livre :**
1. **Architecture du feature store** : online store (techno + dimensionnement : 18 000 lookups/sec en pic combine, < 10 ms), offline store, pipeline de materialisation (batch + streaming pour les features < 1 min). Comment les features partagees sont-elles gouvernees (ownership, definitions, SLA par feature) ?
2. **Dimensionnement online store** : 30M users x 120 features x ~50 octets = combien de RAM ? Avec replication ? Quel cout estime (pose tes hypotheses de prix) et tient-il dans le budget global ?
3. **Serving** : une infra de serving unique (multi-model) ou par equipe ? Compare contre les SLOs (30 ms vs 100 ms vs batch) et l'equipe de 5.
4. **Anti-skew structurel** : concois le mecanisme qui rend le skew IMPOSSIBLE par construction (definitions uniques, point-in-time correctness pour le training, tests de parite online/offline en CI). Explique le piege du data leakage temporel dans la generation des training sets et comment le point-in-time join le resout.
5. **Multi-tenancy et noisy neighbor** : la fraude (30 ms) partage l'online store avec les recos (15K lookups/s). Comment proteger le p99 de la fraude ?
6. **Plan de migration** : les 3 equipes ont des stacks existantes. Sequence la migration (qui en premier ? pourquoi ?) et definis les criteres de succes a 6 mois.

### Criteres de reussite
- [ ] Online store dimensionne : ~180 Go de donnees brutes (30M x 120 x 50 o), ~400-500 Go avec replication et overhead -> cluster Redis/DynamoDB chiffre avec hypotheses de prix explicites
- [ ] La materialisation distingue batch (pricing, quotidien) et streaming (fraude, < 1 min) avec UN SEUL code de definition des features (transformations partagees)
- [ ] Le point-in-time correctness est explique avec un exemple concret de leakage (feature calculee apres l'evenement a predire) et le mecanisme de join temporel
- [ ] Le serving est differencie : pool dedie basse latence pour la fraude, pool mutualise pour les recos, batch sur infra ephemere pour le pricing — justifie par les SLOs
- [ ] Le noisy neighbor a une reponse concrete : rate limiting par client, replicas dedies lecture fraude, ou clusters separes avec le tradeoff cout
- [ ] La migration commence par l'equipe au meilleur ratio risque/demonstration (souvent recos), avec criteres mesurables (0 skew incident, latences SLO, adoption)
- [ ] Au moins 3 tradeoffs explicites avec consequences acceptees, et le budget total est confronte aux 60 K$/mois

---

## Exercice 2 : Sauver un systeme de matching qui se degrade silencieusement

### Objectif
Diagnostiquer une degradation ML en production sans cause evidente, et concevoir le systeme qui l'aurait detectee — sous pression business.

### Consigne
Tu arrives comme tech lead chez un acteur de la livraison. Le modele de matching coursier/commande (en prod depuis 18 mois) semble se degrader : le temps de livraison moyen a augmente de 12% en 6 mois, mais PERSONNE ne sait si c'est le modele, la ville qui a change, ou la flotte.

**Etat des lieux chiffre :**
- Le modele n'a JAMAIS ete re-entraine depuis son deploiement (18 mois)
- Aucun monitoring ML : seulement les metriques infra (CPU, latence API)
- Le training set initial : 6 mois de donnees de 3 villes ; aujourd'hui le service couvre 12 villes
- 800 K matchings/jour ; chaque point de % de temps de livraison ~ 150 K$/an de churn estime
- La direction veut "un fix en 4 semaines", l'equipe data est 3 personnes

**Livre :**
1. **Diagnostic structure** : liste les 4 hypotheses de degradation (drift de distribution, concept drift, degradation des donnees d'entree, facteur externe non-ML) et pour CHACUNE le test concret qui la confirme/infirme avec les donnees disponibles (logs de prod 18 mois).
2. **Quantification du drift** : explique comment tu calcules retroactivement le PSI des features cles entre le training set (3 villes, vieux de 18 mois) et la prod actuelle (12 villes). Quels resultats attends-tu et pourquoi le drift est ici quasi certain ?
3. **Le piege du retraining naif** : "on re-entraine sur les 6 derniers mois et on redeploie" — donne 3 raisons pour lesquelles ca peut AGGRAVER la situation (boucle de feedback : le training set contient les decisions du modele degrade ; nouvelles villes sous-representees ; pas de baseline d'evaluation fiable).
4. **Plan 4 semaines** : sequence realiste semaine par semaine pour 3 personnes : instrumentation minimale d'abord ? backtest ? retraining cible ? Quel livrable a J+28 et qu'est-ce qui est explicitement REPORTE ?
5. **Le systeme cible** : concois le monitoring qui aurait detecte le probleme au mois 3 au lieu du mois 18 : metriques (proxy metrics vs delayed labels), seuils, dashboards, politique de retraining automatique. Chiffre le cout de ce systeme face aux 150 K$/an/point.
6. **Boucle de feedback** : le matching influence les donnees futures (un mauvais matching change le comportement des coursiers). Propose une strategie pour entrainer sans amplifier le biais (exploration/randomisation partielle, holdout geographique).

### Criteres de reussite
- [ ] Les 4 hypotheses ont chacune un test concret et faisable avec les logs (comparaison de distributions, performance par cohorte ville/anciennete, completude des features, saisonnalite)
- [ ] Le PSI retroactif est decrit methodiquement (binning sur le training set, comparaison par fenetres mensuelles) avec l'attente argumentee d'un drift majeur (9 villes jamais vues = hors distribution)
- [ ] Les 3 pieges du retraining naif sont identifies, dont la boucle de feedback (le modele genere ses propres training data)
- [ ] Le plan 4 semaines est realiste et priorise : semaine 1 = instrumentation + backtest baseline simple (heuristique distance) pour situer le modele ; un livrable concret a J+28 ; le reste explicitement reporte
- [ ] Le monitoring cible utilise des proxy metrics immediates (distance des matchings, taux de refus coursier) en plus des labels retardes, avec seuils et retraining declenche
- [ ] L'exploration controlee (ex : 2-5% de matchings randomises ou holdout) est proposee pour casser la boucle de feedback, avec son cout assume
