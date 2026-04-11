# Exercices Medium — Principes fondamentaux

---

## Exercice 1 : Estimation complete — Service de stockage de photos

### Objectif
Mener une estimation back-of-the-envelope complete pour un systeme reel, avec tous les axes (QPS, stockage, bande passante, nombre de serveurs).

### Consigne
Tu dois estimer les besoins pour un service de partage de photos (type Instagram simplifie) :

**Hypotheses de depart :**
- 100 millions de DAU
- 20% des utilisateurs uploadent 2 photos/jour en moyenne
- 80% des utilisateurs consultent leur feed (30 photos par chargement, 5 chargements/jour)
- Taille moyenne d'une photo : 2 Mo (apres compression)
- Metadata par photo : 500 octets
- Retention : illimitee

**Calcule :**
1. QPS en ecriture (upload) — moyen et pic
2. QPS en lecture (feed) — moyen et pic
3. Ratio lecture/ecriture
4. Bande passante en ecriture et en lecture (pic)
5. Stockage necessaire pour 1 an et 5 ans
6. Si un serveur gere 10 000 req/s, combien de serveurs pour le pic de lecture ?

**Bonus** : Identifie quel est le bottleneck principal (CPU, stockage, reseau ?) et propose une solution.

### Criteres de reussite
- [ ] Tous les calculs sont presents et corrects a l'ordre de grandeur pres
- [ ] Le ratio lecture/ecriture est calcule et commente (~200:1 ou plus)
- [ ] Le stockage est en petaoctets pour 5 ans
- [ ] Le bottleneck est identifie avec justification
- [ ] La solution proposee est coherente (CDN, cache, etc.)

---

## Exercice 2 : Tradeoff Analysis — Choisir la bonne DB

### Objectif
Appliquer les principes CAP et consistency models pour choisir une base de donnees adaptee.

### Consigne
Tu concois un systeme e-commerce. Il gere 3 types de donnees :

| Donnee | Caracteristiques |
|---|---|
| **Catalogue produits** | 10M de produits, lu 1000x plus que modifie, structure semi-flexible (attributs variables par categorie) |
| **Commandes** | Transactions critiques, integrite referentielle, ~50K commandes/jour |
| **Sessions utilisateur** | TTL de 30 min, acces tres frequents, perte tolerable |

Pour chaque type de donnee :
1. Choisis une base de donnees (PostgreSQL, MongoDB, Redis, Cassandra, DynamoDB — justifie)
2. Indique le consistency model necessaire (strong, eventual, causal)
3. Explique pourquoi les autres choix seraient inferieurs
4. Identifie le risque principal de ton choix et la mitigation

### Criteres de reussite
- [ ] 3 choix de DB distincts et justifies
- [ ] Consistency model correct pour chaque use case
- [ ] Au moins 1 alternative ecartee avec explication pour chaque choix
- [ ] Risques identifies sont realistes (pas generiques)

---

## Exercice 3 : Latency Budget — Decomposition d'un appel API

### Objectif
Comprendre comment la latence se decompose dans un systeme distribue et comment respecter un SLO.

### Consigne
Ton endpoint `/api/v1/product/{id}` a un SLO de **200ms au p99**. Voici l'architecture actuelle :

```
Client -> API Gateway (5ms) -> Auth Service (15ms) -> Product Service -> Response
                                                          |
                                                    PostgreSQL (20ms)
                                                    Redis Cache (2ms, hit rate 80%)
                                                    Recommendation Service (100ms)
                                                    Price Service (30ms)
```

1. Calcule la latence p99 dans le cas **sans cache** (tous les appels sequentiels)
2. Calcule la latence p99 dans le cas **avec cache** (hit rate 80%)
3. Le SLO de 200ms est-il respecte ? Si non, propose des optimisations.
4. Redesigne le flow pour respecter le SLO. Indique quels appels peuvent etre parallelises.
5. Calcule la nouvelle latence p99 apres tes optimisations.

**Rappel** : p99 signifie le pire cas raisonnable. Pour les services avec cache, p99 = cas cache miss.

### Criteres de reussite
- [ ] Latences calculees correctement (sequentiel = somme, parallele = max)
- [ ] La distinction cache hit vs cache miss est bien prise en compte pour le p99
- [ ] Au moins 2 optimisations proposees (parallelisation, cache, etc.)
- [ ] La latence finale respecte le SLO de 200ms
- [ ] Le raisonnement montre la comprehension que p99 = worst case (cache miss)
