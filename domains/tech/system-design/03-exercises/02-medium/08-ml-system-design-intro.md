# Exercices Medium — ML System Design Intro

---

## Exercice 1 : Point-in-time correctness — debusquer le data leakage

### Objectif
Comprendre concretement comment un join naif fuite le futur et chiffrer l'illusion de performance qui en resulte.

### Consigne
Tu construis un dataset d'entrainement pour un modele qui predit, **au moment d'un clic** (`event_time`), si l'user va convertir. Tu joins une feature `user_total_purchases` depuis une table `purchases` qui s'incremente dans le temps.

**Donnees pour un user :**
- Clic (event) a t = 2026-01-10, label = "n'a pas converti dans les 7 jours"
- Achats de l'user : 2026-01-05 (1 achat), 2026-01-20 (3 achats), 2026-02-01 (2 achats)

**Questions :**
1. Un join naif `SELECT count(*) FROM purchases WHERE user_id = X` (etat d'aujourd'hui) donne quelle valeur de `user_total_purchases` pour l'exemple du 2026-01-10 ?
2. Quelle est la valeur **correcte** (point-in-time) que le modele aurait vue en prod le 2026-01-10 ?
3. Explique en une phrase pourquoi la valeur naive fait fuiter le futur (data leakage).
4. En quoi ce leakage donne une AUC offline excellente mais une chute en prod ? Quel est le mecanisme ?
5. Comment un feature store resout ca (point-in-time join / time-travel) ? Que doit stocker l'offline store pour le permettre ?
6. Donne un test automatique simple qui aurait attrape ce bug en CI (indice : recalcule la feature a `event_time` et compare).

### Criteres de reussite
- [ ] Join naif -> 6 (1+3+2, inclut des achats POSTERIEURS au clic)
- [ ] Valeur correcte point-in-time -> 1 (seul l'achat du 2026-01-05 existait avant le clic)
- [ ] Le leakage est explique : la feature voit des valeurs du futur (apres event_time)
- [ ] Mecanisme : le modele "triche" en offline (correlation avec le futur), info absente en prod -> chute
- [ ] Le feature store fait un point-in-time join (as-of join) ; l'offline store garde l'historique horodate des valeurs
- [ ] Le test CI recalcule la feature a event_time et assert == valeur attendue (catch leakage)

---

## Exercice 2 : Choisir batch vs real-time avec le cout chiffre

### Objectif
Arbitrer batch / real-time / micro-batch en chiffrant cout et fraicheur, pas a l'instinct.

### Consigne
Un e-commerce a 50M users. Il veut des recommandations produit. Deux options :
- **Batch** : recalculer 50 recos/user toutes les 6h via un job Spark. Cout du job : ~$200 par run.
- **Real-time** : scorer a la requete, 1 GPU-inference par home view. 50M users * 5 home views/jour, $0.0002 par scoring.

**Questions :**
1. Cout/jour du batch (4 runs/jour) vs cout/jour du real-time (50M * 5 * $0.0002). Lequel est moins cher ?
2. Quelle est la fraicheur des recos en batch (delai max) vs real-time ?
3. Pour un user qui vient de cliquer "j'aime les chaussures de running", combien de temps avant que le batch en tienne compte ? Et le real-time ?
4. Propose un **hybride** : qu'est-ce qui reste batch (candidate generation) et qu'est-ce qui passe online (re-ranking selon le contexte du moment) ? Pourquoi cet ordre ?
5. Le micro-batch (toutes les N secondes) : dans quel cas l'utiliserais-tu ici ? Donne la contrainte qui le justifie.

### Criteres de reussite
- [ ] Batch : 4 * $200 = $800/jour. Real-time : 50M * 5 * $0.0002 = $50000/jour -> batch ~60x moins cher
- [ ] Fraicheur batch : jusqu'a 6h de retard ; real-time : immediat
- [ ] Le clic "running" : batch jusqu'a 6h ; real-time visible des le prochain home view
- [ ] Hybride : candidate generation en batch (lourd, stable) + re-ranking online (leger, contextuel) -> meilleur ratio cout/fraicheur
- [ ] Micro-batch justifie quand le cout par requete est prohibitif mais qu'on veut une latence de quelques secondes (ex: scoring groupe)

---

## Exercice 3 : Shadow puis canary — sizing du rollout sans risque

### Objectif
Dessiner la sequence shadow -> canary -> 100% et chiffrer ce qu'on observe a chaque etape.

### Consigne
Tu veux promouvoir un modele V2 (AUC offline 0.92 vs V1 0.91). Trafic : 100M predictions/jour. Tu n'as PAS le droit de degrader l'experience.

**Questions :**
1. Decris le shadow deployment : que recoit V2, qu'est-ce qui est renvoye au user, qu'est-ce qu'on logge ?
2. Pendant le shadow, quels 3 signaux tu mesures AVANT meme de parler de qualite metier (indice : ca ne crash pas, ca tient la latence, les predictions ne divergent pas n'importe comment) ?
3. Le shadow montre que V2 "disagree" avec V1 sur 18% des cas. Est-ce alarmant en soi ? Que dois-tu verifier ?
4. Canary 1% -> 10% -> 50% -> 100% : a 1% de 100M = combien de predictions/jour servent reellement V2 ? Pourquoi commencer si bas ?
5. Quels signaux declenchent un **rollback automatique** pendant le canary ? Cite des guardrails (latence p99, error rate, business metric).
6. Pourquoi l'AUC offline superieure ne suffit PAS pour promouvoir directement en prod ?

### Criteres de reussite
- [ ] Shadow : V2 recoit le trafic reel, sa reponse N'EST PAS envoyee au user, predictions loggees + comparees a V1
- [ ] Les 3 signaux pre-qualite : pas de crash, latence/ressources OK, distribution des predictions raisonnable (disagreement rate)
- [ ] 18% disagreement n'est pas alarmant en soi -> verifier si V2 a raison sur les cas de desaccord (eval sur labels)
- [ ] Canary 1% de 100M = 1M predictions/jour servies reellement ; on commence bas pour limiter le blast radius
- [ ] Rollback auto sur degradation de guardrails : p99 latency, error rate, business metric (CTR/conversion)
- [ ] L'AUC offline ne correle pas parfaitement au business -> shadow + canary + mesure online obligatoires
