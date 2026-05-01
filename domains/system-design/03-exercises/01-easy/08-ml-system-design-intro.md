# Exercices Easy — ML System Design Intro

---

## Exercice 1 : Diagnostic training-serving skew

### Objectif
Savoir identifier la cause d'un skew a partir de symptomes terrain.

### Consigne
Pour chacune des situations suivantes, identifie la cause probable du training-serving skew et propose une mitigation concrete :

1. Un modele de detection de fraude atteint 95% de precision en offline mais chute a 70% en production. Les data scientists ont entraine avec un notebook Python qui lit depuis un dump CSV, et l'application appelle une feature calculee en SQL dans une UDF PostgreSQL.
2. Un modele de churn prediction a d'excellentes metriques sur le dernier mois mais s'effondre apres 6 mois en production sans avoir ete re-entraine.
3. Un modele de recommandation utilise la feature `last_order_amount`. En training, les NaN sont remplies par la moyenne. En serving, les NaN sont envoyees brutes au modele.
4. Un modele utilise la feature `average_review_score` d'un produit. En entrainement sur des donnees de 2024, il utilise la moyenne finale de toutes les reviews du produit, meme celles deposees apres la date de l'evenement.

### Livrables
Un tableau avec 3 colonnes : situation / cause / mitigation.

### Criteres de reussite
- [ ] Les 4 situations sont correctement categorisees (code divergent, drift, NaN handling, data leakage)
- [ ] Au moins 2 mitigations citent explicitement un feature store
- [ ] La notion de point-in-time correctness apparait pour le cas 4
- [ ] Le cas 2 mentionne monitoring + retraining automatique

---

## Exercice 2 : Batch ou real-time ?

### Objectif
Choisir la bonne architecture d'inference selon le use case.

### Consigne
Pour chacun de ces produits, indique si tu ferais du **batch**, du **real-time**, ou un **hybride**, et justifie en 2-3 phrases en citant au moins une contrainte chiffree (latence cible, volume, cout).

1. Un moteur de recommandation Netflix (homepage avec 50 lignes de contenus personnalises)
2. Un systeme anti-fraude pour une plateforme de paiement (1000 tx/s, refus bloquant)
3. Un generateur de resumes d'emails quotidiens envoyes le matin
4. Un pricing dynamique pour Uber
5. Un systeme de detection d'anomalies sur des capteurs IoT (5M capteurs, remontee 1x/min)
6. Un correcteur orthographique integre dans un IDE

### Livrables
Un petit tableau : produit / archi / latence cible / justification.

### Criteres de reussite
- [ ] Au moins 1 batch, 1 real-time, 1 hybride
- [ ] Les latences cibles sont realistes (anti-fraude < 100ms, pricing Uber < 500ms, etc.)
- [ ] Netflix est identifie comme hybride (batch pour la majorite + re-ranking online)
- [ ] Le correcteur orthographique est real-time (exige par l'UX)
- [ ] Le volume est mentionne au moins 2 fois comme argument pour batch

---

## Exercice 3 : Dessiner un feature store pour un cas reel

### Objectif
Traduire un probleme metier en architecture feature store concrete.

### Consigne
Tu travailles chez un e-commerce de mode. L'equipe data veut deployer un modele de recommandation de produits. Les features necessaires sont :

- `user_avg_order_value_30d` (moyenne glissante sur 30 jours)
- `user_fav_category` (categorie la plus achetee, recalculee quotidiennement)
- `user_current_cart_size` (en temps reel, met a jour a chaque clic)
- `product_sales_rank_7d` (rang du produit sur les ventes des 7 derniers jours)
- `user_device_type` (desktop/mobile, donnee a la requete)

**Questions :**
1. Pour chaque feature, indique : source (batch/stream/on-demand), stockage offline (ou ?), stockage online (quelle techno et pourquoi ?), TTL approprie.
2. Dessine (en ASCII ou pseudo-schema) le pipeline complet, de la source au model serving.
3. Quel est le point-in-time le plus delicat ? Explique pourquoi.
4. Si Redis online est down, quelle est ta strategie de degradation ?

### Livrables
Un doc markdown avec le tableau des features, le schema, et les reponses aux questions 3 et 4.

### Criteres de reussite
- [ ] `user_current_cart_size` est clairement identifie comme stream ou on-demand
- [ ] `user_avg_order_value_30d` est en batch, avec offline=Parquet/BQ et online=Redis/DynamoDB
- [ ] `user_device_type` n'est PAS stocke (feature on-demand)
- [ ] Le schema montre separation offline/online + materialisation vers l'online
- [ ] La strategie de degradation mentionne : fallback vers features par defaut OU fallback vers un modele plus simple (ex: top sellers)
- [ ] Le point-in-time delicat est identifie (probablement `product_sales_rank_7d` car il varie vite)
