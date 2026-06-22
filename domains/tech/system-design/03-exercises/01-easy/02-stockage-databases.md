# Exercices Easy — Stockage & Databases

---

## Exercice 1 : SQL ou NoSQL — Choisis ta DB

### Objectif
Savoir identifier rapidement le type de base de donnees adapte a un use case.

### Consigne
Pour chacun des systemes suivants, indique si tu choisirais **SQL** ou **NoSQL** (en precisant le type : key-value, document, column-family, graph). Justifie en une phrase.

1. Un systeme de gestion de paie (salaires, bulletins, cotisations)
2. Un cache de resultats de recherche avec TTL de 5 minutes
3. Un catalogue de recettes de cuisine ou chaque recette a des attributs differents (temps de cuisson, allergenes, ingredients en quantites variables)
4. Un systeme de recommandation "les utilisateurs qui ont achete X ont aussi achete Y"
5. Un service de metriques IoT recevant 200K events/sec de 50K capteurs
6. Un backoffice de gestion de contrats d'assurance

### Criteres de reussite
- [ ] 6/6 choix corrects avec le type precis
- [ ] Chaque justification mentionne la contrainte cle qui oriente le choix
- [ ] Au moins une reponse mentionne un tradeoff accepte

---

## Exercice 2 : Index ou pas index ?

### Objectif
Comprendre quand un index est benefique et quand il est contre-productif.

### Consigne
Pour chacune des situations suivantes, indique si tu creerais un index et justifie :

1. Table `users` (50M lignes), colonne `email`, utilisee dans `WHERE email = ?` a chaque login
2. Table `logs` (500M lignes), colonne `level` qui prend 3 valeurs : INFO, WARN, ERROR. La table recoit 10K inserts/sec.
3. Table `orders` (10M lignes), colonnes `customer_id` et `created_at`, utilisees dans `WHERE customer_id = ? ORDER BY created_at DESC`
4. Table `config` (50 lignes), colonne `key`, utilisee dans `WHERE key = ?`
5. Table `events` (1B lignes), insert-only (pas de reads sauf 1 batch job par nuit)

### Criteres de reussite
- [ ] 5/5 decisions correctes
- [ ] La raison "faible cardinalite" est evoquee pour la colonne `level`
- [ ] La notion de composite index est mentionnee pour `(customer_id, created_at)`
- [ ] L'impact des index sur les writes est mentionne au moins une fois

---

## Exercice 3 : Replication — Qui lit quoi ?

### Objectif
Comprendre l'impact du replication lag sur les lectures.

### Consigne
Tu as un systeme PostgreSQL avec 1 leader et 2 followers. Le replication lag moyen est de 200ms. Le pic de lag est de 2 secondes.

Scenario :
1. Un utilisateur modifie son nom de profil (ecriture sur le leader)
2. 100ms plus tard, il rafraichit sa page de profil (lecture)
3. La lecture est redirigee vers un follower par le load balancer

**Questions :**
1. Que voit l'utilisateur ? Pourquoi ?
2. Comment resoudre ce probleme sans forcer TOUTES les lectures vers le leader ?
3. Propose un algorithme simple de "read-your-writes consistency" en pseudo-code
4. Quel est le cout de cette solution en termes de charge sur le leader ?

### Criteres de reussite
- [ ] Le probleme du stale read est correctement identifie
- [ ] La solution "read-your-writes" est expliquee (pas juste nommee)
- [ ] Le pseudo-code est fonctionnel et couvre le cas nominal
- [ ] Le cout en charge supplementaire sur le leader est estime
