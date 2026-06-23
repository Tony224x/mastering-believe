# Exercices Easy — Caching & CDN

---

## Exercice 1 : Quelle strategie de cache ?

### Objectif
Savoir identifier rapidement la strategie de cache adaptee a un use case.

### Consigne
Pour chacun des systemes suivants, indique quelle strategie de cache tu utiliserais (**cache-aside**, **write-through**, **write-behind**, **read-through**). Justifie en une phrase.

1. Un service de profils utilisateur lu 50K fois/sec, mis a jour 10 fois/sec
2. Un compteur de vues sur des videos YouTube (incremente a chaque vue, lu sur la page)
3. Un systeme de gestion de stock e-commerce ou chaque survente coute de l'argent
4. Un service de configuration globale identique pour tous les pods d'un cluster Kubernetes
5. Un dashboard analytics qui aggrege les ventes des 24 dernieres heures
6. Un systeme de sessions utilisateur (login/logout)

### Criteres de reussite
- [ ] 6/6 choix corrects avec justification
- [ ] Cache-aside est propose pour au moins 2 cas (c'est le defaut)
- [ ] Write-behind est propose pour le compteur de vues (write-heavy)
- [ ] Le tradeoff consistance/performance est mentionne au moins une fois

---

## Exercice 2 : Cache-Control headers — Quel header pour quel contenu ?

### Objectif
Comprendre les directives Cache-Control et savoir les appliquer.

### Consigne
Pour chacune des ressources suivantes, ecris le header `Cache-Control` complet et justifie chaque directive :

1. `app.a3f2b1c.js` — fichier JavaScript avec hash dans le nom, deploye sur CDN
2. `/api/me` — endpoint qui retourne le profil de l'utilisateur connecte
3. `/api/products` — catalogue de 10K produits mis a jour toutes les heures
4. `/login` — page HTML de connexion
5. Un fichier PDF de facture telecharge par un utilisateur authentifie

### Criteres de reussite
- [ ] Le JS avec hash utilise `immutable` et `max-age=31536000`
- [ ] L'endpoint `/api/me` utilise `private` (donnee personnalisee)
- [ ] Le catalogue utilise `s-maxage` (pour le CDN) distinct de `max-age` (pour le browser)
- [ ] La facture PDF utilise `private, no-store` ou `private, no-cache`
- [ ] Chaque directive est justifiee (pas juste listee)

---

## Exercice 3 : Dimensionnement memoire Redis

### Objectif
Savoir estimer la memoire Redis necessaire pour un use case donne.

### Consigne
Tu deployes Redis pour stocker les sessions utilisateur d'une application web :

**Donnees :**
- 5 millions d'utilisateurs actifs par jour
- Chaque session contient : user_id (UUID, 36 bytes), role (10 bytes), token (64 bytes), last_seen (timestamp, 8 bytes), preferences (JSON, ~200 bytes)
- TTL des sessions : 30 minutes
- En pic, 40% des users sont connectes simultanement (= 2M sessions en cache)
- Redis a un overhead de ~2.5x sur les donnees brutes (structures internes, pointeurs)

**Questions :**
1. Calcule la taille brute d'une session
2. Calcule la memoire totale necessaire pour 2M sessions avec l'overhead Redis
3. Si tu utilises un Redis Cluster avec 3 masters, combien de RAM par noeud ?
4. Faut-il prevoir des replicas ? Si oui, quel est le total de RAM du cluster ?
5. Quel `maxmemory-policy` utiliser pour les sessions et pourquoi ?

### Criteres de reussite
- [ ] La taille brute est correctement calculee (~318 bytes)
- [ ] La memoire totale inclut l'overhead Redis (~1.5 Go)
- [ ] La RAM par noeud est correcte (~500 Mo par master)
- [ ] Les replicas sont mentionnes (HA) et le total est calcule (~3 Go avec replicas)
- [ ] La policy `volatile-ttl` ou `volatile-lru` est recommandee (pas `allkeys-lru` car les sessions ont un TTL)
