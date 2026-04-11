# Exercices Easy — Design classiques (entretiens)

---

## Exercice 1 : Design Pastebin

### Objectif
Designer un service type Pastebin (partage de textes avec lien court) en suivant le framework en 6 etapes.

### Consigne
**Produit** : un service ou l'utilisateur colle du texte (code snippet, note) et recupere une URL courte partageable. Le paste peut expirer.

**Exigences :**
- 1-10 MB max par paste
- Expiration configurable (1h, 1j, 1 semaine, jamais)
- Optionnel : password-protected
- 1M DAU, ratio reads:writes = 20:1
- Reads ~90% des 10 premieres secondes apres creation puis decroissance

**A rendre :**
1. Liste des questions de **clarification** que tu poserais a l'intervieweur (5+).
2. **Capacity estimation** : QPS reads/writes, storage 1 an, bandwidth.
3. **Architecture high-level** : boites et fleches (texte / ASCII OK). Composants majeurs + flux.
4. **Deep dive** : schema de stockage pour les pastes. Quelle DB choisis-tu ? Pourquoi ?
5. **Cache strategy** : quoi cacher, ou, TTL.
6. **Deux bottlenecks** probables et comment les resoudre.

### Criteres de reussite
- [ ] 5+ questions de clarification pertinentes
- [ ] QPS calcules (~11 reads/s, 0.6 writes/s moyen, peak 3x)
- [ ] Storage ~3-10 TB/an estime (selon taille moyenne choisie)
- [ ] Architecture contient : CDN, LB, App, Cache, Object storage (S3) pour le body, KV pour metadata
- [ ] Choix DB explique (S3 pour le body, Cassandra ou Redis pour les metadata)
- [ ] Cache strategy coherente (hot pastes en Redis, TTL 5-60 min)
- [ ] Deux bottlenecks identifies (ex : hot paste viral, expiration a grande echelle)

---

## Exercice 2 : Design Instagram feed

### Objectif
Designer le feed d'Instagram : poster une photo, suivre des users, voir le feed.

### Consigne
**Produit** : poster des photos (media), suivre des users, voir le feed des users suivis (chronologique pour cet exercice).

**Exigences :**
- 500M DAU
- Chaque user poste 1 photo/jour en moyenne
- Chaque user regarde son feed ~20 fois/jour
- Photos de 2 MB en moyenne (apres compression)
- Latence feed load < 1 seconde

**A rendre :**
1. **Clarification** : 5+ questions.
2. **Capacity estimation** : QPS reads/writes, storage photos 1 an, bandwidth CDN.
3. **Architecture high-level**.
4. **Deep dive** : Explique la difference entre **fanout-on-write** et **fanout-on-read** pour le feed. Quel choix pour Instagram ? Justifie avec des chiffres.
5. **Storage photos** : comment/ou stocker les 2 MB * 500M/jour ? Comment servir les photos rapidement ?
6. **Deux bottlenecks** probables et solutions.

### Criteres de reussite
- [ ] Capacity : ~6K writes/s moyen, ~115K feed reads/s moyen
- [ ] Storage photos : ~350 PB/an (2 MB * 500M * 365) -> OBLIGATOIRE : S3 + CDN
- [ ] Architecture inclut : object storage + CDN pour photos, KV pour metadata, Redis pour feeds
- [ ] Fanout-on-write explique (pre-compute feed en Redis) avec les limites du celebrity problem
- [ ] Solution hybride proposee pour les gros comptes (Kylie Jenner 400M followers)
- [ ] Mentionne le image processing (thumbnails, formats) via un worker async
- [ ] Bottlenecks : celebrity problem, storage cost, hot photos virales

---

## Exercice 3 : Design Notification System

### Objectif
Designer un systeme de notifications qui route des events vers les bons canaux (push mobile, email, SMS, in-app).

### Consigne
**Produit** : un service interne qui recoit des events metier (ex : "new_follower", "payment_received", "birthday_alert") et les livre aux users via les canaux appropries.

**Exigences :**
- 100M users
- ~10 notifications envoyees par user et par jour en moyenne
- 3 canaux : push mobile (FCM/APNs), email (SendGrid), SMS (Twilio)
- Chaque user a des **preferences** : quels types de notifs via quels canaux
- Latence acceptable : < 30 secondes pour les notifs non-critiques, < 3 secondes pour les critiques (ex : 2FA)
- **Deduplication** : ne pas envoyer 2x la meme notif si un event est dupplique
- **Rate limiting** : max 20 push/heure par user (eviter le spam)

**A rendre :**
1. **Clarification** : 5+ questions.
2. **Capacity estimation** : QPS events, QPS notifs generees apres fanout.
3. **Architecture high-level** avec les composants clefs.
4. **Deep dive** : flow complet d'un event "new_follower" jusqu'a la reception sur le telephone du user.
5. **Deduplication strategy** : comment eviter les doubles.
6. **Rate limiting strategy** : ou, quel algo, quelle granularite.
7. **Retry strategy** : que faire si SendGrid est down ?

### Criteres de reussite
- [ ] Capacity : ~1B notifs/jour = 11K/s moyen, 35K peak
- [ ] Architecture inclut : event ingestion (Kafka), preference service, template service, delivery workers par canal, DLQ
- [ ] Le flow event -> livraison est clair et mentionne preferences + templates
- [ ] Deduplication via event_id unique (cache ou DB avec contrainte UNIQUE)
- [ ] Rate limiting via token bucket en Redis par user_id + canal
- [ ] Retry via DLQ + exponential backoff
- [ ] Au moins un tradeoff discute (push vs polling, sync vs async)
