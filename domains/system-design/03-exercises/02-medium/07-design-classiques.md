# Exercices Medium — Design classiques (entretiens)

---

## Exercice 1 : Dimensionner le fanout d'un Twitter timeline

### Objectif
Chiffrer le cout du fanout-on-write vs fanout-on-read et justifier l'hybride avec des nombres, pas des intuitions.

### Consigne
Tu designs la home timeline d'un Twitter-like en mode chronologique.

**Chiffres :**
- 200M DAU, chaque user poste 2 tweets/jour
- Distribution des followers : la mediane est a 200 followers, mais 3 mega-comptes ont 100M followers chacun
- Lectures : chaque user ouvre sa home 50 fois/jour
- 1 entree de timeline cache (sorted set Redis) = ~40 bytes (tweet_id + score)

**Questions :**
1. Calcule le nombre total d'ecritures de fanout-on-write par jour pour un user **median** (200 followers, 2 tweets/jour). Puis pour **un seul** mega-compte (100M followers, 2 tweets/jour).
2. Pourquoi le fanout-on-write est-il insoutenable pour les mega-comptes ? Donne le chiffre d'ecritures par tweet d'un mega-compte.
3. En fanout-on-read pur, combien de timelines un user qui suit 500 comptes doit-il merger a chaque ouverture de home ? Pourquoi est-ce le mauvais defaut a 200M DAU read-heavy ?
4. Decris l'hybride : qui est en write, qui est en read, et comment on construit la home d'un user qui suit a la fois 499 comptes normaux et 1 mega-compte.
5. Estime la taille du cache Redis si on garde 800 tweets par user en timeline (40 bytes/entree, 200M users). CDN/sharding necessaire ?

### Criteres de reussite
- [ ] Median : 200 followers * 2 tweets = 400 ecritures/jour. Mega-compte : 100M * 2 = 200M ecritures/jour (500K ecritures/tweet en peak)
- [ ] Le fanout-on-write des mega-comptes est rejete (100M ecritures par tweet, latence + cout insoutenables)
- [ ] Fanout-on-read pur : merger ~500 timelines par ouverture * 50 ouvertures/jour = trop cher en read-heavy
- [ ] L'hybride : write pour les comptes normaux (timeline precalculee), read pour les mega-comptes (merge a la lecture)
- [ ] La home hybride = LRANGE de la timeline precalculee + pull live des quelques mega-comptes suivis, puis merge trie
- [ ] Cache Redis ~ 800 * 40 B * 200M = ~6.4 TB -> sharding Redis obligatoire (pas une seule instance)

---

## Exercice 2 : Encoding d'un URL shortener — collision et capacite

### Objectif
Raisonner sur l'espace de codes, le birthday paradox, et choisir entre hash et compteur avec des nombres.

### Consigne
Tu designs un URL shortener qui cree 50M URLs/jour. Tu hesites entre `md5(url)[:k]` en base62 et un compteur global + base62.

**Questions :**
1. Combien de codes distincts pour `k=6` et `k=7` caracteres en base62 ? (62^6 et 62^7)
2. En combien d'annees epuises-tu l'espace `k=7` au rythme de 50M URLs/jour ?
3. Birthday paradox : sur un espace de N codes, on attend une premiere collision apres ~`sqrt(N)` tirages. Calcule `sqrt(62^7)` et compare a une journee de trafic (50M). Conclusion sur le hash tronque ?
4. Le compteur global est un single point of failure. Decris le **range allocation** : combien de round-trips au coordinateur par batch de 100K ids, et que se passe-t-il si un serveur crash en plein batch (ids perdus, est-ce grave) ?
5. Le compteur expose un ordre sequentiel devinable (1,2,3 -> aa, ab, ac). Pourquoi ca peut etre un probleme de securite/business, et comment le mitiger sans repasser au hash ?

### Criteres de reussite
- [ ] 62^6 ≈ 5.68e10 (~57 milliards), 62^7 ≈ 3.52e12 (~3.5 trillions)
- [ ] Epuisement k=7 : 3.52e12 / (50e6 * 365) ≈ ~193 ans
- [ ] sqrt(62^7) ≈ 1.88M, soit < 1 jour de trafic (50M) -> hash tronque force des collisions a gerer (verif + retry)
- [ ] Range allocation : 1 round-trip par batch de 100K -> 1 appel coordinateur pour 100K URLs ; un crash perd au pire un batch (trous benins, pas de doublon)
- [ ] La sequentialite devinable (enumeration, fuite de volume) se mitige par offset/permutation/base62 shuffle, sans collision

---

## Exercice 3 : Chat — registry de connexions et ordering

### Objectif
Concevoir le routing d'un message 1-to-1 a travers un parc de serveurs WebSocket stateful et garantir l'ordre.

### Consigne
Tu designs un chat WhatsApp-like : 100M connexions WS simultanees en peak, chaque serveur WS tient ~150K connexions.

**Questions :**
1. Combien de serveurs WebSocket faut-il pour 100M connexions a 150K conns/serveur ? Ajoute un headroom raisonnable.
2. Alice (sur serveur WS-A) envoie a Bob (sur serveur WS-B). Decris les 6 etapes du flow, en incluant le registry `user_id -> server` (Redis) et le pub/sub inter-serveurs.
3. Le registry Redis : quand est-il ecrit/mis a jour ? Que se passe-t-il a la reconnexion de Bob sur un autre serveur (WS-C) ?
4. Les messages d'un groupe arrivent "out of order" a cause du reseau. Comment garantis-tu l'ordre final percu par tous les membres ? (serveur-side timestamp, TIMEUUID, partition Cassandra)
5. Bob est offline. Decris le fallback (FCM/APNs + stockage pour rattrapage) et comment Bob resync au retour avec son `last_message_id`.

### Criteres de reussite
- [ ] 100M / 150K ≈ 667 serveurs ; avec headroom (~30%) ~ 850-900 serveurs
- [ ] Le flow inclut : persist Cassandra -> lookup registry -> publish sur le canal du serveur de Bob -> push WS
- [ ] Le registry est ecrit a chaque connect/disconnect ; la reconnexion sur WS-C met a jour `user_id -> WS-C`
- [ ] L'ordering est fixe par le timestamp serveur (moment de reception) + TIMEUUID + partition par conversation_id
- [ ] Offline -> FCM/APNs + stockage ; resync via `last_message_id` (le client pull les messages manquants)
