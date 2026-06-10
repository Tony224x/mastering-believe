# Exercices Medium — API Design & Patterns

---

## Exercice 1 : Concevoir l'API complete d'un service de reservation

### Objectif
Designer une API REST production-ready de bout en bout (ressources, idempotence, pagination, erreurs, versioning).

### Consigne
Tu designes l'API publique d'un systeme de reservation de salles de reunion (SaaS B2B). Fonctionnalites : lister les salles, consulter les disponibilites, creer/modifier/annuler une reservation, lister les reservations d'un utilisateur.

**A rendre :**
1. La liste des endpoints (methode + path + status codes de succes et d'erreur). Ressources au pluriel, pas de verbes dans les URLs.
2. Le endpoint de creation de reservation doit etre **idempotent** : decris le mecanisme (header, stockage, duree de retention de la cle, comportement si meme cle + body different).
3. La double reservation est interdite : quel status code et quel body d'erreur renvoyer si la salle est deja prise ? Propose un format d'erreur standard (type, code machine-readable, message, details).
4. `GET /reservations` peut retourner 100K+ items pour un gros client : concois la pagination **cursor-based** (format du cursor, parametres, reponse). Pourquoi pas offset ?
5. Tu dois renommer le champ `start` en `start_time` (breaking change). Deroule ta strategie de versioning et de deprecation (durees, headers, communication).

### Criteres de reussite
- [ ] Au moins 6 endpoints REST corrects (GET /rooms, GET /rooms/{id}/availability, POST /reservations, PATCH, DELETE, GET /users/{id}/reservations ou filtre)
- [ ] Idempotency-Key stockee avec la reponse, rejouee si meme cle, 409/422 si meme cle avec body different
- [ ] Conflit de reservation = 409 Conflict avec un code d'erreur machine-readable (ex : room_already_booked)
- [ ] Cursor opaque (base64 d'un id/timestamp), stable face aux insertions, contrairement a l'offset
- [ ] Versioning : /v2/ ou nouveau champ additif d'abord, deprecation annoncee avec 6-12 mois et header Deprecation/Sunset

---

## Exercice 2 : Migrer un monolithe REST vers gRPC interne

### Objectif
Choisir ou et comment introduire gRPC dans une architecture microservices, sans casser les clients externes.

### Consigne
Une plateforme a 12 microservices qui communiquent tous en REST/JSON. Constats : la latence inter-services represente 40% de la latence totale, les payloads JSON font en moyenne 8 Ko, et les equipes perdent du temps sur des contrats d'API ambigus.

1. Quels appels migrer vers gRPC en priorite, et lesquels garder en REST ? Donne la regle generale (interne vs externe, browser ou pas).
2. Estime le gain : protobuf reduit la taille des payloads d'environ 60-80% et la serialisation est 5-10x plus rapide. Si un appel inter-service prend 12 ms (2 ms reseau incompressible + 4 ms serialisation/deserialisation + 6 ms traitement), quelle latence apres migration ?
3. Ecris le fichier `.proto` (service + messages) pour le service `RoomAvailability` : une RPC unaire `CheckAvailability` et une RPC server-streaming `WatchAvailability`.
4. Les clients web (browser) doivent continuer a consommer ces donnees. Quelles sont les 2 options (gRPC-web + proxy, ou gateway REST devant gRPC) et laquelle recommandes-tu ?
5. Comment gerer l'evolution du contrat protobuf sans casser les anciens clients (3 regles de compatibilite protobuf) ?

### Criteres de reussite
- [ ] Regle claire : gRPC pour l'interne service-a-service, REST pour l'API publique/browser
- [ ] Calcul de latence : ~2 + ~0.5 + 6 = ~8.5 ms (la serialisation chute, le reseau et le traitement restent)
- [ ] Le .proto est syntaxiquement plausible : syntax proto3, service, rpc unaire et `stream` cote reponse
- [ ] La recommandation browser est justifiee (gateway REST/gRPC-web selon le controle des clients)
- [ ] Les regles protobuf : ne jamais reutiliser un numero de champ, pas de champ required, ajouter des champs optionnels uniquement, ne pas changer le type d'un champ existant

---

## Exercice 3 : API Gateway — centraliser sans creer un monolithe

### Objectif
Definir ce qui appartient a l'API Gateway et ce qui doit rester dans les services.

### Consigne
Ta plateforme expose 40 endpoints publics servis par 12 microservices. Aujourd'hui, chaque service reimplemente : l'authentification JWT, le rate limiting, le logging des requetes, la validation des API keys, et le CORS.

1. Liste ce qui doit migrer dans l'API Gateway et ce qui doit RESTER dans les services. Donne le critere de decision (infra transverse vs logique metier).
2. L'equipe propose d'ajouter dans le gateway : "si le user est premium, enrichir la reponse avec les recommandations". Accepte ou refuse ? Justifie.
3. Le gateway devient un SPOF et un bottleneck potentiel. Dimensionne : 25 000 req/s en pic, un pod gateway tient 4 000 req/s avec p99 < 5 ms. Combien de pods (avec marge N+2) ? Quelles metriques surveiller ?
4. L'auth JWT au gateway : le service downstream doit-il refaire confiance aveuglement ? Decris le pattern (headers internes signes, ou re-validation legere) et le risque si le reseau interne n'est pas zero-trust.
5. Propose la strategie de rollout : comment migrer 40 endpoints vers le gateway sans big bang ?

### Criteres de reussite
- [ ] Gateway : auth, rate limiting, CORS, logging, routing, TLS. Services : toute logique metier
- [ ] L'enrichissement premium est REFUSE (logique metier dans le gateway = anti-pattern)
- [ ] Calcul pods : ceil(25000/4000) = 7, +2 de marge = 9 pods ; metriques : p99 ajoutee par le gateway, error rate, saturation CPU
- [ ] Le pattern de confiance : header interne (X-User-Id) + signature/mTLS, risque de spoofing si un pod interne est compromis
- [ ] Rollout progressif : endpoint par endpoint via DNS/routing, en commencant par un endpoint non critique
