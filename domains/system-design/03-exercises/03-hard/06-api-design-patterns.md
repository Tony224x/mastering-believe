# Exercices Hard — API Design & Patterns

---

## Exercice 1 : API publique d'une plateforme bancaire (open banking)

### Objectif
Concevoir une API publique critique avec exigences reglementaires, idempotence forte, et engagement de compatibilite long terme.

### Consigne
Tu concois l'API publique d'une banque pour l'open banking : des centaines de fintechs tierces vont initier des virements et consulter des comptes via ton API.

**Contraintes chiffrees :**
- 2 000 req/s en moyenne, 12 000 req/s en pic (debut de mois)
- Initiation de virement : idempotence ABSOLUE (un virement duplique = incident reglementaire, amende possible)
- SLO : lectures 150 ms p99, ecritures 400 ms p99, disponibilite 99.95% (engagement contractuel avec penalites)
- Un virement passe par des etats asynchrones : initiated -> pending_authorization -> executed / rejected (l'execution prend 1 s a 48 h selon le reseau bancaire)
- Compatibilite : une version d'API doit etre supportee 24 mois minimum (exigence reglementaire)
- Quotas par fintech : 100 req/s standard, negociable par contrat

**Livre :**
1. **Design des endpoints virement** : POST d'initiation (avec le mecanisme d'idempotence complet : header, stockage, TTL, collision), GET de statut, et le probleme du suivi asynchrone : polling vs webhooks — concois LES DEUX (les fintechs exigent les webhooks, le regulateur exige le polling).
2. **Webhooks fiables** : tu envoies des webhooks aux fintechs (leurs endpoints sont down 2% du temps). Concois : signature (comment la fintech verifie l'authenticite ?), retry policy, ordering, et le endpoint de reconciliation pour les events manques.
3. **Machine a etats** : specifie les transitions valides du virement et comment l'API les expose (champ status + timestamps ? event log ?). Que repond un PATCH d'annulation selon l'etat courant ?
4. **Idempotence sous concurrence** : deux requetes IDENTIQUES (meme Idempotency-Key) arrivent a 3 ms d'intervalle sur 2 pods differents. Decris le mecanisme qui garantit UN seul virement (lock ? contrainte unique ? quelle reponse pour la 2e requete pendant que la 1ere est encore en cours ?).
5. **Rate limiting contractuel** : concois la reponse 429 (headers, granularite par endpoint ?) et le mecanisme pour qu'un burst legitime de debut de mois (paie) ne soit pas massacre par le quota.
6. **Evolution sur 24 mois** : liste ce que tu t'interdis de changer dans v1, et le processus complet de sortie de v2 (sunset policy, metriques d'adoption, communication).

### Criteres de reussite
- [ ] L'idempotence est complete : Idempotency-Key obligatoire, stockee avec le hash du body + la reponse, TTL >= 24-48h, 409/422 si meme cle + body different, replay de la reponse originale sinon
- [ ] Les webhooks ont : signature HMAC avec secret par fintech (+ rotation), retries avec backoff exponentiel sur 24-72h, livraison at-least-once assumee, et un GET /events depuis un cursor pour la reconciliation
- [ ] La machine a etats est explicite avec transitions interdites ; l'annulation repond differemment selon l'etat (200 si pending, 409 si executed)
- [ ] La concurrence est resolue par une primitive atomique (INSERT unique sur la cle) + reponse de la 2e requete : 409 "processing" ou attente courte puis replay — pas de double virement possible
- [ ] Le rate limiting propose token bucket (burst tolere au-dessus du debit soutenu) + headers standards + 429 avec Retry-After
- [ ] Le plan v2 contient : changements additifs only en v1, versioning par URL ou header, 24 mois de support parallele, header Sunset/Deprecation, suivi du trafic v1 restant avant extinction
- [ ] Au moins 3 tradeoffs explicites (polling vs webhook, granularite des quotas, strictness de l'idempotence vs UX)

---

## Exercice 2 : Unifier 3 APIs heterogenes derriere une facade GraphQL — analyse critique

### Objectif
Evaluer et concevoir une federation d'APIs sous contrainte de latence, en identifiant honnetement ou GraphQL aide et ou il aggrave.

### Consigne
Une marketplace a 3 backends historiques : Catalog (REST, p99 80 ms), Orders (gRPC, p99 60 ms), Reviews (REST legacy, p99 300 ms, fragile : tombe a 50 req/s). Les clients : une app mobile (ecran produit = 1 appel souhaite), un site web, et 40 partenaires B2B qui n'utilisent QUE le catalogue.

**Contraintes chiffrees :**
- Ecran produit mobile : 1 requete, < 250 ms p95, agrege les 3 backends
- Trafic : 8 000 req/s sur l'ecran produit en pic
- Reviews ne supporte que 50 req/s — il faut 8 000 req/s d'ecrans produits AVEC reviews
- Equipe plateforme : 4 devs ; les 3 equipes backend ne changeront pas leurs APIs
- Les partenaires B2B ont des contrats stables (pas question de leur imposer GraphQL)

**Livre :**
1. **Decision d'architecture** : GraphQL federation, BFF par client, ou API Gateway + endpoints composites REST ? Compare les 3 contre les contraintes (latence, equipe de 4, partenaires B2B) et choisis. Le choix "GraphQL partout" doit etre challenge.
2. **Le probleme Reviews (50 req/s vs 8 000 req/s)** : concois la couche d'absorption complete : cache (quel TTL pour des reviews ?), dataloader/batching, valeurs par defaut si indisponible (l'ecran s'affiche sans reviews ?). Calcule le hit rate de cache minimum requis pour ne pas tuer Reviews.
3. **Latence** : montre comment la facade tient 250 ms p95 : parallelisation des 3 backends, timeout par champ/source, partial responses. Que renvoie l'ecran si Reviews timeout ?
4. **Le probleme N+1** : la requete "20 produits avec leurs reviews" declenche 1 + 20 appels Reviews. Explique le mecanisme dataloader et chiffre la reduction.
5. **Protection de la facade** : query complexity / depth limiting (un partenaire malveillant peut composer une requete explosive) — donne un schema de scoring et une limite.
6. **Verdict honnete** : liste 3 choses que cette facade AGGRAVE (debugging, caching HTTP, courbe d'apprentissage...) et comment tu les mitiges.

### Criteres de reussite
- [ ] La comparaison des 3 options est argumentee contre les contraintes ; un choix pragmatique est fait (ex : BFF/gateway composite pour mobile + REST intact pour B2B, GraphQL seulement si le besoin de flexibilite des clients le justifie)
- [ ] Le calcul Reviews est pose : 8 000 req/s d'ecrans -> hit rate cache requis >= 1 - 50/8000 = 99.4% -> TTL longs (minutes-heures) + cache warming + stale-while-revalidate, plus degradation sans reviews
- [ ] La facade parallelise les appels (latence = max, pas somme : ~max(80, 60, 300 cache) et timeout par source avec partial response documentee
- [ ] Le dataloader est explique (collecte des ids dans le tick, 1 appel batch) : 21 appels -> 2 appels
- [ ] La protection donne un scoring concret (cout par champ x profondeur x listes, limite par token/partenaire)
- [ ] 3 inconvenients honnetes avec mitigation (tracing par champ, APQ/persisted queries pour le cache, formation/conventions)
