# Exercices Medium — Capstone

---

## Exercice 1 : Design complet — Uber Eats (livraison de repas)

### Objectif
Derouler le framework 6 etapes en 45 minutes sur un design classique multi-acteurs (clients, restaurants, livreurs).

### Consigne
Concois la plateforme de livraison de repas : les clients commandent, les restaurants preparent, les livreurs sont dispatches en temps reel.

**Chiffres imposes :**
- 10M de commandes/jour, concentrees sur 4h (midi et soir : 80% du volume)
- 2M de livreurs actifs en pic, position GPS toutes les 4 secondes
- Matching commande -> livreur en < 30 secondes
- Tracking temps reel de la livraison pour le client (< 5 s de fraicheur)

**Deroule (45 min chrono recommande) :**
1. **Estimation** : QPS commandes (moyen, pic), QPS GPS ingestion, stockage des positions (retention 30 jours).
2. **High-level design** : services (orders, restaurants, dispatch, tracking, notifications), bases de donnees, queues.
3. **Deep dive — dispatch** : comment trouver les livreurs proches en < 1 s parmi 2M ? (indexation geospatiale : geohash/H3, structure de la requete)
4. **Deep dive — tracking** : comment pousser la position au client (WebSocket ? polling ?) et a quel cout ?
5. **Bottlenecks** : les 2 principaux en pic, avec mitigation.
6. **Extensions** : surge pricing, prevision du temps de preparation.

### Criteres de reussite
- [ ] Estimation : ~115 commandes/s en moyenne, ~580/s en pic (80% sur 4h) ; GPS : 500K events/s en pic ; stockage positions ~30+ To/mois selon hypotheses
- [ ] L'architecture separe le flux transactionnel (commandes, SQL) du flux haute frequence (GPS, Kafka + store in-memory/TSDB)
- [ ] Le dispatch utilise un index geospatial (geohash/H3/S2) en memoire avec recherche par cellules voisines
- [ ] Le tracking client passe par WebSocket/SSE avec throttling (1 update/2-5 s suffit), pas du polling DB
- [ ] Les bottlenecks identifies sont chiffres (ingestion GPS, matching en pic) et au moins 3 tradeoffs sont explicites

---

## Exercice 2 : Design complet — Assistant de code IA (type Copilot backend)

### Objectif
Combiner inference at scale, RAG, caching et latence stricte dans un design IA realiste.

### Consigne
Concois le backend d'un assistant de completion de code dans l'IDE.

**Chiffres imposes :**
- 2M de developpeurs actifs, ~1 completion demandee toutes les 10 secondes de frappe active (3h/jour de frappe active par dev en moyenne)
- Latence percue : < 300 ms au p95 sinon la suggestion est inutile (le dev a deja tape la suite)
- Contexte envoye : fichier courant + imports + symboles voisins (~2K tokens)
- 25% des completions sont acceptees

1. **Estimation** : QPS de completions (moyen/pic x2), tokens/s d'inference necessaires, ordre de grandeur de la fleet GPU (un GPU sert ~10 req/s pour un petit modele optimise).
2. **Latency budget** : decompose les 300 ms (reseau aller-retour, contexte, inference, post-processing). Combien reste-t-il pour l'inference elle-meme ?
3. **Architecture** : place le cache (quel type de cache pour du code en cours de frappe ?), le modele (taille ? pourquoi un modele frontier est exclu ?), le streaming et la cancellation (le dev continue de taper -> la requete devient obsolete).
4. **Multi-region** : pourquoi le deploiement doit etre multi-region (vs un seul datacenter US) ? Chiffre l'impact reseau pour un dev a Singapour.
5. **Qualite** : comment mesurer en continu que les suggestions sont bonnes (metrique online directe, pas un benchmark) ?

### Criteres de reussite
- [ ] Estimation posee : 2M devs x 3h x 360 completions/h = ~2.2B completions/jour, soit ~25K req/s en moyenne et ~50K en pic ; fleet de l'ordre de 5 000 GPUs (50K / 10 req/s) — les hypotheses doivent etre explicites
- [ ] Budget : ~50-100 ms reseau RTT + ~20 contexte + ~150 inference + ~30 marge ; l'inference doit tenir en ~150 ms -> petit modele (1-7B) optimise, frontier exclu (latence et cout)
- [ ] La cancellation des requetes obsoletes est presente (debouncing cote client + abort cote serveur) — c'est LE point distinctif du design
- [ ] Multi-region justifie par le RTT : Singapour -> US = 180-250 ms de RTT, le budget de 300 ms est deja mort
- [ ] Metrique online : taux d'acceptation des suggestions (et retention des suggestions acceptees apres 30 s), suivi par version de modele

---

## Exercice 3 : Arbitrage d'architecture — migrer un produit IA qui explose

### Objectif
Exercer le jugement senior : prioriser des evolutions d'architecture sous contrainte de budget et de temps.

### Consigne
Tu reprends la tech d'une startup dont l'assistant juridique IA explose : 10x d'utilisateurs en 3 mois (5K -> 50K avocats). Stack actuelle : monolithe Python + PostgreSQL + pgvector (2M chunks), appels directs a un provider LLM unique (pas de gateway), pas de cache, pas d'evals, un seul environnement (la prod). Incidents recents : 3 pannes du provider LLM repercutees aux clients, facture LLM x8, latence p95 passee de 4 s a 11 s, un avocat a signale une citation inventee.

Budget : 2 ingenieurs pendant 1 trimestre. Tu ne peux PAS tout faire.

1. Liste 8-10 chantiers possibles, puis selectionne les 4 prioritaires pour le trimestre. Justifie chaque choix ET chaque report par l'impact incident/risque.
2. Pour chacun des 4 retenus : esquisse la solution (2-3 phrases + composants).
3. La citation inventee est le risque le plus grave (responsabilite professionnelle). Quelle reponse complete (court terme : mitigation immediate ; moyen terme : architecture) ?
4. Quels chantiers structurels REFUSES-tu de faire maintenant malgre la pression (ex : microservices, fine-tuning, multi-cloud) et pourquoi ?
5. Definis les 5 metriques que tu mets en place des la semaine 1 pour piloter le trimestre.

### Criteres de reussite
- [ ] Les 4 priorites adressent les incidents reels : gateway LLM (fallback + cache + tracking couts), evals + groundedness/citations verifiees, observabilite/tracing, optimisation latence (cache, routing, eventuellement migration vector DB si pgvector sature)
- [ ] Chaque report est justifie (microservices = pas le bottleneck, le monolithe tient ; fine-tuning = pas de probleme de qualite de base ; multi-cloud = overkill)
- [ ] La citation inventee a une reponse en 2 temps : court terme = verification des citations contre les sources (string match / verification deterministe) + disclaimer ; moyen terme = pipeline de groundedness + gold set juridique
- [ ] Le candidat refuse explicitement au moins 2 chantiers "sexy mais non prioritaires"
- [ ] Les 5 metriques couvrent : latence p95, cout/requete, disponibilite (vue client), taux de citations verifiees, satisfaction/feedback
