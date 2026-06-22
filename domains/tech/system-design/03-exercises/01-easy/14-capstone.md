# Exercices Easy — Capstone

---

## Exercice 1 : Concevoir le backend Spotify Recommendations

### Objectif
Appliquer le framework 6-etapes a un systeme de recommandation musicale de grande echelle.

### Consigne
On te demande en entretien : **"Design Spotify Recommendations"** — la home feed personnalisee, les playlists auto-generees (Discover Weekly), les radios basees sur un titre.

Contraintes implicites :
- 500M MAU, 200M DAU
- Chaque user voit ~50 recos par jour (home + radios + playlists auto)
- Le catalogue contient 100M titres
- La home doit charger en < 500 ms
- Discover Weekly est genere une fois par semaine, mais doit pouvoir integrer des signaux recents
- Tu as un budget : utiliser un mix de recos collaborative filtering + content-based + LLM pour les descriptions humaines

**Travail attendu :**
1. Requirements (functional + non-functional) : 5 bullets chacun
2. Capacity estimation : users, QPS, storage recos, bandwidth
3. High-level design : schema ASCII des composants (min 6 boxes)
4. Deep dive sur : feature store, batch pipeline Discover Weekly, real-time scoring de la home
5. 3 bottlenecks + mitigations
6. 2 extensions pour aller plus loin

### Livrables
Un doc markdown complet de 1500-2500 mots avec le schema, les calculs et la logique.

### Criteres de reussite
- [ ] Les requirements distinguent functional (recos, radios, playlists) et non-functional (latence, scale, cost)
- [ ] La capacity donne un ordre de grandeur de peak QPS (~5K / s sur la home), storage (~ To), bandwidth
- [ ] Le design comporte un feature store, un serving layer, un batch pipeline, un online scoring
- [ ] Le batch Discover Weekly est identifie comme job Spark / Flink tournant chaque semaine
- [ ] Le real-time scoring utilise une feature store online (Redis / Dynamo) pour les features fraiches
- [ ] Au moins un bottleneck sur le feature store ou sur la DB de recos est identifie
- [ ] Le role du LLM est borne (ex: generation de descriptions, NOT le scoring lui-meme)

---

## Exercice 2 : Concevoir le backend GitHub Copilot

### Objectif
Maitriser un design LLM a fort enjeu de latence et de cout.

### Consigne
Design the backend of **GitHub Copilot** (completion de code inline dans un IDE).

Contraintes :
- 10M devs actifs / jour
- 30 completions par user par heure de coding
- Latence cible : < 300 ms end-to-end (IDE perçoit rapidement)
- Le modele voit le contexte du fichier courant + fichiers voisins (recuperes via un "context engine")
- Le modele principal est un LLM specialise en code
- Le produit doit fonctionner en environnement offline partiel (VPN, mauvaise connexion)
- Confidentialite : certains clients enterprise ne veulent pas que leur code quitte leur VPC

**Travail attendu :**
1. Requirements + contraintes specifiques (privacy)
2. Capacity estimation : calls/s, tokens/s, cost mensuel
3. High-level design avec plusieurs tiers : client IDE, edge router, LLM serving, context retrieval
4. Deep dive : comment on tient < 300 ms p95 ? Comment on gere le cache ? Quel modele pour quel tier ?
5. 3 challenges specifiques : streaming token-by-token, cancellation des requetes, privacy enterprise
6. Extensions : mode chat, agentic multi-file edit, inline tests

### Livrables
Doc markdown complet.

### Criteres de reussite
- [ ] La latence < 300 ms est abordee explicitement (streaming TTFT)
- [ ] Il y a un "context engine" pour retrieve les fichiers voisins pertinents
- [ ] Le modele est plutot "code-specialized" (Codex, StarCoder, Codellama ou equivalent)
- [ ] La privacy enterprise est resolue via self-host ou VPC-peered deployment
- [ ] Le cache est aborde : prefix cache, embedding de context similaire, ou cache de completions identiques
- [ ] La cancellation est citee (si user tape une nouvelle touche, ancienne completion annulee)
- [ ] Le cost est calcule, et au moins une optimisation citee (tiers, smaller model, quantization)

---

## Exercice 3 : Concevoir un Autonomous Research Agent

### Objectif
Construire un systeme complet d'agent de recherche autonome.

### Consigne
**Design an autonomous research agent** : un user donne un sujet ("state of the art in battery technology 2026"), l'agent planifie des sous-questions, cherche sur le web, lit des sources, evalue la credibilite, ecrit un rapport structure de 10-20 pages avec citations verifiables.

Contraintes :
- Un "run" peut prendre 5-30 minutes (async, non-bloquant)
- Budget par run : $5-20 en LLM cost, 100-500 search queries
- Le rapport doit inclure des citations verifiables (URL + quote exacte)
- Robustesse : l'agent doit survivre aux sites down, timeouts, rate limits
- Observability : l'utilisateur peut voir la progression en temps reel

**Travail attendu :**
1. Requirements + definition precise de "done" (quand l'agent s'arrete ?)
2. Architecture multi-agent : quelle hierarchie ? quels specialistes ? quel handoff ?
3. Deep dive sur : planning (comment on decompose une question en sous-questions), sources evaluation (credibilite), citation verification, memory management
4. 3 failure modes specifiques + comment les gerer : boucles infinites, source hallucination, budget explosion
5. Observability : comment le user voit la progression, comment debugger si ca rate
6. Extensions : collaboration entre agents, interruption humaine, reutilisation de recherches passees

### Livrables
Doc markdown + schema ASCII de l'architecture.

### Criteres de reussite
- [ ] L'architecture est multi-agent (supervisor + specialists : planner, searcher, reader, writer, critic)
- [ ] Le stopping criteria est explicite (budget depasse, critic valide, user interrompt, completeness score atteint)
- [ ] La memoire est separee : working memory (contexte courant) + long-term (recherches passees)
- [ ] Les 3 failure modes sont couverts (budget, loops, citations)
- [ ] La citation verification est abordee : l'agent doit retrouver le passage source (pas juste l'URL)
- [ ] L'observability inclut une trace temps reel (Langfuse / LangSmith) et une progression affichee au user
- [ ] Une extension sur l'interruption humaine (pause/resume/redirect) est proposee
