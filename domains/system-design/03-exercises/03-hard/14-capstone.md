# Exercices Hard — Capstone (extensions)

> Ces exercices ETENDENT les 2 designs de reference du J14 (Dropbox et LLM Support Assistant)
> sur des scenarios a fort enjeu. On ne re-concoit pas tout : on dimensionne une evolution
> majeure et on anticipe les failure modes, comme en entretien senior.

---

## Exercice 1 : Concevoir le rollback et la migration zero-downtime du LLM Support Assistant

### Objectif
Etendre le design J14 avec une strategie de deploiement industrielle : migrer le modele/prompt en prod sans downtime, avec rollback instantane et garde-fous qualite. Reutilise J11 (infra), J13 (observability/MLOps) et J14 (design).

### Consigne
Le LLM Support Assistant du J14 tourne en prod (500K conversations/jour, faithfulness > 90% requise, < $0.10/conversation). Tu dois faire evoluer 3 choses **sans casser la prod** :
1. Migrer le modele principal vers un modele plus recent (meilleur en benchmark, prix different)
2. Mettre a jour le system prompt (3000 -> 1800 tokens, restructure)
3. Re-indexer la knowledge base avec un nouvel embedding model

**Livre :**

1. **Strategie de deploiement** :
   - Concois le pipeline de migration (shadow -> canary -> promote) pour les 3 changements.
   - Lesquels peuvent etre deployes ensemble, lesquels doivent etre isoles ? Justifie.
   - Comment evites-tu un incident type "RAG embedding mismatch" (cf J10) lors de la re-indexation ?

2. **Rollback** :
   - Concois un rollback INSTANTANE pour chacun des 3 changements (modele, prompt, index).
   - Que gardes-tu actif en parallele pour pouvoir revenir en arriere en < 1 min ?
   - Le rollback du prompt et celui de l'index ont-ils les memes contraintes ? (versionning)

3. **Gates qualite (J13)** :
   - Quels gates DOIVENT passer avant chaque promotion ? (gold set, faithfulness, cost, latence)
   - Comment detectes-tu une regression de qualite en canary AVANT qu'elle touche trop d'users ?
   - Sur quels signaux declenches-tu un rollback automatique ?

4. **Cout & portabilite (J11)** :
   - Le nouveau modele change le prix : comment verifies-tu que tu restes < $0.10/conversation ?
   - Le prompt qui marche sur l'ancien modele peut mal marcher sur le nouveau : comment tu testes ?

5. **A/B testing (J13)** :
   - Comment structures-tu l'A/B (split, metrique primary, guardrails, duree) pour decider modele A vs B ?
   - Pourquoi le "meilleur en benchmark offline" ne suffit pas a decider ?

6. **Failure modes** :
   - La re-indexation prend 8h ; pendant ce temps, l'ancien et le nouveau index coexistent. Quel risque, comment l'eviter ?
   - Le canary du nouveau modele degrade la faithfulness a 84%. Que se passe-t-il ?
   - Le rollback lui-meme echoue (le vieux modele a ete deprecie par le provider). Comment tu t'en sors ?

### Criteres de reussite
- [ ] Le pipeline suit shadow -> canary (1%->...) -> promote, avec un gate qualite a chaque etape
- [ ] Les 3 changements sont isoles (un changement a la fois) pour pouvoir attribuer une regression ; sinon impossible de savoir lequel a casse
- [ ] La re-indexation evite le mismatch : re-embedder TOUT le corpus dans un index neuf (blue/green), basculer atomiquement (cf J10)
- [ ] Le rollback est instantane via feature flags + double-run (ancien index + ancien prompt gardes actifs jusqu'a validation)
- [ ] Les gates incluent gold set recall/faithfulness >= baseline, cost <= cible, latence p95 OK
- [ ] L'A/B a une metrique primary business + guardrails + duree 2-4 semaines ; le benchmark offline ne prouve pas l'impact prod
- [ ] Les failure modes sont traites : coexistence d'index (router clairement vers le bon par version), rollback auto si faithfulness chute, plan B si le vieux modele est deprecie (garder une version pinnee / un fallback alternatif)

---

## Exercice 2 : Fusionner les deux designs — Dropbox + Assistant IA documentaire

### Objectif
Combiner les 2 designs de reference du J14 en un seul systeme coherent : un Dropbox-like avec un assistant IA qui repond sur le contenu des fichiers de l'utilisateur. C'est l'exercice de synthese du domaine : storage + RAG + agents + infra LLM + observability.

### Consigne
On greffe sur le Dropbox-like du J14 un **assistant IA** (inspire du LLM Support Assistant) qui permet a chaque user de poser des questions sur SES fichiers ("resume-moi le contrat signe le mois dernier", "quels documents parlent du projet X ?").

Rappel des contraintes pertinentes :
- 50M users actifs/jour, fichiers stockes en blocs dedupliques (S3-like)
- Documents confidentiels et PERSONNELS (isolation stricte par user obligatoire)
- L'assistant doit citer ses sources (les fichiers de l'user), faithfulness elevee
- Cible de cout maitrisee, latence raisonnable (< 5s pour une reponse)

**Livre :**

1. **Architecture combinee** :
   - Dessine (ASCII) comment le RAG se greffe sur le storage existant. Quelles briques du J14 (Dropbox) et lesquelles du J14 (Support) reutilises-tu ?
   - Quand indexes-tu les fichiers (a l'upload ? a la demande ? en batch) ? Tradeoffs.

2. **Isolation multi-user (critique)** :
   - Chaque user ne doit JAMAIS voir un chunk d'un autre user. Comment garantis-tu ca dans le retrieval ? (rappelle le post-mortem RAG du J10)
   - Ou pousses-tu le filtre d'isolation (pre-filter vs post-filter) ?

3. **Indexation a l'echelle** :
   - 50M users, combien de chunks au total (hypothese : 50 docs/user, 20 chunks/doc) ? Sizing de l'index.
   - Comment geres-tu le fait que la plupart des users ne posent jamais de question (indexer a la demande ?) ?

4. **Cout** :
   - Indexer 100% des fichiers de tous les users a l'upload est-il rentable si 5% seulement utilisent l'assistant ?
   - Propose une strategie lazy (indexer au premier usage) et chiffre l'economie d'embedding.

5. **Reutilisation des patterns** :
   - Liste explicitement quels patterns des J10-J13 tu reutilises (hybrid search, reranker, semantic cache, guardrails, tracing, drift).
   - Quel guardrail de sortie est NON NEGOCIABLE ici (et pourquoi) ?

6. **Failure modes** :
   - L'assistant cite un fichier que l'user vient de supprimer. Comment evites-tu ca ?
   - L'index d'un gros user (1M fichiers) noie les petits. Comment isoles-tu ?
   - Un user pose une question sur un fichier qu'il n'a pas le droit de lire (partage revoque). Que se passe-t-il ?

### Criteres de reussite
- [ ] L'architecture reutilise le block store + metadata DB du Dropbox-like ET le LLM Gateway + RAG engine du Support Assistant
- [ ] Le moment d'indexation est discute (upload vs lazy on-demand) avec ses tradeoffs cout/fraicheur
- [ ] L'isolation par user est garantie par un pre-filter user_id DANS la requete vector DB + test d'isolation (rappel explicite du post-mortem J10)
- [ ] Le sizing est calcule (50M * 50 * 20 = 50G chunks en theorie -> d'ou la necessite d'indexer a la demande)
- [ ] La strategie lazy est chiffree : indexer 5% des users economise ~95% du cout d'embedding initial
- [ ] Les patterns reutilises sont listes (hybrid+RRF+reranker, semantic cache par user, guardrails, tracing, drift) ; le groundedness check est identifie comme non negociable
- [ ] Les 3 failure modes sont traites : suppression (invalidation de l'index/citations), noisy neighbor (isolation des gros users), ACL revoquee (pre-filter a jour -> plus de retrieval)
