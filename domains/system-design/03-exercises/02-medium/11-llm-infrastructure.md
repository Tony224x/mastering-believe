# Exercices Medium — LLM Infrastructure

---

## Exercice 1 : Concevoir un LLM gateway complet

### Objectif
Assembler les briques d'une couche d'infra LLM (routing, cache, guardrails, fallback) dans le bon ordre.

### Consigne
Ta plateforme fait **2M de requetes LLM/jour** vers 3 providers (un modele frontier, un modele mid-tier, un modele open-source self-hosted). Tu construis le gateway interne par lequel TOUTES les equipes passent.

1. **Ordre des middlewares** : place dans le bon ordre et justifie : semantic cache, guardrails input, guardrails output, routing, rate limiting/quotas par equipe, logging/tracing, fallback. (Indice : ou placer le cache pour maximiser les economies sans servir de reponses non valides ?)
2. **Routing par tier** : 60% des requetes sont simples (classification, extraction), 30% moyennes, 10% complexes (raisonnement). Propose la politique de routing (comment classifier la requete ? regles statiques, modele leger, ou declaration par l'appelant ?) et calcule l'economie si les couts sont 15 $/M tokens (frontier), 2 $/M (mid), 0.3 $/M (self-hosted) — aujourd'hui tout part sur le frontier.
3. **Semantic cache** : seuil de similarite a 0.95 vs 0.85 — decris ce que chaque choix change (hit rate vs faux hits). Quels types de requetes EXCLURE du cache ?
4. **Fallback chain** : le provider frontier a une panne de 45 min. Decris le comportement attendu du gateway etape par etape (circuit breaker, bascule, qualite degradee signalee ?).
5. **Quotas** : une equipe consomme 80% du budget. Comment implementer des quotas par equipe sans bloquer brutalement un service en prod ?

### Criteres de reussite
- [ ] Ordre coherent : auth/rate limiting -> guardrails input -> cache lookup -> routing -> appel + fallback -> guardrails output -> cache write -> logging (tracing partout)
- [ ] Economie calculee : cout pondere = 0.6x0.3 + 0.3x2 + 0.1x15 = 2.28 $/M vs 15 $/M, soit ~85% d'economie
- [ ] Cache : 0.95 = hit rate faible mais sur, 0.85 = plus de hits mais risque de reponse a cote ; exclure : requetes personnalisees/contextuelles, donnees temps reel, contenus sensibles
- [ ] Fallback : N erreurs/timeouts -> circuit open -> bascule mid-tier automatique + header/flag "degraded" + alerte + retour progressif (half-open)
- [ ] Quotas : budgets mensuels avec alerte a 80%, throttling progressif (puis downgrade de tier plutot que blocage), dashboard par equipe

---

## Exercice 2 : Tracker et reduire le cout par feature

### Objectif
Mettre en place la comptabilite analytique des tokens et un plan de reduction des couts chiffre.

### Consigne
Ta facture LLM mensuelle atteint **~91 000 $**. Le breakdown par feature manque, mais tu sais que : le chatbot support fait 1.5M req/mois (prompt moyen 3 000 tokens dont 2 200 de system prompt + contexte RAG, completion 250 tokens), la generation de resumes fait 400K req/mois (prompt 8 000, completion 600), et l'autocomplete fait 30M req/mois (prompt 500, completion 30). Tarifs du modele utilise partout : 3 $/M input, 15 $/M output.

1. Calcule le cout mensuel de chaque feature et son poids dans la facture. Quelle surprise probable ?
2. Pour chacune des 3 features, propose l'optimisation LA PLUS rentable parmi : prompt caching (input cache a -90% sur la partie cachee), downgrade vers un modele 10x moins cher, batching API (-50%), reduction du system prompt. Chiffre le gain de chaque optimisation choisie.
3. Le prompt caching exige que la partie stable soit un PREFIXE. Reorganise le prompt du chatbot (system prompt, contexte RAG, historique, question) pour maximiser le cache hit.
4. Quelles 3 metriques de cout integres-tu au dashboard pour eviter que ca derive a nouveau ?
5. Quel garde-fou automatique poses-tu contre une explosion accidentelle (bug de boucle d'une equipe qui genere 100x le trafic) ?

### Criteres de reussite
- [ ] Couts calcules : chatbot ~19.1K$ (13.5K input + 5.6K output), resumes ~13.2K$ (9.6K + 3.6K), autocomplete ~58.5K$ (45K + 13.5K) ; la surprise : l'autocomplete "petite feature" represente ~64% de la facture car le VOLUME domine
- [ ] Optimisations adaptees : autocomplete -> modele 10x moins cher (gain maximal), chatbot -> prompt caching sur les 2 200 tokens stables, resumes -> batching si async
- [ ] Prompt reorganise : system prompt fixe en premier, puis contexte RAG, puis historique, question en dernier
- [ ] Metriques : cout/jour par feature et par equipe, cout par requete, tokens in/out par requete (tendance)
- [ ] Garde-fou : budget cap par API key avec alerte + kill switch / throttling automatique au-dela d'un seuil journalier

---

## Exercice 3 : Guardrails sous contrainte de latence

### Objectif
Concevoir une chaine de guardrails efficace sans detruire la latence percue.

### Consigne
Ton assistant grand public stream ses reponses (les tokens s'affichent au fil de l'eau). SLO : premier token affiche < 1 s. Tu dois implementer : detection PII (input et output), filtre toxicite, validation de format JSON pour les appels internes, et blocage des sujets interdits (conseil medical).

1. Classe chaque guardrail selon sa technique : regex/deterministe, classifier ML leger, LLM-judge. Donne la latence typique de chaque technique (ordre de grandeur).
2. **Probleme du streaming** : le filtre de toxicite sur l'output complet exigerait d'attendre la fin de la generation. Propose 2 strategies compatibles avec le streaming et leur risque residuel.
3. Ou places-tu chaque guardrail (avant le LLM, pendant le stream, apres) pour respecter le SLO de 1 s ?
4. Le guardrail "sujets interdits" en LLM-judge ajoute 400 ms en serie. Propose une optimisation (parallelisation avec la generation, classifier en cascade, cache des verdicts).
5. Que fait le systeme quand un guardrail declenche en PLEIN stream (l'utilisateur a deja vu 200 tokens) ? Decris l'UX et le logging.

### Criteres de reussite
- [ ] Classification correcte : PII = regex + NER leger (~1-10 ms), toxicite = classifier (~10-50 ms), JSON = validation deterministe (~1 ms), sujets interdits = classifier ou LLM-judge (~100-500 ms)
- [ ] Streaming : verification par fenetres/chunks au fil du stream, ou buffer des N premiers tokens avant affichage ; risque residuel : contenu toxique revele en debut de fenetre / latence ajoutee par le buffer
- [ ] Placement : input guards avant l'appel (parallelisables entre eux), output guards en chunked streaming, JSON validation a la fin (non-streame pour les appels internes)
- [ ] Optimisation : cascade (classifier rapide filtre 95%, LLM-judge seulement sur les cas ambigus) et/ou run en parallele du debut de generation
- [ ] En cas de declenchement mid-stream : couper le stream, remplacer par un message generique, logger l'evenement complet (prompt, output partiel, regle declenchee) pour revue
