# Exercices Hard — RAG Architecture

---

## Exercice 1 : RAG reglemente a l'echelle — assistant medical interne

### Objectif
Concevoir un RAG complet sous contraintes reglementaires, de fraicheur et de fiabilite extremes.

### Consigne
Un groupe hospitalier (40 000 soignants) veut un assistant qui repond aux questions sur les protocoles de soin internes.

**Contraintes chiffrees :**
- Corpus : 800 000 documents (protocoles, notices medicaments, procedures), mis a jour en continu — un protocole PERIME servi = risque patient, fraicheur exigee < 5 minutes apres publication
- 50 000 requetes/jour, pics a 30 req/s (releves de garde)
- Latence : < 3 s au p95 pour une reponse complete avec citations
- ZERO reponse non sourcee : chaque affirmation doit citer un document verifiable, sinon refuser de repondre
- Audit : chaque reponse doit etre rejouable a l'identique 5 ans apres (quel index, quels docs, quel modele)
- Hebergement on-premise uniquement (donnees de sante) ; budget GPU : 8x A100
- Taux de "je ne sais pas" tolere : < 15% (au-dela, les soignants abandonnent l'outil)

**Livre :**
1. **Architecture complete** : pipeline d'ingestion (fraicheur < 5 min : comment ?), index (hybrid ? dimensionne pour 800K docs ~ 16M chunks), serving (8 GPUs a repartir entre embedding, reranking, generation : propose la repartition).
2. **Zero reponse non sourcee** : concois le mecanisme de bout en bout : groundedness check (comment verifier que CHAQUE phrase est appuyee par un chunk ?), seuil de confiance de retrieval en-dessous duquel on refuse, format de citation verifiable (lien direct au paragraphe). Quel est le tradeoff avec le taux de refus < 15% ?
3. **Le probleme des documents perimes** : un protocole v2 remplace la v1, mais la v1 reste dans l'index 4 minutes. Pire : des chunks v1 et v2 peuvent etre retrieves ENSEMBLE. Concois la solution (versioning des docs dans l'index, filtre at-query-time, invalidation prioritaire des docs critiques).
4. **Auditabilite 5 ans** : que faut-il snapshotter pour rejouer une reponse de 2026 en 2031 ? (index versionne ? logs des chunks servis ? poids du modele ?) Estime le cout de stockage de cette auditabilite.
5. **Evaluation specifique au risque** : construis le plan d'eval avant mise en prod : gold set (qui le construit ? combien de questions ?), metriques avec seuils de no-go, et red teaming specifique (questions piege : dosages, contre-indications).
6. **3 tradeoffs explicites** dont au moins un entre securite (refus) et adoption (15%).

### Criteres de reussite
- [ ] L'ingestion < 5 min est event-driven (webhook/CDC du DMS -> queue prioritaire -> upsert), avec invalidation IMMEDIATE de l'ancienne version avant meme la fin du re-embedding (tombstone/filtre)
- [ ] La repartition GPU est posee et justifiee (ex : 1 embedding, 1 reranker, 5 generation, 1 marge/eval) avec un calcul de capacite vs 30 req/s
- [ ] Le groundedness check est concret : verification phrase-par-phrase contre les chunks cites (NLI/entailment ou LLM-judge), seuil de retrieval en-dessous duquel -> refus avec redirection humaine
- [ ] Le versioning d'index est explicite : doc_id + version + valid_from/valid_to, filtre at-query (only latest), et la fenetre de coexistence v1/v2 est eliminee par tombstone synchrone
- [ ] L'audit snapshot : log immuable par reponse (query, chunks + versions, prompt, modele + version, reponse), cout estime (50K req/jour x ~50 Ko ~ 2.5 Go/jour ~ 4.5 To/5 ans, negligeable) — pas besoin de snapshotter tout l'index a chaque requete si les chunks sont versionnes
- [ ] Le plan d'eval inclut un gold set construit AVEC des soignants (100-500 questions), des seuils no-go chiffres (groundedness, recall), et du red teaming sur les cas dangereux
- [ ] 3 tradeoffs dont securite vs adoption (seuil de refus calibre sur le gold set)

---

## Exercice 2 : Refondre un RAG qui coute trop cher et repond trop lentement

### Objectif
Optimiser un pipeline RAG existant sur 3 axes simultanes (cout, latence, qualite) avec des contraintes contradictoires.

### Consigne
Tu reprends le RAG d'un editeur SaaS (assistant utilisateur sur la doc + tickets). Etat actuel mesure :

**Pipeline actuel (sequentiel) :**
- Query rewriting (LLM frontier) : 600 ms, 0.4 cent/req
- Retrieval dense top-20 (vector DB managee) : 350 ms
- Reranking LLM (frontier, "note chaque doc de 1 a 10") : 1 800 ms, 1.2 cent/req
- Generation (frontier, contexte = 20 chunks entiers ~ 12K tokens) : 2 700 ms, 4.5 cents/req
- **Total : ~5.5 s p50, ~9 s p95, ~6.1 cents/requete, 1.2M req/mois = ~73 K$/mois**

**Objectifs imposes par la direction :**
- p95 < 2.5 s (les users decrochent a 3 s)
- Cout < 20 K$/mois (le pricing du produit ne supporte pas plus)
- Qualite : le CSAT des reponses (72%) ne doit PAS baisser de plus de 2 points
- Delai : 8 semaines, 2 ingenieurs

**Livre :**
1. **Audit du pipeline** : pour chaque etape, identifie le levier (supprimer ? remplacer par moins cher ? paralleliser ? reduire ?) et estime l'impact latence + cout de chaque changement.
2. **Pipeline cible** : propose le nouveau pipeline avec chiffres etape par etape (latence p50/p95, cout/req). Verifie que tu tiens les 3 objectifs SIMULTANEMENT. Les choix attendus a discuter : cross-encoder dedie vs reranking LLM, generation sur modele plus petit, compression du contexte (moins de chunks ? chunks tronques ?), cache semantique (quel hit rate supposer ?), query rewriting conditionnel.
3. **Le piege qualite** : chaque optimisation (petit modele, moins de contexte) risque de faire chuter le CSAT. Concois le protocole de validation : eval offline sur gold set AVANT, A/B test online APRES, et les seuils de rollback automatique.
4. **Risque du cache semantique** : un user demande "comment supprimer mon compte" et recoit la reponse cachee de "comment supprimer un projet". Quelles protections (seuil, scope par tenant/page, exclusions) ?
5. **Sequencement 8 semaines** : ordonne les chantiers par ratio gain/risque ; quel quick win en semaine 1-2 ? Que fais-tu si a la semaine 6 le CSAT a baisse de 3 points ?
6. **Calcul final** : presente le tableau avant/apres (latence, cout mensuel, CSAT attendu) et les 3 tradeoffs majeurs acceptes.

### Criteres de reussite
- [ ] L'audit identifie les 2 postes dominants : reranking LLM (1.8 s, remplacable par cross-encoder ~80 ms et ~100x moins cher) et generation sur 12K tokens (reductible : top-5 apres rerank ~ 3K tokens)
- [ ] Le pipeline cible est chiffre etape par etape et tient les objectifs : ex. rewriting conditionnel/petit modele (~100 ms), retrieval parallele a autre chose si possible, cross-encoder top-20 -> top-5 (~80 ms), generation modele mid-tier sur 3K tokens (~1.2 s), total p50 ~1.7 s, cout ~1-1.5 cent/req -> ~15 K$/mois avec cache
- [ ] Le cache semantique est borne par des protections explicites (seuil eleve 0.95+, scope tenant + page, exclusion des requetes avec contexte personnel) apres l'incident decrit
- [ ] Le protocole qualite est complet : gold set + LLM-judge en offline, A/B avec CSAT comme metrique primaire, rollback automatique si -2 points
- [ ] Le sequencement met le cross-encoder et la reduction de contexte en premier (gain maximal, risque modere) et prevoit le plan B a la semaine 6 (revenir au modele superieur pour la generation, garder les gains de reranking)
- [ ] Le tableau avant/apres est presente avec les 3 tradeoffs assumes
