# Exercices Medium — Observabilite & MLOps

---

## Exercice 1 : Instrumenter un pipeline RAG de bout en bout

### Objectif
Concevoir le tracing complet d'une application LLM multi-etapes et definir les metriques derivees.

### Consigne
Ton application RAG de support traite 50K requetes/jour : query rewriting -> retrieval hybride -> reranking -> generation -> guardrails output. Aujourd'hui, tu n'as que les logs applicatifs et le taux d'erreur HTTP. Un utilisateur se plaint : "la reponse d'hier a 15h32 etait fausse". Tu es incapable d'investiguer.

1. **Modele de trace** : definis la structure trace/spans pour une requete (un span par etape). Pour chaque span, liste les attributs a capturer (inputs, outputs, latence, tokens, scores...).
2. **Investigation** : avec ce tracing en place, deroule les etapes pour diagnostiquer la reponse fausse d'hier 15h32 (quel span regardes-tu en premier ? quels sont les 3 diagnostics possibles d'une reponse fausse en RAG ?).
3. **Metriques derivees** : a partir des traces, definis 6 metriques agregees pour le dashboard (latence par etape, qualite, cout). Lesquelles en alerte ?
4. **Echantillonnage** : tracer 100% des requetes avec inputs/outputs complets coute cher et pose des questions PII. Propose une politique d'echantillonnage intelligente (que garder a 100% ? quoi sampler ?).
5. **Evaluation continue** : sur quel sous-ensemble fais-tu tourner un LLM-judge quotidien, et quelles 3 dimensions note-t-il ?

### Criteres de reussite
- [ ] La trace contient >= 5 spans avec attributs pertinents par type (retrieval : query, top-k docs + scores ; generation : prompt, completion, tokens, modele ; guardrails : verdicts)
- [ ] Les 3 diagnostics RAG : mauvais retrieval (docs hors sujet), bon retrieval mais hallucination du LLM, ou question hors corpus ; l'investigation commence par le span retrieval
- [ ] 6 metriques plausibles : latence p95 par etape, cout/requete, taux de guardrail triggers, score de retrieval moyen, taux de reponses sans source, feedback negatif ; alertes sur erreurs, latence, guardrails
- [ ] Echantillonnage : 100% des erreurs/guardrail triggers/feedback negatifs et des metadata, sampling (1-10%) des payloads complets, avec masquage PII
- [ ] LLM-judge sur echantillon aleatoire quotidien (ex : 200 traces) notant groundedness, pertinence, completude

---

## Exercice 2 : Lancer et analyser un A/B test de modele

### Objectif
Concevoir un A/B test ML rigoureux et interpreter des resultats ambigus.

### Consigne
Tu remplaces le modele de ranking de ton site e-commerce (v3 -> v4, meilleur de +4% en NDCG offline). Trafic : 500K sessions/jour, taux de conversion ~3%, revenu moyen par session 2.50 €.

1. **Design du test** : definis la metrique primaire, 3 guardrail metrics, l'unite de randomisation (requete, session ou utilisateur ? pourquoi ?), et le split.
2. **Duree** : tu veux detecter un uplift relatif de +2% sur la conversion (3.00% -> 3.06%). Avec ~250K sessions/jour par bras, l'ordre de grandeur requis est de plusieurs centaines de milliers de sessions par bras. Sans formule exacte, explique POURQUOI un petit uplift sur une metrique rare exige un echantillon enorme, et pourquoi arreter le test au bout de 3 jours "parce que c'est deja significatif" est un piege (peeking).
3. **Resultats apres 3 semaines** : conversion +1.8% (p=0.04), revenu/session +0.5% (p=0.31), latence p95 +40 ms, et -6% de conversion sur le segment mobile bas de gamme. Decision : ship, no-ship, ou autre chose ? Argumente.
4. **Inference causale piegee** : le PM note que "les users exposes a v4 qui ont clique sur une reco achetent 2x plus". Pourquoi cette comparaison est-elle invalide ?
5. **Rollout** : tu decides de shipper. Decris le plan de rollout progressif avec les criteres de rollback automatique.

### Criteres de reussite
- [ ] Primaire = conversion (ou revenu/session) ; guardrails = latence p95, taux d'erreur, CTR ou engagement ; randomisation par UTILISATEUR (experience coherente, pas de contamination intra-session)
- [ ] L'explication relie variance des proportions rares et taille d'echantillon ; le peeking est identifie comme inflation du risque de faux positif
- [ ] La decision discute le conflit : metrique primaire OK mais regression segment mobile + latence -> investiguer le segment avant ship global (ship partiel ou fix d'abord est acceptable, ship aveugle non)
- [ ] La comparaison "cliqueurs" est un biais de selection (conditionnement post-traitement) : les cliqueurs ne sont pas un groupe randomise
- [ ] Rollout : 5% -> 25% -> 50% -> 100% avec fenetres d'observation et rollback automatique sur seuils (conversion, latence, erreurs)

---

## Exercice 3 : Pipeline de retraining automatique declenche par le drift

### Objectif
Concevoir la boucle complete monitoring -> drift -> retraining -> validation -> deploiement.

### Consigne
Ton modele de scoring credit tourne en production depuis 8 mois. Les performances se degradent lentement (les labels arrivent avec 60 jours de retard). Tu dois automatiser la boucle de maintenance.

1. **Detection** : tu calcules le PSI hebdomadaire de chaque feature. Donne les seuils standard d'interpretation du PSI et definis la politique de declenchement (1 feature a 0.3 ? 5 features a 0.12 ? le score lui-meme ?).
2. **Faux declencheurs** : donne 2 situations ou le PSI s'envole SANS que le modele soit perime (et comment les filtrer avant de retrainer pour rien).
3. **Pipeline de retraining** : decris les etapes automatisees (extraction donnees fraiches, training, evaluation offline, comparaison champion/challenger, registry). Quels criteres objectifs pour que le challenger remplace le champion ?
4. **Validation sans labels frais** : les labels ont 60 jours de retard — le challenger est entraine sur des donnees dont les labels datent. Quels risques et quelles validations complementaires avant de deployer ?
5. **Deploiement** : shadow puis canary. Que compares-tu en shadow (sans labels !) et quel critere de promotion ?

### Criteres de reussite
- [ ] Seuils PSI : < 0.1 stable, 0.1-0.25 a surveiller, > 0.25 drift significatif ; la politique combine nombre de features et importance des features driftees + PSI du score
- [ ] Faux declencheurs : changement de mix marketing/saisonnalite (nouveaux segments legitimes), bug ou changement de schema dans le pipeline de donnees upstream
- [ ] Le pipeline contient une comparaison champion/challenger sur un meme jeu de test fige + criteres chiffres (ex : AUC challenger >= champion + epsilon, pas de degradation par segment)
- [ ] Risque identifie : le drift recent n'est pas couvert par les labels disponibles ; validations : backtesting glissant, stabilite des distributions de scores, revue des features importances
- [ ] Shadow : comparer distributions de scores, taux d'approbation simules, desaccords champion/challenger sur les memes dossiers ; promotion si stable sur N semaines puis canary avec rollback
