# Exercices Easy — LLM Infrastructure

---

## Exercice 1 : Rediger une politique de routing

### Objectif
Savoir definir un routeur de prompts realiste pour un produit donne.

### Consigne
Tu travailles sur un produit "assistant email" qui fait 5 types de taches :

1. **Classify** : classer un email dans une categorie (personal, work, spam, newsletter...)
2. **Summarize** : resumer un thread d'emails en 3 lignes
3. **Draft reply** : proposer une reponse courte basee sur l'historique
4. **Extract** : extraire des donnees (date, lieu, montant) d'un email de facture
5. **Answer question** : repondre a une question factuelle basee sur une knowledge base interne

**Donnees :**
- Les modeles disponibles sont : nano (Haiku), mini (gpt-5.4-mini), std (Sonnet 4.6), frontier (Opus 4.6)
- Cout relatif : 1x / 4x / 15x / 80x
- Volume mensuel : 2M appels au total, reparti ~50% classify, ~25% extract, ~15% summarize, ~8% draft, ~2% answer

**Questions :**
1. Pour chaque tache, propose un modele primaire + conditions de basculement vers un tier superieur
2. Calcule le cout relatif total de ta politique vs une politique "frontier pour tout"
3. Pour quelle tache accepterais-tu de passer en frontier et dans quelles conditions ?

### Livrables
Un tableau + les calculs.

### Criteres de reussite
- [ ] Classify et Extract sont mappes sur nano
- [ ] Summarize et Draft reply sont sur mini (sauf exceptions)
- [ ] Answer question est sur std avec fallback vers frontier
- [ ] Le cout total est divise par au moins 10x vs "frontier pour tout"
- [ ] Les conditions de basculement sont explicites (ex: "si la confidence du classifier nano < 0.8, router vers mini")

---

## Exercice 2 : Dessiner une politique de guardrails

### Objectif
Savoir identifier les guardrails necessaires selon le contexte metier.

### Consigne
Pour chacun des produits suivants, liste les guardrails d'**input** et d'**output** que tu mettrais en place, et explique ce qui se passe en cas d'echec (bloquer / retry / fallback / escalation humaine) :

1. **Chatbot medical** qui repond a des questions de patients
2. **Generateur de CV automatique** qui prend les donnees du user et genere un CV
3. **Support customer d'un e-commerce** avec acces a la commande du user
4. **Assistant juridique** qui extrait des clauses de contrats et les commente

### Livrables
Un tableau : produit / input guards / output guards / politique d'echec.

### Criteres de reussite
- [ ] Le chatbot medical mentionne disclaimer "pas un conseil medical" et escalation humaine obligatoire
- [ ] Le generateur de CV a un scrubbing PII et une verification d'output (pas d'hallucination d'experiences)
- [ ] Le support e-commerce a une verification que les commandes citees appartiennent bien au user (authz)
- [ ] L'assistant juridique a une verification de groundedness (chaque clause citee existe bien dans le doc)
- [ ] Chaque produit a au moins un mecanisme de prompt injection detection
- [ ] Au moins 2 produits ont une validation JSON structuree

---

## Exercice 3 : Semantic cache tuning

### Objectif
Comprendre les effets du threshold de similarite sur un semantic cache.

### Consigne
Ton produit est un assistant interne pour des devs qui repond a des questions techniques ("comment deployer un service sur K8s", "comment debugger un memory leak Python", etc.).

Tu as active un semantic cache avec threshold=0.90. Apres 2 semaines :
- Hit rate = 45%
- Mais 3 incidents ou le cache a retourne une reponse incorrecte car la question etait subtilement differente (ex: "comment deployer sur K8s" vs "comment **mettre a jour** un deploy sur K8s")

**Questions :**
1. Quelle est la cause racine probable et quel est le dilemme central ?
2. Liste 4 techniques differentes pour reduire les faux positifs **sans** sacrifier le hit rate
3. Comment mesures-tu si tes changements ameliorent ou degradent la qualite ?
4. A partir de quel moment devrais-tu envisager de desactiver le cache pour cette tache ?

### Livrables
Une analyse structuree avec les 4 techniques et un plan de mesure.

### Criteres de reussite
- [ ] Le dilemme threshold haut (faux negatifs) vs bas (faux positifs) est explicite
- [ ] Les techniques incluent au moins : threshold adaptatif par type de query, distance word-level en plus du cosine, TTL raccourci, segmentation du cache par topic
- [ ] La solution "LLM-as-a-judge" pour valider le cache hit est mentionnee comme option
- [ ] Un plan de mesure inclut : taux de faux positif mesure (gold set) + hit rate + user feedback (thumbs down)
- [ ] La decision de desactiver le cache est conditionnee sur un seuil de faux positifs (ex: > 1%)
