# Exercices Hard — LLM Infrastructure

---

## Exercice 1 : LLM gateway d'entreprise — 99.95% de dispo avec des providers a 99.5%

### Objectif
Concevoir la couche d'infra LLM d'une entreprise entiere avec un SLA superieur a celui de chacune de ses dependances.

### Consigne
Tu concois le gateway LLM central d'un groupe (15 equipes produit, 40 applications). TOUT le trafic LLM du groupe passe par toi.

**Contraintes chiffrees :**
- 8M de requetes/jour, pic a 400 req/s ; 30% du trafic est latence-sensible (chat users), 70% tolerant (jobs, enrichissement)
- SLA du gateway envers les equipes : 99.95% de succes, < 150 ms de latence AJOUTEE au p99
- Providers disponibles : Provider A (frontier, 99.5% dispo historique, 10 $/M tokens), Provider B (comparable, 99.5%, 12 $/M), modele self-hosted (8B, qualite inferieure, 99.9% si bien opere, ~1 $/M)
- Les providers ont des rate limits : A = 500 req/s contractuel, B = 200 req/s
- Budget mensuel : la facture actuelle (sans gateway) est de 280 K$/mois ; la direction exige -40% en 6 mois
- Conformite : certaines equipes (RH, juridique) interdisent que leurs prompts sortent vers les providers US

**Livre :**
1. **Calcul de disponibilite** : montre qu'un provider seul ne tient pas le SLA. Concois la fallback chain (A -> B -> self-hosted) et calcule la disponibilite composee theorique (hypotheses d'independance a discuter : quand les pannes sont-elles correlees ?). Ou la chaine s'arrete-t-elle pour les equipes "conformite" ?
2. **Architecture du gateway** : tous les etages (auth/quotas, routing par politique d'equipe, cache semantique, guardrails, fallback, observabilite) ET sa propre haute disponibilite (le gateway lui-meme ne doit pas devenir le SPOF : deploiement, nombre d'instances pour 400 req/s a 2K req/s/instance, multi-AZ).
3. **Le plan -40%** : decompose la reduction de 280 K$ a 168 K$ : routing par tier (quelle part du trafic peut descendre sur le 8B ? comment le determiner objectivement par eval, pas au doigt mouille ?), caching (hit rate realiste par type de trafic), prompt caching, batching du trafic tolerant. Chiffre chaque levier et montre que la somme atteint -40%.
4. **Rate limits des providers** : au pic (400 req/s), avec A limite a 500 et B a 200, que se passe-t-il si A tombe ? Concois le mecanisme (shedding du trafic tolerant, queue avec backpressure, degradation vers self-hosted) qui protege le trafic latence-sensible.
5. **Failure drill complet** : Provider A degrade (latence x10, pas de panne franche — le cas vicieux). Decris la detection (quel signal ? latence p99 par provider ?), la decision (circuit breaker sur la latence, pas seulement les erreurs), la bascule, et le retour.
6. **Gouvernance** : comment les 15 equipes declarent leurs politiques (latence vs cout vs conformite) sans que tu deviennes le goulot organisationnel ?

### Criteres de reussite
- [ ] La dispo composee est calculee (1 - 0.005 x 0.005 x 0.001 avec independance ~ 99.9999%, puis nuancee : pannes correlees possibles — meme region cloud, meme incident upstream) et la chaine conformite s'arrete au self-hosted EU/on-prem
- [ ] Le gateway est dimensionne (400/2000 -> 1 instance suffit MAIS 3+ multi-AZ pour la dispo) et stateless (cache/quotas externalises dans Redis)
- [ ] Le plan -40% est un tableau chiffre dont la somme atteint ~112 K$ d'economie, avec un mecanisme objectif de tiering (eval par tache : gold sets par equipe, bascule si qualite >= seuil)
- [ ] Le scenario "A tombe au pic" est resolu : 400 req/s > B(200), donc shedding/batching differe du trafic tolerant (70%) + self-hosted, le latence-sensible (120 req/s) tient sur B
- [ ] Le circuit breaker declenche sur latence p99 ET error rate (fenetres glissantes, half-open progressif) — la degradation lente est explicitement geree
- [ ] La gouvernance est self-service (config declarative par equipe : tier, fallback autorise, residence des donnees) avec defaults sains
- [ ] 3 tradeoffs explicites avec consequences acceptees

---

## Exercice 2 : Lancement grand public — survivre au jour 1 (cold start a 50x)

### Objectif
Preparer l'infrastructure LLM d'un lancement produit a fort risque de surcharge, avec arbitrages cout/experience en temps reel.

### Consigne
Ta startup lance dans 6 semaines un assistant IA grand public adosse a un partenariat media majeur. Le marketing prevoit entre 100K et 5M de visiteurs le jour 1 (incertitude totale : x50).

**Contraintes chiffrees :**
- Experience nominale : chat streaming, TTFT < 800 ms, sessions de ~6 messages, ~1.5K tokens in / 300 tokens out par message
- Capacite contractuelle provider frontier : 300 req/s (impossible d'augmenter en moins de 3 semaines) ; self-hosted possible : jusqu'a 40x H100 livrables en 2 semaines (8B ou 32B)
- Si 5M de visiteurs : pic estime 8 000 sessions simultanees, ~2 200 req/s de messages
- Tresorerie : le jour 1 peut couter cher, mais le RUN mensuel post-lancement doit rester < 60 K$/mois pour 300K users actifs attendus
- Un crash public le jour 1 = echec du partenariat (la dispo percue prime sur la qualite percue)

**Livre :**
1. **Arithmetique de l'ecart** : 2 200 req/s vs 300 req/s de capacite frontier : l'ecart est x7. Quelles sont les SEULES strategies possibles (reduire la demande admise, augmenter l'offre self-hosted, degrader le cout unitaire) ? Chiffre chacune.
2. **Architecture d'admission** : concois la waiting room (file d'attente d'acces equitable, estimation du temps d'attente, et l'UX d'attente qui ne fait pas fuir) + les quotas par session (nombre de messages ?) pour borner la demande. A quel niveau de charge actives-tu chaque mecanisme ?
3. **Strategie de capacite mixte** : repartis le trafic entre frontier (300 req/s) et self-hosted (40 GPUs : calcule la capacite avec un 8B a ~15 req/s/GPU vs un 32B a ~4 req/s/GPU). Qui recoit quel modele (premiers messages ? users en file ? tout le monde pareil ?) et le user le sait-il ?
4. **Plan de charge par paliers** : definis 4 paliers (100K / 500K / 2M / 5M visiteurs) avec, pour chacun : config de routing, mecanismes actives, experience degradee assumee, cout estime du jour 1.
5. **Le RUN post-lancement** : 300K users actifs, < 60 K$/mois : calcule le cout par user et la config cible (mix de modeles, cache, quotas free tier) qui tient le budget. Que deviennent les 40 GPUs ?
6. **War room jour 1** : les 5 metriques sur l'ecran principal, les 3 leviers actionnables a chaud (sans deploy), et le critere GO/NO-GO pour ouvrir le palier suivant.

### Criteres de reussite
- [ ] L'arithmetique est posee : frontier seul = ~14% de la demande pic ; self-hosted 8B = 40 x 15 = 600 req/s, 32B = 160 req/s ; meme tout combine (300+600) < 2 200 -> l'admission control n'est PAS optionnelle, c'est mathematique
- [ ] La waiting room est concue (file FIFO/token, ETA affiche, contenu d'attente) avec seuils d'activation chiffres et quotas par session (ex : 10 messages) — la demande est bornee par construction
- [ ] La strategie mixte est justifiee par la contrainte "dispo percue > qualite percue" : 8B pour le volume, frontier reserve (premiers messages pour le wow, ou users payants/presse) — choix argumente et transparent ou non (tradeoff discute)
- [ ] Les 4 paliers existent avec configs et couts distincts, et le palier 5M assume une degradation explicite (modele 8B majoritaire + file d'attente) plutot qu'un crash
- [ ] Le RUN est calcule : 300K users x usage estime -> mix cache (20-30%) + 8B majoritaire + frontier minoritaire, confronte aux 60 K$/mois ; les GPUs sont rendus/downsizes selon le contrat
- [ ] La war room a 5 metriques operationnelles (req/s, TTFT p95, taux d'admission, error rate par provider, cout/heure) et 3 leviers a chaud (quotas, ratio de routing, taille de la file)
- [ ] 3 tradeoffs explicites, dont qualite vs disponibilite le jour 1
