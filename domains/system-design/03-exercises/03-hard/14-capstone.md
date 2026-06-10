# Exercices Hard — Capstone

---

## Exercice 1 : Design complet — Plateforme de trading retail (type Robinhood), 60 min

### Objectif
Capstone classique de niveau senior : un systeme transactionnel temps reel ou TOUT compte (latence, exactitude, conformite, pics extremes).

### Consigne
Concois la plateforme de trading actions pour particuliers : consultation des cours en temps reel, passage d'ordres, portefeuille, et conformite.

**Contraintes chiffrees :**
- 10M d'utilisateurs, 2M actifs par jour ; 5M d'ordres/jour en regime normal
- Jour de panique (krach, meme stock GameStop) : 20x les ordres en 30 min, 50x les consultations de cours
- Cours temps reel : 8 000 symboles, ticks recus du market data feed a ~500K updates/sec en pic, affiches aux users avec < 500 ms de fraicheur
- Ordre : soumis -> route vers le broker d'execution en < 100 ms p99 ; AUCUN ordre perdu ou duplique
- Reglementaire : best execution tracable, audit complet 7 ans, et le systeme ne doit JAMAIS afficher un solde faux (cash + positions)
- SLO : 99.95% de dispo sur le passage d'ordres (les pannes pendant un krach = unes de presse + class action)

**Deroule le framework complet (60 min) :**
1. **Clarification et estimation** (8 min) : QPS ordres (normal/panique), fan-out du market data (500K ticks/s vers combien de clients connectes ?), stockage audit 7 ans.
2. **High-level design** : separation claire des plans — market data (read, broadcast), trading (write, transactionnel), portefeuille (consistency forte), audit (append-only).
3. **Deep dive — market data fan-out** : 500K ticks/s en entree, 2M clients WebSocket. Pourquoi on ne forward PAS chaque tick a chaque client (calcul du fan-out naif en messages/sec !) ; concois l'agregation (conflation : dernier prix par symbole a 2-4 Hz par client, subscriptions par symbole).
4. **Deep dive — chemin de l'ordre** : de l'app au broker : validation (solde suffisant : comment verifier sous concurrence ? reservation du cash), idempotence, machine a etats de l'ordre, et que se passe-t-il si le broker timeout (ordre passe ou pas ?).
5. **Le jour de panique** : quel composant tombe en premier a 20x ? Concois les protections (queue d'admission sur les ordres ? jamais ! ou alors ? degradation des cours temps reel d'abord ?), et la communication aux users. Discute le choix delicat : refuser des ordres vs les mettre en file.
6. **Bottlenecks, conformite et extensions** : audit immuable, best execution, et extension crypto/24-7.

### Criteres de reussite
- [ ] Estimation posee : ordres normal ~58/s en moyenne, panique ~20x sur 30 min concentre (~2 300/s+) ; fan-out naif calcule (500K x 2M = 10^12 msg/s = impossible) qui FORCE la conflation
- [ ] Les plans sont separes avec des modeles de consistency differents : market data = eventual/conflated, ordres + cash = fortement consistant (reservation atomique du solde)
- [ ] La conflation est concue : etat courant par symbole, push du dernier prix a frequence bornee par client, subscriptions — et le calcul du fan-out apres conflation est refait (2M clients x ~10 symboles x 2 Hz = 40M msg/s, traite par une fleet de gateways)
- [ ] Le chemin d'ordre traite le timeout broker explicitement : statut UNKNOWN + reconciliation par order_id idempotent aupres du broker, jamais de re-soumission aveugle
- [ ] Le jour de panique a une position claire : on protege le passage d'ordres (jamais de perte silencieuse ; file persistante avec ack explicite OU refus franc et affiche), on degrade le market data en premier (frequence reduite)
- [ ] La verification du solde sous concurrence est resolue (reservation/escrow atomique, pas de check-then-act)
- [ ] 4+ tradeoffs explicites et 3 failure modes avec mitigation

---

## Exercice 2 : Design complet — Copilote d'entreprise sur les donnees internes (type Glean/enterprise assistant), 60 min

### Objectif
Capstone IA de niveau senior : assembler RAG, agents, LLM infra, permissions et observabilite en UN systeme coherent pour 100 entreprises clientes.

### Consigne
Concois un copilote SaaS B2B qui repond aux questions des employes sur TOUTES les donnees internes de leur entreprise (Drive, Slack, Notion, Jira, CRM) et peut executer des actions simples (creer un ticket, poser une reunion).

**Contraintes chiffrees :**
- 100 entreprises clientes, de 200 a 20 000 employes ; total 400K utilisateurs, ~1.2M requetes/jour
- Connecteurs : 5 sources par tenant en moyenne, 10M de documents pour le plus gros tenant ; fraicheur < 15 min
- Permissions : le copilote ne doit JAMAIS reveler un document que l'employe ne peut pas ouvrir lui-meme (ACLs natives de chaque source, qui CHANGENT en continu)
- Latence : reponse complete < 4 s p95 ; actions (creer un ticket) avec confirmation explicite de l'utilisateur
- Cout : marge brute > 70% sur un prix de 20 $/user/mois pour le tier standard — donc cout infra+LLM < 6 $/user/mois
- Securite : un incident de fuite cross-tenant ou cross-ACL = perte du business ; SOC2 + audit des actions

**Deroule le framework complet (60 min) :**
1. **Clarification et estimation** (8 min) : QPS (moyen/pic), volume d'indexation (100 tenants, fraicheur 15 min), budget LLM par requete pour tenir 6 $/user/mois (calcule : 1.2M req/jour / 400K users = 3 req/user/jour -> combien de $ par requete max ?).
2. **High-level design** : pipeline d'ingestion multi-connecteurs, index par tenant, retrieval + generation, agent d'actions, gateway LLM, observabilite. Isolation multi-tenant a CHAQUE etage.
3. **Deep dive — permissions dynamiques** : les ACLs changent en continu (un employe quitte un projet a 14h, ne doit plus voir les docs a 14h15). Compare : ACL copiees dans l'index (sync en continu) vs verification live a la requete (latence ?) vs hybride (filtre a l'index + re-check des top-k). Choisis, chiffre la latence et le risque residuel.
4. **Deep dive — budget par requete** : decompose ta cible cout/requete (retrieval, rerank, generation, tracing) et montre les leviers pour tenir la marge (routing par complexite, cache, petit modele par defaut). Que fais-tu pour le tenant qui abuse (10x la moyenne) ?
5. **Deep dive — actions avec confirmation** : concois le flux d'action sur (proposition structuree -> preview -> confirmation -> execution avec token OAuth de L'UTILISATEUR, pas un compte de service global) et l'audit trail. Pourquoi le token utilisateur est-il le seul choix sur ?
6. **Bottlenecks et risques** : les 3 pires (re-indexation d'un gros tenant, ACL sync en retard, fuite de prompt entre tenants via le cache) et leurs mitigations. Extensions : analytics d'adoption, memoire personnelle par employe.

### Criteres de reussite
- [ ] L'estimation aboutit a un budget par requete : 6 $/user/mois / ~90 req/user/mois = ~6-7 cents max par requete TOUT compris — ce chiffre doit structurer le design LLM (pas de frontier systematique)
- [ ] L'isolation multi-tenant est presente a chaque etage (index/namespace par tenant, cles de cache prefixees par tenant, quotas, chiffrement par tenant) et le cache semantique est explicitement scope par tenant (la fuite via cache est nommee)
- [ ] Le choix de permissions est l'hybride : filtre ACL a l'index (rapide, parfois stale) + re-verification live des top-k avant generation (~50-100 ms) ; le risque residuel (fenetre de staleness sur le filtre initial) est borne et chiffre (15 min max, re-check live le couvre)
- [ ] Le flux d'action utilise l'OAuth de l'utilisateur (le copilote ne peut pas faire plus que l'employe lui-meme — c'est la SEULE garantie structurelle) avec preview + confirmation + audit log immuable
- [ ] Le budget est decompose avec leviers chiffres (cache 20-30%, modele leger par defaut + escalade, rerank cross-encoder dedie) et une reponse au tenant abusif (quotas du contrat, throttling, upsell)
- [ ] Les 3 risques sont traites avec mitigation concrete (file d'indexation par tenant avec fairness, monitoring du lag ACL avec alerte, scoping strict du cache teste en CI)
- [ ] 4+ tradeoffs explicites (fraicheur vs cout d'indexation, latence vs verification ACL, qualite vs marge, autonomie de l'agent vs confirmation)
