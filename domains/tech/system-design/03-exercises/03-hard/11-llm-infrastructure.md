# Exercices Hard — LLM Infrastructure

---

## Exercice 1 : Concevoir une LLM Gateway pour 50M requetes/jour

### Objectif
Concevoir une LLM Gateway complete, multi-provider, multi-tenant, capable de tenir le volume, le cout, la securite et la fiabilite. Design d'entretien senior.

### Consigne
Tu es l'architecte de la plateforme LLM interne d'une grosse boite. Toutes les equipes produit passent par ta Gateway.

**Chiffres :**
- 50M requetes/jour, pic x5 a certaines heures
- 40 equipes produit (tenants internes), profils tres differents (chatbot, batch, code, classification)
- Budget LLM global : $400K/mois, a repartir et a controler par equipe
- Providers : 3 externes + 2 modeles self-hosted (GPU cluster)
- Contraintes : SLA 99.95%, aucun PII ne doit fuiter dans les logs, chaque equipe a un quota

**Livre :**

1. **Architecture** :
   - Dessine les briques de la Gateway (ASCII) et le flow d'une requete.
   - Quelles briques sont sur le chemin critique (latence) et lesquelles sont async ?
   - Comment rends-tu la Gateway elle-meme scalable et sans single point of failure ?

2. **Routing & cout** :
   - Comment routes-tu 50M requetes vers le bon modele/provider en respectant le budget de chaque equipe ?
   - Comment empeches-tu une equipe de cramer le budget d'une autre ?
   - Estime le QPS pic et le nombre d'instances de Gateway necessaires (1 instance ~ 2000 req/s).

3. **Cache** :
   - Quelle place pour un semantic cache a cette echelle ? Par tenant ? Global ?
   - Comment estimes-tu le ROI du cache (cout du cache vs economie LLM) ?

4. **Fiabilite (SLA 99.95%)** :
   - Concois la chaine de fallback + circuit breaker pour atteindre 99.95%.
   - Que fais-tu quand TOUS les providers externes sont degrades en meme temps ?
   - Comment geres-tu les rate limits imposes par les providers externes ?

5. **Securite & gouvernance** :
   - Comment garantis-tu qu'aucun PII ne finit dans les logs/traces ?
   - Comment factures-tu/attribues-tu le cout par equipe (showback/chargeback) ?
   - Comment versionnes-tu et deploies-tu les prompts de maniere controlee ?

6. **Observability & failure modes** :
   - Les 8 metriques critiques de la Gateway + leurs seuils.
   - Que se passe-t-il si le GPU cluster self-hosted tombe en plein pic ?
   - Le runbook d'urgence en 5 etapes si le cout horaire explose (boucle infinie d'un agent client).

### Criteres de reussite
- [ ] L'architecture separe le chemin critique (routing, cache lookup, call) de l'async (logging, scoring, billing)
- [ ] La Gateway est stateless + horizontalement scalable (cache/quotas dans un store partage Redis)
- [ ] Le routing respecte un budget par tenant avec quota + rate limiting (token bucket par equipe)
- [ ] Le QPS pic est calcule (~2900 req/s avg, ~14500 pic) et le nombre d'instances en decoule
- [ ] Le semantic cache est par tenant (pas global, pour le PII) avec un ROI chiffre
- [ ] La chaine de fallback atteint 99.95% ; un mode degrade existe quand tout est down (file d'attente, reponse cache, message clair)
- [ ] Le PII est scrub AVANT logging ; chargeback par tenant via spans `gen_ai.*` agreges
- [ ] Le runbook anti-explosion de cout commence par couper/limiter le tenant fautif (hard cap par session)

---

## Exercice 2 : Post-mortem — La facture LLM de $80K en une nuit

### Objectif
Analyser un incident de cout (runaway cost) sur une infrastructure LLM, identifier la cascade, et concevoir les garde-fous budgetaires.

### Consigne
Voici le rapport d'incident (resume).

**Contexte** : Une equipe a deploye un agent "auto-resolver" qui traite les tickets support. L'agent peut appeler des tools (lire le CRM, chercher dans la doc, escalader) et boucle jusqu'a resoudre le ticket. Le system prompt fait 6000 tokens. Pas de prompt caching active. Budget par session : non implemente. Le modele utilise est le tier "frontier" ($15/1M in, $75/1M out).

**Timeline de l'incident :**

| Heure | Evenement |
|---|---|
| 22:00 | Un client envoie un ticket ambigu et contradictoire ("annule ma commande mais garde l'article"). |
| 22:00 | L'agent ne trouve pas de resolution. Il re-cherche, reformule, re-appelle ses tools. |
| 22:01 | A chaque tour, l'agent re-envoie tout le contexte (system prompt 6000 tok + historique croissant) SANS prompt caching. |
| 22:01 - 22:40 | L'agent entre en boucle : 1 ticket = ~400 tours. Aucune condition d'arret sur le nombre de tours. |
| 22:40 | 50 autres tickets ambigus declenchent le meme comportement en parallele. |
| 22:40 - 06:00 | Pendant la nuit, sans personne pour surveiller, les agents bouclent. Le contexte de chaque session gonfle a 80K+ tokens. |
| 06:00 | L'equipe arrive : alerte de facturation du provider ($80K consommes en 8h vs $300/jour habituel). |
| 06:05 | Personne ne sait quel agent/tenant consomme. Les logs n'ont pas le cost par session. |
| 06:30 | L'equipe coupe l'agent entierement (pas de kill switch granulaire). |
| 09:00 | Post-mortem : $80K brules, et le support a tourne en mode degrade 8h. |

**Questions :**

1. **Root cause analysis** :
   - Reconstitue la cascade complete (pas un seul facteur).
   - Pour chaque maillon, identifie le garde-fou manquant.
   - Classe les causes : architecture agent, budget/cost, observability, processus.

2. **Les multiplicateurs de cout** :
   - Identifie les 4 decisions qui ont multiplie le cout (modele, caching, boucle, contexte croissant).
   - Estime l'ordre de grandeur de chacune (ex: tier frontier vs mini = combien de x ?).

3. **Budget guardrails** :
   - Concois un budget PAR SESSION avec soft cap et hard cap. A quels seuils ?
   - Comment un middleware peut-il tracker le cost cumule en temps reel et couper ?
   - Pourquoi un budget par user (quotidien) n'aurait PAS suffi ici ?

4. **Architecture agent** :
   - Quelles conditions d'arret manquaient ? (steps, tokens, temps, no-progress)
   - Comment le prompt caching aurait reduit la facture ? Chiffre l'ordre de grandeur.
   - Faut-il vraiment le tier frontier pour un auto-resolver de tickets ?

5. **Observability & reponse** :
   - Quelles 4 metriques/attributs auraient permis d'alerter en < 15 min ?
   - Concois un kill switch granulaire (par tenant/agent, pas global).
   - Propose un runbook de 7 etapes pour un runaway cost LLM.

### Criteres de reussite
- [ ] La cascade complete est reconstituee : ticket ambigu -> pas de stop condition -> boucle -> contexte croissant sans caching -> tier frontier -> pas de budget/alerte -> nuit sans surveillance -> $80K
- [ ] Les 4 multiplicateurs sont chiffres (frontier ~50-100x un nano ; pas de caching ~10x sur le system prompt repete ; boucle 400 tours ; contexte qui gonfle)
- [ ] Le budget par session a soft cap (avertir/resumer) + hard cap (couper/forcer nouveau thread) avec des seuils concrets
- [ ] Le tracking temps reel utilise des spans avec cost delta agreges par session_id (cf cours J11/J13)
- [ ] Le budget par user est explicitement insuffisant (une seule session peut bruler le budget en minutes)
- [ ] Les conditions d'arret manquantes sont listees (max_steps, max_tokens, timeout, no-progress detector)
- [ ] Le caching et le downgrade de tier sont chiffres ; le kill switch est granulaire ; le runbook commence par couper le tenant fautif
