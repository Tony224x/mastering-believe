# Exercices Hard — API Design & Patterns

---

## Exercice 1 : Concevoir l'API + le Gateway d'une plateforme multi-tenant

### Objectif
Concevoir une API publique complete avec API Gateway, BFF, auth, rate limiting par tenant, versioning et idempotence — le niveau "design d'une plateforme".

### Consigne
Tu conçois l'API d'une plateforme SaaS B2B (type Stripe/Twilio) :

**Contexte :**
- API publique REST consommee par des milliers de clients (tenants), chacun avec ses propres limites
- 3 clients internes : web app, mobile app, partenaires (webhooks)
- Microservices internes en gRPC : accounts, billing, messaging, analytics
- 100K req/s en pic, multi-region
- Operations critiques (creation de ressources facturees, envoi de messages payants) qui doivent etre idempotentes

**Livre :**

1. **Couches** :
   - Dessine le chemin requete : client → ... → microservices. Ou place-tu le gateway, le BFF, le LB ?
   - Quelles responsabilites au gateway et lesquelles surtout PAS (piege du gateway "monolith cache") ?

2. **Auth & multi-tenant** :
   - Comment authentifies-tu un tenant (API key, OAuth, JWT) ? Que passe le gateway aux services internes ?
   - Comment appliques-tu un rate limit DIFFERENT par tenant (free tier vs enterprise) ? Quelle structure ?

3. **Idempotence a l'echelle** :
   - L'idempotency key doit fonctionner en multi-region (un retry peut tomber sur une autre region). Quel est le defi et comment le resous-tu ?

4. **Versioning & evolution** :
   - Strategie de versioning pour des milliers de clients qu'on ne controle pas.
   - Un tenant est bloque sur v1, un autre sur v2 : comment routes-tu ? Comment forces-tu la migration sans casser personne du jour au lendemain ?

5. **BFF (Backend for Frontend)** :
   - Pourquoi un BFF par client (web/mobile) plutot qu'une API unique ? Donne un exemple concret d'aggregation que le BFF mobile fait et pas le web.

6. **Protocole interne** :
   - REST externe → gRPC interne : ou se fait la traduction ? Quel cout/benefice ?

7. **Resilience & contrats** :
   - Comment garantis-tu qu'un service interne ne casse pas ses consommateurs (contract testing) ?
   - OpenAPI : a quoi sert-il concretement ici (3 usages) ?

### Criteres de reussite
- [ ] Chemin clair : client → CDN/LB → API Gateway → BFF → services gRPC ; le gateway fait infra (auth, rate limit, routing, observability), PAS de business logic
- [ ] Auth tenant via API key/OAuth → le gateway valide et injecte tenant_id/scopes dans les headers internes
- [ ] Rate limit par tenant via une config (tier → quota) appliquee dans un store partage (Redis), differencie free/enterprise
- [ ] Idempotence multi-region : store de cles repartie/repliquee OU routage sticky par idempotency key (defi de coherence cross-region identifie)
- [ ] Versioning /v1//v2/ + routage par version + deprecation progressive (Sunset header, preavis 6-12 mois)
- [ ] BFF justifie (aggregation specifique par client) avec exemple concret, zero business logic
- [ ] Traduction REST→gRPC au gateway/BFF, benefice (typage interne) vs cout (mapping)
- [ ] Contract testing (OpenAPI/protobuf) + 3 usages OpenAPI (doc, codegen, validation/contract test)

---

## Exercice 2 : Post-mortem — Le breaking change qui a casse 4000 integrations

### Objectif
Analyser un incident d'evolution d'API : un changement "innocent" qui casse les clients en silence, et concevoir une politique d'evolution sure.

### Consigne
Voici le rapport d'incident (resume) :

**Contexte** : Une API publique REST (`/v1/`) avec ~4000 integrations actives de clients tiers. Le champ `status` d'une ressource `order` etait un entier (`status: 1`). Une equipe a decide de le rendre plus lisible en le changeant en string (`status: "paid"`) — "c'est plus propre". Deploye un mardi a 11h, directement en `/v1/`, sans nouvelle version.

**Timeline de l'incident :**

| Heure | Evenement |
|---|---|
| 11:00 | Deploiement. Le champ `order.status` passe de `1` (int) a `"paid"` (string) dans toutes les reponses `/v1/orders`. |
| 11:02 | Les clients qui faisaient `if (order.status === 1)` ou un `switch(status)` sur des entiers commencent a casser silencieusement (la condition n'est jamais vraie). |
| 11:05 | Certains clients ont un parsing strict (typed deserialization, ex : `int status`) → exception → leur integration crash en 500. |
| 11:10 | Les clients NE remontent PAS d'erreur cote API (la reponse est 200 OK valide). L'API provider ne voit aucune alerte : ses metriques (latence, 5xx) sont vertes. |
| 11:30 | Premiers tickets support : "vos webhooks renvoient des donnees invalides", "mon systeme de fulfillment ne traite plus les commandes payees". |
| 12:00 | L'ampleur monte : ~4000 integrations potentiellement affectees, dont des clients qui facturent leurs propres utilisateurs sur la base de ce statut. |
| 12:30 | Decision : rollback. Mais certains clients ont DEJA adapte leur code au nouveau format (string) entre 11h et 12h30 → le rollback CASSE ces clients-la. |
| 13:00 | Etat chaotique : impossible de satisfaire les deux populations (ceux sur int, ceux passes a string). |
| 14:00 | Rollback effectue. Communication d'urgence. Mais la confiance est entamee : un client majeur menace de partir. |
| J+1 | Post-mortem : aucune perte de donnees, mais des heures de fulfillment casse chez des centaines de clients, et un cout reputationnel majeur. |

**Questions :**

1. **Root cause analysis** :
   - Pourquoi ce changement etait-il un breaking change, alors que "ca passe les tests internes" ?
   - Identifie la chaine causale et les guardrails manquants. Classe : processus, architecture, monitoring.
   - Pourquoi l'incident etait-il INVISIBLE cote provider (metriques vertes) ?

2. **Additif vs breaking** :
   - Quelle est la regle qui distingue un changement additif (safe) d'un breaking change ?
   - Comment aurait-on pu ajouter `status` lisible SANS casser personne ? (indice : nouveau field)

3. **Le rollback a casse les clients qui s'etaient adaptes** :
   - Pourquoi le rollback a-t-il fait des nouvelles victimes ?
   - Quelle propriete d'une bonne strategie d'evolution aurait evite ce piege ?

4. **Detection** :
   - Pourquoi les metriques classiques (5xx, latence) n'ont rien vu ?
   - Quels mecanismes auraient detecte un breaking change AVANT/PENDANT le deploiement ? (contract testing, consumer-driven contracts, schema diff en CI, canary sur un sous-ensemble de clients)

5. **Politique d'evolution corrigee** :
   - Concois la politique complete : versioning, deprecation, additive-only dans une version, schema validation en CI, communication, sunset.
   - Comment introduire un VRAI breaking change a l'avenir, etape par etape, sur 6-12 mois ?

6. **Runbook** :
   - Un runbook de 7 etapes pour "breaking change deploye par erreur en prod sur une API publique".

### Criteres de reussite
- [ ] Le changement de TYPE d'un field existant est identifie comme breaking (les tests internes ne testent pas les clients tiers)
- [ ] Chaine causale + guardrails manquants classes (process : pas de revue d'impact ; archi : pas de versioning/additive-only ; monitoring : pas de contract testing)
- [ ] L'invisibilite est expliquee : la reponse reste 200 OK, l'erreur est cote CLIENT, donc les metriques provider sont vertes
- [ ] La regle additif vs breaking est claire (ajouter = safe car clients ignorent les fields inconnus ; renommer/supprimer/changer un type = breaking)
- [ ] La solution non-cassante : ajouter un NOUVEAU field (ex : status_label) en gardant status int → additif
- [ ] Le rollback fait des nouvelles victimes car le changement n'etait pas backward-compatible dans les deux sens → leçon : additive-only évite ce double-bind
- [ ] La detection : contract testing / consumer-driven contracts / schema diff en CI / canary clients (au moins 2)
- [ ] Politique complete : versioning + additive-only + schema validation CI + deprecation 6-12 mois + Sunset header
- [ ] Le runbook traite le double-bind (ne pas rollback aveuglement ; communiquer ; servir les deux formats temporairement si possible)
