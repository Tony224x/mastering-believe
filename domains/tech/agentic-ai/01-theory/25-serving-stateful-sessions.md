# J25 — Serving stateful & sessions : agents a l'echelle en production

> **Temps estime** : 3h | **Prerequis** : J1-J24
> **Objectif** : maitriser les strategies de gestion d'etat de sessions en production — choix du backend de checkpointing, scaling horizontal stateless, gestion de sessions multi-utilisateurs et monitoring de derive en ligne.

---

## 1. Le probleme du stateful en production

Un agent conversationnel doit se souvenir des tours precedents : c'est le **thread-scoped state** — l'historique de messages, les variables de contexte, les resultats d'outils enregistres. En developpement, tout ca vit en memoire. En production, trois forces detruisent cette approche :

- **Les pods meurent** : un crash, un redemarrage, une mise a jour → tout l'etat en memoire disparait.
- **Le trafic scale** : 1 instance ne suffit plus → plusieurs workers, chacun avec son propre espace memoire. L'utilisateur U peut envoyer le message 1 au worker W1 et le message 2 au worker W2 qui ne connait pas W1.
- **Les sessions sont longues** : une conversation peut durer des jours ou des semaines. Personne ne garde un process Python actif aussi longtemps.

**La solution universelle** : externaliser l'etat. Les workers deviennent **stateless** — ils ne gardent rien en memoire entre les requetes. L'etat de chaque session vit dans un **store externe** (base de donnees, cache distribue) accessible par tous les workers.

```
              ┌──────────┐   ┌──────────┐   ┌──────────┐
 Messages  -> │ Worker 1 │   │ Worker 2 │   │ Worker 3 │  <- Workers stateless
              └────┬─────┘   └────┬─────┘   └────┬─────┘
                   │              │               │
                   └──────────────┼───────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │      Checkpoint Store       │
                    │  (SQLite / Postgres / Redis) │
                    └─────────────────────────────┘
```

Chaque requete : (1) le worker charge l'etat du thread depuis le store, (2) traite le message, (3) sauvegarde le nouvel etat. Le worker lui-meme n'a aucune memoire entre deux requetes.

---

## 2. Sessions et threads : le modele conceptuel

### 2.1 Thread = une conversation

Un **thread** est une conversation continue identifiee par un `thread_id` unique. Il contient :
- L'historique des messages (user/assistant/tool)
- L'etat de l'agent (etape courante dans le graphe LangGraph, valeurs des noeuds)
- Les metadonnees (created_at, user_id, config)

Un meme utilisateur peut avoir plusieurs threads (plusieurs conversations independantes). C'est le **multi-thread par user**.

### 2.2 Checkpoint = un snapshot d'etat

A chaque etape de l'agent, un **checkpoint** est sauvegarde : c'est un snapshot complet de l'etat du thread a ce moment. Rappel J6 : LangGraph fait ca automatiquement via son interface `BaseCheckpointSaver`. Ici on se concentre sur **comment choisir et deployer le bon backend**.

### 2.3 Thread-scoped vs cross-thread memory

| Type | Scope | Exemple |
|------|-------|---------|
| **Thread-scoped** | Une seule conversation | Historique des messages d'un chat |
| **Cross-thread** | Plusieurs conversations d'un meme user | Preferences persistantes de l'utilisateur |
| **Global** | Tous les users | Base de connaissances partagee |

Le cross-thread et le global memory necessitent un **store separe** (Vector DB, SQL) en plus du checkpointer de session. On ne les confond pas : le checkpointer gere la continuite d'UNE conversation, le store partagé gere la memoire qui traverse les conversations.

---

## 3. Backends de checkpointing : comparatif

Le choix du backend est la decision architecturale centrale. Voici les 4 options principales :

| Backend | Durabilite | Latence | Scaling horizontal | Quand l'utiliser |
|---------|-----------|---------|-------------------|-----------------|
| **MemorySaver** | Aucune (process) | <1 ms | Non (1 process) | Dev local, tests, demos |
| **SQLite** | Fichier local | 1-5 ms | Non (1 machine) | Prototype, single-node, edge |
| **PostgresSaver** | Forte (ACID) | 2-20 ms | Oui (connection pool) | Production standard, millions de sessions |
| **RedisSaver** | Configurable (AOF/RDB) | <1 ms | Oui (cluster) | Haute freq, sessions courtes, cache L1 |

### 3.1 MemorySaver

```python
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()
```

Stocke les checkpoints dans un `dict` Python. Instantane, zero dependance. Mais : si le process redémarre, tout est perdu. Strictement reserve au dev.

### 3.2 SQLiteCheckpointer

```python
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver.from_conn_string("./sessions.db")
```

SQLite est fiable, ACID, zero config serveur. Limite : un seul writer a la fois (WAL mode ameliore ca). Parfait pour un serveur single-node ou un edge deployment (notebook, Raspberry Pi, Lambda stateful).

### 3.3 PostgresSaver

```python
from langgraph.checkpoint.postgres import PostgresSaver
checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:pass@host/db"
)
```

Postgres est le choix de production standard. Connection pooling avec `psycopg2` ou `asyncpg`. Supporte le scaling horizontal : N workers se partagent la meme base. Index sur `(thread_id, checkpoint_id)` pour des lectures rapides.

**Trade-offs** :
- Latence reseau (2-20 ms) vs 0 ms en memoire → verifiable en prod avec du tracing (J12)
- Necessite un serveur Postgres gere (RDS, Cloud SQL, Supabase)
- ACID complet : pas de perte de checkpoint meme si le worker crash en plein milieu

### 3.4 RedisSaver (langgraph-redis)

```python
from langgraph_redis import RedisSaver
checkpointer = RedisSaver.from_conn_string("redis://localhost:6379")
```

Redis offre des latences sub-milliseconde. Utile si les sessions sont nombreuses et courtes (chatbot haute frequence, assistant en temps reel). Attention a la persistance : Redis est par defaut in-memory avec durabilite optionnelle (AOF/RDB). En cluster, les checkpoints sont partitionnes par `thread_id`.

**Pattern hybride** : Redis comme cache L1 (derniers N checkpoints) + Postgres comme store durable. Les workers lisent depuis Redis, Redis sync vers Postgres en arriere-plan.

### 3.5 Alternative : le runtime manage (Vertex AI Agent Engine & co.)

Les 4 backends ci-dessus supposent que **tu heberges** le store (self-hosted). Une autre voie : deleguer le runtime ET l'etat a un service manage. **Vertex AI Agent Engine** (Google Cloud, GA 2025) est l'exemple de reference du programme Google Agents ; AWS Bedrock AgentCore et les "assistants" heberges jouent dans la meme categorie.

Ce qu'un runtime manage prend en charge a ta place :
- **Sessions** (memoire court terme = historique de conversation) gerees automatiquement.
- **Scaling** et disponibilite du runtime (pas de pods a operer).
- **Memory Bank** (memoire long terme manage) — voir J16 §5.4 : extraction et **consolidation automatiques** des faits par le modele.

| | Self-hosted (sections 3.1-3.4) | Runtime manage (Agent Engine…) |
|---|---|---|
| **Controle** | Total (backend, schema, locks) | Limite (boite plus fermee) |
| **Ops** | A ta charge (DB, scaling, backups) | Delegues au provider |
| **Portabilite** | Forte (Postgres/Redis partout) | Faible (lock-in cloud) |
| **Time-to-prod** | Plus long | Tres court |
| **Cout** | Infra + maintenance | Fees runtime (souvent plus cher a l'echelle) |

> **Regle de choix** : self-hosted quand tu veux le controle, la portabilite et maitriser le cout a l'echelle (le cas par defaut de ce cours). Runtime manage quand le time-to-prod prime et que tu es deja sur le cloud du provider. Le concept de **session / thread** (section 2) reste identique des deux cotes — seul l'operateur du store change.

---

## 4. Scaling horizontal : workers stateless

### 4.1 Le principe

```
Client ──> Load Balancer ──> Worker 1 ─┐
                        ──> Worker 2 ─┤──> Checkpoint Store (Postgres/Redis)
                        ──> Worker 3 ─┘
```

Chaque requete arrive avec un `thread_id`. Le worker :
1. `checkpoint = store.load(thread_id)` — charge le dernier etat
2. Traite le message, execute les outils
3. `store.save(thread_id, new_checkpoint)` — sauvegarde le nouvel etat
4. Retourne la reponse → aucune memoire conservee

Le worker suivant qui reçoit une requete pour ce `thread_id` chargera exactement le bon etat.

> **Analogie** : comme un bibliothecaire. Chaque jour, n'importe quel bibliothecaire peut reprendre ton dossier de pret parce que tout est dans le registre central (le store). Le bibliothecaire lui-meme n'a pas besoin de se souvenir de toi — il consulte le registre.

### 4.2 Sticky sessions vs etat partage

**Sticky sessions** : le load balancer route toujours le meme `user_id` vers le meme worker. L'etat peut rester en memoire. Avantages : zero latence de lecture, pas de store externe. Inconvenients : si le worker crash, la session est perdue ; le scaling est inegale (un user tres actif surcharge un worker).

**Etat partage** : n'importe quel worker peut traiter n'importe quelle session. Necessite un store externe. Avantages : resilience totale, scaling uniforme, zero perte sur crash. Inconvenients : latence reseau pour chaque load/save.

**En production, prefer l'etat partage** : la resilience l'emporte sur les quelques ms de latence.

### 4.3 Concurrence sur un thread

Que se passe-t-il si deux requetes arrivent simultanement pour le meme `thread_id` ?
- **Postgres** : les transactions ACID et les row-level locks gerent ca. La deuxieme requete attend ou echoue avec un conflict error.
- **Redis** : utiliser des transactions MULTI/EXEC ou des scripts Lua pour garantir l'atomicite.
- **Pattern recommande** : une seule requete active par `thread_id` a la fois (queue FIFO par thread). Le client reessaie si conflit.

---

## 5. Sessions multi-utilisateurs : isolation et securite

### 5.1 Namespace par user

Chaque thread doit etre isole par `user_id`. Le `thread_id` seul ne suffit pas : un utilisateur ne doit pas pouvoir acceder au thread d'un autre.

```python
# Pattern : thread_id = f"{user_id}:{session_id}"
# Ou : verifier dans le store que thread.owner == user_id courant
config = {
    "configurable": {
        "thread_id": f"{user_id}:{session_id}",
        "user_id": user_id,
    }
}
```

### 5.2 Expiration des sessions

Les sessions anciennes consomment de la place. Strategie :
- **TTL Redis** : `EXPIRE thread:{thread_id} 86400` (24h)
- **Postgres** : job de nettoyage (`DELETE FROM checkpoints WHERE updated_at < NOW() - INTERVAL '30 days'`)
- **Archivage** : deplacer les sessions expirees vers un storage froid (S3) avant suppression

### 5.3 Store cross-thread

Pour les preferences utilisateur qui traversent toutes les sessions (langue preferee, contexte metier, historique de feedback) : utiliser un **store separe** adresse par `user_id`, pas par `thread_id`.

```python
# LangGraph Store API (cross-thread)
store.put(("users", user_id), "preferences", {"language": "fr"})
prefs = store.get(("users", user_id), "preferences")
```

---

## 6. Online eval & drift en production

Rappel J11 : l'eval offline teste le modele sur un dataset fixe avant deployment. En production, l'environment change : les queries evoluent, le comportement du LLM derive, les nouveaux edge cases emergent.

**Online eval** = mesurer la qualite en temps reel sur du trafic reel.

### 6.1 Metriques a monitorer

| Metrique | Signal | Comment la mesurer |
|----------|--------|-------------------|
| **Taux de succes** | L'agent a-t-il complete la tache ? | Feedback user (thumbs up/down), LLM-as-judge |
| **Nombre de tours** | L'agent divague-t-il ? | Nb de messages par session avant resolution |
| **Taux d'erreur outil** | Les tools echouent-ils plus souvent ? | Logs des tool calls (J12) |
| **Latence P95** | Le service est-il lent ? | Traces (OpenTelemetry) |
| **Session abandonnee** | L'utilisateur part sans reponse | Pas de message final dans le thread |

### 6.2 Detection de drift

Le **drift** arrive quand la distribution des queries change : nouveaux sujets, nouvelles langues, nouvelles intentions. Signes :
- Taux de succes qui baisse progressivement
- Clusters de queries jamais vus en eval offline (embedding + clustering)
- LLM-as-judge qui signale plus de reponses hors-scope

**Fenetre glissante** : calculer les metriques sur les N derniers appels (ex: 1000) et alerter si la valeur sort de la plage historique de ±2 ecarts-types.

### 6.3 Boucle feedback → reval

```
Prod trafic ──> Logs ──> Sampler (5%) ──> LLM-as-judge ──> Score
                                                          └──> Dashboard
                                                          └──> Alert si score < seuil
                                                          └──> Collect → dataset offline
```

Les exemples signales par le judge (score bas) deviennent automatiquement des candidats pour le dataset d'eval offline. La boucle est fermee.

---

## 7. Flash Cards — Test de comprehension

**Q1 : Pourquoi un MemorySaver est-il inutilisable en production horizontalement scalee ?**
> R : MemorySaver stocke l'etat dans un `dict` Python en memoire du process. Si N workers tournent en parallele, chacun a son propre dictionnaire — ils ne se voient pas. De plus, un restart du pod efface tout. En production scalee, l'etat doit vivre dans un store externe accessible par tous les workers (Postgres, Redis).

**Q2 : Quelle est la difference entre thread-scoped memory et cross-thread memory ? Donnez un exemple de chaque.**
> R : **Thread-scoped** = etat d'UNE conversation (historique des messages, variables de l'agent pendant ce dialogue). Ex : "l'utilisateur a mentionne qu'il cherche un vol pour Paris dans ce chat". **Cross-thread** = informations qui persistent entre plusieurs conversations du meme user. Ex : "la langue preferee de cet utilisateur est le francais" — vraie dans toutes ses sessions. Le checkpointer gere le thread-scoped ; un store separe (adresse par user_id) gere le cross-thread.

**Q3 : Quand choisir Redis plutot que Postgres comme backend de checkpointing ?**
> R : Redis quand la latence est critique (sub-ms) et les sessions sont courtes ou nombreuses (chatbot haute frequence). Postgres quand la durabilite ACID est primordiale et que les sessions sont longues (risque inacceptable de perte). En pratique : pattern hybride Redis (cache L1 chaud) + Postgres (durabilite) donne le meilleur des deux mondes.

**Q4 : Qu'est-ce que le "drift" en production et comment le detecte-t-on ?**
> R : Le drift est le changement de la distribution des requetes reelles par rapport a ce sur quoi le systeme a ete evalue. Il se manifeste par une baisse progressive du taux de succes, des clusters de queries jamais vus, ou plus d'outputs hors-scope signales par un LLM-as-judge. On le detecte avec une fenetre glissante sur les metriques cles et on alerte quand une metrique sort de ±2 sigma de la plage historique.

**Q5 : Pourquoi les sticky sessions sont-elles deconseillees en production HA ?**
> R : Les sticky sessions routent toujours le meme user vers le meme worker pour garder l'etat en memoire. Mais : (a) si ce worker crashe, toutes ses sessions sont perdues ; (b) un user tres actif surcharge un seul worker pendant que les autres sont sous-utilises ; (c) les deployments rolling necessitent des precautions speciales pour "drainer" les sessions. L'etat partage (store externe) evite ces 3 problemes au prix de quelques ms de latence reseau.

**Q6 : Self-hosted vs runtime manage (type Vertex AI Agent Engine) : qu'est-ce qui change et qu'est-ce qui reste pareil ?**
> R : Ce qui change = l'operateur du store. Self-hosted (3.1-3.4) : tu heberges Postgres/Redis, controle total + portabilite, mais tu portes les ops (DB, scaling, backups). Runtime manage : sessions + scaling + memoire (Memory Bank) delegues au provider, time-to-prod tres court, mais lock-in cloud et cout souvent plus eleve a l'echelle. Ce qui reste identique : le modele conceptuel **thread / session** (section 2). Regle : self-hosted par defaut (controle + cout), manage quand le time-to-prod prime et qu'on est deja sur le cloud du provider.

---

## Points cles a retenir

- **Workers stateless + store externe** : le pattern universel pour scaler des agents stateful. Chaque requete charge l'etat depuis le store, traite, sauvegarde.
- **4 backends de checkpointing** : MemorySaver (dev only) < SQLite (single-node) < PostgresSaver (production standard) < RedisSaver (haute freq / faible latence).
- **Thread-scoped vs cross-thread** : le checkpointer gere une session ; un store separe (user_id) gere les preferences persistantes inter-sessions.
- **Sticky sessions** deconseillees en HA : resilience zero si le worker crash.
- **Isolation par user_id** : le thread_id seul ne suffit pas — toujours verifier que l'owner correspond au user authentifie.
- **Expiration des sessions** : TTL Redis ou job de nettoyage Postgres pour eviter la croissance infinie du store.
- **Online eval** : taux de succes, nb de tours, taux d'erreur outil, latence P95, sessions abandonnees — sur fenetre glissante.
- **Drift** : baisse progressive du taux de succes ou queries hors-distribution → alerter, collecter des exemples, relancer l'eval offline.
- **Boucle feedback** : les exemples malscores par le LLM-as-judge en prod alimentent le dataset d'eval offline → amelioration continue.
- **Self-hosted vs runtime manage** : un service comme Vertex AI Agent Engine delegue sessions + scaling + memoire (Memory Bank, J16 §5.4) au provider — time-to-prod court contre lock-in cloud ; le modele thread/session ne change pas.

---

## Pour aller plus loin

- LangChain, "LangGraph — Persistence" https://docs.langchain.com/oss/python/langgraph/persistence
- LangChain, "Checkpoints API reference" https://reference.langchain.com/python/langgraph/checkpoints
- Redis, "langgraph-redis" (2025) https://github.com/redis-developer/langgraph-redis
- Google Cloud, "Vertex AI Agent Engine — overview" (2025) https://docs.cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/overview
