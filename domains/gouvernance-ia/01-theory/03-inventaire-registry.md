# Inventaire & registry d'agents

## Pourquoi ce module

Tu sais (J1-J2) qu'une flotte d'agents non gouvernee est un risque et qu'un agent gouvernable repose sur 4 piliers. Reste la question operationnelle : **comment savoir, a tout instant, combien d'agents tournent, qui les possede et lesquels sont orphelins ?** Ce module construit le registry qui repond a cette question — pas un tableur, un controle vivant.

---

## 1. Le tableur qui ment (exemple concret)

Lundi, une DSI demande a son equipe « la liste de nos agents IA ». On lui rend un fichier Excel : 14 lignes, propre, un onglet par equipe. Jeudi, un agent de l'equipe Finance declenche par erreur 2 300 emails fournisseurs. On retourne au tableur : l'agent **n'y est pas**. Il a ete deploye mardi par un stagiaire via Copilot Studio. Le tableur etait vrai lundi, faux mercredi, dangereux jeudi.

Le probleme n'est pas le contenu du tableur — c'est sa **nature**. Un export fige est une photo ; la flotte d'agents est une video. Microsoft chiffre l'ampleur du decalage : **80 % des entreprises du Fortune 500 utilisent des agents IA actifs** (telemetrie first-party Copilot Studio / Agent Builder, fenetre nov. 2025) [Microsoft Security — Cyber Pulse, Issue 1, 2026]. Quand le rythme de creation depasse le rythme de recensement manuel, le tableur est structurellement en retard.

Le bon objet n'est pas une liste, c'est un **registry** : une source de verite **interrogeable** et **mise a jour en continu**, qui sait repondre — par requete, pas a la main — a « combien d'agents ? », « qui possede celui-ci ? », « lesquels n'ont pas d'owner ? ».

> **Key takeaway** — Un inventaire d'agents fige (tableur, export) est faux des qu'un agent est cree ou retire. La gouvernance d'une flotte exige un **registry vivant et interrogeable**, pas une photo.

---

## 2. Le registry comme controle « live », pas un artefact mort

Distinguons trois objets que l'on confond souvent :

| Objet | Nature | Probleme |
|-------|--------|----------|
| **Tableur / export CSV** | Photo a l'instant T, edite a la main | Faux des le prochain deploiement ; pas de requete |
| **Catalogue documentaire** | Liste descriptive (wiki, Confluence) | Decoratif : personne ne le consulte avant d'agir |
| **Registry** | Source de verite **interrogeable**, reliee au cycle de vie | C'est ce qu'on veut |

Un **registry** est un **controle** au sens gouvernance : il **conditionne** des decisions (un agent absent du registry ne devrait pas pouvoir agir en production) et il **repond a des questions** sans intervention humaine. C'est la difference entre « on a documente nos agents » (passif) et « le registry est la source d'autorite — s'il n'y est pas, il n'existe pas » (actif).

Concretement, un registry minimal stocke, **par agent**, les attributs de gouvernance vus en J2 :

- `agent_id` — identifiant unique et stable (le pilier *identite*) ;
- `owner` — un humain responsable, nomme (le pilier *owner*) ;
- `permissions` / `scopes` — ce que l'agent a le droit de faire (pilier *permissions*) ;
- `risk_tier` — le niveau de risque (utile pour prioriser l'attention) ;
- `status` — actif / suspendu / decommissionne (le cycle de vie) ;
- metadonnees : date d'enrolement, derniere mise a jour.

Ce sont exactement les champs que manipulera le code du jour. Le CSA decrit ce besoin comme la base d'un IAM purpose-built pour agents : « les agents ont besoin d'identites de premiere classe, decouvrables et gerables sur tout leur cycle de vie » [Cloud Security Alliance — Agentic AI Identity and Access Management, 2025].

> **Key takeaway** — Le registry est un **controle vivant** : source d'autorite interrogeable, reliee au cycle de vie. Regle d'or : *pas dans le registry ⇒ pas en production*.

---

## 3. Decouverte : comment le registry se remplit (sans compter sur la bonne volonte)

Si remplir le registry repose sur la discipline (« chacun declare ses agents »), il sera incomplet — c'est exactement ce qui produit le shadow AI. Deux strategies coexistent, complementaires :

**a) Enrolement (push, declaratif).** L'owner inscrit son agent au moment du deploiement : il fournit `agent_id`, `owner`, `permissions`, `risk_tier`. C'est propre mais incomplet : il suffit d'un oubli pour creer un orphelin.

**b) Decouverte (pull, par telemetrie/scan).** On observe l'environnement — logs d'appels d'API, identites machine creees, connexions a des outils — et on **detecte** des agents qui agissent sans etre enregistres. Tout ce qui agit et qui n'est pas dans le registry est un **candidat orphelin** a reconcilier.

La combinaison des deux donne la **reconciliation** : on confronte « ce qui est declare » (registry) a « ce qui agit » (telemetrie). L'ecart, ce sont les agents fantomes. C'est la version agentique du rapprochement d'inventaire physique : on ne fait pas confiance au registre seul, on le confronte au reel.

Une **pratique emergente** outille cette idee cote produit : Microsoft **Entra Agent ID** (*Preview*, dec. 2025) attribue aux agents une identite de premiere classe dans l'annuaire ; Google publie l'**Agent Card** via le protocole **A2A** (avr. 2025), un document JSON ou l'agent **declare** son identite, ses capacites et son authentification. **Attention** : ce sont des pratiques **non figees** (preview, specs jeunes) — ce qui est durable, c'est le **principe** (un registre interrogeable + une carte d'agent declarative), pas tel produit precis [Microsoft Entra Agent ID, Preview 2025 ; Google A2A Agent Card, 2025].

> **Key takeaway** — Un registry se remplit par **enrolement** (declaratif) *et* **decouverte** (telemetrie). La **reconciliation** des deux fait apparaitre les agents orphelins. Les produits (Entra Agent ID, A2A) sont emergents ; le principe reconductible est : registre interrogeable + Agent Card declarative.

---

## 4. L'Agent Card : la fiche d'identite declarative

Avant d'enregistrer un agent, encore faut-il une **forme** standard pour le decrire. C'est le role de l'**Agent Card** : un document structure (typiquement JSON) ou un agent declare *qui il est* et *ce qu'il peut faire*. Pense-la comme la carte d'identite + le permis de l'agent.

Un Agent Card minimal contient :

- une **identite** (nom, identifiant, version) ;
- des **capacites / skills** (ce que l'agent sait faire) ;
- un **mode d'authentification** (comment on prouve son identite) ;
- un **endpoint** (ou on lui parle).

L'interet de gouvernance : l'Agent Card rend l'enrolement **machine-readable**. Au lieu de retranscrire a la main un agent dans un tableur, on **ingere** sa carte dans le registry. Cote securite, le CSA insiste : ces declarations doivent etre **verifiables** (signees, attestees), sinon n'importe quel agent peut se declarer ce qu'il veut — d'ou l'usage de *verifiable credentials* dans une vraie architecture [Cloud Security Alliance — Agentic AI Identity and Access Management, 2025].

Dans le code du jour, on garde l'idee sans la cryptographie : un Agent Card = un `dict` JSON que le registry sait **valider** (champs obligatoires presents ?) puis **ingerer** en une entree de gouvernance.

> **Key takeaway** — L'**Agent Card** est la fiche declarative (identite + capacites + auth) qui rend l'enrolement automatisable. En production, elle doit etre **verifiable** ; en mini, on valide au moins la presence des champs critiques avant d'ingerer.

---

## 5. Les requetes de gouvernance : ce qu'un registry doit savoir repondre

Un registry n'a de valeur que par les **questions** qu'il rend instantanees. Un comite de gouvernance ne lit pas 400 lignes ; il pose des questions. Les trois requetes fondatrices :

1. **`by_owner`** — « tous les agents possedes par X ». Sert a l'accountability : quand X quitte l'entreprise, qui herite de ses 12 agents ? Sert aussi a equilibrer la charge (un seul humain owner de 50 agents = goulot).
2. **`orphans`** — « les agents sans owner (ou owner inconnu/parti) ». **La** requete critique de gouvernance : un orphelin est ingerable et non-imputable. La question fondatrice du domaine — « combien d'agents tournent, et qui les possede ? » — se mesure ici directement [Microsoft Security — Cyber Pulse, Issue 1, 2026].
3. **`by_risk`** — « les agents par tier de risque ». Permet de **prioriser** : on n'audite pas un agent de resume de notes comme un agent qui declenche des paiements.

De ces trois primitives derivent les **indicateurs** que l'on porte au board : *taux de couverture de gouvernance* (= part des agents avec owner ET permissions ET risk_tier renseignes), *nombre d'orphelins*, *concentration d'ownership*, *exposition par tier*. Le registry n'est pas une fin : c'est le **socle** qui alimente le scoring de risque (J4), l'audit (J9) et le reporting board (J15).

> **Key takeaway** — La valeur d'un registry = les requetes qu'il rend instantanees : `by_owner` (accountability), `orphans` (le controle critique), `by_risk` (priorisation). Elles produisent les indicateurs de gouvernance portes au board.

---

## 6. Cycle de vie : enroler, muter, decommissionner

Un agent n'est pas eternel. Un registry credible suit son **cycle de vie**, sinon il se remplit de zombies (des agents `status=actif` qui ne tournent plus, ou pire qui tournent encore sans surveillance).

Les transitions minimales :

- **Enrolement** — creation de l'entree (via Agent Card ou formulaire). Etat initial : `active`.
- **Mutation** — changement d'owner (depart d'un employe), elargissement/reduction de permissions, re-classification de risque. Chaque mutation doit etre **horodatee** (qui a change quoi, quand) — c'est le pont vers l'audit trail de J9.
- **Suspension** — gel temporaire (`suspended`) sans suppression : l'agent ne doit plus agir mais son historique reste.
- **Decommission** — retrait (`decommissioned`). **On ne supprime pas la ligne** : on change le statut. Supprimer effacerait la tracabilite (qui a possede quoi, quand) dont on a besoin en cas d'incident.

Regle pratique : le registry est **append-friendly** — on ajoute et on transitionne, on n'efface presque jamais. La persistance se fait sur un support durable (ici un fichier JSON ; en production une base + un journal). Deloitte rappelle l'enjeu : seulement **21 % des organisations declarent une gouvernance agentique mature**, alors que **74 % prevoient un deploiement d'agents d'ici 2027** — sans gestion du cycle de vie, l'ecart se creuse [Deloitte Insights — State of AI in the Enterprise 2026, 2026].

> **Key takeaway** — Un registry suit le **cycle de vie** : enroler → muter (horodate) → suspendre → decommissionner. On **transitionne le statut**, on ne supprime pas — sinon on perd la tracabilite necessaire en cas d'incident.

---

## Spaced repetition

1. **Q :** Pourquoi un tableur Excel d'agents IA est-il structurellement un mauvais outil de gouvernance, meme s'il est exact au moment ou on le cree ?
   **R :** Parce que c'est une **photo** d'une realite qui change en continu : des qu'un agent est cree ou retire, le tableur devient faux. Il n'est ni mis a jour automatiquement, ni interrogeable. La gouvernance exige un **registry vivant** (source de verite reliee au cycle de vie + requetes).

2. **Q :** Qu'est-ce que la « reconciliation » dans le remplissage d'un registry, et que revele-t-elle ?
   **R :** Confronter ce qui est **declare** (enrolement, push) a ce qui **agit** (decouverte par telemetrie/scan, pull). L'ecart revele les **agents orphelins / fantomes** : ils agissent sans etre enregistres ni rattaches a un owner.

3. **Q :** Pourquoi la requete `orphans` est-elle « la » requete critique de gouvernance ?
   **R :** Un agent sans owner est **ingerable** (personne pour le superviser/decommissionner) et **non-imputable** (personne de responsable en cas d'incident). C'est la mesure directe de la question fondatrice « combien d'agents, et qui les possede ? ».

4. **Q :** Pourquoi ne supprime-t-on PAS la ligne d'un agent decommissionne, mais change-t-on seulement son `status` ?
   **R :** Pour preserver la **tracabilite** : qui a possede quel agent, avec quelles permissions, sur quelle periode. Supprimer effacerait l'historique necessaire a une investigation d'incident (pont vers l'audit trail, J9).

5. **Q :** Entra Agent ID et l'Agent Card A2A sont cites comme « pratiques emergentes » — qu'est-ce qui est durable la-dedans, et qu'est-ce qui ne l'est pas ?
   **R :** **Durable** : le *principe* — un registre interrogeable + une carte d'agent declarative (identite, capacites, auth). **Non fige** : les *produits* eux-memes (Entra Agent ID en Preview, A2A specs jeunes) peuvent changer. On code le principe, on cite le produit en exemple.
