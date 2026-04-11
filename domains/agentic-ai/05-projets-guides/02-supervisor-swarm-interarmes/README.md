# Projet 02 — Supervisor + Swarm interarmes (PROJET PHARE)

## Contexte metier

Une operation **interarmes** engage simultanement plusieurs composantes : infanterie, appui feu (artillerie / mortiers), reconnaissance (drones), genie. C'est la situation ou le commandement est le plus exigeant, parce qu'il faut :
- **Coordonner** les efforts (le drone doit marquer la cible avant que l'artillerie tire)
- **Deconflicter** les espaces (evite les tirs fratricides, ne pas bombarder ou passe l'infanterie)
- **Reagir** quand la situation change (un peloton prend feu, il faut basculer la priorite)

Dans SWORD, ca se traduit souvent par un **etat-major de brigade** qui commande 3-4 bataillons et plusieurs unites specialisees. C'est typiquement la ou les LLM apportent le plus : le raisonnement multi-etapes, le langage naturel, l'adaptation.

**But du projet** : construire un systeme multi-agent qui combine les deux patterns canoniques de LangGraph :
1. **Supervisor** (hierarchique) — le commandant de brigade orchestre ses subordonnes
2. **Swarm** (horizontal) — les subordonnes peuvent se passer le controle entre eux directement, sans repasser par le commandant, pour les coordinations tactiques rapides (ex: le drone dit a l'artillerie "cible marquee, tire maintenant")

C'est le cas d'usage ideal pour illustrer **quand utiliser quoi**.

## Objectif technique

Construire un graphe LangGraph qui orchestre 4 agents :

1. **BrigadeCommander** (supervisor) — recoit l'OPORD, decompose en missions, assigne aux subordonnes
2. **InfantryLead** (worker) — commande l'infanterie, peut passer la main a Recon ou Artillery
3. **ReconDroneLead** (worker) — commande les drones, peut passer la main a Artillery (swarm handoff)
4. **ArtilleryLead** (worker) — commande les tirs d'appui, repond aux demandes

## Architecture

```
                       [USER / Formateur]
                             |
                             v OPORD
                    +-----------------+
                    | BrigadeCommander|  <-- supervisor
                    |   (orchestre)   |
                    +-----+---+---+---+
                          |   |   |
            +-------------+   |   +--------------+
            |                 |                  |
            v                 v                  v
     +-----------+     +------------+     +-------------+
     | Infantry  |<--->| Recon Drone|<--->|  Artillery  |
     |   Lead    |     |    Lead    |     |    Lead     |
     +-----------+     +------------+     +-------------+
         ^                   ^                  ^
         |                   |                  |
         +--- swarm handoff -+------------------+
         (les 3 workers peuvent se passer la main directement,
          sans retourner au supervisor, pour les coordinations
          rapides comme "marquage -> tir")
```

**Regles de routing** :
- Par defaut, un worker qui finit sa sous-tache rend la main au **supervisor** (pattern supervisor classique)
- Mais un worker peut aussi appeler un **tool `handoff_to(agent)`** qui transfere directement le controle a un autre worker (pattern swarm)
- Le supervisor peut a tout moment reprendre la main (emergency override) si l'etat indique une urgence

## Consigne

Livrables :
- `solution/state.py` — definition du State partage
- `solution/agents.py` — les 4 agents
- `solution/tools.py` — les tools metier + le tool `handoff_to`
- `solution/graph.py` — assemblage du graphe
- `solution/demo.py` — scenario demo + trace

## Pattern 1 — Supervisor (rappel)

Le supervisor est un agent LLM **routeur** : il decide quel worker invoquer ensuite, ou `FINISH`. C'est lui qui "tient" la mission.

```python
def brigade_commander(state):
    # LLM choisit le prochain worker a partir de l'etat
    response = llm.invoke([sys, state.messages])
    next_node = parse_routing(response)  # "infantry" | "recon" | "artillery" | "FINISH"
    return {"next": next_node, "messages": [response]}
```

## Pattern 2 — Swarm (handoff direct)

Un handoff swarm se fait via **un tool special** que l'agent en cours peut appeler. Ce tool, au lieu de renvoyer un resultat classique, renvoie une **`Command`** qui change le `next_node` du graphe.

```python
@tool
def handoff_to_artillery(reason: str) -> Command:
    return Command(
        goto="artillery_lead",
        update={"messages": [ToolMessage(f"Handoff: {reason}", ...)], "active_agent": "artillery"},
    )
```

C'est la cle : **le tool renvoie un `Command`, pas une string**. LangGraph interpreter `Command(goto=...)` comme un reroutage.

## Etapes guidees

1. **State** — `messages: list[BaseMessage]`, `active_agent: str`, `mission_phase: str`, `enemy_observed: list`, `support_requested: bool`. Le supervisor maintient `mission_phase` (PLAN / RECON / STRIKE / ASSAULT / DONE).
2. **Supervisor** — agent LLM avec system prompt "tu es le commandant de brigade". Expose un tool `assign(worker, task)` pour deleguer.
3. **Workers** — chacun a son propre system prompt et ses propres tools metier. Ils ont tous en plus le tool `handoff_to_<other>` pour la coordination horizontale, et le tool `report_to_commander` pour remonter au supervisor.
4. **Routing** — conditional edge depuis chaque worker : si le dernier AIMessage a appele un handoff, goto worker cible ; sinon, retour au supervisor.
5. **Stop** — supervisor appelle `finish()` quand la mission est accomplie.
6. **Trace structuree** — chaque handoff logge (from, to, reason) pour visualiser la danse entre agents.

## Scenario demo

```
OPORD : Prendre le village de 4521. Suspecter presence OPFOR (blindes legers).

Deroule attendu :
1. Supervisor -> Recon : "va marquer les positions OPFOR"
2. Recon envoie drone, observe 2 blindes en 4521-NE
3. Recon handoff SWARM -> Artillery : "marquage pret, tire maintenant"
   (pas de retour au supervisor, coordination directe)
4. Artillery tire, neutralise 1 blinde, report
5. Artillery retour au Supervisor
6. Supervisor -> Infantry : "assaut sur 4521, cover d'Artillery a la demande"
7. Infantry progresse, contact
8. Infantry handoff SWARM -> Artillery : "demande tir d'appui 4521-sud"
9. Artillery tire, Infantry continue
10. Infantry report mission complete
11. Supervisor -> FINISH
```

## Criteres de reussite

- Les 4 agents sont invoques au moins une fois
- Au moins 2 handoffs "swarm" horizontaux (sans passer par le supervisor)
- Le supervisor reprend le controle apres chaque coordination worker-worker
- La trace montre clairement "[from] -> [to] : reason" pour chaque handoff
- Stub mode fonctionne (scenario scripte deterministique)
- Live mode fonctionne avec claude-haiku-4-5

## Points didactiques

- **Quand supervisor, quand swarm** ?
  - Supervisor : decisions strategiques, allocation de ressources, priorisation
  - Swarm : coordination tactique rapide (marquage-tir, demande d'appui au plus proche)
- **Risque du swarm** : perte de vue globale. Si Infantry et Artillery se coordonnent seuls pendant 5 min, le supervisor n'a plus de vue. Il faut des **reports periodiques** (tool `report_to_commander`).
- **Risque du supervisor** : goulot d'etranglement. Tout passe par lui = latence elevee. D'ou l'interet du swarm pour les interactions frequentes.

## Questions de revue

- Qu'est-ce qui empeche un worker de se mettre en boucle infinie avec un autre worker (handoff ping-pong) ?
- Comment le supervisor reprend le controle en urgence ?
- Si un worker crash, que fait le supervisor ?
- Comment tu scales ca a 10 subordonnees (bataillons + sections specialisees) ?

## Solution

Voir `solution/` pour le code complet et commente.

## Pour aller plus loin

- **Checkpointer** : persister l'etat du graphe pour un rejeu AAR
- **Human-in-the-loop** : le supervisor demande validation au formateur avant chaque handoff critique
- **Agents specialises par doctrine** : un supervisor "doctrine francaise" et un "doctrine US", choisi selon le client MASA
- **Adversarial** : en plus, un agent OPFOR qui joue l'ennemi. Deux systems multi-agents qui s'affrontent.
