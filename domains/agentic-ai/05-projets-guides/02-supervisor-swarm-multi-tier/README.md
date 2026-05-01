# Projet 02 — Supervisor + Swarm multi-flotte (PROJET PHARE)

## Contexte metier

Une **operation multi-flotte** dans un entrepot LogiSim engage simultanement plusieurs flottes heterogenes : tri (sorters + conveyors), transport (AGV), inventaire (drones), et une supervision OCC. C'est la situation ou la coordination est la plus exigeante, parce qu'il faut :
- **Coordonner** les efforts (le drone identifie un colis fragile *avant* que l'AGV vienne le prendre)
- **Deconflicter** les zones (eviter les collisions inter-flotte, ne pas envoyer un AGV dans une allee deja occupee par un sorter mobile)
- **Reagir** quand la situation change (un sorter tombe en panne, un pic d'arrivee sur le quai 4 — il faut basculer la priorite)

Dans FleetSim, ca se traduit souvent par un **OCC de shift** qui supervise 3-4 escouades de robots et plusieurs unites specialisees. C'est typiquement la ou les LLM apportent le plus de valeur : raisonnement multi-etapes, langage naturel, adaptation au contexte.

**But du projet** : construire un systeme multi-agent qui combine les deux patterns canoniques de LangGraph :
1. **Supervisor** (hierarchique) — le coordinateur de shift orchestre ses subordonnes
2. **Swarm** (horizontal) — les subordonnes se passent le controle entre eux directement, sans repasser par le superviseur, pour les coordinations tactiques rapides (ex: le drone dit a l'AGV "colis fragile localise en B-12-NE, viens le prendre maintenant")

C'est le cas d'usage ideal pour illustrer **quand utiliser quoi**.

## Objectif technique

Construire un graphe LangGraph qui orchestre 4 agents :

1. **ShiftCoordinator** (supervisor) — recoit le Work Order Plan du WMS, decompose en phases, assigne aux subordonnes
2. **SortingFleetLead** (worker) — pilote la flotte de tri (sorters, conveyors), peut passer la main a Drone ou AGV
3. **InventoryDroneLead** (worker) — pilote les drones d'inventaire, peut passer la main a AGV (swarm handoff)
4. **AGVFleetLead** (worker) — pilote les AGV de transport, repond aux demandes de pickup et de transport

## Architecture

```
                       [USER / OCC]
                             |
                             v Work Order Plan
                    +-------------------+
                    | ShiftCoordinator  |  <-- supervisor
                    |    (orchestre)    |
                    +-----+----+----+---+
                          |    |    |
            +-------------+    |    +--------------+
            |                  |                   |
            v                  v                   v
     +----------+      +-------------+     +------------+
     | Sorting  |<---->| InventoryDr |<--->|    AGV     |
     |   Lead   |      |     Lead    |     |    Lead    |
     +----------+      +-------------+     +------------+
         ^                    ^                   ^
         |                    |                   |
         +--- swarm handoff --+-------------------+
         (les 3 workers peuvent se passer la main directement,
          sans retourner au supervisor, pour les coordinations
          rapides comme "marquage -> pickup")
```

**Regles de routing** :
- Par defaut, un worker qui finit sa sous-tache rend la main au **supervisor** (pattern supervisor classique)
- Mais un worker peut aussi appeler un **tool `handoff_to(agent)`** qui transfere directement le controle a un autre worker (pattern swarm)
- Le supervisor peut a tout moment reprendre la main (emergency override) si l'etat indique une urgence (FAULT critique, COLLISION)

## Consigne

Livrables :
- `solution/state.py` — definition du State partage
- `solution/agents.py` — les 4 agents
- `solution/tools.py` — les tools metier + le tool `handoff_to`
- `solution/graph.py` — assemblage du graphe
- `solution/demo.py` — scenario demo + trace

## Pattern 1 — Supervisor (rappel)

Le supervisor est un agent LLM **routeur** : il decide quel worker invoquer ensuite, ou `FINISH`. C'est lui qui "tient" le shift.

```python
def shift_coordinator(state):
    # LLM choisit le prochain worker a partir de l'etat
    response = llm.invoke([sys, state.messages])
    next_node = parse_routing(response)  # "sorting" | "drone" | "agv" | "FINISH"
    return {"next": next_node, "messages": [response]}
```

## Pattern 2 — Swarm (handoff direct)

Un handoff swarm se fait via **un tool special** que l'agent en cours peut appeler. Ce tool, au lieu de renvoyer un resultat classique, renvoie une **`Command`** qui change le `next_node` du graphe.

```python
@tool
def handoff_to_agv(reason: str) -> Command:
    return Command(
        goto="agv_lead",
        update={"messages": [ToolMessage(f"Handoff: {reason}", ...)], "active_agent": "agv"},
    )
```

C'est la cle : **le tool renvoie un `Command`, pas une string**. LangGraph interprete `Command(goto=...)` comme un reroutage.

## Etapes guidees

1. **State** — `messages: list[BaseMessage]`, `active_agent: str`, `shift_phase: str`, `parcels_observed: list`, `pickup_requested: bool`. Le supervisor maintient `shift_phase` (PLAN / SCAN / DISPATCH / FULFILL / DONE).
2. **Supervisor** — agent LLM avec system prompt "tu es le coordinateur de shift". Expose un tool `assign(worker, task)` pour deleguer.
3. **Workers** — chacun a son propre system prompt et ses propres tools metier. Ils ont tous en plus le tool `handoff_to_<other>` pour la coordination horizontale, et le tool `report_to_coordinator` pour remonter au supervisor.
4. **Routing** — conditional edge depuis chaque worker : si le dernier AIMessage a appele un handoff, goto worker cible ; sinon, retour au supervisor.
5. **Stop** — supervisor appelle `finish()` quand le shift est complete.
6. **Trace structuree** — chaque handoff logge (from, to, reason) pour visualiser la danse entre agents.

## Scenario demo

```
Work Order Plan : Decharger le camion 4 sur le quai B avant 14h.
                  Zone B-12 a vider, suspicion de palettes mal etiquetees (anomalies).

Deroule attendu :
 1. Supervisor -> Drone : "scanner la zone B-12, identifier les anomalies"
 2. Drone lance un drone d'inventaire, observe 2 palettes mal etiquetees en B-12-NE
 3. Drone handoff SWARM -> AGV : "palettes prioritaires localisees, viens les prendre"
    (pas de retour au supervisor, coordination directe)
 4. AGV execute le pickup, embarque une palette, report
 5. AGV retour au Supervisor
 6. Supervisor -> Sorting : "trier la zone B-12 vers les lignes de tri 3 et 4"
 7. Sorting progresse, detecte un colis fragile bloquant
 8. Sorting handoff SWARM -> AGV : "demande pickup colis fragile en B-12-sud"
 9. AGV vient chercher le colis, Sorting continue
10. Sorting report shift complete
11. Supervisor -> FINISH
```

## Criteres de reussite

- Les 4 agents sont invoques au moins une fois
- Au moins 2 handoffs "swarm" horizontaux (sans passer par le supervisor)
- Le supervisor reprend le controle apres chaque coordination worker-worker
- La trace montre clairement "[from] -> [to] : reason" pour chaque handoff
- Stub mode fonctionne (scenario scripte deterministe)
- Live mode fonctionne avec claude-haiku-4-5

## Points didactiques

- **Quand supervisor, quand swarm** ?
  - Supervisor : decisions strategiques, allocation de ressources, priorisation des Work Orders
  - Swarm : coordination operationnelle rapide (marquage-pickup, demande de transport au plus proche)
- **Risque du swarm** : perte de vue globale. Si Sorting et AGV se coordonnent seuls pendant 5 min, le supervisor n'a plus de vue. Il faut des **reports periodiques** (tool `report_to_coordinator`).
- **Risque du supervisor** : goulot d'etranglement. Tout passe par lui = latence elevee. D'ou l'interet du swarm pour les interactions frequentes.

## Questions de revue

- Qu'est-ce qui empeche un worker de se mettre en boucle infinie avec un autre worker (handoff ping-pong) ?
- Comment le supervisor reprend le controle en urgence (ex: alerte COLLISION inter-flotte) ?
- Si un worker crash, que fait le supervisor ?
- Comment tu scales ca a 10 subordonnees (escouades + sections specialisees) ?

## Solution

Voir `solution/` pour le code complet et commente.

## Pour aller plus loin

- **Checkpointer** : persister l'etat du graphe pour un rejeu EOD Review
- **Human-in-the-loop** : le superviseur demande validation a l'OCC humain avant chaque handoff critique
- **Agents specialises par SOP** : un supervisor "SOP client A" et un "SOP client B", choisi selon le client LogiSim
- **Adversarial** : en plus, un agent flotte tierce (livreur externe) qui partage les zones. Deux systemes multi-agents qui doivent se coordonner.
