# Contexte LogiSim — pour les projets guides

> Document partage par tous les dossiers `05-projets-guides/` des domaines. Sert de reference metier unique pour les cas d'etudes.

## L'entreprise

**LogiSim** est un editeur fictif de logiciels de simulation et de pilotage pour la **logistique automatisee** (entrepots, plateformes 3PL, distribution last-mile). Il sert de contexte metier pour les projets guides de ce repo. Son produit phare permet de simuler et orchestrer la flotte robotisee d'un entrepot (drones, AGV, convoyeurs, postes de tri) en temps reel, et de rejouer des shifts pour optimiser les SOPs (Standard Operating Procedures).

LogiSim est deploye on-premise chez ses clients (operateurs logistiques, 3PL, retailers) qui exigent souvent une connectivite reduite et une isolation reseau forte (donnees de flux client = sensibles).

## Produits phares

### FleetSim
Simulation operationnelle d'une plateforme logistique automatisee. Modelise une flotte heterogene (drones de comptage, AGV de transport, robots de picking, lignes de tri, convoyeurs) avec interactions temps reel. Les unites recoivent des ordres operationnels en langage naturel ("decharge le camion 4 sur le quai B avant 14h, priorise les colis frais") et adaptent leur trajectoire et leurs handoffs quand un imprevu survient (panne, pic d'arrivee, congestion).

- Reduit le temps de design d'une nouvelle plateforme de **40-60%**
- Permet de tester un changement de SOP sans interrompre l'exploitation
- Simule des milliers de robots autonomes et de colis en temps reel
- Couvre exploitation nominale, pics saisonniers, modes degrades

### NetworkSim
Extension multi-site de FleetSim pour modeliser un **reseau de plateformes** (ex : 12 entrepots regionaux + 3 hubs + flotte inter-sites), avec routing inter-sites, gestion des stocks repartis, et coordination multi-OCC.

### AutonomyAI SDK
SDK de comportements autonomes pour robots logistiques, en developpement depuis plus de 10 ans. Permet de creer des agents **reactifs, opportunistes, cooperatifs** capables d'arbitrer entre objectifs contradictoires (latence vs cout energetique vs risque collision). C'est le moteur cognitif sous FleetSim.

## Stack technique (ce qu'on peut raisonnablement supposer)

- **Core simulation** : C++, moteur deterministique tick-based (10-50 Hz typique en simulation operationnelle)
- **Orchestration / outillage** : Python pour scripts d'analyse, pipelines de donnees, evals
- **IA moderne (la couche que ces projets explorent)** : agents LLM pour l'aide a la decision OCC, EOD report automatique, NLP des Work Orders, RAG sur les SOPs et historiques d'incidents
- **Integration** : OPC-UA et MQTT (standards SCADA / IIoT), connecteurs WMS / TMS, couplage avec un digital twin tiers pour le rendu 3D temps reel
- **Deploiement** : on-premise (reseau client souvent isole), parfois VDI

## Vocabulaire metier a connaitre

| Terme | Definition |
|---|---|
| **Simulation operationnelle** | Simulation ou les robots/colis sont simules, pas les operateurs humains. Oppose a "digital twin" (rendu visuel) et "physical run" (exploitation reelle). |
| **OCC** | Operations Control Center — la salle de controle qui supervise la plateforme. |
| **Work Order** | Ordre operationnel emis par le WMS, format standardise (origine, destination, deadline, contraintes). |
| **Routing Plan** | Option de cheminement consideree par le planner pour un colis ou une flotte. |
| **AAR / EOD Review** | After-Action Review / End-of-Day debrief base sur les traces de la journee operationnelle. |
| **Own fleet / External fleet** | Flotte interne (FleetSim) / flotte tiers (sous-traitants, livreurs externes). |
| **Safety Policy** | Regles d'engagement des robots (vitesse max en zone humaine, priorite pieton, etc.). |
| **Coverage** | Couverture sensorielle entre deux points (camera / lidar), critique pour detection / tracking. |
| **Partial telemetry** | Information imparfaite sur l'etat de certains robots (latence, perte de paquets). |
| **Wear / Breakdown** | Usure cumulee d'un robot pendant un shift (battery, cycles). |
| **Pcollision** | Probabilite qu'un mouvement entraine une collision (fonction densite, vitesse, visibilite). |

## Cas d'usage recurrents (ce qui alimente les projets guides)

1. **Planification de plateforme** — un consultant cree un scenario, le client teste plusieurs designs d'entrepot, FleetSim simule l'exploitation.
2. **Evaluation de SOP** — comparer plusieurs Routing Plans via simulations Monte-Carlo.
3. **Rejeu et EOD Review** — rejouer une journee operationnelle pour identifier les decisions cles et les bottlenecks.
4. **Integration physique-virtuelle** — un operateur en realite augmentee interagit avec des robots FleetSim simules pour s'entrainer.
5. **Pic saisonnier / mode degrade** — meme moteur, scenarios de Black Friday, panne d'energie, surcharge.

## Contraintes particulieres logistique automatisee

- **Determinisme** : meme seed = meme resultat (pour reconstitution d'incident et certification).
- **Connectivite limitee** : pas de cloud public garanti, certains sites en quasi air-gap. Tout doit pouvoir tourner on-premise.
- **Confidentialite** : les flux et donnees clients sont contractuellement sensibles, jamais de telemetrie sortante non-anonymisee.
- **Certification** : les modeles (cinematique, batterie, capacites) doivent etre traceables et validables (clients audites ISO 9001 / SOC 2).
- **Interop norms** : OPC-UA, MQTT, GS1 EPCIS pour la tracabilite des colis.

## Profil AI Engineer chez un editeur de ce type — role suppose

**AI Engineer** — amener les techniques IA modernes (LLM, agents, RAG, RL) dans un produit historiquement base sur l'IA symbolique / regles operationnelles. Les projets guides illustrent ce qu'on est susceptible de construire dans les 6 premiers mois :
- Outiller l'OCC (generation de scenario, EOD Review automatique)
- Enrichir AutonomyAI SDK avec des agents LLM-driven pour les unites high-level (chef de flotte)
- Pipelines de donnees sur les traces de shifts
- Prototypes de RAG sur le corpus SOP et l'historique d'incidents

## Comment lire chaque projet guide

Chaque projet suit la meme structure :

1. **Contexte metier** — pourquoi ce probleme existe chez LogiSim
2. **Objectif technique** — ce qu'il faut construire
3. **Consigne** — specs precises, entrees/sorties attendues
4. **Etapes guidees** — decomposition en sous-taches
5. **Criteres de reussite** — comment juger que c'est bon
6. **Corrige** — solution commentee, avec les raisons des choix
7. **Pour aller plus loin** — extensions pour approfondir

## Schema d'event canonique (reference partagee)

Plusieurs projets manipulent des "events" issus d'un shift FleetSim (SD 02, NN 01, NN 02, NN 03, Agentic 03). Pour eviter la derive, voici le **schema unique** que tous les projets doivent respecter.

```python
# Event canonique LogiSim — format pivot, utilise partout
CanonicalEvent = {
    "id": int,             # identifiant unique dans le shift, monotone croissant
    "shift_id": int,       # identifiant du shift (journee operationnelle)
    "seq": int,            # numero de sequence, monotone par shift (detection drop)
    "t_sim": float,        # temps simule en secondes depuis le debut du shift
    "tick": int,           # numero du tick de simulation (entier)
    "unit_id": str,        # identifiant de l'unite source, ex "AGV-12", "Drone-3", "Sorter-A"
    "zone": str,           # "DOCK" | "STORAGE" | "PICKING" | "SORTING" | "STAGING" | "EXTERIOR"
    "kind": str,           # type d'event : voir table ci-dessous
    "payload": dict,       # donnees specifiques au kind (key-value)
}
```

### Kinds d'events standards

| Kind | Signification | Payload typique |
|---|---|---|
| `MOVE` | Deplacement d'une unite | `{"from": "grid", "to": "grid", "speed": float}` |
| `DETECT` | Detection d'une cible (colis, obstacle) | `{"target_id": str, "dist_m": float, "confidence": float}` |
| `ORDER` | Work Order recu | `{"from": "WMS", "order_type": str, "details": str}` |
| `PICKUP` | Prise en charge d'un colis | `{"parcel_id": str, "from_slot": str, "weight_kg": float}` |
| `DROPOFF` | Depose d'un colis | `{"parcel_id": str, "to_slot": str, "ok": bool}` |
| `COLLISION` | Collision detectee | `{"with_unit": str, "severity": float, "stopped": bool}` |
| `FAULT` | Panne ou degradation | `{"code": str, "severity": "minor\|major\|critical"}` |
| `REPORT` | Status report d'une unite | `{"summary": str, "severity": "routine\|urgent"}` |
| `MARK` | Zone marquee (laser, beacon) | `{"grid": str, "tag": str}` |
| `HANDOFF` | Unite passe une tache a une autre | `{"to_unit": str, "task": str}` |

### Regle d'or

**Tout projet qui lit ou ecrit des events utilise strictement ce schema.** Si un besoin n'est pas couvert, on ajoute une cle dans `payload`, jamais au top-level. Le top-level reste stable pour que les outils (pipeline EOD, LLM, agents) n'aient pas a brancher sur le kind.

## Fil rouge — comment les 12 projets s'emboitent

Les projets ne sont pas isoles : ils forment trois couches d'un mini-FleetSim.

```
                    LogiSim / FleetSim — fil rouge des 12 projets guides

  +------------------------- RUNTIME de simulation ----------------------+
  |                                                                     |
  |   [Algo 01]       [Algo 02]       [Algo 03]                         |
  |    A* path         Sensor          Operations                       |
  |    finding         coverage        event queue                      |
  |       \              |              /                              |
  |        \             |             /                               |
  |         v            v            v                                |
  |        +------------ SD 01 ---------+     <-- architecture          |
  |        |    Simulation distribuee  |          (tick sync, hand-off) |
  |        +---------------+-----------+                                |
  |                        |                                            |
  |                        | produit un flux                            |
  |                        v d'events canoniques                        |
  |                  +----------+                                       |
  |                  |  SD 02   |                                       |
  |                  | Pipeline |                                       |
  |                  | EOD Rev  |<--- Agentic 01 (fleet brain) ecrit ici|
  |                  +----+-----+                                       |
  |                       |                                             |
  +-----------------------+---------------------------------------------+
                          |
  +---------------------- INTELLIGENCE (consomme les events) -----------+
  |                       |                                             |
  |                       v                                             |
  |    +--------+   +--------+   +--------+                             |
  |    | NN 01  |   | NN 02  |   | NN 03  |                             |
  |    |colli-  |   |imita-  |   | LLM    |                             |
  |    |sion det|   |tion    |   | EOD    |                             |
  |    +--------+   +--------+   +---+----+                             |
  |                                  |                                  |
  |                                  v                                  |
  |                         +-----------------+                         |
  |                         |  Agentic 03     |                         |
  |                         | EOD conversat.  |                         |
  |                         +-----------------+                         |
  |                                                                     |
  |           +----------------------------------+                      |
  |           |       Agentic 02 (PHARE)        |                      |
  |           | Supervisor + Swarm multi-tier   |                      |
  |           |  (orchestre des Agentic 01)     |                      |
  |           +----------------------------------+                      |
  |                                                                     |
  +---------------------------------------------------------------------+

  +-------------------- PLATFORM (delivery) ---------------------------+
  |   SD 03 — Multi-tenant on-prem : livraison chez les 20+ clients    |
  |   logistique. Couche ops/supply-chain, transverse aux deux couches |
  |   precedentes.                                                     |
  +--------------------------------------------------------------------+
```

**Lectures de bout en bout** :

1. **Chaine "data EOD"** : un shift genere des events via `SD 01` qui les persiste dans `SD 02`. `NN 03` et `Agentic 03` les relisent pour produire le rapport, `NN 01` les scanne pour detecter les collisions / quasi-collisions.
2. **Chaine "runtime operationnel"** : `Algo 01/02/03` sont les primitives utilisees dans le moteur `SD 01`. Les "brains" des unites sont des `Agentic 01` ; quand on orchestre plusieurs flottes, on utilise `Agentic 02`.
3. **Chaine "modelisation comportementale"** : `NN 02` apprend d'operateurs experts pour enrichir `Agentic 01` avec des politiques apprises plutot que scripted.

Le lecteur n'est pas oblige de faire les projets dans cet ordre, mais comprendre le fil rouge aide a voir pourquoi chaque brique existe.
