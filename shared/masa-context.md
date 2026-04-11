# Contexte Masa Group — pour les projets guides

> Document partage par tous les dossiers `05-projets-guides/` des domaines. Sert de reference metier unique pour les cas d'etudes.

## L'entreprise

**MASA Group** (masasim.com) est un editeur francais de logiciels de simulation, base a Paris, spin-off d'un laboratoire de bio-informatique. Leader mondial sur la simulation constructive pour l'entrainement militaire, MASA est aussi actif en securite civile, gestion de crise et serious games. Deploye dans **plus de 20 pays** (France, Singapour, Canada, Nigeria, etc.).

## Produits phares

### SWORD
**Simulated Wargaming for Operational Readiness and Doctrine**. Simulation constructive tactique utilisee par les armees pour entrainer les etats-majors du **bataillon a la division**. Les unites (peloton, compagnie, batterie) sont autonomes : elles recoivent des ordres en langage militaire, les executent sans micro-management, et adaptent leur comportement quand la situation evolue.

- Reduit le nombre d'operateurs d'exercice jusqu'a **70%**
- Reduit le temps de preparation de **50%**
- Simule des milliers d'unites autonomes en temps reel
- Couvre haute intensite, asymetrique, appui aux populations

### TARAN
Extension strategique de SWORD pour entrainer au niveau **division a corps d'armee**, co-developpe avec le Ministere francais des Armees.

### Direct AI
SDK de comportements autonomes, 15+ annees de R&D, actuellement en v5. Permet de creer des agents **reactifs, adaptatifs, opportunistes** capables de trouver un compromis quand plusieurs objectifs sont contradictoires. C'est le moteur cognitif qui tourne sous SWORD.

## Stack technique (ce qu'on peut raisonnablement supposer)

- **Core simulation** : C++, moteur deterministique tick-based (20-50 Hz typique en constructive sim)
- **Orchestration / outillage** : Python pour scripts d'analyse, pipelines de donnees, tests
- **IA moderne (ou Anthony arrive)** : agents LLM pour l'aide au commandement, AAR automatique, NLP d'ordres, RAG sur corpus doctrinal
- **Integration** : HLA (High Level Architecture), DIS, couplage avec VBS4 (Bohemia Interactive) pour le rendu virtuel
- **Deploiement** : on-premise (reseau defense, souvent air-gapped), parfois VDI

## Vocabulaire metier a connaitre

| Terme | Definition |
|---|---|
| **Constructive simulation** | Simulation ou les unites sont simulees, pas les soldats individuels. Oppose a "virtual" (VBS4, casque VR) et "live" (terrain reel). |
| **Echelon** | Niveau de commandement : section (~30), peloton (~30-40), compagnie (~150), bataillon (~800), brigade, division, corps. |
| **OPORD** | Ordre d'operation, format standard OTAN en 5 paragraphes (situation, mission, execution, logistique, commandement). |
| **COA** | Course of Action — option de manoeuvre consideree par l'etat-major. |
| **C2** | Command & Control. |
| **AAR** | After-Action Review — debriefing apres exercice base sur les traces. |
| **BLUFOR / OPFOR** | Forces amies (bleues) / forces adverses (rouges). |
| **ROE** | Rules of Engagement. |
| **LOS** | Line of Sight — visibilite entre deux points, critique pour detection/tir. |
| **Fog of war** | Information imparfaite sur les unites adverses. |
| **Attrition** | Pertes cumulees sur une unite pendant un engagement. |
| **PK / Pkill** | Probabilite d'un tir de neutraliser la cible (fonction distance, arme, protection). |

## Cas d'usage recurrents (ce qui alimente les projets guides)

1. **Entrainement etat-major** — un formateur cree un scenario, les stagiaires jouent a tour de role chef de brigade, SWORD simule l'execution.
2. **Evaluation de doctrine** — comparer plusieurs COA via simulations Monte-Carlo.
3. **Rejeu et AAR** — rejouer un exercice pour identifier les decisions cles et les erreurs.
4. **Integration live-virtual-constructive (LVC)** — un soldat en VBS4 interagit avec des unites SWORD constructives.
5. **Crisis management** — meme moteur, scenarios de protection civile (inondation, attentat).

## Contraintes particulieres defense

- **Determinisme** : meme seed = meme resultat (pour debriefing et certification).
- **Air-gap** : pas d'internet, pas de cloud public. Tout doit tourner on-premise.
- **Classification** : les scenarios et donnees client sont parfois classifies, jamais de telemetrie sortante.
- **Certification** : les modeles (balistique, deplacement) doivent etre traceables et validables.
- **Interop norms** : HLA (IEEE 1516), DIS (IEEE 1278), MSDL pour les scenarios.

## Anthony chez MASA — role suppose

**AI Engineer** — amener les techniques IA modernes (LLM, agents, RAG, RL) dans un produit historiquement base sur l'IA symbolique/comportementale. Les projets guides illustrent ce que tu es susceptible de construire dans les 6 premiers mois :
- Outiller les formateurs (generation de scenario, AAR auto)
- Enrichir Direct AI avec des agents LLM-driven pour les unites high-level (EM)
- Pipelines de donnees sur les traces d'exercices
- Prototypes de RAG sur le corpus doctrinal

## Comment lire chaque projet guide

Chaque projet suit la meme structure :

1. **Contexte metier** — pourquoi ce probleme existe chez MASA
2. **Objectif technique** — ce qu'il faut construire
3. **Consigne** — specs precises, entrees/sorties attendues
4. **Etapes guidees** — decomposition en sous-taches
5. **Criteres de reussite** — comment juger que c'est bon
6. **Corrige** — solution commentee, avec les raisons des choix
7. **Pour aller plus loin** — extensions pour approfondir

## Schema d'event canonique (reference partagee)

Plusieurs projets manipulent des "events" issus d'un exercice SWORD (SD 02, NN 01, NN 02, NN 03, Agentic 03). Pour eviter la derive, voici le **schema unique** que tous les projets doivent respecter.

```python
# Event canonique MASA — format pivot, utilise partout
CanonicalEvent = {
    "id": int,             # identifiant unique dans l'exercice, monotone croissant
    "exercise_id": int,    # identifiant de l'exercice
    "seq": int,            # numero de sequence, monotone par exercise (detection drop)
    "t_sim": float,        # temps simule en secondes depuis le debut de l'exercice
    "tick": int,           # numero du tick de simulation (entier)
    "unit_id": str,        # identifiant de l'unite source, ex "Alpha-2", "OPFOR-7"
    "side": str,           # "BLUFOR" | "OPFOR" | "NEUTRAL" | "CIV"
    "kind": str,           # type d'event : voir table ci-dessous
    "payload": dict,       # donnees specifiques au kind (key-value)
}
```

### Kinds d'events standards

| Kind | Signification | Payload typique |
|---|---|---|
| `MOVE` | Deplacement d'une unite | `{"from": "grid", "to": "grid", "speed": float}` |
| `DETECT` | Detection d'une cible | `{"target_id": str, "dist_m": float, "confidence": float}` |
| `ORDER` | Ordre recu | `{"from": "HQ-id", "order_type": str, "details": str}` |
| `FIRE` | Tir d'une unite | `{"weapon": str, "rounds": int, "mode": "preemptive\|ordered\|reactive"}` |
| `IMPACT` | Impact d'un projectile | `{"target_id": str, "hit": bool, "damage": float}` |
| `DAMAGE` | Pertes subies | `{"casualties": int, "wounded": int, "source": "enemy\|friendly\|unknown"}` |
| `NEUTRALIZED` | Unite hors de combat | `{"by": "unit_id", "reason": str}` |
| `REPORT` | Sitrep remonte par une unite | `{"summary": str, "severity": "routine\|urgent"}` |
| `MARK` | Cible marquee (drone, laser) | `{"grid": str, "laser_code": str}` |
| `HANDOFF` | Unite passe sa zone a une autre | `{"to_unit": str, "area": str}` |

### Regle d'or

**Tout projet qui lit ou ecrit des events utilise strictement ce schema.** Si un besoin n'est pas couvert, on ajoute une cle dans `payload`, jamais au top-level. Le top-level reste stable pour que les outils (pipeline AAR, LLM, agents) n'aient pas a brancher sur le kind.

## Fil rouge — comment les 12 projets s'emboitent

Les projets ne sont pas isoles : ils forment trois couches d'un mini-SWORD.

```
                    MASA / SWORD — fil rouge des 12 projets guides

  +------------------------- RUNTIME de simulation ----------------------+
  |                                                                     |
  |   [Algo 01]       [Algo 02]       [Algo 03]                         |
  |    A* path         LOS             Event                            |
  |    finding         Bresenham       queue                            |
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
  |                  |   AAR    |<--- Agentic 01 (unit brain) ecrit ici |
  |                  +----+-----+                                       |
  |                       |                                             |
  +-----------------------+---------------------------------------------+
                          |
  +---------------------- INTELLIGENCE (consomme les events) -----------+
  |                       |                                             |
  |                       v                                             |
  |    +--------+   +--------+   +--------+                             |
  |    | NN 01  |   | NN 02  |   | NN 03  |                             |
  |    |fratri- |   |imita-  |   | LLM    |                             |
  |    |cide det|   |tion    |   | AAR    |                             |
  |    +--------+   +--------+   +---+----+                             |
  |                                  |                                  |
  |                                  v                                  |
  |                         +-----------------+                         |
  |                         |  Agentic 03     |                         |
  |                         | AAR conversat.  |                         |
  |                         +-----------------+                         |
  |                                                                     |
  |           +----------------------------------+                      |
  |           |       Agentic 02 (PHARE)        |                      |
  |           | Supervisor + Swarm interarmes   |                      |
  |           |  (orchestre des Agentic 01)     |                      |
  |           +----------------------------------+                      |
  |                                                                     |
  +---------------------------------------------------------------------+

  +-------------------- PLATFORM (delivery) ---------------------------+
  |   SD 03 — Multi-tenant air-gap : livraison chez les 20+ clients    |
  |   defense. Couche ops/supply-chain, transverse aux deux couches    |
  |   precedentes.                                                     |
  +--------------------------------------------------------------------+
```

**Lectures de bout en bout** :

1. **Chaine "data AAR"** : un exercice genere des events via `SD 01` qui les persiste dans `SD 02`. `NN 03` et `Agentic 03` les relisent pour produire le rapport, `NN 01` les scanne pour detecter les fratricides.
2. **Chaine "runtime tactique"** : `Algo 01/02/03` sont les primitives utilisees dans le moteur `SD 01`. Les "brains" des unites sont des `Agentic 01` ; quand on orchestre plusieurs bataillons, on utilise `Agentic 02`.
3. **Chaine "modelisation comportementale"** : `NN 02` apprend d'experts humains pour enrichir `Agentic 01` avec des politiques apprises plutot que scripted.

Anthony n'est pas oblige de faire les projets dans cet ordre, mais comprendre le fil rouge aide a voir pourquoi chaque brique existe.
