# Projet 01 — Registry de gouvernance & audit des agents orphelins

> Difficulte : **medium** — contexte metier LogiSim/FleetSim : voir [`shared/logistics-context.md`](../../../../shared/logistics-context.md).

## Contexte metier

Tu es AI Engineer en mission chez un client **LogiSim**, dans une salle de controle (OCC) qui pilote une plateforme logistique automatisee avec **FleetSim**. Depuis un an, plusieurs equipes ops ont chacune deploye, de leur cote, des agents LLM « fleet brain » — les coordinateurs de flotte du [projet Agentic 01](../../../agentic-ai/05-projets-guides/01-agent-fleet-coordinator/README.md) — pour piloter leurs flottes d'AGV, de drones de comptage et de lignes de tri.

Lundi matin, le DSI du client pose une question banale au comite : **« combien d'agents IA tournent chez nous, et qui les possede ? »** Personne ne sait repondre. Il y a « plusieurs fleet brains », certains deployes par des gens partis depuis. L'un d'eux a meme ete branche sur une **flotte externe** (un sous-traitant 3PL) avec un acces a la telemetrie client — alors que le site est **on-premise isole** et que la **confidentialite des flux client** est une obligation contractuelle (clients audites ISO 9001 / SOC 2).

Le probleme n'est pas le code des agents. C'est une **absence de gouvernance** : pas d'inventaire vivant, pas d'owner clair, des agents fantomes. Ta mission : construire le **registry** qui rend cette flotte gouvernable.

## Objectif technique

Construire un **registry de gouvernance** en Python stdlib (aucune dependance, tourne sur un site isole sans cloud) qui :

1. **Ingest** une flotte d'agents (liste de dicts en dur, de la forme d'un `fleet.json` exporte) melangeant **own fleet** et **external fleet**, certains sans owner, certains sur-permissionnes, un usurpant une identite humaine.
2. **Valide les 4 piliers** par agent — identite unique non-humaine, owner humain nomme, permissions au moindre privilege bornees par scopes, presence d'audit — en classant chaque pilier **OK / PARTIEL / ABSENT**.
3. **Detecte les orphelins** (agents sans owner nomme = *shadow agents*) et les rend **VISIBLES** (on ne les supprime jamais : supprimer effacerait la tracabilite).
4. **Calcule la couverture de gouvernance** = agents pleinement gouvernes / total.
5. **Imprime un tableau registry** clair (`agent_id | owner | piliers OK | statut`) + la liste des orphelins + le taux de couverture, le tout termine par un **verdict** actionnable.

## Consigne

Le pipeline a la forme suivante :

```
   FLEET (liste de dicts, facon fleet.json)
        |
        v
  +------------+   (1) INGEST : charge chaque agent dans un Registry vivant
  | Registry   |        -> on n'ecarte PAS les agents incomplets, on les admet
  +-----+------+
        |
        v
  +------------+   (2) VALIDATE : les 4 piliers par agent
  | 4 piliers  |        identity / owner / permissions / audit -> OK|PARTIEL|ABSENT
  +-----+------+        + drapeaux rouges metier (egress client interdit)
        |
        v
  +------------+   (3) ORPHANS    : agents sans owner nomme (rendus visibles)
  | Requetes   |   (4) COVERAGE   : gouvernes / total
  +-----+------+
        |
        v
     [ tableau registry + orphelins + couverture + verdict ]   (5) REPORT
```

Entrees / sorties attendues :

- **Entree** : un `FLEET: list[dict]` module-level. Chaque agent a au minimum `agent_id`, `owner` (peut etre `None`), `fleet_kind` (`"own"` | `"external"`), `scopes` (liste), `risk_tier`, `has_audit` (bool).
- **Sortie** : un rapport texte imprime sur stdout (tableau + bloc orphelins + bloc drapeaux rouges + couverture + verdict). Exit code **0**.

Regles de validation des piliers :

| Pilier | OK | PARTIEL | ABSENT |
|---|---|---|---|
| **Identite** | `agent_id` present, non-humain | — | absent, ou prefixe humain (`user:`, `human:`, …) = usurpation |
| **Owner** | personne nommee (own fleet) | personne nommee mais **external** (redevabilite moindre) | vide, ou boite generique (`team`, `it`, `ops-team`, …) = orphelin |
| **Permissions** | scopes explicites et bornes | scope **wildcard** (`*`) = present mais non borne | aucun scope declare |
| **Audit** | `has_audit = True` | — | `has_audit = False` |

**Ancre metier** : un scope d'egress (`export:client_telemetry`, …) est un **drapeau rouge** dans un contexte isole — **CRITIQUE** si l'agent est une `external` fleet. Un agent peut cocher les 4 piliers ET rester dangereux : la couverture « pleinement gouverne » exige donc *aussi* l'absence de drapeau rouge.

## Etapes guidees

1. **Modeliser l'agent** — une `@dataclass RegistryAgent` avec un constructeur `from_raw(dict)` **tolerant** : un export reel est imparfait (cles manquantes). On admet les agents incomplets pour que leurs trous deviennent comptables — sinon on recree le *shadow AI* qu'on veut eliminer.
2. **Coder les 4 controles** — une fonction par pilier, retournant un etat ternaire `OK | PARTIEL | ABSENT`. Le ternaire (et pas un booleen) est ce qui permet de **prioriser** la remediation : un wildcard n'est pas « absent », c'est « scope present mais non borne ».
3. **Centraliser la doctrine** — les ensembles `_HUMAN_ID_PREFIXES`, `_NON_OWNERS`, `_WILDCARD_SCOPES`, `_FORBIDDEN_EGRESS_SCOPES` encodent les regles de gouvernance. Les isoler permet de les auditer / faire evoluer sans toucher la logique.
4. **Construire le Registry** — un objet (pas un tableur) qui ingere, puis repond a des **requetes** : `orphans()` (la requete critique), `red_flags()`, `coverage()`. Regle d'or du domaine : *pas dans le registry => pas en production*.
5. **Rendre le rapport** — trier le tableau **pires en premier** (un comite lit le haut, pas 400 lignes), afficher orphelins et drapeaux rouges en blocs distincts, finir sur un **verdict** (un board decide, il ne lit pas des tables brutes).
6. **Probe adversariale** — verifier qu'une **flotte vide** ne plante pas (couverture `0/0` sans `ZeroDivisionError`).

## Criteres de reussite

- Le script tourne avec `python registry_audit.py` et **exit 0**, sans aucune dependance externe ni cle API.
- Le tableau registry distingue clairement les 4 statuts : `GOUVERNE`, `ORPHELIN`, `DRAPEAU ROUGE`, `INCOMPLET`.
- Les **2 agents orphelins** sont detectes ET rendus visibles : l'agent sans owner (`owner = None`) **et** celui dont l'owner est une boite generique (`ops-team`) — pas seulement le premier.
- L'agent `external` portant `export:client_telemetry` est marque **drapeau rouge CRITIQUE** (confidentialite des flux client en site isole).
- L'agent usurpant une identite humaine (`user:…`) sort le pilier **Identite = ABSENT**.
- La **couverture de gouvernance** est calculee et affichee (ici 3/8 = 38 %), et le verdict pointe l'action prioritaire.
- La probe « flotte vide » ne plante pas.

## Solution

Voir [`solution/registry_audit.py`](solution/registry_audit.py). Le fichier est commente avec le **POURQUOI** de chaque choix de gouvernance (pourquoi un etat ternaire, pourquoi on n'ecarte pas les agents incomplets, pourquoi un owner externe vaut PARTIEL, pourquoi l'egress client est un drapeau rouge a part).

## Questions de reflexion

- Le registry repose ici sur un **export declaratif** (`fleet.json`). Que manque-t-il pour passer a la **reconciliation** (confronter « ce qui est declare » a « ce qui agit reellement » via telemetrie/scan) ? Quel type d'agent ce rapprochement ferait-il apparaitre que l'export seul rate ?
- Un agent coche les **4 piliers** mais c'est une `external` fleet avec `export:client_telemetry`. Pourquoi le statut « pleinement gouverne » doit-il quand meme echouer ? Que dit cela sur la difference entre **gouvernance generique** et **regles metier** specifiques au client ?
- L'agent `legacy-99` a `owner = "ops-team"`. Pourquoi traite-t-on une **equipe** comme un orphelin, au meme titre qu'un `owner = None` ? (Indice : escalade, kill-switch, imputabilite.)
- On **n'efface jamais** un agent du registry, meme orphelin ou decommissionne. Quel risque d'investigation d'incident la suppression ferait-elle courir ?

## Pour aller plus loin

- Ajouter une requete `by_owner(owner)` (accountability : quand un employe part, qui herite de ses N agents ?) et `by_risk(tier)` (priorisation : on n'audite pas un agent de resume comme un agent qui dispatch des AGV).
- Brancher l'**Agent Card** : au lieu d'un `dict` en dur, ingerer une carte JSON declarative (identite + scopes + auth) et **valider** la presence des champs critiques avant ingestion (le principe durable derriere Microsoft Entra Agent ID / Google A2A).
- Ajouter le **cycle de vie** : un champ `status` (`active` | `suspended` | `decommissioned`) et des transitions **horodatees** (pont vers l'audit trail), en **transitionnant le statut** plutot qu'en supprimant la ligne.
- Exporter le rapport en **JSON** en plus du texte, pour le differ mois apres mois et suivre la couverture de gouvernance dans le temps.
