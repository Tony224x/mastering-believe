# Projet 03 — Operations event queue (simulation discrete-event)

## Contexte metier

FleetSim n'est pas une simulation au tick brutal ou "tout le monde fait une action par tick". C'est une **simulation a evenements discrets** : on a une queue priorisee par temps, on pop l'evenement le plus proche dans le futur, on avance le temps simule, on applique l'evenement, qui peut en generer d'autres. C'est ce qui permet de simuler 2000 robots sans iterer sur les 2000 chaque seconde.

Pour un cycle de pickup/depose typique, les evenements sont :
- `MOVE_START(unit, from, to, t)` — un AGV demarre un trajet
- `MOVE_DONE(unit, to, t + dt_move)` — l'AGV arrive a destination
- `PICKUP(unit, parcel, t)` — le robot prend en charge un colis
- `DROPOFF(unit, parcel, slot, t)` — le robot depose le colis
- `RECHARGE_COMPLETE(unit, t + dt_recharge)` — la flotte est prete a repartir

Ton job : coder le scheduler.

## Objectif technique

Implementer une classe `EventLoop` qui :
- Stocke les evenements dans un heap priorise par `(timestamp, sequence)`
- Expose `schedule(event, at)` pour planifier un evenement dans le futur
- Expose `run_until(t_end)` qui pop et applique tous les evenements jusqu'a `t_end`
- Maintient un temps simule monotone (ne revient jamais en arriere)
- Est **deterministe** : meme entree = meme ordre d'execution meme en cas de ties

## Consigne

```python
@dataclass
class Event:
    kind: str
    payload: dict
    def apply(self, loop: "EventLoop") -> None: ...

class EventLoop:
    def __init__(self) -> None: ...
    def now(self) -> float: ...
    def schedule(self, event: Event, at: float) -> None: ...
    def run_until(self, t_end: float) -> int:  # retourne nb d'events traites
        ...
```

## Etapes guidees

1. **Heap** — `heapq` avec des tuples `(t, seq, event)`. Le `seq` est un compteur monotone, tie-breaker deterministe.
2. **Invariants** — `self._now` ne doit jamais decroitre. Rejeter (ou lever) si `schedule(event, at=self._now - 1)`.
3. **apply_in_order** — dans `run_until`, popper tant que `t <= t_end`. Bien penser au cas `t == t_end`.
4. **Reactions** — `event.apply(loop)` peut appeler `loop.schedule(new_event, ...)`. Le nouvel evenement arrive dans le heap, sera traite si `t <= t_end`.
5. **Integration** — creer un mini-cycle : 2 AGV, l'un fait un pickup et un dropoff, l'autre rate son slot et reessaie. Verifier l'ordre des evenements dans le log.

## Criteres de reussite

- Tests de base (scheduling, ordre, determinisme) passent
- Un scenario "duel pickup" (fourni) produit **exactement** le meme log sur deux runs
- 100 000 evenements planifies et traites en **moins de 200 ms**
- Pas de fuite : aucune reference vers un event deja traite apres `run_until`
- Robuste au cas pathologique : schedule in-event d'un event a `t == now` (doit etre traite au tick suivant, pas en infinite loop)

## Piege tie-break

Deux events planifies a **exactement** le meme `t` doivent s'appliquer dans l'ordre d'insertion (FIFO). C'est ce qu'assure le `seq`. Sans lui, Python compare les payloads (dicts, dataclasses) et peut lever un `TypeError` ou produire un ordre aleatoire.

## Solution

Voir `solution/event_loop.py` pour la correction et le scenario "duel pickup" commente.

## Pour aller plus loin

- **Cancellation** — marquer un event comme annule sans le repop du heap (lazy deletion)
- **Priorites non-temporelles** — un event `EMERGENCY` doit s'executer avant un `ROUTINE` meme a meme `t`
- **Snapshots** — sauvegarder l'etat complet de l'event loop a un `t` donne pour rejeu / EOD Review
- **Multi-thread** — partitionner par zone de l'entrepot, une event loop par zone, synchro barrier
