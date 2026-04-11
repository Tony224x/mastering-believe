# Reference design — Simulation distribuee tick-based

## Architecture globale

```
                  +------------------+
                  |   Tick master    |<---- commande formateur (OPORD injection)
                  |  (barrier + seed)|
                  +---------+--------+
                            |
        +------+------+------+------+------+
        |      |      |      |      |      |
        v      v      v      v      v      v
     [W1]   [W2]   [W3]   [W4]   [W5]   [Observer]
     zone   zone   zone   zone   zone    (read-only
      1      2      3      4      5      snapshots)
      |<----- ghost boundary replication ---->|
            (chaque worker publie sa zone
             a ses voisins immediats)
```

**Roles** :
- **Tick master** — coordonne la progression globale. Emet "TICK n" a 10 Hz, attend ACK de tous les workers, puis "COMMIT n" pour autoriser la publication cross-zone. Pas de logique simu, juste l'orchestration.
- **Worker** — authoritative sur une zone. Simule les unites dans sa zone, execute les events, publie son etat frontier aux voisins.
- **Observer** — client read-only (GUI formateur, service AAR). N'influence pas la simu.

## Decoupage spatial — choix : grille reguliere statique

Considere :
1. **Grille reguliere** — simple, predictible, facile a debugger. Mauvais si l'action se concentre sur une seule zone (un worker sature quand les 4 autres sont idle).
2. **Quadtree adaptatif** — redistribue dynamiquement. Plus complexe, casse le determinisme si on rebalance en cours d'exercice.
3. **Zones metier** — "forest area", "urban area" definies a la main. Flexible mais pas general, demande de la config par scenario.

**Choix : grille reguliere** pour la v1. Simplicite > optimalite sur un prototype. On acceptera qu'un worker sature en cas de concentration — mesurable et documentable.

## Boucle de tick — lockstep avec barrier

Chaque tick passe par 3 phases :

```
Phase 1 : EXEC
  master -> all workers : "TICK n, seed=HASH(base_seed, n)"
  worker :
    - applique les events de la priority queue locale jusqu'a t == n*dt
    - les events generes restent locaux ou sont buffered pour phase 2
    - envoie "DONE n" a master

Phase 2 : EXCHANGE (barrier : master attend tous les DONE)
  master -> all workers : "EXCHANGE n"
  worker :
    - publie son etat frontier (unites dans la bande de X metres) aux voisins
    - recoit l'etat frontier des voisins
    - resout les events cross-zone (hand-off, combat a la frontiere)

Phase 3 : COMMIT
  master -> all workers : "COMMIT n"
  worker : sauvegarde snapshot incremental, prepare tick n+1
```

Latence pour un tick complet = max(duree_exec_workers) + 2 * RTT + duree_exchange. Sur LAN datacenter (RTT < 1 ms) et tick de 100 ms, ca passe largement.

## Hand-off cross-zone — "frontier propose, voisin valide"

Probleme : une unite a `(r=100, c=100)` dans la zone 1 avance vers `(r=100, c=101)` qui est dans la zone 2.

Protocole :
1. En phase 1, W1 constate que l'unite va sortir et marque l'event `HANDOFF(unit_id, target_zone=2)` dans son buffer cross-zone.
2. En phase 2, W1 envoie l'event + l'etat complet de l'unite a W2.
3. W2 valide (pas de collision, zone accepte), confirme a W1.
4. W1 retire l'unite de son etat, W2 l'ajoute.
5. Si le tick master n'a pas recu le ACK de W2 dans la fenetre, W1 garde l'unite (abort du hand-off). Retry au prochain tick.

**Cle** : un seul worker est authoritative a la fois. Pas de region overlap sur l'autorite, juste sur la lecture (ghost frontier).

## Determinisme — les 5 regles

1. **Seed par tick, pas par worker** : `seed_n = hash(base_seed, n)`. Chaque worker derive son PRNG du seed global, pas d'une horloge locale.
2. **Ordre stable des events a meme t** : trier par `(t, hash_stable(event_id))` avant application. `event_id = hash(unit_id, kind, creation_tick)`.
3. **Pas de timer OS dans la simu** : tout est event-driven, le seul "temps" qui compte est `tick_number`.
4. **Pas d'iteration sur `set()`** : `set` est non ordonne en Python. Utiliser `sorted(set)` ou `list`.
5. **Snapshot avant/apres chaque tick** : hashable. En CI, rejeu deux fois, compare les hashes tick par tick. Premier divergence = bug determinisme.

## Crash d'un worker

Options :
1. **Abort exercise** — le plus simple, acceptable pour la v1. L'exercice est perdu, le formateur recommence.
2. **Snapshot + reprise** — chaque worker persiste son snapshot a chaque COMMIT (disque local). Un worker de secours peut reprendre la zone a partir du dernier snapshot.
3. **Etat replique** — chaque zone a un follower qui copie l'etat en temps reel. Instantane en cas de bascule, mais x2 la RAM et la bande.

**Choix v1 : option 1**. Document pour la v2 : option 2.

## SLOs

| SLO | Valeur | Mesure |
|---|---|---|
| Tick completion p99 | < 80 ms | histogramme par tick |
| Determinisme rejeu | 100% bit-exact | test CI nightly |
| Crash recovery | n/a v1, < 10 s v2 | manuel |
| Desync drift | 0 ticks | compteur de ticks par worker |

## Questions de revue — reponses

**Pourquoi pas un gros serveur unique ?**
Pour 5000 unites, on atteint les limites d'un seul thread Python/C++ non-vectorise. Le GIL Python serait bloquant. Un seul noeud = unique SPOF. La distribution est aussi une competence que le client defense valorise (resilience).

**Taille max d'une zone ?**
Empirique : quand le cout de l'exchange depasse 20% du temps de tick, la zone est trop grande. Dans notre cas ca tombe autour de 1500 unites / zone.

**Test CI de determinisme ?**
Bench : rejouer un scenario fixe 2 fois en parallele, comparer hash(snapshot) a chaque tick. Premier hash divergent = fail CI.

**Qu'est-ce qui casse le determinisme ?**
- `time.time()` pour un random
- `set` / `dict.keys()` non ordonne (OK en Python 3.7+ pour `dict`, pas pour `set`)
- Ordonnancement d'events generes par des handlers non stables
- Flottants et ordre d'application (a+b+c != a+c+b en IEEE 754)

**Crash de la zone la plus active ?**
C'est le cas qui fait le plus mal : on perd le plus d'unites. La v2 doit prioriser la replication pour les zones a forte densite.

## Trade-offs non choisis

- **Jefferson / virtual time / rollback** — trop complexe pour le gain sur LAN. Valable si WAN ou tick rate > 60 Hz.
- **HLA/DIS natif** — les normes IEEE 1516 / 1278 font deja le lockstep. MASA les supporte deja en realite. Pour un exo pedagogique, construire from scratch est instructif.
- **Actor model (Akka / Orleans)** — beau, mais cache la boucle de tick qu'on veut controler explicitement.
