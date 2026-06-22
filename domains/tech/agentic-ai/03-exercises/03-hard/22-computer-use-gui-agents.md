# Exercices Hard — Computer use & GUI agents (J22)

> Theme : passer du jouet a l'agent GUI **defendable en production**. On cable une boucle robuste
> (retries, budget d'etapes, detection de boucle, gate HITL, echec gracieux) puis un mini-harnais
> facon WebArena (sections 7 et 8 du cours) avec erreurs cumulatives et une **ablation** qui mesure
> le gain du set-of-marks vs le grounding pixel brut. Tout tourne hors-ligne, en stdlib, sur le
> `VirtualScreen` ASCII du module et un MockLLM deterministe.

---

## Exercice 1 : Runner d'agent GUI robuste (retries + step budget + loop-detect + HITL + echec gracieux)

### Objectif
Assembler en un seul runner les garde-fous de production de la section 8 : budget d'etapes strict,
detection de boucle (meme etat d'ecran revisite), retries avec re-perception, **gate HITL avant toute
action irreversible** (submit/delete/buy), et un echec **gracieux** qui ne lance jamais d'exception non
maitrisee. On valide le runner sur trois scenarios aux issues differentes.

### Consigne
En t'appuyant sur les primitives de `02-code/22-computer-use-gui-agents.py` :

1. Construis un `RobustGUIAgent` avec un `step_budget` (ex : 12) et :
   - **Loop detection** : hache l'etat de l'ecran (valeurs des inputs + dernier bouton active) ;
     si le **meme** hash revient `loop_threshold` fois (defaut 3) → abandon `status="loop_aborted"`.
   - **Retries** : chaque action de clic re-percoit (`set_of_marks`) et valide le label resolu ;
     en cas d'echec, retente jusqu'a `max_retries` (defaut 2), sinon `status="failed"`.
   - **Gate HITL** : avant toute action dont le tool est dans `IRREVERSIBLE = {"Submit", "Delete", "Buy"}`,
     appeler un callback `confirm(intent) -> bool`. Si refus → `status="hitl_blocked"`, l'action n'est PAS executee.
   - **Echec gracieux** : `run(...)` retourne toujours un dict `{status, steps, log, last_screen}` ;
     aucune exception ne doit remonter.
2. Le LLM est un **MockLLM** deterministe : il recoit le screenshot SoM + le but et renvoie l'`Intent`
   suivante (par label). Fournis 3 politiques scriptees pour les 3 scenarios ci-dessous.
3. **Scenario A — succes** : login complet (`Username`/type/`Password`/type/`Submit`) avec un `confirm`
   qui dit oui sur `Submit`. Attendu : `status="success"`, les deux inputs remplis, `Submit` active.
4. **Scenario B — boucle detectee** : politique pathologique qui re-clique le **meme** input sans jamais
   taper, donc l'etat ne change pas. Attendu : `status="loop_aborted"` avant d'epuiser le budget.
5. **Scenario C — HITL bloque** : login jusqu'a `Submit`, mais `confirm` renvoie `False` sur `Submit`.
   Attendu : `status="hitl_blocked"`, le bouton `Submit` n'a **jamais** ete active (verifie sur l'ecran).

### Criteres de reussite
- [ ] `RobustGUIAgent.run` retourne toujours un dict structure (echec gracieux, zero exception)
- [ ] La detection de boucle stoppe le scenario B avant la fin du budget (`loop_aborted`)
- [ ] Le gate HITL bloque `Submit` dans le scenario C ; le bouton n'est jamais active
- [ ] Le scenario A reussit : inputs remplis et `Submit` active
- [ ] Le budget d'etapes est respecte (jamais depasse) et les retries re-percoivent l'ecran
- [ ] Les 3 scenarios tournent dans un seul script stdlib, code 0

---

## Exercice 2 : Mini-harnais WebArena + erreurs cumulatives + ablation SoM vs pixels

### Objectif
Reproduire a petite echelle le constat de la section 7.1 (WebArena) : sur une tache **multi-pages**,
les erreurs de grounding **s'accumulent** (un mauvais clic mene a une page inattendue, qui mene a un
nouveau mauvais clic...). On mesure ensuite, par **ablation**, que le set-of-marks fait monter le taux
de succes par rapport au grounding pixel brut bruite.

### Consigne
1. Modelise un **parcours multi-pages** : une liste ordonnee de `Page`, chacune avec son `VirtualScreen`
   et une `goal_label` (l'element a cliquer pour passer a la page suivante). Ex : `Search → Product →
   Cart → Checkout` (4 pages, 4 clics corrects = succes).
2. Implemente deux modeles de grounding **bruites et deterministes** (seedes par `random.Random(seed)`,
   PAS de hasard global), chacun renvoyant le mark/element clique pour une `goal_label` :
   - `RawPixelGrounding(error_rate=p)` : avec proba `p`, clique un element **voisin** au lieu de la cible
     (simule l'imprecision pixel). Sinon clique juste.
   - `SoMGrounding(error_rate=q)` avec `q < p` : le choix d'un id discret reduit l'erreur (section 6).
3. Implemente la **propagation d'erreur cumulative** : un clic errone fait echouer la page (et donc la
   tache) — l'agent ne peut pas atteindre la page suivante. Compte une tache comme **reussie** seulement
   si **toutes** les pages sont franchies correctement.
4. Lance une **ablation** : `N=200` essais (seeds 0..199) pour chaque strategie, calcule le `success_rate`,
   et verifie par assertion que `success_rate(SoM) > success_rate(pixels)`.
5. Ajoute une **courbe d'erreur cumulative** : montre analytiquement et empiriquement que le succes
   ~ `(1 - error_rate) ** n_pages` (le succes chute exponentiellement avec le nombre d'etapes), et
   affiche un petit tableau `n_pages → success_rate` pour les deux strategies.

### Criteres de reussite
- [ ] Le parcours a >= 4 pages, succes = tous les clics corrects (erreur cumulative)
- [ ] Les deux groundings sont deterministes (seedes via `random.Random`, pas d'etat global)
- [ ] `SoMGrounding` a un taux d'erreur strictement inferieur a `RawPixelGrounding`
- [ ] L'ablation sur N essais montre `success_rate(SoM) > success_rate(pixels)` (assertion)
- [ ] La decroissance empirique suit ~ `(1 - error_rate) ** n_pages` (verifiee approximativement)
- [ ] Un tableau `n_pages → success_rate` est affiche, stdlib pure, code 0
