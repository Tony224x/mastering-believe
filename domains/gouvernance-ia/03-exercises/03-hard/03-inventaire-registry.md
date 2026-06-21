# Exercice (hard) — Registry de gouvernance versionne & board report

## Objectif

Transformer le registry en **source d'autorite auditable** : chaque mutation est horodatee et conservee (jamais de suppression destructive), le cycle de vie est gere proprement (suspension, decommission), et le registry produit un **rapport de gouvernance board-ready** (indicateurs + alertes) a partir de requetes. C'est l'ossature qui alimentera le scoring de risque (J4) et l'audit (J9).

## Consigne

1. Pars d'un registry avec les 4 piliers + `status` + `enrolled_at` + `updated_at` (timestamps ISO UTC).
2. **Cycle de vie sans suppression** : implemente `transition(agent_id, *, status=None, owner=None)` qui modifie statut et/ou owner, **met a jour `updated_at`**, et **n'efface jamais** la ligne. Valeurs de `status` autorisees : `active`, `suspended`, `decommissioned`. Decommissionner = changer le statut, pas supprimer.
3. **Journal de mutations (append-only)** : maintiens une liste `history` ou chaque mutation pousse une entree `{timestamp, agent_id, change}`. Le journal ne doit **jamais** etre reduit (on ajoute seulement). C'est le pont vers l'audit trail de J9.
4. **Requetes de gouvernance avancees** :
   - `by_risk(min_tier)` — agents au tier `min_tier` ou au-dessus, tries du plus risque au moins risque.
   - `ownership_concentration()` — `dict {owner: nombre_d_agents}`, pour reperer un humain surcharge (goulot d'accountability).
   - `active_orphans()` — orphelins **dont le statut est `active`** (un orphelin actif est le pire cas : il agit et n'est imputable a personne).
5. **Board report** : `build_report() -> dict` qui assemble : `coverage_pct`, `total_active`, `n_active_orphans`, `ownership_concentration`, `high_risk_active` (nb d'agents `high` et `active`), et une liste `alerts` (texte) declenchee par des seuils — par ex. « N orphelins actifs », « owner X possede > K agents ».
6. **Probe adversariale** : prouve qu'apres un `decommission`, (a) la ligne existe toujours, (b) `updated_at` a change, (c) le `history` a grossi, (d) l'agent decommissionne **ne compte plus** dans `total_active`.
7. `if __name__ == "__main__":` : monte une flotte de ~6 agents, applique mutations + un decommission, imprime le `build_report()` et verifie les invariants de la probe.

## Criteres de reussite

- [ ] `transition` met a jour `updated_at`, refuse un `status` invalide, et **ne supprime jamais** d'entree.
- [ ] Decommissionner un agent : la ligne subsiste (`status="decommissioned"`) et l'agent sort de `total_active`.
- [ ] `history` est strictement append-only (sa longueur ne fait que croitre au fil des mutations).
- [ ] `by_risk` trie correctement (plus risque d'abord) ; `ownership_concentration` compte juste.
- [ ] `active_orphans()` ne renvoie que des orphelins `active`.
- [ ] `build_report()` produit les indicateurs ET genere au moins une `alert` quand un seuil est franchi (test : forcer un orphelin actif => alerte presente).
- [ ] La probe adversariale verifie les 4 invariants du point 6 (assertions qui passent).
- [ ] Script en **stdlib uniquement**, `python <fichier>` exit 0, `python -m py_compile` passe.
