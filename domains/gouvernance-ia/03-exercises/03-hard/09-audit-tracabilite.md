# Exercice (hard) — Reconstruction d'incident & ancrage par checkpoint

## Objectif
Aller au-dela de la simple chaine de hash : **correler** une action a travers plusieurs agents via un `trace_id`, **reconstruire** l'incident en un recit defendable, et **ancrer** periodiquement l'integrite par un checkpoint qui resiste a un attaquant capable de recalculer toute la chaine.

## Consigne
1. Pars d'un journal chaine par hash (reutilise ta classe du medium, ou l'API du `02-code/09-audit-tracabilite.py`). Chaque entree porte au minimum : `trace_id`, `span_id`, `parent_span`, `agent_id`, `owner`, `action`, `params`, `scope`, `policy`, `decision`, `status`.
2. Journalise une requete de bout en bout partagee par **un seul `trace_id`** (au moins 3 spans : un orchestrateur -> un agent metier -> un appel d'outil), plus **au moins une** action sans rapport sous un `trace_id` different.
3. Ecris `reconstruct_incident(trace_id)` qui renvoie, **dans l'ordre**, toutes les entrees du trace demande (et seulement celles-la).
4. Ecris `narrate(trace_id)` qui produit une phrase de recit pour l'action sensible (ex. le `bank_transfer`) au format : *« A {when}, l'agent {agent}({owner}) a execute {action}({params}) ; scope {scope}, policy {policy} -> {decision} ; status {status}. »*. La fonction doit **d'abord** appeler `verify()` et refuser de narrer (ou prefixer un avertissement) si l'integrite est cassee.
5. Ajoute un mecanisme d'**ancrage** : une methode `checkpoint()` qui renvoie `{checkpoint_at, entries, head_hash}`. Simule un attaquant qui (a) modifie une entree passee PUIS (b) **recalcule toute la chaine** pour la rendre coherente a nouveau. Montre que `verify()` ne suffit plus a le detecter, mais que la comparaison du `head_hash` courant avec le `head_hash` du **checkpoint anterieur** revele la falsification.

## Criteres de reussite
- [ ] `reconstruct_incident(trace_id)` renvoie uniquement les spans du bon trace, dans l'ordre d'insertion.
- [ ] `narrate` appelle `verify()` avant de produire le recit et signale toute integrite cassee.
- [ ] Le recit cite explicitement l'**autorisation** (scope + policy + decision), pas seulement l'action.
- [ ] `checkpoint()` renvoie un `head_hash` qui change apres un nouvel append.
- [ ] Le scenario « attaquant qui recalcule toute la chaine » est detecte par comparaison au checkpoint anterieur (et tu expliques en commentaire pourquoi `verify()` seul ne le voit pas).
- [ ] Tout tourne en stdlib seule, sans erreur.
