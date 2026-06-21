# Exercice (hard) — Chaîne de délégation & atténuation des privilèges

## Objectif

Ajouter à ton PDP la dimension la plus subtile de l'IAM d'agents : la **délégation**. Tu vas modéliser une chaîne `humain → agent → agent`, garantir qu'elle remonte toujours à un **principal humain**, et faire respecter l'**atténuation des privilèges** (un délégué ne peut jamais gagner un scope que son délégant n'avait pas). C'est l'invariant « les humains restent ultimement responsables » rendu exécutable.

## Consigne

Pars de ton `authorize()` de l'exercice medium et enrichis-le.

1. Ajoute au jeton un champ `delegated_by` : une **séquence ordonnée** de maillons (du principal racine vers le délégué direct). Chaque maillon porte au minimum : `principal` (id), `is_human` (booléen), `scopes` (les scopes que ce principal détenait au moment de déléguer).
2. Ajoute une **4ᵉ vérification** dans `authorize()`, après le check de scope :
   1. si `delegated_by` est **vide** → autorisé, mais renvoie un motif signalant l'absence de racine humaine (visibilité de gouvernance) ;
   2. sinon, le **premier** maillon doit avoir `is_human=True` → sinon DENY (« chaîne ne remonte pas à un humain ») ;
   3. **atténuation** : pour le scope demandé, **chaque** maillon de la chaîne doit déjà détenir ce scope → sinon DENY en nommant le maillon fautif. (Les privilèges ne peuvent que rétrécir le long de la chaîne, jamais apparaître.)
3. Écris une fonction `root_principal(token) -> str` qui remonte la chaîne et renvoie l'identifiant du **principal humain racine** (ou `<self:agent_id>` si pas de délégation).
4. Démontre **quatre** cas :
   - délégation valide `humain → orchestrateur → data-agent`, scope détenu par toute la chaîne → ALLOW, et `root_principal` renvoie bien l'humain ;
   - chaîne dont le premier maillon **n'est pas** humain → DENY ;
   - **escalade « magique »** : le jeton du data-agent accorde `payments:execute`, mais aucun maillon de sa chaîne ne détenait ce scope → DENY (atténuation violée) ;
   - rappel de régression : un jeton **expiré** avec une délégation par ailleurs valide → DENY pour expiration (les checks précédents priment, *fail-closed*).
5. **Probe adversariale** : construis une chaîne où le maillon intermédiaire (agent) détient le scope mais où **l'humain racine ne le détient pas** → doit DENY (l'agent ne peut pas avoir plus de droits que l'humain qui l'a mandaté).

## Critères de réussite

- [ ] Le jeton porte une chaîne `delegated_by` ordonnée (racine → délégué direct) avec `principal`, `is_human`, `scopes` par maillon.
- [ ] La chaîne non vide est **rejetée** si son premier maillon n'est pas humain.
- [ ] L'**atténuation** est vérifiée scope-par-scope sur **tous** les maillons ; le motif de refus nomme le maillon fautif.
- [ ] `root_principal()` renvoie l'humain racine pour une chaîne, et `<self:...>` en l'absence de délégation.
- [ ] Les quatre cas démontrés donnent ALLOW / DENY (non-humain) / DENY (escalade) / DENY (expiré) conformément à l'ordre *fail-closed*.
- [ ] La probe « humain racine sans le scope » renvoie DENY (atténuation), pas ALLOW.
