# Exercices Moyens — Les 4 piliers d'un agent gouvernable (J2)

> Stack : Python 3.11+ stdlib uniquement. Point de depart : `02-code/02-quatre-piliers.py`.

---

## Exercice 1 : Enforcer le moindre privilege a l'execution

### Objectif
Transformer le pilier Permissions en garde-fou actif : bloquer une action hors scope **avant** qu'elle s'execute, et la tracer.

### Consigne
1. En partant de la methode `act()` de `GovernedAgent`, ecris une fonction `attempt(agent, action, scope_required, **context)` qui :
   - autorise l'action si `scope_required` est dans `agent.permissions`,
   - la refuse sinon,
   - **dans les deux cas**, ecrit une entree dans l'`audit_log` (identite, action, scope, resultat `executed`/`denied`).
2. Rejoue l'incident bancaire : un agent `read:disputes` / `write:dispute_notes` qui tente `funds_transfer` (scope `transfer:funds`). L'action doit etre **refusee**.
3. Ajoute une notion de **borne** sur un scope : meme avec `transfer:funds`, refuse si `amount` depasse une limite (ex. 1 000 €). Modelise la limite comme une donnee de l'agent (ex. un dict `limits`).

### Criteres de reussite
- [ ] Une action hors scope est refusee (`result == "denied"`)
- [ ] Une action dans le scope est executee (`result == "executed"`)
- [ ] Les actions refusees **et** acceptees laissent une trace dans l'audit_log
- [ ] Un montant au-dessus de la borne est refuse meme si le scope est present
- [ ] Le test rejoue l'incident : le `funds_transfer` de 4 200 € est bien bloque

---

## Exercice 2 : Serialiser / deserialiser une Agent Card (JSON)

### Objectif
Materialiser les 4 piliers dans une Agent Card persistable et la recharger sans perte.

### Consigne
1. Ajoute a `GovernedAgent` deux methodes :
   - `to_card() -> dict` : produit un dict des metadonnees de gouvernance (id, owner, permissions, et un champ `audit_entries` = nombre d'entrees, **pas** tout le log).
   - `from_card(cls, data: dict)` (classmethod) : reconstruit un agent depuis ce dict.
2. Serialise une Agent Card en JSON (`json.dumps`), recharge-la (`json.loads` + `from_card`), et verifie que l'agent reconstruit **passe toujours** `check_governance`.
3. Ajoute une validation a `from_card` : si une cle de pilier manque (ex. pas d'`owner`), leve une `ValueError` explicite (« missing pillar: owner »).

### Criteres de reussite
- [ ] `to_card()` produit un dict serialisable en JSON (testé avec `json.dumps`)
- [ ] Le round-trip (card -> JSON -> card) reconstruit un agent equivalent
- [ ] L'agent reconstruit passe `check_governance` (liste vide)
- [ ] Une card incomplete leve une `ValueError` nommant le pilier manquant
- [ ] Le nombre d'entrees d'audit est preserve dans la card

---

## Exercice 3 : Distinguer ASI01 — usurpation vs abus de privilege

### Objectif
Outiller la categorie OWASP **ASI01 (Identity and Privilege Abuse)** en deux sous-detecteurs distincts.

### Consigne
1. Ecris deux fonctions de detection separees :
   - `detect_impersonation(agent) -> bool` : l'`agent_id` usurpe-t-il une identite humaine (prefixe `user:`/`human:`/`person:`/`employee:`) ?
   - `detect_privilege_abuse(agent) -> bool` : l'agent detient-il un scope dangereux **non justifie** par son role ? Modelise un role attendu (ex. un dict `ROLE_ALLOWED_SCOPES = {"summarizer": {"read:tickets", "write:ticket_summary"}}`) et signale tout scope **hors** de cet ensemble (ou un wildcard).
2. Sur une petite flotte mixte, produis un tableau : pour chaque agent, `impersonation` (oui/non) et `privilege_abuse` (oui/non).
3. Conclus : compte combien d'agents declenchent **au moins un** des deux signaux ASI01.

### Criteres de reussite
- [ ] `detect_impersonation` distingue un `agent://...` (faux) d'un `user:...` (vrai)
- [ ] `detect_privilege_abuse` signale un scope hors du role attendu et le wildcard
- [ ] Le tableau couvre toute la flotte avec les deux colonnes
- [ ] Le decompte final des agents « ASI01 » est correct sur ton echantillon
- [ ] Un agent conforme a son role ne declenche **aucun** des deux signaux
