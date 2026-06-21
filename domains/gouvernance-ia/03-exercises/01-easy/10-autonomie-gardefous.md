# Exercices Faciles — Autonomie, garde-fous & operations (J10)

---

## Exercice 1 : Calibrer le niveau d'autonomie

### Objectif
Associer une action a son niveau d'autonomie (in / on / out-of-the-loop) selon son risque, sans coder de politique complexe.

### Consigne
En partant de l'idee du `risk_score` de `02-code/10-autonomie-gardefous.py` :

1. Ecris une fonction `autonomy_level(impact: int, irreversible: bool) -> str` qui retourne `"out-of-the-loop"`, `"on-the-loop"` ou `"in-the-loop"`.
2. Regle interne : calcule un score = `impact` (+2 si `irreversible`). Score <= 2 -> out-of-the-loop ; 3-4 -> on-the-loop ; >= 5 -> in-the-loop.
3. Teste sur 3 actions : `("classer un e-mail", impact=1, reversible)`, `("router un ticket", impact=3, reversible)`, `("emettre un virement", impact=4, irreversible)`.
4. Affiche pour chacune : nom de l'action -> niveau retourne.

### Criteres de reussite
- [ ] La fonction retourne exactement une des trois chaines attendues.
- [ ] Le classement d'e-mail tombe en `out-of-the-loop`.
- [ ] Le virement irreversible tombe en `in-the-loop`.
- [ ] Le routage de ticket tombe en `on-the-loop`.

---

## Exercice 2 : Un mini garde-fou d'action (action rail)

### Objectif
Comprendre qu'un garde-fou tranche `ALLOW / DENY / ESCALATE` au moment de l'action.

### Consigne
1. Ecris `refund_guardrail(amount: float) -> str` qui retourne `"ALLOW"`, `"ESCALATE"` ou `"DENY"`.
2. Regles : `amount <= 0` -> `"DENY"` (montant invalide) ; `0 < amount <= 100` -> `"ALLOW"` ; `100 < amount <= 2000` -> `"ESCALATE"` (validation humaine) ; `amount > 2000` -> `"DENY"` (hors perimetre).
3. Teste sur les montants : `-5`, `20`, `500`, `5000`.
4. Affiche `montant -> decision` pour chaque cas.

### Criteres de reussite
- [ ] Un montant negatif ou nul renvoie `DENY`.
- [ ] Un petit montant (20) renvoie `ALLOW`.
- [ ] Un montant moyen (500) renvoie `ESCALATE`.
- [ ] Un tres gros montant (5000) renvoie `DENY`.

---

## Exercice 3 : Kill-switch fail-safe

### Objectif
Implementer un kill-switch externe qui, en cas de doute, arrete l'agent (default deny).

### Consigne
1. Cree un `dict` `switch = {"refund-bot": "active"}`.
2. Ecris `can_run(switch: dict, agent_id: str) -> bool` : retourne `True` uniquement si l'etat de l'agent vaut exactement `"active"`. Pour tout autre etat **ou agent inconnu**, retourne `False`.
3. Teste : agent `"active"` -> doit pouvoir tourner ; passe l'etat a `"killed"` -> ne doit plus tourner ; interroge un agent absent du dict (`"ghost-bot"`) -> ne doit pas tourner.
4. Affiche les trois resultats.

### Criteres de reussite
- [ ] Un agent `"active"` peut tourner.
- [ ] Un agent `"killed"` ne peut pas tourner.
- [ ] Un agent **absent** du dict ne peut pas tourner (fail-safe).
- [ ] `can_run` ne leve jamais d'exception, meme pour un agent inconnu.
