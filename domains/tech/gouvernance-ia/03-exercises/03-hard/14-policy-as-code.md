# J14 — Exercice difficile : MCP permission gate + test de politique + drift

## Objectif

Pousser le moteur jusqu'a un dispositif realiste : un **gate de permission facon MCP** (consentement explicite avant `tools/call`), une **suite de tests de politique** (la politique est du code, donc on la teste), et une detection de **drift** entre deux versions de politique. C'est la mecanique qu'on assemblera dans le capstone.

## Consigne

1. Pars du PDP/PEP de l'exercice moyen (regles + precedence de surete + journal d'audit).
2. **MCP permission gate** — ecris une classe `MCPPermissionGate` qui enveloppe le PEP et expose `call_tool(action, agent, ctx, run_tool, user_consented=False, human_approves=False)` :
   - elle maintient un ensemble `consent_required` d'outils sensibles (ex. `{"issue_refund", "export_data"}`) ;
   - si l'outil appele est sensible **et** `user_consented` est faux → renvoie un verdict `deny` avec la regle `"mcp_consent"`, **sans** jamais appeler `run_tool`, et journalise quand meme la tentative ;
   - sinon → delegue au PEP. Documente en commentaire que cela modelise la section *Security & Trust* de la spec MCP (consentement explicite, permissions par outil).
3. **Suite de tests de politique** — ecris une fonction `run_policy_tests(pdp)` qui prend une liste de cas `(action, agent, ctx, verdict_attendu)` et verifie que `pdp.evaluate(...)` renvoie bien le verdict attendu pour chacun. Elle renvoie le nombre de cas passes/echoues et imprime tout cas qui echoue (action, attendu, obtenu). Fournis au moins **6 cas** couvrant `allow`/`deny`/`oblige`.
4. **Detection de drift** — ecris une fonction `detect_drift(pdp_old, pdp_new, actions)` qui, pour une meme liste d'actions/agents, signale toute action dont le **verdict change** entre l'ancienne et la nouvelle politique. Construis deux PDP qui different par une regle (par ex. abaisser le plafond de remboursement de 1000 a 500) et montre au moins un verdict qui bascule.
5. Smoke test final : execute le gate sur quelques scenarios (dont un refus pour absence de consentement), lance la suite de tests, puis affiche le rapport de drift.

## Criteres de reussite

- [ ] Un appel a un outil sensible **sans** consentement est refuse au gate (`deny`, regle `mcp_consent`) et `run_tool` n'est **jamais** invoque.
- [ ] Le gate delegue correctement au PEP quand le consentement est donne (la politique s'applique ensuite normalement).
- [ ] `run_policy_tests` detecte un verdict attendu errone et imprime le detail du cas qui echoue (verifie-le en introduisant volontairement un cas faux, puis corrige-le).
- [ ] `detect_drift` identifie au moins une action dont le verdict change entre deux versions de politique et l'affiche.
- [ ] Chaque tentative (y compris les refus au gate) figure dans le journal d'audit.
- [ ] Le script s'execute sans erreur (`python`, stdlib uniquement) et passe `python -m py_compile`.
