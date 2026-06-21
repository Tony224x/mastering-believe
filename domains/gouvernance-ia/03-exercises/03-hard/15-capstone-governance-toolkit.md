# J15 — Exercice difficile : le toolkit complet, jusqu'au verdict board-ready

## Objectif

Assembler le toolkit de gouvernance de bout en bout : `ingest → enforce → log → score → map → report`. Le livrable final est un rapport board-ready (markdown **et** JSON) qui se termine sur un **verdict actionnable**, ou la conformite est **gatee par des preuves live** (integrite de l'audit, presence d'un owner nomme). C'est le capstone reduit a l'essentiel.

## Consigne

1. Pars du moteur de l'exercice moyen (`GovernedAgent`, `ingest`, policy engine, `AuditTrail`).
2. **Consentement MCP** — ajoute au moteur un ensemble `consent_required` d'outils sensibles. Si une action porte sur un tel outil **sans** `user_consented=True` → `DENY` (regle `mcp_consent`), sans jamais executer l'outil, mais journalise quand meme. (Commente que cela modelise la section *Security & Trust* de la spec MCP.)
3. **SCORE** — ecris `score_agent(agent)` : `criticality = likelihood × impact` (echelles 1..5 derivees du `risk_tier`), **modulee** par le contexte agentique — `handles_irreversible` → impact +1, `autonomous` → likelihood +1 (cape a 5). Classe en `TREAT` (≥12) / `MONITOR` (6..11) / `ACCEPT` (<6) et conserve une `rationale` expliquant les modulateurs.
4. **MAP** — ecris un mini-crosswalk : un catalogue d'au moins 6 `Requirement(framework, ref, label, mandatory)` couvrant EU AI Act (mandatory), NIST AI RMF et ISO/IEC 42001 (volontaires). Definis des `Control` **gates sur preuve** : le controle « audit trail » ne compte que si `audit.verify()` passe ; le controle « human oversight » (EU AI Act Art. 14) ne compte que si **aucun** agent n'est orphelin. Calcule la couverture par referentiel et la liste des **trous obligatoires**.
5. **REPORT** — ecris `render_markdown(result)` et `render_json(result)`. Le markdown doit afficher couverture, orphelins, integrite de l'audit, posture de risque, enforcement (attempts/blocked), couverture de conformite, et **se terminer sur un VERDICT** : si l'audit est casse → halte ; sinon s'il existe un trou obligatoire → remediation requise ; sinon s'il reste des orphelins → assigner des owners ; sinon → cleared to scale.
6. Smoke test final : lance le pipeline complet sur une flotte d'exemple **incluant un orphelin**, imprime le rapport markdown, et verifie que (a) le verdict signale bien le trou obligatoire Art. 14 du a l'orphelin, et (b) une edition silencieuse d'une entree d'audit fait basculer `verify()` en `(False, index)`.

## Criteres de reussite

- [ ] Un agent `tier=high`, `autonomous=True`, `handles_irreversible=True` obtient `criticality=25` et `decision=TREAT`, avec les deux modulateurs dans sa `rationale`.
- [ ] Un appel a un outil sensible sans consentement est refuse (`mcp_consent`) et journalise, sans executer l'outil.
- [ ] Le crosswalk laisse EU AI Act Art. 14 **non couvert** tant qu'il reste un orphelin, et ce trou apparait comme **MANDATORY**.
- [ ] Le verdict final reflete l'etat reel : trou obligatoire → « remediation required » ; aucun trou + aucun orphelin → « cleared to scale ».
- [ ] `render_json` produit un JSON valide (parsable par `json.loads`) contenant couverture, risque et trous obligatoires.
- [ ] Une edition silencieuse d'une entree d'audit fait passer `verify()` a `(False, index_exact)`.
- [ ] Le script s'execute sans erreur (`python`, stdlib uniquement) et passe `python -m py_compile`.
