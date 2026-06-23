# J14 — Exercice moyen : un PDP/PEP avec precedence de surete

## Objectif

Assembler plusieurs regles en un **moteur de decision (PDP)** qui les combine avec la precedence de surete **deny > oblige > allow**, puis brancher un **point d'enforcement (PEP)** qui execute (ou non) l'outil selon le verdict — et qui journalise chaque decision.

## Consigne

1. Reprends (ou recrée) les structures `Agent` et `Action` de l'exercice facile, plus une structure `Decision` portant au minimum `verdict` (`"allow"`/`"deny"`/`"oblige"`), `rule` (nom de la regle) et `reason`.
2. Ecris **au moins trois** regles, chacune une fonction `(action, agent, ctx) -> Decision | None` qui renvoie `None` quand elle ne s'applique pas :
   - `rule_scope` (deny si scope manquant) ;
   - `rule_refund_cap` (oblige si remboursement > 1000) ;
   - `rule_high_risk_irreversible` (oblige si `agent.risk_tier == "high"` et l'outil figure dans `ctx["irreversible_tools"]`).
3. Ecris une classe `PolicyDecisionPoint` avec une methode `evaluate(action, agent, ctx)` qui :
   - collecte toutes les decisions des regles qui se sont declenchees ;
   - si aucune ne s'est declenchee, renvoie un `allow` par defaut ;
   - sinon, applique la **precedence de surete** : `deny` l'emporte sur `oblige`, qui l'emporte sur `allow`. (Astuce : ordonne tes verdicts et prends le plus severe.)
4. Ecris une classe `PolicyEnforcementPoint` avec `enforce(action, agent, ctx, run_tool, human_approves=False)` qui :
   - appelle le PDP ;
   - si `allow` → execute `run_tool(action)` ;
   - si `oblige` → execute **seulement** si `human_approves` est vrai ;
   - si `deny` → n'execute jamais ;
   - **journalise** chaque tentative (agent, outil, verdict, regle, executed) dans une liste append-only.
5. Ecris un petit smoke test (au moins 5 actions) couvrant les trois verdicts et les deux cas d'`oblige` (avec et sans approbation humaine), puis imprime le journal d'audit final.

## Criteres de reussite

- [ ] Chaque regle renvoie `None` quand elle ne s'applique pas (et n'a aucun effet de bord).
- [ ] `PolicyDecisionPoint.evaluate` applique correctement la precedence **deny > oblige > allow** quand plusieurs regles se declenchent.
- [ ] Un `oblige` non approuve par un humain n'execute **pas** l'outil ; un `oblige` approuve l'execute.
- [ ] Chaque tentative apparait dans le journal d'audit avec son verdict et le nom de la regle declenchante.
- [ ] Le smoke test couvre les trois verdicts et imprime un journal lisible.
- [ ] Le script s'execute sans erreur (`python`, stdlib uniquement) et passe `python -m py_compile`.
