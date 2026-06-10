# Exercices Medium — Multi-Agent Patterns (J9)

---

## Exercice 1 : Supervisor avec re-planification sur feedback du reviewer

### Objectif
Rendre le supervisor adaptatif : quand le reviewer signale un probleme, le supervisor doit inserer une etape de correction au lieu de terminer betement son plan fixe.

### Consigne
En partant de `02-code/09-multi-agent-patterns.py` :

1. Modifie `MockLLM._reviewer` pour retourner un verdict structure : `"VERDICT: REJECT | reason: missing error handling"` au premier passage sur un code donne, puis `"VERDICT: APPROVE"` si le code contient `try:` (le coder corrige est passe dans le contexte)
2. Ajoute a `MockLLM._coder` un mode correction : si le contexte contient un feedback de reject, il retourne une version corrigee du code (avec `try/except`)
3. Modifie `SupervisorPattern.run` :
   - Apres chaque etape `reviewer`, parse le verdict
   - Si `REJECT` : insere dynamiquement 2 etapes dans le plan : `coder` (avec le feedback en contexte) puis `reviewer` (re-verification)
   - Limite : maximum 2 cycles de correction par run -> au-dela, termine avec le statut `"escalated_to_human"`
4. La trace doit montrer le plan initial, chaque insertion dynamique, et le statut final
5. Teste 2 scenarios : (a) le coder corrige et le 2e review approuve ; (b) un reviewer mock qui rejette toujours -> escalade apres 2 cycles

### Criteres de reussite
- [ ] Le verdict du reviewer est parse de maniere fiable (format structure)
- [ ] Le plan s'allonge dynamiquement apres un REJECT (visible dans la trace)
- [ ] Le coder recoit le feedback du reviewer dans son contexte
- [ ] Le scenario (a) se termine avec un code approuve contenant la correction
- [ ] Le scenario (b) s'arrete a 2 cycles avec le statut `escalated_to_human`

---

## Exercice 2 : Hierarchie a 2 niveaux — equipes et sous-superviseurs

### Objectif
Implementer le pattern hierarchique : un superviseur racine delegue a des equipes, chaque equipe ayant son propre sous-superviseur et ses workers.

### Consigne
1. Cree une classe `Team` : un sous-superviseur + une liste de workers + un domaine (`"research"` ou `"engineering"`)
   - `Team.run(subtask) -> str` : le sous-superviseur planifie 2 etapes max avec SES workers et retourne une synthese d'equipe
2. Cree un `TopSupervisor` qui :
   - Decoupe la tache globale en sous-taches par domaine (mock : "Write a documented function that parses logs" -> research: "find best practices for log parsing", engineering: "implement + review the function")
   - Delegue chaque sous-tache a l'equipe du bon domaine
   - Assemble les syntheses d'equipes en un livrable final
3. Equipes de la demo :
   - Team Research : workers `researcher`, `writer` (resume des findings)
   - Team Engineering : workers `coder`, `reviewer`
4. La trace doit etre indentee par niveau pour montrer la hierarchie :
   ```
   [TOP] plan: 2 subtasks
     [research] subsupervisor plan: researcher -> writer
       [researcher] ...
   ```
5. Ajoute un comptage des appels LLM par niveau (top / sous-superviseurs / workers) et affiche le total
6. Asserts : chaque equipe ne voit QUE sa sous-tache (pas la tache globale ni l'autre equipe), et le livrable final contient la contribution des 2 equipes

### Criteres de reussite
- [ ] La structure a 2 niveaux est explicite dans le code (TopSupervisor -> Team -> workers)
- [ ] Le decoupage par domaine route les sous-taches vers la bonne equipe
- [ ] La trace indentee reflete la hierarchie reelle des appels
- [ ] L'isolation des equipes est verifiee par un assert
- [ ] Le comptage des appels par niveau est exact

---

## Exercice 3 : Debat multi-rounds avec convergence

### Objectif
Etendre le debate pattern a plusieurs rounds : les agents voient les arguments des autres et revisent leur position jusqu'a converger (ou jusqu'a la limite de rounds).

### Consigne
1. Cree un `MultiRoundDebate(llm, agents, max_rounds=3, epsilon=1.0)` :
   - Round 1 : chaque agent donne un score 0-10 + un argument sur la proposition
   - Rounds suivants : chaque agent recoit les scores + arguments des AUTRES agents du round precedent et revise son score (mock deterministe : chaque agent bouge de 30% vers la moyenne des autres, arrondi a 1 decimale)
   - **Convergence** : si `max(scores) - min(scores) <= epsilon`, on arrete avant `max_rounds`
2. Le moderateur produit le verdict final sur la moyenne du dernier round : `accept` (>= 6), `reject` (< 4), `no_consensus` sinon — et redige une synthese qui cite le score initial et final de chaque agent
3. Demo avec 3 agents aux positions initiales eloignees : researcher=8, coder=9, reviewer=3 :
   - Affiche chaque round : scores, ecart max, statut de convergence
   - La demo doit converger en 2-3 rounds (avant d'atteindre la limite)
4. Teste aussi : (a) accord immediat (scores initiaux 7, 7, 8 -> arret au round 1), (b) divergence persistante avec `epsilon=0.1` -> arret a max_rounds avec `no_consensus` si la moyenne est entre 4 et 6
5. Retourne un objet `DebateResult` : `rounds: list[dict]`, `converged: bool`, `final_scores`, `verdict`, `summary`

### Criteres de reussite
- [ ] La revision des scores suit la regle des 30% vers la moyenne des autres (deterministe)
- [ ] La convergence arrete le debat des que l'ecart passe sous epsilon
- [ ] Le cas accord immediat ne fait qu'un round
- [ ] Le cas divergence s'arrete proprement a max_rounds avec le bon verdict
- [ ] `DebateResult` contient l'historique complet des rounds pour audit
