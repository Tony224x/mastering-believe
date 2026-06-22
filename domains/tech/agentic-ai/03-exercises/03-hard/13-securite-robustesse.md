# Exercices Hard — Securite & Robustesse (J13)

---

## Exercice 1 : Defense en profondeur orchestree + harnais de red-team OWASP

### Objectif
Cabler les 5 couches de la defense en profondeur (section 3 du cours) en un pipeline unique, puis le valider avec un harnais de red-team qui mappe chaque attaque sur l'OWASP LLM Top 10 (section 10) et mesure le taux d'attaques bloquees.

### Consigne
Construis un `DefenseInDepthPipeline` qui empile, dans l'ordre, les 5 couches :

1. **Layer 1 — Input guardrails** : injection patterns, longueur, rate limit (reutilise le code du J13)
2. **Layer 2 — Trust boundaries** : marque le contenu issu de tools/web comme non-fiable, le scanne pour injection indirecte
3. **Layer 3 — Tool guardrails** : whitelist + validation d'arguments + sandbox + HITL sur les actions dangereuses
4. **Layer 4 — Output guardrails** : canary token (fuite de system prompt), PII, schema
5. **Layer 5 — Monitoring + audit** : log de chaque decision, kill switch si trop d'attaques d'un meme user

Puis construis un **`RedTeamHarness`** :
- Une suite d'au moins 8 attaques, chacune annotee avec son `owasp_id` (ex: `LLM01` prompt injection, `LLM02` insecure output, `LLM06` sensitive info disclosure, `LLM08` excessive agency...)
- Pour chaque attaque : la passer dans le pipeline et verifier qu'elle est bloquee a la BONNE couche
- Produire un rapport : `{total, blocked, blocked_by_layer, owasp_coverage, attack_success_rate}`
- L'`attack_success_rate` doit etre 0% (toutes bloquees) ; si une attaque passe → le rapport l'identifie

Exemples d'attaques a inclure : injection directe, injection indirecte via tool output, tentative d'appel d'un tool non whiteliste, exfiltration de canary, exces d'agence (enchainer des actions destructives), DoS par input geant, PII leak en sortie, jailbreak roleplay.

### Criteres de reussite
- [ ] Les 5 couches sont chainees et chacune intercepte une classe d'attaque differente
- [ ] Le harnais couvre >= 4 categories OWASP LLM distinctes
- [ ] Chaque attaque est bloquee a la couche attendue (verifie par assertion)
- [ ] Le kill switch s'active apres N attaques du meme user
- [ ] Le rapport donne un `attack_success_rate` de 0% et la `owasp_coverage`
- [ ] L'audit log trace chaque blocage avec la couche et la raison

---

## Exercice 2 : Budget d'agence (action budget) + confirmation par tiers de risque + rollback

### Objectif
Limiter l'"excessive agency" (OWASP LLM08) : un agent ne doit pas pouvoir enchainer un nombre illimite d'actions a effet de bord. Implementer un budget d'actions par tiers de risque, une confirmation graduee (auto / HITL / double-approbation), et un mecanisme de rollback transactionnel.

### Consigne
Construis un `AgencyController` qui gouverne l'execution des tools a effet de bord :

1. **Tiers de risque** par tool : `read` (gratuit), `write` (cout 1, auto sous budget), `destructive` (cout 5, HITL obligatoire), `irreversible` (cout 10, double-approbation : 2 approbateurs distincts)
2. **Budget d'agence par run** : un budget total (ex: 15 points). Chaque action consomme son cout. Quand le budget est epuise → toute nouvelle action a effet de bord est refusee (l'agent peut encore lire)
3. **Confirmation graduee** :
   - `write` sous budget → execute directement
   - `destructive` → callback HITL simple ; refus → action annulee
   - `irreversible` → exige 2 approbations de 2 approbateurs differents ; une seule → refus
4. **Transaction + rollback** : chaque action `write`/`destructive` enregistre une operation inverse (`undo`). Si une action ULTERIEURE echoue dans la meme transaction, on rollback toutes les actions precedentes dans l'ordre inverse (compensating transactions)
5. Teste un scenario : l'agent fait 2 writes (OK), tente un destructive (HITL yes), tente un irreversible avec 1 seule approbation (refus), epuise le budget, puis une action echoue → rollback complet verifie

### Criteres de reussite
- [ ] Chaque tool a un tier de risque et un cout d'agence
- [ ] Le budget d'agence est decremente et bloque les actions une fois epuise
- [ ] `destructive` exige une approbation HITL, `irreversible` en exige deux distinctes
- [ ] Une seule approbation sur un `irreversible` → action refusee
- [ ] Le rollback rejoue les undo dans l'ordre inverse quand une action echoue
- [ ] Apres rollback, l'etat est revenu a celui d'avant la transaction (verifie par assertion)
