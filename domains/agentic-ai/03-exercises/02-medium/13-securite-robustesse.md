# Exercices Medium — Securite & Robustesse (J13)

---

## Exercice 1 : Detection d'injection par score pondere (au-dela du binaire)

### Objectif
Remplacer la detection binaire pattern-matche par un scoring pondere multi-signaux avec 3 zones de decision (pass / flag / block), et mesurer le taux de faux positifs sur un corpus sain.

### Consigne
En partant de `InputGuardrail` de `02-code/13-securite-robustesse.py` :

1. Cree un `ScoredInjectionDetector` qui combine plusieurs signaux, chacun avec un poids :
   - Patterns directs (`ignore previous instructions`, `you are now`, `system prompt`) : +0.5 chacun
   - Marqueurs de roleplay (`pretend you are`, `act as`, `jailbreak`) : +0.3
   - **Contenu encode** : presence d'une longue chaine base64-like (>= 20 chars de `[A-Za-z0-9+/=]` d'affilee avec decodage base64 qui reussit) : +0.4
   - **Texte invisible/homoglyphes** : caracteres hors ASCII imprimable dans un texte par ailleurs anglais (zero-width space `​`, homoglyphes cyrilliques) : +0.4
   - Imperatifs suspects diriges vers l'agent (`do not tell`, `send ... to`, `delete`) : +0.2
2. Decision par seuils : score < 0.3 -> `PASS` ; 0.3 <= score < 0.7 -> `FLAG` (on laisse passer mais on journalise et on marque le contenu untrusted) ; >= 0.7 -> `BLOCK`
3. Le verdict retourne le detail : signaux declenches avec leur contribution, score total, decision
4. Construis 2 corpus de test :
   - 8 textes malveillants varies (direct, roleplay, base64, zero-width, combinaisons)
   - 10 textes sains dont des pieges a faux positifs ("Can you delete the duplicate line in my code?", "explain what a system prompt is")
5. Mesure et affiche : detection rate sur le corpus malveillant, taux de faux positifs BLOCK sur le corpus sain (exige 0), taux de FLAG sur le sain (tolere)
6. Ajuste les poids si necessaire pour atteindre : 100% des malveillants en FLAG ou BLOCK, 0% des sains en BLOCK

### Criteres de reussite
- [ ] Les 5 familles de signaux sont implementees et testees individuellement
- [ ] La detection base64 valide par decodage reel (pas juste la regex)
- [ ] Les zero-width/homoglyphes sont detectes
- [ ] Aucun texte sain n'est BLOCK (faux positifs bloquants = 0)
- [ ] Tous les textes malveillants sont au minimum FLAG
- [ ] Le rapport detaille les signaux par texte

---

## Exercice 2 : Pipeline PII — detection, redaction typee et vault reversible

### Objectif
Proteger les donnees personnelles dans les deux sens : redacter les PII avant l'appel LLM, et pouvoir restaurer les valeurs pour les destinataires autorises via un vault de correspondances.

### Consigne
1. Cree un `PIIDetector.detect(text) -> list[PIIMatch]` avec des regex pour :
   - Email, telephone FR (+33/0 + 9 chiffres avec separateurs optionnels), IBAN-like (`FR76` + 20+ chars alphanum), cle API (`sk-`, `key-`, `Bearer ` suivis de 16+ chars), nom de personne precede d'un titre (`Mr/Mrs/Dr <Capitalized>`)
   - Chaque `PIIMatch` : type, valeur, span (start, end)
2. Cree un `PIIVault` :
   - `redact(text) -> tuple[str, str]` : remplace chaque PII par un placeholder type et numerote `<EMAIL_1>`, `<PHONE_1>`, `<EMAIL_2>`... et retourne `(texte_redacte, redaction_id)`
   - La correspondance placeholder -> valeur est stockee dans le vault sous le `redaction_id`
   - **Stabilite** : la meme valeur dans le meme texte recoit le meme placeholder
   - `restore(text, redaction_id, authorized: bool) -> str` : si autorise, re-substitue les valeurs ; sinon leve `PermissionError`
3. Integre dans un flux agent simule : message user avec 2 emails + 1 telephone + 1 cle API -> redaction -> "appel LLM" mock (qui cite `<EMAIL_1>` dans sa reponse) -> restauration autorisee pour l'affichage final a l'utilisateur
4. Verifie qu'AUCUNE PII brute n'atteint le mock LLM (assert sur le prompt recu par le mock)
5. La cle API est un cas special : `restore` ne la restaure JAMAIS (politique : secrets non restaurables), elle reste `<API_KEY_1>` meme autorise
6. Teste aussi le round-trip sur un texte sans PII (passthrough sans vault)

### Criteres de reussite
- [ ] Les 5 types de PII sont detectes avec leurs spans corrects
- [ ] Les placeholders sont types, numerotes et stables pour les valeurs repetees
- [ ] Le mock LLM ne recoit jamais une PII brute (assert)
- [ ] La restauration fonctionne pour les autorises et leve PermissionError sinon
- [ ] Les cles API ne sont jamais restaurees, meme avec autorisation
- [ ] Le texte sans PII traverse le pipeline inchange

---

## Exercice 3 : Workflow d'approbation HITL avec politiques par niveau de risque

### Objectif
Industrialiser le human-in-the-loop : chaque tool a un niveau de risque qui determine sa politique d'approbation, avec file d'attente, timeout par defaut-refus et journal des decisions.

### Consigne
1. Etends `ToolSpec` avec `risk_level: str` (`"low"`, `"medium"`, `"high"`) et definis les politiques :
   - `low` (ex: `search_docs`) : auto-approve, journalise seulement
   - `medium` (ex: `send_email`) : approbation requise ; si pas de decision avant `timeout_s` (simule par horloge injectable), **refus par defaut** (`deny_by_timeout`)
   - `high` (ex: `delete_record`, `run_sql`) : approbation requise PLUS justification ecrite de l'approbateur ; approbation sans justification -> refus
2. Cree une `ApprovalQueue` :
   - `submit(request) -> request_id` : empile `{tool, args, user_id, risk, submitted_at}`
   - `pending() -> list` ; `decide(request_id, approved, approver, justification="")`
   - `resolve(request_id, now) -> Decision` : applique la politique (timeout inclus)
3. Cree un `PolicyHITLGate.call(name, args, user_id, now)` qui orchestre : consulte la politique, passe par la queue si necessaire, execute le tool seulement si la decision finale est APPROVE
4. Journal des decisions : chaque resolution ecrit une entree (tool, risk, decision, approver, justification, latence de decision simulee) ; methode `decisions_report()`
5. Scenario de demo avec une horloge simulee :
   - `search_docs` -> auto-approve immediat
   - `send_email` approuve par un humain a t+10s -> execute
   - `send_email` sans decision a t+timeout -> refuse par timeout, jamais execute
   - `delete_record` approuve AVEC justification -> execute
   - `delete_record` approuve SANS justification -> refuse avec la raison `missing_justification`
6. Asserts : les tools refuses n'ont jamais ete executes (compteur d'executions par tool), le rapport contient les 5 decisions avec les bonnes raisons

### Criteres de reussite
- [ ] Les 3 politiques de risque sont appliquees correctement
- [ ] Le timeout refuse par defaut sans executer le tool
- [ ] L'exigence de justification est verifiee pour le niveau high
- [ ] La queue expose les demandes en attente et leur resolution
- [ ] Le journal des decisions est complet et exact (asserts)
