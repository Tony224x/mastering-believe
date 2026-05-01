# Exercices Faciles — Securite & Robustesse (J13)

---

## Exercice 1 : Detecter les instructions indirectes dans les tool outputs

### Objectif
Comprendre qu'un tool result peut contenir des injections que le LLM suivra s'il n'est pas protege.

### Consigne
En partant de `02-code/13-securite-robustesse.py` :

1. Cree une classe `ToolOutputGuardrail` qui scanne le resultat d'un tool avant qu'il ne soit mis dans le contexte du LLM
2. Reutilise les `INJECTION_PATTERNS` et ajoutes-en 3 nouveaux orientes "contenu externe" :
   - `ignore the user`
   - `do not tell the user`
   - `send .* to .*@`
3. Si une injection est detectee, la methode `sanitize(output: str) -> str` doit :
   - Retourner le texte **entoure** de balises `[UNTRUSTED CONTENT]...[/UNTRUSTED CONTENT]`
   - Prefixer avec un warning `[GUARDRAIL: potential injection detected]`
   - Ne PAS supprimer le texte (l'agent peut avoir besoin de le lire)
4. Teste avec 3 tool outputs dont 2 contiennent des injections
5. Affiche le texte avant et apres sanitization

### Criteres de reussite
- [ ] `ToolOutputGuardrail.sanitize` detecte les 3 nouveaux patterns
- [ ] Le texte injecte est encapsule dans des balises `[UNTRUSTED CONTENT]`
- [ ] Le prefix warning est present
- [ ] Les outputs "propres" ne sont pas modifies
- [ ] Le test montre clairement les 3 cas

---

## Exercice 2 : Whitelist dynamique par user role

### Objectif
Implementer un controle d'acces aux tools base sur le role de l'utilisateur.

### Consigne
1. Etends `SandboxedRegistry` pour supporter des **roles** par tool :
   - Ajoute un champ `allowed_roles: list[str]` a `ToolSpec` (par defaut `["user"]`)
   - Modifie `call` pour accepter un parametre `user_role: str`
   - Si `user_role` n'est pas dans `allowed_roles`, retourne `{"error": "forbidden: role X cannot call tool Y"}`
2. Definis 3 roles : `"guest"`, `"user"`, `"admin"`
3. Enregistre 3 tools :
   - `search_docs` : roles `["guest", "user", "admin"]`
   - `send_email` : roles `["user", "admin"]`
   - `delete_record` : roles `["admin"]`
4. Teste la matrice 3x3 (3 users x 3 tools) et affiche les verdicts
5. Verifie que l'escalade de privileges est bloquee (un guest qui tente delete_record)

### Criteres de reussite
- [ ] `ToolSpec` a un champ `allowed_roles`
- [ ] `call` verifie le role avant d'executer
- [ ] Les 9 combinaisons sont testees
- [ ] Les 3 verdicts attendus (ok, forbidden, ok) sont corrects
- [ ] Le message d'erreur mentionne le role et le tool

---

## Exercice 3 : Audit log append-only

### Objectif
Implementer un journal d'audit qui ne peut pas etre altere une fois ecrit.

### Consigne
1. Cree une classe `AuditLog` :
   - `__init__(self, filepath: str)` : ouvre un fichier en mode append
   - `log(self, event: dict)` : ajoute un event JSON a la fin, avec timestamp et hash de la ligne precedente
   - Chaque ligne a un champ `prev_hash` qui est le SHA-256 de la ligne precedente (hash chaining)
   - La premiere ligne a `prev_hash = "0" * 64`
2. Cree une methode `verify(self) -> bool` qui relit le fichier et verifie que chaque `prev_hash` correspond bien au hash de la ligne precedente
3. Ajoute une methode `detect_tampering(self) -> list[int]` qui retourne les numeros des lignes modifiees
4. Teste :
   - Log 5 events -> verify() doit retourner True
   - Modifie manuellement une ligne dans le fichier -> verify() doit retourner False, detect_tampering() doit lister la ligne
5. Utilise `hashlib.sha256` et `json.dumps` (avec `sort_keys=True` pour la determinisme)

### Criteres de reussite
- [ ] `log` ecrit des lignes avec `prev_hash` correct
- [ ] `verify` retourne True sur un fichier non-altere
- [ ] `verify` retourne False sur un fichier altere
- [ ] `detect_tampering` identifie la ligne altere
- [ ] Le cycle complet fonctionne sur un fichier temporaire
