# Exercices — Sandboxing & execution sure (J23)

> **Prerequis** : avoir lu `01-theory/23-sandboxing-execution-sure.md` et execute `02-code/23-sandboxing-execution-sure.py`.
> **Objectif global** : ancrer les concepts infra d'isolation en implementant des variantes du code de cours.

---

## Exercice 1 : Revocation de CapabilityToken

### Objectif

Comprendre que le moindre privilege ne suffit pas sans revocation : un token valide qui fuite devient une porte ouverte. Implémenter la revocation centralisee.

### Consigne

En partant de `02-code/23-sandboxing-execution-sure.py` :

1. Cree une classe `CapabilityStore` qui gere un dictionnaire `{token_id -> CapabilityToken}`.
2. Ajoute une methode `issue(tool, ttl, max_calls) -> CapabilityToken` qui cree et enregistre un token.
3. Ajoute une methode `revoke(token_id: str) -> bool` qui invalide immediatement le token (sans modifier `ttl`, par exemple via un ensemble `_revoked: set[str]`).
4. Ajoute une methode `is_valid(token: CapabilityToken) -> bool` qui verifie a la fois la validite interne du token ET qu'il n'est pas revoque.
5. Adapte `CapabilityRegistry.call` pour accepter un `store: CapabilityStore` optionnel et verifier `store.is_valid(token)` avant d'executer.
6. Teste le cycle complet :
   - Emet un token pour `"send_email"` (TTL 60s, max_calls 5).
   - Appel 1 : doit reussir.
   - Revoque le token.
   - Appel 2 : doit etre refuse avec un message mentionnant "revoked".
   - Affiche clairement les deux resultats.

### Criteres de reussite

- [ ] `CapabilityStore.issue` cree et stocke un token avec un `_token_id` unique
- [ ] `CapabilityStore.revoke` marque le token comme invalide sans modifier son TTL
- [ ] `CapabilityStore.is_valid` retourne `False` pour un token revoque, meme si le TTL n'est pas ecoule
- [ ] Le premier appel reussit, le second (post-revocation) est refuse
- [ ] Le message de refus mentionne "revoked" ou "revocation"

---

## Exercice 2 : Proxy avec journalisation structuree et rapport d'audit

### Objectif

Un proxy d'egress sans audit utilisable est inutile. Implémenter un rapport d'audit structure qui permet de detecter les tentatives d'exfiltration.

### Consigne

En partant de `NetworkAllowlistProxy` dans `02-code/23-sandboxing-execution-sure.py` :

1. Enrichis le log de chaque requete avec :
   - `session_id` : identifiant de session passe au constructeur
   - `seq` : numero de sequence (1, 2, 3...) dans la session
   - Un hash SHA-256 du champ `url` (simule l'empreinte de la requete)
2. Ajoute une methode `summary() -> dict` qui retourne :
   - `total_requests` : nombre total de requetes
   - `denied_requests` : nombre de requetes refusees
   - `unique_domains_denied` : liste des domaines distincts refuses
   - `denial_rate` : pourcentage de refus (0.0 a 1.0)
3. Ajoute une methode `suspicious_sessions(threshold: float = 0.5) -> bool` qui retourne `True` si le taux de refus depasse le seuil (session potentiellement compromise).
4. Teste avec au moins 8 requetes (melange de domaines autorises et non autorises), affiche le summary et la detection.

### Criteres de reussite

- [ ] Chaque entree de log a `session_id`, `seq`, et `url_hash`
- [ ] `summary()` retourne les 4 champs attendus avec les bonnes valeurs
- [ ] `suspicious_sessions(0.5)` retourne `True` si plus de 50% des requetes sont refusees
- [ ] `suspicious_sessions(0.5)` retourne `False` si le taux est inferieur au seuil
- [ ] La liste `unique_domains_denied` ne contient pas de doublons

---

## Exercice 3 : Sandbox multi-niveaux avec allowlist de chemins

### Objectif

Simuler l'isolation filesystem d'un sandbox (bubblewrap) en validant les chemins avant execution, et combiner avec le timeout pour un sandbox a deux couches.

### Consigne

1. Cree une fonction `validate_path(path: str, allowed_paths: list[str]) -> tuple[bool, str]` :
   - Retourne `(True, "")` si le chemin est dans un des dossiers autorises (verifie avec `os.path.commonpath` ou `Path.is_relative_to` pour eviter les traversals `../../`)
   - Retourne `(False, raison)` si le chemin sort de l'arborescence autorisee ou contient `..`
2. Cree une classe `FilesystemSandbox` avec :
   - `__init__(self, allowed_dirs: list[str])` : liste des repertoires autorises
   - `read(self, path: str) -> str` : lit le fichier si `validate_path` l'autorise, sinon retourne une erreur
   - `write(self, path: str, content: str) -> str` : ecrit si autorise, sinon erreur
3. Combine `FilesystemSandbox` et `run_in_subprocess` (de `02-code/23-sandboxing-execution-sure.py`) : cree une fonction `sandboxed_exec(code: str, work_dir: str, timeout: float)` qui :
   - Valide que `work_dir` est dans `/tmp`
   - Execute `code` via `run_in_subprocess` avec `timeout`
   - Retourne un dict avec `stdout`, `timed_out`, `path_validated`
4. Teste 3 scenarios :
   - Lecture d'un fichier dans `/tmp` : autorisee
   - Lecture de `/etc/passwd` : bloquee par `validate_path`
   - Execution d'un code qui tourne 5s avec timeout 1s : tue par timeout

### Criteres de reussite

- [ ] `validate_path` bloque `../../etc/passwd` et `/etc/passwd` si seul `/tmp` est autorise
- [ ] `validate_path` autorise `/tmp/work/output.txt` si `/tmp` est dans `allowed_dirs`
- [ ] `FilesystemSandbox.read` retourne un message d'erreur clair pour les chemins non autorises
- [ ] `sandboxed_exec` valide le `work_dir` avant tout
- [ ] Le scenario timeout retourne `timed_out=True` et `returncode != 0`
