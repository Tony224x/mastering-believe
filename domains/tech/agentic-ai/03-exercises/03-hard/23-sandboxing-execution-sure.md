# Exercices Hard — Sandboxing & execution sure (J23)

> **Note securite** : tout est **simule en memoire**. On ne lance jamais de commande hostile, pas de reseau, pas d'evasion filesystem reelle. Le sandbox = moteur de politique ; les attaques = actions classifiees/bloquees.

---

## Exercice 1 : Sandbox defense-en-profondeur + matrice d'ablation prouvant chaque couche

### Objectif

Cabler une defense en profondeur conforme aux 7 couches du cours (section 7) — ici 4 couches techniques : **fs-jail**, **egress control**, **resource caps**, **capability drop** — en un seul `LayeredSandbox`, puis prouver par une **matrice d'ablation** que **chaque couche est necessaire** : desactiver une couche fait passer exactement l'attaque que cette couche etait censee arreter (et seulement celle-la).

### Consigne

En reutilisant les briques du cours (`02-code/23-sandboxing-execution-sure.py`) et des exercices medium :

1. Construis un `LayeredSandbox` avec 4 couches activables/desactivables via un dict `layers = {"fs_jail": True, "egress": True, "resource_caps": True, "capability_drop": True}` :
   - **fs_jail** : valide qu'un chemin demande reste sous `allowed_dirs` (bloque `../` traversal et chemins absolus hors jail).
   - **egress** : valide la destination reseau contre une allowlist (reutilise la logique wildcard).
   - **resource_caps** : refuse une action dont le `cpu` ou `wall` demande depasse le cap.
   - **capability_drop** : l'action demandee doit etre dans l'ensemble des capacites conservees (ex: `ptrace`, `mount` ont ete *drop*).
2. Implemente `evaluate(self, action: dict) -> dict` ou `action` decrit une tentative (`{"type": "fs"|"net"|"compute"|"syscall", ...}`) et retourne `{"allowed": bool, "blocked_by": <layer|None>}`. Une couche desactivee est **transparente** (ne bloque rien).
3. Definis 4 **attaques canoniques**, une par couche :
   - `attack_fs` : `{"type": "fs", "path": "../../etc/passwd"}` (path traversal).
   - `attack_net` : `{"type": "net", "domain": "evil-c2.example.com"}` (exfiltration).
   - `attack_cpu` : `{"type": "compute", "cpu": 10_000}` (epuisement CPU).
   - `attack_syscall` : `{"type": "syscall", "name": "ptrace"}` (escalade via syscall dangereux).
4. Construis une **matrice d'ablation** : pour chaque couche L, desactive **uniquement** L, rejoue les 4 attaques, et verifie que :
   - l'attaque correspondant a L **passe** (la couche manquante ne la bloque plus),
   - les 3 autres attaques restent **bloquees** (les autres couches tiennent).
   Avec **toutes** les couches actives, les 4 attaques doivent etre bloquees.
5. Produis un `ablation_report` : un dict `{layer_disabled: {attack: allowed_bool}}` et imprime-le lisiblement.

### Criteres de reussite

- [ ] `LayeredSandbox.evaluate` route chaque type d'action vers sa couche et retourne `blocked_by`
- [ ] Avec les 4 couches actives, les 4 attaques canoniques sont **toutes** bloquees
- [ ] Desactiver une couche fait passer **exactement** l'attaque associee (verifie par assertion)
- [ ] Les 3 autres attaques restent bloquees quand une seule couche est desactivee (independance des couches)
- [ ] `ablation_report` est une matrice complete couche x attaque, imprimee

---

## Exercice 2 : Suite de tests adversariale (sandbox-escape) + rapport de robustesse

### Objectif

Ecrire un **harnais red-team** qui rejoue une batterie de tentatives d'evasion de sandbox (cf. limites listees dans le cours : egress sections 6.2, fs traversal, fork-bomb, env leak, DNS exfil) — **toutes simulees** — contre un `SandboxGuard`, et qui produit un **rapport de robustesse** ou le taux d'evasion reussie doit etre 0%.

### Consigne

1. Construis un `SandboxGuard` regroupant plusieurs verifications **pures** (aucune execution reelle) :
   - `check_path(path)` : bloque `..`, chemins absolus hors `/tmp` jail, et symlink simules (un dict `{"path": "...", "symlink_target": "/etc/shadow"}` dont la cible sort du jail).
   - `check_fork(requested_procs)` : bloque si `requested_procs > max_procs` (fork-bomb).
   - `check_env(env: dict)` : bloque/retire les cles secretes (`ANTHROPIC_API_KEY`, `AWS_*`, `DATABASE_URL`, `*_TOKEN`, `*_SECRET`) — un agent ne doit pas voir l'environnement de l'hote.
   - `check_egress(url)` : bloque les domaines hors allowlist ET les patterns de DNS-exfil (`<longue_chaine>.attacker.com`, sous-domaine encode).
2. Definis une suite d'au moins **6 attaques** sous forme de dataclass `EscapeAttempt(name, kind, payload, expected="blocked")`, couvrant : path traversal `../`, symlink escape, fork-bomb, env leak, DNS exfil, et un cas **benin** (`expected="allowed"`) pour verifier que le guard ne sur-bloque pas (pas de faux positif).
3. Implemente `run_red_team(guard, attempts) -> dict` qui passe chaque tentative dans le bon `check_*`, compare au resultat attendu, et produit :
   - `{"total", "blocked", "allowed", "escapes": [...noms d'attaques hostiles passees...], "false_positives": [...benins bloques...], "escape_rate": float}`.
4. **Assertions de robustesse** : `escape_rate == 0.0` (aucune attaque hostile ne passe), `false_positives == []` (le cas benin passe). Imprime le rapport.

### Criteres de reussite

- [ ] Au moins 6 tentatives couvrant >= 4 vecteurs distincts (traversal, symlink, fork-bomb, env leak, DNS exfil)
- [ ] Chaque verification est **pure/simulee** : aucune commande, aucun fichier, aucun socket reel n'est touche
- [ ] Toutes les tentatives hostiles sont bloquees → `escape_rate == 0.0`
- [ ] Le cas benin est autorise → `false_positives == []` (pas de sur-blocage)
- [ ] `check_env` retire bien les cles secretes de l'environnement expose
- [ ] Le rapport de robustesse est imprime avec total / blocked / escape_rate
