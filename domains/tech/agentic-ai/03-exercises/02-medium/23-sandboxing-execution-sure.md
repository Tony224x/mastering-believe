# Exercices Medium — Sandboxing & execution sure (J23)

> **Prerequis** : avoir lu `01-theory/23-sandboxing-execution-sure.md` et execute `02-code/23-sandboxing-execution-sure.py`.
> **Note securite** : on ne lance JAMAIS de commande hostile reelle. Le sandbox est modelise comme un **moteur de politique** (allowlist/denylist, limites de ressources, capacites) qui classe et bloque des actions **simulees**.

---

## Exercice 1 : Politique en couches allowlist + denylist + modele de capacites

### Objectif

Aller au-dela du simple `CapabilityRegistry` du cours (section 5) : construire un moteur de politique qui combine une **allowlist** (autorise par defaut tout ce qui matche), une **denylist** prioritaire (interdit meme si l'allowlist matche), et une verification de **capacite** (le syscall/command demande doit etre couvert par un `CapabilityToken`). On prouve qu'une commande autorisee passe alors qu'une commande denylistee ou non-capacitee est bloquee — sans rien executer.

### Consigne

En partant de `02-code/23-sandboxing-execution-sure.py` (classes `CapabilityToken`, `CapabilityRegistry`) :

1. Cree une classe `PolicyEngine` avec `__init__(self, allow: set[str], deny: set[str])` ou `allow`/`deny` sont des ensembles de noms de commandes/syscalls (ex: `"read"`, `"socket"`, `"ptrace"`, `"execve"`).
2. Implemente `classify(self, action: str, token: CapabilityToken | None) -> dict` qui applique l'ordre de decision **deny-first** :
   - Si `action` est dans `deny` → `{"decision": "deny", "reason": "denylist", "layer": "denylist"}` (la denylist gagne meme si l'allowlist matche aussi).
   - Sinon si `action` n'est pas dans `allow` → `{"decision": "deny", "reason": "not_in_allowlist", "layer": "allowlist"}`.
   - Sinon si `token is None` ou `token.tool != action` ou `not token.is_valid()` → `{"decision": "deny", "reason": "capability_missing", "layer": "capability"}`.
   - Sinon → consomme le token (`token.consume()`) et retourne `{"decision": "allow", "layer": "ok"}`.
3. Ajoute une methode `enforce(self, action, token) -> bool` qui retourne `True` seulement si la decision est `"allow"`, et logge chaque decision dans une liste `self.audit`.
4. Teste 4 scenarios (aucune execution reelle) :
   - `"read"` (allowed) avec un token `read` valide → autorise.
   - `"ptrace"` (denylist) avec un token `ptrace` valide → bloque par la denylist (la denylist prime sur la capacite).
   - `"socket"` (ni allow ni deny) avec token valide → bloque par l'allowlist.
   - `"read"` sans token → bloque par la capacite.

### Criteres de reussite

- [ ] La denylist prime : une `action` denylistee est bloquee **meme** si elle est aussi dans l'allowlist et qu'un token valide est fourni
- [ ] Une action absente de l'allowlist est bloquee avec `layer == "allowlist"`
- [ ] Une action allowlistee sans capacite valide est bloquee avec `layer == "capability"`
- [ ] Une action allowlistee + denylist-clean + token valide est autorisee et le token est consomme
- [ ] `self.audit` contient une entree par decision avec son `layer`

---

## Exercice 2 : Executeur metere avec limites de ressources (CPU / wall-clock / sortie)

### Objectif

Simuler le `ResourceLimiter` du cours (section 3 / couche 3 de la defense en profondeur) **sans** lancer de process reel : un executeur metere qui consomme un budget de CPU virtuel, de temps mural, et de taille de sortie, et qui **tue proprement** une tache emballee (boucle infinie, bombe memoire) quand un budget est depasse.

### Consigne

1. Cree une dataclass `ResourceBudget` avec `max_cpu_ticks: int`, `max_wall_ticks: int`, `max_output_bytes: int`.
2. Cree une classe `MeteredExecutor(budget: ResourceBudget)` qui execute une **tache simulee**. Une tache est une liste d'`Op` (dataclass) avec `cpu: int` (ticks de CPU consommes), `wall: int` (ticks de temps mural), `emit: int` (octets produits). Une tache emballee peut etre representee par une liste tres longue (boucle infinie simulee) — tu ne dois jamais l'iterer entierement, tu t'arretes des qu'un budget est depasse.
3. Implemente `run(self, ops: Iterable[Op]) -> dict` qui itere les ops en accumulant `cpu_used`, `wall_used`, `output_bytes` et s'arrete (kill gracieux) des qu'une limite est franchie. Retourne `{"status": "completed" | "killed", "killed_by": <"cpu"|"wall"|"output"|None>, "cpu_used", "wall_used", "output_bytes", "ops_executed"}`.
4. Le kill doit etre **gracieux** : pas d'exception non geree, on renvoie un dict propre, et `ops_executed` reflete combien d'ops ont tourne avant le kill.
5. Teste 4 scenarios :
   - Tache courte sous tous les budgets → `status == "completed"`.
   - Boucle (quasi) infinie : un generateur d'ops `cpu=1` sans fin → `killed_by == "cpu"` (ou `"wall"`), borne atteinte, pas de blocage.
   - Bombe de sortie : des ops qui emettent beaucoup d'octets → `killed_by == "output"`.
   - Verifie que `ops_executed` est strictement inferieur au nombre total d'ops pour les taches tuees.

### Criteres de reussite

- [ ] `MeteredExecutor.run` s'arrete des qu'**un** budget (cpu, wall ou output) est depasse
- [ ] Une tache (quasi) infinie est tuee proprement (pas de boucle infinie reelle, pas d'exception) avec `status == "killed"`
- [ ] `killed_by` identifie correctement la ressource qui a declenche le kill
- [ ] Une tache sous tous les budgets retourne `status == "completed"` avec les compteurs exacts
- [ ] `ops_executed` < nombre total d'ops pour une tache tuee

---

## Exercice 3 : Politique d'egress avec detection de tentative d'exfiltration

### Objectif

Etendre le `NetworkAllowlistProxy` du cours (section 6) : autoriser un hote allowliste, bloquer une destination inconnue, et surtout **detecter une tentative d'exfiltration** vers un domaine pourtant autorise (donnees encodees dans l'URL / volume anormal), conformement aux limites de l'egress filtering decrites dans le cours (section 6.2).

### Consigne

En partant de `NetworkAllowlistProxy` dans `02-code/23-sandboxing-execution-sure.py` :

1. Cree une classe `EgressPolicy(allowed_domains: set[str])` reutilisant la logique de matching exact + wildcard `*.example.com` du cours.
2. Implemente `check(self, url: str, payload_bytes: int = 0) -> dict` qui retourne un verdict en deux temps :
   - **Allowlist** : si le domaine n'est pas autorise → `{"action": "block", "reason": "domain_not_allowed"}`.
   - **Heuristique d'exfiltration** (meme sur un domaine autorise) : leve un flag si l'URL contient un parametre suspect (`data=`, `exfil`, `dump`, une longue chaine base64-like de >= 40 chars dans la query) OU si `payload_bytes > self.max_payload_bytes` (defaut 100_000). Dans ce cas → `{"action": "alert", "reason": "possible_exfiltration", "domain": ...}` (autorise techniquement mais signale).
   - Sinon → `{"action": "allow", "domain": ...}`.
3. Tiens un compteur `self.alerts` et `self.blocks`, et une methode `report() -> dict` retournant `{"allowed", "blocked", "alerted"}`.
4. Teste au moins 5 requetes : un appel API legitime (allow), une destination inconnue (block), une exfil vers `evil.com` (block car domaine), une exfil **vers un domaine autorise** avec `?data=<long_b64>` (alert), et un gros payload vers un domaine autorise (alert).

### Criteres de reussite

- [ ] Un domaine non allowliste est bloque (`action == "block"`)
- [ ] Un appel legitime vers un domaine allowliste passe (`action == "allow"`)
- [ ] Une tentative d'exfil vers un domaine **autorise** (URL suspecte ou gros payload) est signalee (`action == "alert"`), pas silencieusement laissee passer
- [ ] Le matching wildcard `*.wikipedia.org` fonctionne
- [ ] `report()` retourne les bons compteurs allowed / blocked / alerted
