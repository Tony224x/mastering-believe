# Exercices Faciles — MCP (J10)

---

## Exercice 1 : Ajouter un tool "multiply" avec validation d'arguments

### Objectif
Comprendre comment enregistrer un nouveau tool et faire une vraie validation des arguments cote serveur.

### Consigne
En partant de `02-code/10-mcp.py` :

1. Ajoute un tool `multiply(a: int, b: int) -> int` sur le serveur de demo
2. Cote serveur, avant d'appeler le handler, valide que :
   - Les deux arguments sont presents
   - Les deux arguments sont des entiers (pas des strings, pas des floats)
   - Si invalide, retourne une reponse JSON-RPC error avec code `-32602` et un message clair
3. Ajoute des tests dans le demo :
   - `multiply(6, 7)` → 42
   - `multiply("a", 7)` → erreur `-32602`
   - `multiply(6)` → erreur `-32602`
4. Affiche les 3 cas dans le demo output

### Criteres de reussite
- [ ] Le tool `multiply` est enregistre et appele avec succes
- [ ] La validation rejette les mauvais types et les arguments manquants
- [ ] Les reponses d'erreur respectent le format JSON-RPC (`error.code`, `error.message`)
- [ ] Les 3 cas de test affichent un resultat clair

---

## Exercice 2 : Resource dynamique (time://)

### Objectif
Comprendre qu'une resource MCP peut calculer son contenu a la demande, pas juste retourner un fichier statique.

### Consigne
1. Ajoute une resource `time://now` au serveur de demo
2. Son reader doit retourner l'heure actuelle au format ISO 8601 (`datetime.now().isoformat()`)
3. Dans le demo, lis cette resource 2 fois avec un petit delai entre les deux lectures (0.01s suffit) et verifie que les deux valeurs sont differentes
4. Ajoute aussi une resource `time://uptime` qui retourne depuis combien de secondes le serveur tourne (stocke un `start_time` dans le serveur a l'initialisation)
5. Affiche les deux resources dans le demo

### Criteres de reussite
- [ ] Les deux resources `time://now` et `time://uptime` sont enregistrees
- [ ] `time://now` retourne une string ISO 8601 valide
- [ ] Deux lectures successives de `time://now` donnent des valeurs differentes
- [ ] `time://uptime` retourne bien un nombre qui croit avec le temps
- [ ] Aucun appel `datetime.now()` au moment de l'enregistrement (sinon le lazy serait casse)

---

## Exercice 3 : Prompt parametre avec plusieurs arguments

### Objectif
Comprendre comment un prompt MCP peut etre un template avec plusieurs variables.

### Consigne
1. Ajoute un prompt `code_review(language, file_content, focus)` au serveur de demo
2. Les arguments :
   - `language` : "python", "typescript", "rust"...
   - `file_content` : le code a review
   - `focus` : "bugs", "performance", "style"
3. Le builder doit retourner une liste de 2 messages :
   - Un `system` qui inclut le langage et le focus
   - Un `user` qui contient le code a analyser
4. Teste avec :
   - `code_review("python", "def foo(): return 1", "bugs")`
   - `code_review("typescript", "const x = 1;", "style")`
5. Dans le demo, affiche les 2 prompts resultants avec leurs messages

### Criteres de reussite
- [ ] Le prompt est enregistre avec 3 arguments dans la definition
- [ ] Le builder genere un system prompt qui mentionne le langage ET le focus
- [ ] Le builder genere un user message qui contient le code
- [ ] Les 2 invocations de test produisent des prompts differents
- [ ] Le demo les affiche proprement
