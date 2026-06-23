# Exercices Faciles — Agent complet (J7, capstone semaine 1)

---

## Exercice 1 : Ajouter un 4e outil (calculator)

### Objectif
Etendre l'agent pour supporter une nouvelle operation : comparer deux densites.

### Consigne
En partant du code de `02-code/07-agent-complet.py` :

1. Ajoute un outil `compare_tool(a: float, b: float) -> str` qui retourne "a > b" / "a < b" / "a == b" plus l'ecart en pourcentage
2. Ajoute une nouvelle action `compare:density_africa_vs_paris` dans le planner pour la question "Quelle densite est plus elevee, Afrique ou Paris ?"
3. Gere cette action dans l'`executor_node` (nouveau `elif action == "compare"`)
4. Le synthesizer doit produire une reponse du type "Paris has density X, Africa has density Y, Paris is Z% denser."
5. Teste avec la question : "Compare the population density of Africa and Paris."

### Criteres de reussite
- [ ] Le nouvel outil `compare_tool` est une fonction pure et testable seule
- [ ] L'executor reconnait l'action `compare:...`
- [ ] L'agent execute correctement les 3 etapes : recherche Afrique, recherche Paris, compare
- [ ] La reponse finale contient les 2 densites ET le verdict de comparaison
- [ ] Aucune modification du planner ne casse les 2 questions precedentes (Africa/Paris density seules)

---

## Exercice 2 : Changer la strategie de planning

### Objectif
Passer d'un planner rule-based a un planner "LLM-inspired" qui produit des plans plus riches a partir d'une liste de templates.

### Consigne
Remplace la logique du `planner_node` par un systeme de templates :

1. Cree un dict `PLAN_TEMPLATES` qui mappe un pattern de question a une liste de steps, ex :
   ```python
   PLAN_TEMPLATES = {
       "density_of_X": ["search:X area", "search:X population", "compute:density", "format:final_answer"],
       "population_of_X": ["search:X population", "format:final_answer"],
       "area_of_X": ["search:X area", "format:final_answer"],
   }
   ```
2. Le planner doit :
   - Detecter le pattern de la question (regex ou keywords)
   - Extraire l'entite X (par exemple "africa", "paris")
   - Substituer X dans les steps
   - Afficher le template utilise pour que l'utilisateur comprenne
3. Ajoute une 3e question au main : "What is the area of Africa?" qui doit declencher le template `area_of_X`
4. Verifie que les 3 templates fonctionnent et que les plans generes sont differents

### Criteres de reussite
- [ ] Les 3 templates sont definis et testables independamment
- [ ] Le planner logue clairement quel template a matche
- [ ] Les 3 questions produisent 3 plans differents
- [ ] Le plan pour "area of Africa" ne contient PAS de step `compute`
- [ ] Le code reste compatible avec l'executor existant

---

## Exercice 3 : Brancher un vrai LLM (si API key dispo)

### Objectif
Remplacer le mock du planner par un vrai LLM quand une cle API est disponible, sinon garder le fallback deterministe.

### Consigne
1. Ecris une fonction `make_planner_llm()` qui retourne un callable :
   - Si `ANTHROPIC_API_KEY` est set ET `anthropic` est installe, retourne une fonction qui appelle Claude
   - Sinon, retourne une fonction deterministe (la logique actuelle)
2. Le vrai appel LLM doit :
   - Construire un prompt qui demande "Decompose this question into 3-5 concrete steps"
   - Parser la reponse pour extraire les steps
   - Fallback en cas d'erreur (timeout, rate limit) sur le deterministe
3. Affiche en debut de programme si le planner est REAL ou MOCK
4. Verifie que dans les 2 modes, le reste de l'agent (executor, analyzer, synthesizer) fonctionne sans changement

### Criteres de reussite
- [ ] Le code tourne sans cle API (mode MOCK par defaut)
- [ ] Le code utilise le vrai LLM si la cle est presente (mode REAL)
- [ ] Le fallback sur erreur est silencieux (pas de crash)
- [ ] Le reste de l'agent ne voit AUCUNE difference entre les 2 modes
- [ ] Un print explicite indique quel mode est utilise
