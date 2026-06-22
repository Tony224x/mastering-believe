# Exercices Faciles — Planning & Reasoning (J4)

---

## Exercice 1 : CoT vs direct sur un probleme de logique

### Objectif
Mesurer l'ecart de qualite entre un prompt direct et un prompt Chain-of-Thought sur un probleme de logique multi-etapes.

### Consigne
En partant du `MockLLM` dans `02-code/04-planning-reasoning.py` :

1. Cree 3 questions de logique differentes (arithmetique, enigme, logique de deduction)
2. Pour chaque question, appelle le LLM **deux fois** :
   - Une fois avec un prompt direct : `f"Question: {q}\nReponds directement."`
   - Une fois avec un prompt CoT : `f"Question: {q}\nLet's think step by step."`
3. Etends le `MockLLM` pour repondre a ces 3 nouvelles questions (2 versions de reponses : une rapide/fausse, une CoT/correcte)
4. Affiche un tableau recapitulatif :
   ```
   Question             | Direct | CoT
   ---------------------+--------+----
   Q1 (arithmetique)    | FAUX   | OK
   Q2 (enigme)          | FAUX   | OK
   Q3 (deduction)       | OK     | OK
   ```
5. Commente pourquoi Q3 n'a pas eu besoin de CoT.

### Criteres de reussite
- [ ] 3 questions de logique ajoutees, chacune avec 2 reponses dans le MockLLM
- [ ] Le tableau recapitulatif s'affiche clairement
- [ ] Au moins 2 questions montrent une difference direct vs CoT
- [ ] Une explication en 2-3 phrases de quand le CoT est utile ou pas

---

## Exercice 2 : Plan-and-execute avec une nouvelle question

### Objectif
Adapter le pattern plan-and-execute a une nouvelle question, pour comprendre comment le planner se deroule et ou les couplages sont implicites.

### Consigne
Modifie `plan_and_execute` pour gerer une nouvelle question : **"Quel est le ratio surface / habitant de Lyon ?"**

1. Ajoute dans le `MockLLM` :
   - Un plan pour cette question (4 steps : search population, search area, compute ratio, format)
   - Les reponses du executor pour chaque step
   - Les donnees dans `mock_search_tool` (Lyon : 520000 hab, 48 km2)
2. Lance `plan_and_execute` sur cette question
3. Verifie que le scratchpad contient bien `population=520000` et `area=48` apres l'execution
4. Le synthesizer doit produire une reponse contenant le ratio (~92 m2/habitant)
5. Ajoute un `assert` final qui verifie que le chiffre attendu est dans la reponse finale

### Criteres de reussite
- [ ] Nouvelle question traitee de bout en bout sans modifier la boucle
- [ ] Les donnees Lyon sont dans le `mock_search_tool`
- [ ] Le plan est parse correctement en 4 steps
- [ ] L'assert passe (ratio ~92 m2/hab dans la reponse finale)
- [ ] Le script reste deterministe (pas de flakiness)

---

## Exercice 3 : Reflexion avec critere concret

### Objectif
Remplacer la critique "libre" de Reflexion par une critique **basee sur un critere concret**, pour eviter le probleme du LLM qui critique sans ancrage.

### Consigne
Ecris une fonction `reflexion_with_checklist(llm, question, checklist, max_retries=3)` :

1. `checklist` est une liste de strings, chaque string etant un critere concret ex :
   - "Contient un chiffre"
   - "Contient une unite (hab/km2, m2/hab, etc.)"
   - "Contient une reference temporelle (annee, date)"
2. Apres chaque tentative, verifie **localement** (sans appeler le LLM) combien de criteres de la checklist sont presents dans la reponse
3. Si tous les criteres sont presents, retourne la reponse
4. Sinon, construis une critique basee sur les criteres manquants et re-invoque le LLM pour reviser
5. Teste avec la question "Quelle est la densite de Paris ?" et une checklist de 3 items
6. Montre qu'apres 1-2 iterations, tous les criteres sont presents

### Criteres de reussite
- [ ] La verification des criteres est 100% locale (zero appel LLM pour juger)
- [ ] La critique genere une liste claire des criteres manquants
- [ ] Le loop s'arrete des que tous les criteres passent
- [ ] Le cas "tous les criteres deja satisfaits au 1er coup" ne fait pas de retry
- [ ] Le cas "max_retries atteint" retourne la meilleure tentative avec un warning
