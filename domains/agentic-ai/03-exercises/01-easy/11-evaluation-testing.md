# Exercices Faciles — Evaluation & Testing (J11)

---

## Exercice 1 : Ordered trajectory check

### Objectif
Comprendre comment enrichir un trajectory evaluator pour verifier non seulement la presence des etapes mais aussi leur ordre.

### Consigne
En partant de `02-code/11-evaluation-testing.py` :

1. Cree une sous-classe `OrderedTrajectoryEvaluator(TrajectoryEvaluator)` qui verifie en plus que les outils attendus apparaissent dans **le bon ordre** dans la trajectoire reelle
2. Ajoute une methode `_is_subsequence(expected, actual) -> bool` qui verifie que `expected` est une sous-sequence (pas forcement contigue) de `actual`
3. Le cas `revenue-acme` s'attend a `["search_docs", "extract_number"]` dans cet ordre
4. Ajoute un cas de test `AgentTestCase` avec `expected_tools=["search_docs", "summarize"]` pour la question "Who is the main French competitor" et verifie que l'ordre est bien respecte
5. Teste avec un agent qui inverse l'ordre (mock) pour voir le fail

### Criteres de reussite
- [ ] `OrderedTrajectoryEvaluator` etend correctement `TrajectoryEvaluator`
- [ ] `_is_subsequence` retourne True si `expected=[A, B]` et `actual=[X, A, Y, B, Z]`
- [ ] Retourne False si l'ordre est inverse
- [ ] Un cas de test qui a l'ordre correct passe, un cas avec ordre inverse echoue
- [ ] Le message d'erreur mentionne "out of order"

---

## Exercice 2 : LLM judge with 3-criteria rubric

### Objectif
Etendre un LLM-as-judge simple vers un rubric multi-criteres avec scores ponderes.

### Consigne
1. Modifie `MockJudgeLLM` pour accepter un rubric :
   ```python
   rubric = {
       "accuracy": 0.5,      # poids 50%
       "completeness": 0.3,  # poids 30%
       "conciseness": 0.2,   # poids 20%
   }
   ```
2. Chaque critere donne un sous-score 0-5 :
   - `accuracy` : score 5 si tous les keywords attendus sont presents, 3 si partiel, 1 sinon
   - `completeness` : score 5 si la reponse fait > 20 mots, 3 si 10-20, 1 sinon
   - `conciseness` : score 5 si < 50 mots, 3 si 50-100, 1 si > 100
3. Le score final est la somme ponderee des sous-scores (entre 1.0 et 5.0)
4. Cree un `FinalAnswerEvaluatorV2` qui utilise ce rubric et retourne un dict avec :
   - `score` (float)
   - `sub_scores` (dict des scores par critere)
   - `passed` (bool, seuil a 3.5)
5. Teste sur les 4 cas du demo et affiche les sub-scores

### Criteres de reussite
- [ ] Le rubric est configurable (on peut changer les poids)
- [ ] Les 3 sous-scores sont calcules independamment
- [ ] Le score final est la somme ponderee correcte
- [ ] Les sub_scores sont visibles dans le rapport
- [ ] Au moins un cas passe et un cas echoue

---

## Exercice 3 : Export des resultats en CSV

### Objectif
Comprendre comment exporter les resultats d'un eval pour les analyser dans un spreadsheet.

### Consigne
1. Ecris une fonction `export_results_to_csv(results: list[CaseResult], filepath: str)` qui exporte les resultats dans un fichier CSV
2. Les colonnes doivent etre :
   - `id`
   - `verdict` (PASS / FAIL)
   - `final_answer_passed`
   - `final_answer_score`
   - `final_answer_reason`
   - `trajectory_passed`
   - `trajectory_reason`
   - `metrics_passed`
   - `metrics_reason`
3. Utilise `csv.DictWriter` de la stdlib (pas de pandas)
4. Ajoute une fonction `load_results_from_csv(filepath) -> list[dict]` qui relit le fichier
5. Teste le cycle complet : run -> export -> load -> compare

### Criteres de reussite
- [ ] Le CSV est valide et ouvrable dans un spreadsheet
- [ ] Toutes les colonnes sont presentes avec les bonnes valeurs
- [ ] Les chaines avec virgules sont correctement quotees (csv.DictWriter le fait par defaut)
- [ ] Le load fonctionne et retourne des dicts avec les memes cles
- [ ] Le test round-trip (run -> export -> load) passe un assert
