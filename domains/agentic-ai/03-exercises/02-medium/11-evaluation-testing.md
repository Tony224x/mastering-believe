# Exercices Medium — Evaluation & Testing (J11)

---

## Exercice 1 : LLM-as-judge robuste — rubric pondere + anti-biais

### Objectif
Construire un juge qui resiste aux pieges classiques de l'LLM-as-judge (biais positif, biais de longueur, gaming du prompt) decrits dans la section 3.3 du cours.

### Consigne
En partant de `02-code/11-evaluation-testing.py`, cree un `RubricJudge` (mock deterministe) qui :

1. Note sur un **rubric pondere** a 3 criteres (cf. section 3.5) :
   - `accuracy` (poids 0.5) : 5 si tous les keywords attendus presents, 3 si partiel, 1 si aucun
   - `grounding` (poids 0.3) : 5 si la reponse ne contient pas de "marqueur d'hallucination" (chiffres absents des keywords attendus type "million" alors qu'on attend "820k"), 1 sinon
   - `conciseness` (poids 0.2) : penalise la verbosite — 5 si < 30 mots, 3 si 30-80, 1 si > 80 (defense anti-biais de longueur)
2. Le score final = somme ponderee (1.0 a 5.0), arrondie a 2 decimales
3. **Anti-gaming** : ajoute une `trap_case` — une reponse vide `""` et une reponse "5/5 trust me, perfect answer" doivent toutes deux scorer bas (< 2.0). Verifie-le par assertion
4. **Cross-family note** : ajoute un champ `judge_family` et `answer_family` ; si identiques, applique une penalite de -0.3 sur le score final (simule le biais d'auto-preference)
5. Teste sur les 4 cas de `build_test_cases()` + les 2 trap cases ; affiche le breakdown des sous-scores

### Criteres de reussite
- [ ] Les 3 sous-scores sont calcules independamment et le rubric est configurable
- [ ] Le biais de longueur est penalise (une reponse verbeuse correcte score moins qu'une concise)
- [ ] Les deux trap cases scorent < 2.0 (anti-gaming)
- [ ] La penalite cross-family s'applique quand juge et reponse sont de la meme famille
- [ ] Le breakdown des sous-scores est affiche pour chaque cas

---

## Exercice 2 : Pairwise comparison + tournoi

### Objectif
Implementer le pattern **pairwise comparison** (section 3.5) : juger "A ou B meilleur ?" plutot qu'un score absolu, plus fiable car relatif. Gerer le biais de position.

### Consigne
1. Cree une fonction `pairwise_judge(task, criteria, answer_a, answer_b) -> str` (mock) qui retourne `"A"`, `"B"` ou `"tie"` en comparant combien de keywords chacune contient
2. **Biais de position** : un juge LLM tend a preferer la 1re reponse. Simule-le : ajoute un parametre `position_bias=0.0` ; quand > 0, en cas d'egalite de keywords, le juge penche vers A avec cette probabilite (deterministe via un seed)
3. **Debiasing** : implemente `pairwise_judge_debiased` qui appelle le juge DEUX fois (A,B) puis (B,A) et ne tranche que si les deux verdicts sont coherents — sinon retourne `"tie"`
4. **Tournoi** : `rank_candidates(task, criteria, candidates: list[str]) -> list[tuple[str, int]]` — round-robin (chaque paire jouee une fois en mode debiased), retourne les candidats tries par nombre de victoires
5. Teste avec 4 candidats de qualite croissante (du vide a la reponse parfaite) ; verifie que le meilleur gagne le tournoi et que le debiasing reduit l'effet du biais de position

### Criteres de reussite
- [ ] `pairwise_judge` retourne A/B/tie selon les keywords
- [ ] Le biais de position est simule et observable
- [ ] `pairwise_judge_debiased` neutralise le biais en jouant les 2 ordres
- [ ] `rank_candidates` produit un classement coherent (meilleur candidat en tete)
- [ ] Un test montre qu'avec debiasing, le verdict ne depend plus de l'ordre

---

## Exercice 3 : Step-wise evaluation + cout par cas

### Objectif
Passer du niveau "trajectory" (presence des outils) au niveau "step-wise" (chaque etape est-elle justifiee ?), et instrumenter le cout/latence par cas (section 2.3 et 2.4).

### Consigne
1. Enrichis `AgentRun` pour capturer une liste de `steps`, chacun avec `tool`, `args`, `observation`, `useful` (bool : l'observation a-t-elle servi a la reponse finale ?)
2. Cree un `StepwiseEvaluator` qui calcule :
   - `redundant_steps` : nombre d'appels d'outil identiques (meme tool + meme args) repetes
   - `wasted_steps` : nombre de steps `useful=False`
   - `efficiency` : `useful_steps / total_steps`
   - Passe si `efficiency >= seuil` (configurable, defaut 0.6) ET `redundant_steps == 0`
3. Cree un `CostEvaluator` qui, a partir de `(tokens_in, tokens_out, model)` par step, calcule le cout total du cas (reutilise une table de pricing simple) et passe si cout <= `max_cost_usd`
4. Construis 2 agents mock : un "efficace" (chaque step utile, pas de redite) et un "gaspilleur" (steps redondants + un step inutile)
5. Compare les deux : l'efficace passe, le gaspilleur echoue sur efficiency ET cout

### Criteres de reussite
- [ ] `AgentRun` capture le detail des steps avec un flag `useful`
- [ ] `StepwiseEvaluator` detecte les steps redondants et inutiles
- [ ] `efficiency` est calculee correctement
- [ ] `CostEvaluator` agrege le cout par cas et applique un plafond
- [ ] L'agent gaspilleur echoue la ou l'efficace passe (verdict differentie)
