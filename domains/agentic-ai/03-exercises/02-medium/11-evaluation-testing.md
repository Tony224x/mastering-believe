# Exercices Medium — Evaluation & Testing (J11)

---

## Exercice 1 : Juge pairwise avec mitigation du biais de position

### Objectif
Implementer la comparaison A/B de deux agents par un LLM judge, en neutralisant le biais de position (les juges favorisent souvent la premiere reponse presentee).

### Consigne
En partant de `02-code/11-evaluation-testing.py` :

1. Cree un `MockPairwiseJudge` avec une methode `compare(task, answer_a, answer_b) -> dict` retournant `{"winner": "A"|"B"|"TIE", "reason": ...}` :
   - Heuristique mock : gagne la reponse qui contient le plus de keywords attendus ; en cas d'egalite, la plus concise
   - **Injecte volontairement un biais de position** : si les 2 reponses sont a egalite de keywords ET de longueur similaire (±20%), le mock favorise la position A (c'est le biais a detecter)
2. Ecris `debiased_compare(judge, task, answer_1, answer_2) -> dict` :
   - Appelle le juge 2 fois : (answer_1, answer_2) puis (answer_2, answer_1)
   - Si les 2 verdicts sont coherents (le meme contenu gagne dans les 2 sens) -> verdict ferme
   - S'ils se contredisent (A gagne dans les 2 sens = biais de position) -> verdict `"TIE"` avec `"position_bias_detected": True`
3. Construis 4 paires de reponses de test :
   - 2 paires avec un gagnant clair (verdict ferme attendu, dans les 2 ordres)
   - 1 paire quasi identique (le biais doit etre detecte -> TIE)
   - 1 paire ou answer_2 est meilleure (verifie que l'ordre d'appel n'influence pas le verdict final)
4. Lance la comparaison entre `FakeAgent(buggy=False)` et `FakeAgent(buggy=True)` sur les cas de `build_test_cases()` et affiche : verdicts par cas, score global (wins/ties), nombre de biais detectes

### Criteres de reussite
- [ ] Chaque comparaison fait exactement 2 appels juge (ordre normal + inverse)
- [ ] Le gagnant clair est confirme quel que soit l'ordre de presentation
- [ ] La paire quasi identique declenche `position_bias_detected`
- [ ] Le score global agent-sain vs agent-buggy favorise l'agent sain
- [ ] Le rapport affiche verdicts, raisons et stats de biais

---

## Exercice 2 : Regression gate pour la CI

### Objectif
Construire le mecanisme qui empeche de deployer une version d'agent qui regresse : comparaison contre une baseline persistee, classification des ecarts, et code de sortie exploitable en CI.

### Consigne
1. Ecris `save_baseline(results, filepath)` qui persiste les resultats d'un run en JSON : pour chaque cas `id`, `verdict`, `final_answer_score`, et un champ global `created_at` + `agent_version`
2. Ecris `regression_gate(baseline_path, new_results, policy) -> GateReport` qui classifie chaque cas :
   - `REGRESSION` : passait en baseline, echoue maintenant
   - `IMPROVEMENT` : echouait, passe maintenant
   - `STABLE_PASS` / `STABLE_FAIL` : pas de changement
   - `NEW_CASE` : absent de la baseline
   - `REMOVED_CASE` : present en baseline, absent du run (doit etre signale !)
3. La `policy` est un dict : `{"max_regressions": 0, "min_pass_rate": 0.75, "allow_removed_cases": False}` — le gate retourne `passed: bool` + les violations precises
4. Le `GateReport` genere un rapport Markdown (string) avec un tableau des cas et une section "Decision: GO / NO-GO" + les raisons
5. Simule le cycle complet :
   - Run 1 : `FakeAgent(buggy=False)` -> save baseline
   - Run 2 : `FakeAgent(buggy=True)` -> le gate detecte les regressions -> NO-GO
   - Run 3 : agent sain avec un cas de test supplementaire -> GO avec 1 NEW_CASE
6. Implemente `exit_code(report) -> int` (0 = GO, 1 = NO-GO) et affiche-le comme le ferait un job CI

### Criteres de reussite
- [ ] La baseline persiste et se recharge correctement (round-trip JSON)
- [ ] Les 6 classifications sont correctes sur les scenarios simules
- [ ] La policy est configurable et chaque violation est listee avec sa raison
- [ ] Le rapport Markdown est lisible (tableau + decision)
- [ ] Les 3 runs produisent les bons verdicts GO/NO-GO et exit codes

---

## Exercice 3 : Generation de cas de test synthetiques depuis un corpus

### Objectif
Automatiser la creation du jeu d'eval : generer des cas de test (question + keywords attendus) a partir des documents du corpus, puis valider le jeu genere en le faisant tourner dans le harness.

### Consigne
1. Pars d'un mini-corpus de 4 documents factuels (contexte Acme : revenue, produit, equipe, concurrent) — chaque doc a un `id` et un `text` contenant au moins un fait chiffre ou un nom propre
2. Ecris `generate_cases_from_corpus(corpus) -> list[AgentTestCase]` avec 3 templates de generation **deterministes** :
   - **Fait chiffre** : detecte un pattern `<entite> ... <nombre> <unite>` -> question "What is the <metric> of <entity>?", expected_keywords = [nombre, entite]
   - **Definition** : premier nom propre du doc -> "What is <X>?" avec keywords = les 2 termes les plus distinctifs du doc (TF-IDF simplifie : termes presents dans ce doc et rares ailleurs)
   - **Negatif / hors-domaine** : genere 1 cas dont la reponse n'est PAS dans le corpus ("What is the revenue of Globex?") avec `expected_behavior="abstain"` (l'agent doit dire qu'il ne sait pas)
3. Ajoute une etape de **deduplication** : deux cas generes avec des questions quasi identiques (meme ensemble de keywords) -> n'en garder qu'un
4. Etends `AgentTestCase` (sous-classe ou champ optionnel) pour supporter `expected_behavior: str = "answer"` et adapte l'evaluation : un cas `abstain` passe si la reponse contient "do not know" / "no information" et AUCUN chiffre
5. Genere le jeu (attendu : 6-8 cas apres dedup), affiche chaque cas avec son template d'origine, puis lance le harness avec le `FakeAgent` etendu pour repondre aux nouvelles questions (mock)
6. Verifie : le cas abstain passe avec un agent honnete et echoue avec un agent qui hallucine (mock qui invente un chiffre pour Globex)

### Criteres de reussite
- [ ] Les 3 templates generent des cas valides et deterministes depuis le corpus
- [ ] La deduplication elimine les quasi-doublons (demontre avec un doc piege)
- [ ] Le cas hors-domaine teste l'abstention, pas la reponse
- [ ] Le jeu genere tourne dans le harness sans modification de celui-ci (ou minime et justifiee)
- [ ] L'agent halluciant echoue sur le cas abstain, l'agent honnete passe
