# Exercices Hard — Evaluation & Testing (J11)

---

## Exercice 1 : CI gate complet avec stratification, flakiness et budget de regression

### Objectif
Construire le gate d'evaluation qu'on branche en CI : il ne se contente pas d'un pass rate global, il stratifie par difficulte/tag, mesure la flakiness, et applique une politique de regression nuancee (section 4 du cours).

### Consigne
Cree un `CIEvalGate` qui orchestre tout :

1. **Stratification par tag** : a partir des `tags` de chaque `AgentTestCase` (`easy`/`medium`/`hard`, `rag`/`docs`...), calcule un pass rate PAR strate. La policy par difficulte (cf. section 4.2) :
   - `easy` doit etre a 100%
   - `medium` >= 80%
   - `hard` >= 50%
   - Le gate echoue si une strate est sous son seuil
2. **Flakiness** : chaque cas est execute `n_runs=5` fois (l'agent mock a un comportement legerement non-deterministe controle par un seed). Un cas est `flaky` si son verdict varie entre les runs. Le gate REPORTE les cas flaky meme s'ils passent en moyenne (un test flaky est un test casse)
3. **Budget de regression** : compare au baseline. Politique :
   - 0 regression sur les cas `easy` → tolerance zero (bloque)
   - <= 1 regression `medium`/`hard` toleree SI compensee par >= 2 fixes (net positif)
   - Sinon → bloque
4. **Verdict structure** : retourne un dict `{passed, per_tag_rates, flaky_cases, regressions, fixes, blocking_reasons}`
5. Teste avec : un baseline propre, puis un candidat qui (a) regresse 1 cas easy → doit bloquer ; (b) un autre candidat qui regresse 1 medium mais fixe 2 → doit passer ; (c) un cas flaky doit toujours etre signale

### Criteres de reussite
- [ ] Le pass rate est calcule par strate (tag) avec des seuils differencies
- [ ] La flakiness est detectee sur plusieurs runs et reportee meme si le cas "passe"
- [ ] La policy de regression bloque toute regression `easy`
- [ ] La policy tolere une regression `medium`/`hard` compensee par assez de fixes
- [ ] Le verdict final liste des `blocking_reasons` explicites et actionnables
- [ ] Les 3 scenarios (regression easy, regression compensee, flaky) donnent le bon verdict

---

## Exercice 2 : RAG eval — faithfulness + context precision/recall sans LLM externe

### Objectif
Implementer les metriques cles de l'evaluation RAG (faithfulness, context precision, context recall, answer relevance) facon RAGAS, mais en mode offline et deterministe — pour evaluer le pipeline RAG agentique du J8.

### Consigne
Tu disposes, pour chaque cas, de : la `question`, l'`answer` generee, les `retrieved_contexts` (chunks recuperes) et les `ground_truth_contexts` (chunks pertinents de reference). Implemente un `RagEvaluator` avec :

1. **Decomposition en claims** : `extract_claims(answer) -> list[str]` (mock : split en phrases sur `.`). Chaque claim est une affirmation atomique
2. **Faithfulness** : fraction des claims de la reponse qui sont "supportes" par au moins un chunk recupere (mock du support : overlap de tokens significatif, seuil configurable). Une reponse qui invente des claims non supportes a une faithfulness < 1.0 → signe d'hallucination
3. **Context precision** : parmi les chunks recuperes, fraction qui sont reellement pertinents (presents dans `ground_truth_contexts`). Mesure le bruit du retriever
4. **Context recall** : parmi les chunks de reference, fraction qui ont ete recuperes. Mesure les trous du retriever
5. **Answer relevance** : overlap entre les tokens de la question et ceux de la reponse (mesure si la reponse repond bien a la question posee)
6. Calcule un `composite_score` pondere et un verdict par cas. Teste sur 3 cas :
   - Cas ideal (faithfulness=1, bonne precision/recall)
   - Cas hallucination (un claim non supporte → faithfulness < 1)
   - Cas mauvais retriever (recall faible : un chunk de reference manquant)

### Criteres de reussite
- [ ] Les 4 metriques (faithfulness, context precision, context recall, answer relevance) sont calculees
- [ ] Le cas hallucination est detecte par une faithfulness < 1.0
- [ ] Le cas mauvais retriever est detecte par un context recall < 1.0
- [ ] Le cas ideal score haut sur toutes les metriques
- [ ] Le composite est pondere et le verdict differentie les 3 cas
- [ ] Tout est deterministe et offline (aucun LLM externe, aucune dependance)
