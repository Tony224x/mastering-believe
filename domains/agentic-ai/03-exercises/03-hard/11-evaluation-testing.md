# Exercices Hard — Evaluation & Testing (J11)

---

## Exercice 1 : Eval statistique — runs repetes, intervalles de confiance et detection de flakiness

### Objectif
Traiter l'eval d'agent comme une mesure statistique : un agent stochastique ne se juge pas sur 1 run. Implementer les runs repetes, le bootstrap pour les intervalles de confiance, et l'identification des cas flaky.

### Consigne
1. Cree un `StochasticAgent` : wrapper autour de `FakeAgent` qui, selon un `random.Random(seed)` injecte, echoue aleatoirement certains cas :
   - Cas "stables-pass" : 100% de reussite
   - Cas "stables-fail" : 0%
   - Cas "flaky" : ~60% de reussite (au moins 2 cas flaky dans le jeu)
2. Ecris `repeated_eval(agent_factory, cases, n_runs=20) -> EvalMatrix` :
   - Chaque run utilise un seed different mais derive d'un seed maitre (reproductibilite totale)
   - `EvalMatrix` stocke la matrice booleenne `runs x cases`
3. **Bootstrap CI** : `bootstrap_ci(pass_rates_per_run, n_boot=1000, confidence=0.95) -> (low, high)` :
   - Re-echantillonne les runs avec remise, calcule le pass rate moyen de chaque echantillon, retourne les percentiles 2.5/97.5
   - Implemente les percentiles a la main (pas de numpy requis)
4. **Detection de flakiness par cas** : pour chaque cas, calcule le taux de reussite sur les runs ; classe :
   - `STABLE_PASS` (>= 95%), `STABLE_FAIL` (<= 5%), `FLAKY` (entre les deux)
   - Pour les FLAKY, calcule un score d'instabilite : `4 * p * (1 - p)` (max a p=0.5)
5. **Decision de gate statistique** : le deploiement est GO si la **borne basse** du CI du pass rate >= 0.70 (pas la moyenne !). Montre un cas ou la moyenne passe le seuil mais pas la borne basse -> NO-GO
6. Rapport final : pass rate moyen, CI 95%, liste des cas flaky avec leurs taux, decision de gate, et le seed maitre pour rejouer
7. Asserts : la classification retrouve exactement les cas conçus comme flaky ; deux executions avec le meme seed maitre donnent des resultats identiques

### Criteres de reussite
- [ ] La matrice runs x cases est entierement reproductible via le seed maitre
- [ ] Le bootstrap CI est implemente a la main et donne un intervalle plausible
- [ ] Les cas flaky concus sont detectes, les stables ne sont pas faussement flagges
- [ ] La decision de gate utilise la borne basse du CI et le cas limite est demontre
- [ ] Le rapport contient tout ce qu'il faut pour rejouer et auditer la mesure

---

## Exercice 2 : Calibration du juge — meta-evaluation contre un golden set humain

### Objectif
Evaluer l'evaluateur : mesurer l'accord entre un LLM judge et des labels humains, identifier ses biais systematiques (notamment le biais de verbosite), et recalibrer son seuil de decision pour maximiser le F1.

### Consigne
1. Construis un **golden set** de 20 paires (reponse, label humain PASS/FAIL) sur des questions Acme :
   - 8 bonnes reponses courtes, 4 bonnes reponses longues
   - 4 mauvaises reponses courtes, 4 mauvaises reponses longues ET verbeuses (beaucoup de mots, zero contenu correct — le piege a verbosite)
2. Cree un `BiasedMockJudge` qui score 1-5 avec un biais de verbosite integre : `score = base_score(keywords corrects) + bonus_longueur` (+1 si > 40 mots) — plafond a 5
3. Ecris `meta_evaluate(judge, golden_set, threshold) -> dict` qui calcule :
   - Matrice de confusion (TP, FP, TN, FN) avec PASS si `score >= threshold`
   - `accuracy`, `precision`, `recall`, `F1`
   - **Accord global** judge vs humain (pourcentage)
4. **Diagnostic de biais** : compare le taux de FP entre reponses courtes et longues :
   - `verbosity_bias = FP_rate(longues) - FP_rate(courtes)` — doit etre nettement positif avec ce juge
   - Affiche les exemples de reponses verbeuses faussement validees
5. **Recalibration** : balaye `threshold` de 1.0 a 5.0 (pas de 0.5), affiche la courbe F1(threshold) en ASCII, et selectionne le seuil qui maximise le F1
6. **Correction du biais** : cree un `DebiasedJudge` qui retire le bonus de longueur (normalise le score par la densite de keywords plutot que le compte brut) et re-mesure : le F1 au meilleur seuil doit s'ameliorer et le verbosity_bias se rapprocher de 0
7. Rapport comparatif final : juge biaise vs debiase (accord, F1, verbosity_bias, seuil optimal)

### Criteres de reussite
- [ ] Le golden set contient bien les 4 categories (court/long x bon/mauvais)
- [ ] La matrice de confusion et les metriques sont calculees correctement (verifiable a la main)
- [ ] Le biais de verbosite est quantifie et illustre par des exemples concrets
- [ ] La recalibration trouve le seuil F1-optimal et la courbe ASCII est lisible
- [ ] Le juge debiase ameliore mesurablement le F1 et reduit le verbosity_bias
- [ ] L'ensemble est deterministe et les asserts sur les metriques passent
