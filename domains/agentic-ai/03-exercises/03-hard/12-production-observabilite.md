# Exercices Hard — Production & Observabilite (J12)

---

## Exercice 1 : Mini-APM — agregation de traces et detection d'anomalies

### Objectif
Construire un mini systeme APM (Application Performance Monitoring) pour agents : ingerer un volume de traces, etablir des baselines par operation, detecter automatiquement les anomalies et generer un rapport d'incident exploitable.

### Consigne
1. **Generateur de trafic** : produis ~200 traces simulees deterministes (seed fixe) d'un agent 4 etapes (`plan -> search -> llm_step -> synthesize`), avec des distributions realistes par operation (latences et couts de base + bruit), PUIS injecte 3 anomalies precises :
   - A partir de la trace 120 : `search` voit sa latence x8 (degradation de dependance)
   - Traces 140-150 : `llm_step` coute 5x plus cher (prompt qui a gonfle)
   - Traces 160+ : `synthesize` a un taux d'erreur de 30% (au lieu de ~1%)
2. **Baseline** : `build_baseline(traces, until_index=100) -> dict` : pour chaque operation, moyenne et ecart-type de la latence et du cout, taux d'erreur (calcules a la main)
3. **Detection** : `detect_anomalies(traces, baseline, window=10) -> list[Anomaly]` en fenetres glissantes :
   - Latence : moyenne de fenetre > baseline + 3 sigma -> `LATENCY_SPIKE`
   - Cout : > 3x la baseline -> `COST_SPIKE`
   - Erreurs : taux de fenetre > 10x le taux baseline (et >= 2 erreurs) -> `ERROR_BURST`
   - Chaque `Anomaly` : type, operation, fenetre (indices de traces), valeur observee vs baseline, severite (1-3)
4. **Rapport d'incident** : pour chaque anomalie detectee, genere une section : resume, operation suspecte, chronologie (premiere/derniere trace touchee), spans representatifs (le pire exemple), hypotheses de cause (mapping type -> hypotheses predefinies), et actions recommandees
5. **Verification** : les 3 anomalies injectees sont detectees avec le bon type et la bonne operation ; aucune fausse alerte sur les 100 premieres traces saines (precision = 100% sur ce scenario)
6. Affiche : tableau des baselines, liste des anomalies, et le rapport d'incident complet du LATENCY_SPIKE

### Criteres de reussite
- [ ] Le trafic simule est reproductible et les anomalies injectees sont controlees
- [ ] Les baselines (moyenne, sigma, taux d'erreur) sont calculees correctement
- [ ] Les 3 anomalies sont detectees avec le bon type, la bonne operation et une fenetre plausible
- [ ] Zero fausse alerte sur la periode saine
- [ ] Le rapport d'incident contient chronologie, exemple de span et hypotheses
- [ ] Tout tourne en < 5 secondes sans dependance externe

---

## Exercice 2 : Controleur de deploiement canary avec promotion/rollback automatique

### Objectif
Implementer le pattern canary release pour un agent : router une fraction du trafic vers la nouvelle version, comparer les metriques des deux versions, et decider automatiquement de promouvoir ou de rollback.

### Consigne
1. **Deux versions d'agent** (mocks deterministes via seed) :
   - `agent_v1` (stable) : latence ~400ms, taux d'erreur 2%, score qualite ~0.82
   - `agent_v2_good` : latence ~350ms, erreurs 1%, qualite ~0.86
   - `agent_v2_bad` : latence ~380ms, erreurs 12%, qualite ~0.60 (la mauvaise release)
2. **Router pondere** : `CanaryRouter(stable, canary, canary_pct)` — l'assignation utilise un hash du request_id (`hash(request_id) % 100 < canary_pct`), PAS de random : un meme request_id va toujours du meme cote (sticky)
3. **Collecte** : pour chaque requete, enregistre version, latence, erreur, score qualite. `compare() -> dict` calcule par version : p50 latence, taux d'erreur, qualite moyenne, n
4. **Controleur a paliers** : `CanaryController(router, stages=[5, 25, 50, 100], min_samples=30)` :
   - A chaque palier, attend `min_samples` requetes canary puis evalue les **regles de promotion** :
     - taux d'erreur canary <= taux stable + 2 points
     - qualite canary >= qualite stable - 0.03
     - p50 canary <= p50 stable * 1.3
   - Toutes OK -> palier suivant (5 -> 25 -> 50 -> 100 = PROMOTED)
   - Une regle violee -> ROLLBACK immediat (canary_pct = 0) avec la raison precise
5. **Simulation** : envoie un flux de requetes (request_ids deterministes) et deroule :
   - Scenario A : v2_good -> doit franchir les 4 paliers et finir PROMOTED
   - Scenario B : v2_bad -> doit se faire rollback au premier palier evalue, avec la regle violee dans le rapport
6. **Journal de deploiement** : chaque evaluation de palier est journalisee (palier, n echantillons, metriques des 2 versions, decision) ; affiche le journal des 2 scenarios
7. Asserts : v2_good finit a 100%, v2_bad finit a 0%, le sticky routing est verifie (le meme request_id retombe toujours sur la meme version), et aucune evaluation n'a lieu avant min_samples

### Criteres de reussite
- [ ] Le routage est sticky et respecte les pourcentages a ±3 points
- [ ] Les 3 regles de promotion sont evaluees a chaque palier avec min_samples respecte
- [ ] v2_good est promue palier par palier, v2_bad est rollback avec la raison exacte
- [ ] Le journal de deploiement permet de rejouer toute la decision
- [ ] La simulation est entierement deterministe (seeds et hashs stables)
