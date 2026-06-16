# Exercices Hard — Observabilite & MLOps

---

## Exercice 1 : Concevoir l'observability + MLOps d'une plateforme ML/LLM en prod

### Objectif
Concevoir le systeme complet d'observability et de MLOps d'une plateforme qui sert a la fois des modeles ML classiques et des LLMs. Tracing, drift, A/B, CI/CD modele. Design d'entretien senior.

### Consigne
Tu es responsable de la fiabilite ML d'une plateforme qui sert :
- 5 modeles ML classiques (fraud, reco, ranking, churn, pricing) : 100M predictions/jour
- 3 produits LLM (support, search, summarization) : 20M requetes/jour
- Equipe : 6 data scientists qui deploient ~10 modeles/mois au total

**Contraintes :**
- Les pannes silencieuses (modele qui derive, LLM qui hallucine) doivent etre detectees < 1h
- Conformite : certains modeles (fraud, pricing) sont en domaine regule -> audit trail obligatoire
- Budget observability raisonnable (pas une 2eme infra aussi chere que la prod)

**Livre :**

1. **Architecture d'observability** :
   - Les 3 piliers (tracing, metrics, logs) : que mets-tu dans chacun pour ML vs LLM ?
   - Comment unifies-tu l'observability ML classique ET LLM sans 2 stacks separees ?
   - Quelles metriques specifiques LLM vs ML classique ?

2. **Drift & data quality** :
   - Quelle strategie de drift pour les 5 modeles ML (PSI/KL) vs les 3 LLMs (drift sur prompts/responses) ?
   - Comment detectes-tu un drift en < 1h ? (frequence, echantillon, alerting)
   - Distingue le drift (PSI) du data quality (nulls, types) : que monitorer pour chacun ?

3. **Detection des pannes silencieuses** :
   - Un modele de fraud derive lentement (concept drift). Comment l'attrapes-tu AVANT le ticket ?
   - Un LLM se met a halluciner sur un nouveau segment d'users. Comment le detectes-tu sans labels ?

4. **A/B testing & deploiement** :
   - Concois le pipeline CI/CD modele (de commit a prod) avec les gates.
   - Comment geres-tu les deploiements regules (fraud/pricing) avec audit trail ?
   - Canary + rollback automatique : sur quels signaux declenches-tu le rollback ?

5. **Cout & gouvernance** :
   - Comment evites-tu que l'observability coute aussi cher que la prod ? (sampling, TTL, hash)
   - Comment geres-tu le PII dans les logs LLM en domaine regule ?

6. **Failure modes** :
   - L'alerting drift envoie 50 fausses alertes/jour (alert fatigue). Comment tu corriges ?
   - Le pipeline promeut un modele qui passe l'offline eval mais casse en prod. Quel gate manquait ?

### Criteres de reussite
- [ ] Les 3 piliers sont remplis differemment pour ML (features, predictions, PSI) et LLM (spans, tokens, cost, faithfulness)
- [ ] L'unification passe par OpenTelemetry (semantique gen_ai.* pour le LLM) -> une stack commune
- [ ] La strategie de drift distingue PSI/KL (features numeriques ML) et drift sur prompts/responses (longueur, topic, sentiment) pour les LLMs
- [ ] La detection < 1h repose sur un job frequent + alerting sur seuils (PSI > 0.25, faithfulness en chute)
- [ ] Le concept drift fraud est attrape par monitoring de perf sur labels retardes + drift sur la relation, pas juste les inputs
- [ ] Le pipeline CI/CD a des gates (offline eval >= baseline, shadow, canary 1%->100%, rollback auto) + manual gate pour les modeles regules avec audit trail
- [ ] Le cout est maitrise (sampling des traces, TTL agressif, hash+sample des prompts) et le PII est scrub ; l'alert fatigue est traitee (seuils calibres, dedup, severites)

---

## Exercice 2 : Post-mortem — Le modele qui a derive en silence pendant 3 semaines

### Objectif
Analyser un incident de drift silencieux non detecte, reconstituer la cascade, et concevoir le systeme d'observability qui l'aurait attrape.

### Consigne
Voici le rapport d'incident (resume) d'un modele de scoring de fraude.

**Contexte** : Un modele de detection de fraude tourne en prod depuis 1 an. Il bloque les transactions au-dessus d'un score de risque. Pas de drift monitoring. Pas d'eval continue (le modele a ete eval une fois au deploiement). Les labels de fraude (transaction reellement frauduleuse ou non) arrivent avec **3 semaines de retard** (le temps des chargebacks). Les metriques surveillees : latence, error rate, throughput (toutes "vertes").

**Timeline de l'incident :**

| Date | Evenement |
|---|---|
| Semaine 0 | Un nouveau pattern de fraude emerge (concept drift : la relation input->fraude change). |
| Semaine 0 | Le modele ne reconnait pas le nouveau pattern : il laisse passer les transactions frauduleuses (faux negatifs en hausse). |
| Semaine 0-3 | Les metriques techniques restent vertes (latence, errors OK). Le modele renvoie des 200. La fraude passe inapercue. |
| Semaine 1 | En parallele, la distribution des montants de transaction change (nouveau marche) : data drift NON monitore. |
| Semaine 3 | Les premiers chargebacks (labels) arrivent. L'equipe finance remarque une explosion des pertes de fraude. |
| Semaine 3 | Investigation : le modele a un recall de fraude tombe de 92% a 61% depuis 3 semaines. |
| Semaine 3 | Personne n'avait l'info car aucune eval continue, aucun drift monitoring, et les labels arrivaient trop tard pour servir d'alerte directe. |
| Semaine 3 | Post-mortem : pertes de fraude estimees a $2M sur 3 semaines. |

**Questions :**

1. **Root cause analysis** :
   - Reconstitue la cascade complete.
   - Pour chaque maillon, le garde-fou manquant.
   - Classe : monitoring, MLOps/process, architecture de detection.

2. **Le piege des metriques vertes** :
   - Pourquoi latence/error rate/throughput verts ne disaient RIEN du probleme ?
   - Quelle est la difference entre une panne d'API classique et une panne de modele ? (vu au J13)

3. **Le probleme des labels retardes** :
   - Le recall ne peut etre calcule qu'avec les labels (3 semaines de retard). Comment detecter le probleme SANS attendre les labels ?
   - Propose 2 signaux proxy (sans labels) qui auraient alerte des la semaine 0-1.

4. **Concept drift vs data drift** :
   - Ici il y a LES DEUX. Identifie lequel est la cause principale des pertes et lequel est secondaire.
   - Pour chacun, l'action corrective (recalibration vs re-entrainement).

5. **Systeme corrige** :
   - Concois le monitoring qui aurait attrape ca en < 1 jour.
   - Concois un gate qui force une eval continue (pas juste au deploiement).
   - Propose un runbook de 7 etapes pour un drift de modele detecte en prod.

### Criteres de reussite
- [ ] La cascade complete est reconstituee : nouveau pattern de fraude (concept drift) -> faux negatifs -> metriques techniques vertes (panne silencieuse) -> data drift non monitore en parallele -> labels retardes de 3 semaines -> detection tardive -> $2M
- [ ] Le piege des metriques vertes est explique : un modele qui se trompe renvoie un 200 (la difference panne API vs panne modele du J13)
- [ ] La detection sans labels repose sur des proxies : drift sur les inputs (PSI), drift sur la distribution des scores de sortie, taux de transactions bloquees qui chute
- [ ] Le concept drift est identifie comme cause PRINCIPALE (pertes de fraude) ; le data drift comme secondaire
- [ ] Les actions distinguent recalibration (data drift) et re-entrainement (concept drift)
- [ ] Le systeme corrige inclut drift monitoring quotidien (PSI inputs + distribution des scores) + eval continue des que les labels arrivent
- [ ] Le runbook est actionable et commence par contenir le risque (ex: durcir le seuil / review manuelle) en attendant le re-entrainement
