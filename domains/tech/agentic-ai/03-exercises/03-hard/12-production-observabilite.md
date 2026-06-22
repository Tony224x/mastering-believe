# Exercices Hard — Production & Observabilite (J12)

---

## Exercice 1 : Pipeline d'observabilite complet — SLO, error budget, alerting, anomaly detection

### Objectif
Construire le tableau de bord SRE d'un agent en prod : suivi des SLO, consommation de l'error budget, detection d'anomalies sur la latence/cout, et regles d'alerting — a partir d'un flux de spans (sections 4.4 et 5 du cours).

### Consigne
A partir d'un flux de traces (genere offline), construis un `ObservabilityPipeline` :

1. **SLO tracking** : definis des SLO (ex: `availability >= 99%`, `latency_p95 <= 1500ms`, `cost_per_request <= 0.02$`). Calcule pour une fenetre de traces si chaque SLO est respecte
2. **Error budget** : pour l'availability SLO a 99%, l'error budget est 1% des requetes. Calcule le budget consomme (`failed / total`) et le budget restant. Si le budget est epuise → recommande un "feature freeze"
3. **Anomaly detection** : sur la serie temporelle des latences, detecte les anomalies par z-score (rolling mean + std sur une fenetre glissante). Un point a |z| > 3 est une anomalie. Gere le cas demarrage (fenetre incomplete)
4. **Alerting avec deduplication** : un `AlertManager` qui leve une alerte quand un SLO est viole OU une anomalie detectee, mais **deduplique** : pas plus d'une alerte du meme type par fenetre de cooldown (eviter le spam d'alertes)
5. **Rapport** : produit un dict `{slo_status, error_budget, anomalies, alerts_fired}`

Scenario de test : genere ~50 traces dont une rafale d'echecs (epuise l'error budget) et un pic de latence (anomalie). Verifie que le pipeline detecte les deux, alerte une seule fois par type, et recommande le freeze.

### Criteres de reussite
- [ ] Les SLO (availability, p95, cost) sont evalues sur une fenetre de traces
- [ ] L'error budget est calcule et l'epuisement declenche une recommandation de freeze
- [ ] L'anomaly detection par z-score identifie le pic de latence (et pas les points normaux)
- [ ] L'AlertManager deduplique : 1 alerte max par type/cooldown
- [ ] Le rapport final agrege tout de maniere actionnable
- [ ] Tout est deterministe et offline (pas de Langfuse, pas de reseau)

---

## Exercice 2 : Load shedding adaptatif + concurrency limiter sous saturation

### Objectif
Implementer la protection d'un agent sous charge : un limiteur de concurrence (semaphore-like) qui rejette ou met en file selon la priorite, plus un load-shedder adaptatif qui largue le trafic basse priorite quand la latence se degrade (concept de graceful degradation pousse a l'echelle systeme).

### Consigne
Construis un `AdmissionController` simule (tout en mono-thread, on modelise le temps logiquement) :

1. **Concurrency limiter** : `max_in_flight` requetes simultanees. Au-dela, les requetes sont soit mises en file (priorite haute) soit rejetees (priorite basse) — pattern "shed low-priority first"
2. **Adaptive load shedding** : le controleur observe une `observed_latency_ms` (fournie par le simulateur). Quand elle depasse un seuil (`degraded_threshold_ms`), il passe en mode "shed" : il rejette une fraction croissante du trafic basse priorite (ex: 50% si latence 1.5x le seuil, 100% si 2x). Quand la latence redescend, il revient en mode normal (avec hysteresis pour eviter le flapping)
3. **Priority classes** : `critical` (jamais shed), `high` (file d'attente), `low` (shed en premier)
4. **Metriques** : compte les requetes `admitted`, `queued`, `shed`, par classe de priorite
5. Simule une rafale de 100 requetes mixtes (30 critical, 30 high, 40 low) avec une latence observee qui monte puis redescend ; verifie que :
   - Aucune `critical` n'est jamais shed
   - Les `low` sont shed pendant la periode degradee
   - Le systeme revient en mode normal apres recovery (hysteresis respectee)

### Criteres de reussite
- [ ] Le concurrency limiter borne les requetes en vol et met en file les overflow haute priorite
- [ ] Le load shedding s'active au-dela du seuil de latence et largue d'abord le trafic basse priorite
- [ ] Les requetes `critical` ne sont JAMAIS shed
- [ ] La fraction de shed augmente avec la severite de la degradation
- [ ] L'hysteresis empeche le flapping entre normal et degrade
- [ ] Les metriques par classe de priorite sont correctes et le retour en mode normal est verifie
