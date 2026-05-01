# Exercices Easy — Observabilite & MLOps

---

## Exercice 1 : Concevoir un dashboard LLM

### Objectif
Definir les metriques et alertes essentielles pour un produit LLM en prod.

### Consigne
Tu operes un chatbot LLM pour un SaaS qui fait 50K conversations / jour. Ton CTO te demande de creer un dashboard operationnel.

**Travail :**
1. Liste les 8-12 metriques cles a afficher, groupees en 3-4 sections (performance, couts, qualite, fiabilite)
2. Pour chacune, indique : unite, seuil d'alerte, pourquoi elle est importante
3. Propose 4 alertes automatiques (pager ou Slack) avec leurs conditions exactes
4. Quelles metriques sont "leading indicators" (previennent les problemes) vs "lagging indicators" (constatent) ?

### Livrables
Un doc markdown structure : sections dashboard + alerts + leading vs lagging.

### Criteres de reussite
- [ ] Section Performance : p50/p99 latency, TTFT, throughput
- [ ] Section Costs : cost per request, tokens per request, daily burn
- [ ] Section Qualite : faithfulness score, user feedback, cache hit rate
- [ ] Section Fiabilite : error rate par provider, fallback rate, guardrail rejections
- [ ] Au moins une alerte sur drift / quality degradation
- [ ] Leading indicators cites : queue depth, fallback rate, drift (avant que les users ne se plaignent)
- [ ] Lagging indicators : user feedback thumbs down, customer complaints

---

## Exercice 2 : Interpreter un PSI

### Objectif
Savoir lire et interpreter les metriques de drift d'un modele en production.

### Consigne
Ton modele de credit scoring en production tourne depuis 6 mois. Tu mesures le PSI mensuel de 5 features :

| Feature | PSI Mois 1 | Mois 2 | Mois 3 | Mois 4 | Mois 5 | Mois 6 |
|---|---|---|---|---|---|---|
| monthly_income | 0.02 | 0.03 | 0.05 | 0.04 | 0.06 | 0.08 |
| debt_ratio | 0.01 | 0.02 | 0.04 | 0.12 | 0.18 | 0.22 |
| age | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.01 |
| num_late_payments | 0.05 | 0.08 | 0.15 | 0.21 | 0.28 | 0.35 |
| employment_type | 0.10 | 0.11 | 0.10 | 0.11 | 0.12 | 0.11 |

**Questions :**
1. Pour chaque feature, qualifie l'etat (no drift / watch / act) pour le dernier mois
2. Laquelle est la plus inquietante et pourquoi ?
3. Quelle(s) action(s) prends-tu immediatement ?
4. Quelle feature est stable mais deja "borderline" sans degradation ? Est-ce un probleme ?
5. Construis un plan de rollout d'un nouveau modele : quels criteres offline doit-il atteindre, comment tu le deploies, quelle est ta strategie de rollback ?

### Livrables
Un rapport structure avec les reponses aux 5 questions + un plan de remediation.

### Criteres de reussite
- [ ] num_late_payments est identifiee comme la plus inquietante (PSI = 0.35 et trend croissant)
- [ ] debt_ratio est "act" a 0.22 proche du seuil
- [ ] age est stable (no drift)
- [ ] employment_type est "borderline stable" a ~0.11 : pas d'action urgente mais a noter
- [ ] Le plan de rollout inclut : offline eval gate, shadow, canary, rollback auto sur metriques
- [ ] Une action immediate est proposee : investigation root cause (saisonnalite ? event externe ? bug pipeline ?)

---

## Exercice 3 : Design d'une CI/CD ML

### Objectif
Savoir dessiner un pipeline complet de deploiement d'un modele.

### Consigne
Ton equipe passe du "deploy manuel chaque vendredi" a une CI/CD automatisee. Le systeme :

- Est un classifieur de sentiment pour des tickets support
- Tourne sur 4 regions AWS
- Doit etre retrain chaque semaine
- 10K predictions / min en peak
- SLA : p99 < 200 ms, accuracy > 85%

**Travail :**
1. Decris chaque etape du pipeline (data -> train -> eval -> register -> deploy -> monitor) avec les outils que tu choisirais
2. Definis les "gates" (criteres bloquants) a chaque etape
3. Comment tu deploies sur 4 regions sans tout casser en meme temps ?
4. Quelle est ta strategie de rollback si l'accuracy degrade a +72h de deploiement ?
5. Quel monitoring declenche un retrain force en dehors du schedule hebdo ?

### Livrables
Un schema ASCII du pipeline + un doc decrivant les gates et la strategie multi-region.

### Criteres de reussite
- [ ] Pipeline complet : data validation -> train -> offline eval -> register -> staging -> shadow -> canary -> prod
- [ ] Gates offline incluent : accuracy > baseline + 2%, p99 latency < 200 ms sur un bench
- [ ] Deploiement multi-region : progressif (1 region en canary, puis propagation), jamais atomic 4 regions
- [ ] Rollback automatique base sur feature flag + observability alert (accuracy prod < 83%, ou user complaints spike)
- [ ] Au moins un trigger de retrain cite : drift PSI > 0.25 sur une feature cle, ou accuracy prod < 84%
- [ ] MLflow / DVC / Airflow / Kubeflow / feature flags sont mentionnes par leurs noms
