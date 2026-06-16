# Exercices Medium — Observabilite & MLOps

---

## Exercice 1 : Calculer et interpreter le PSI pour declencher une action

### Objectif
Maitriser le calcul du PSI a la main, l'interpreter avec les seuils, et distinguer data drift de concept drift pour decider de l'action.

### Consigne
Tu surveilles une feature numerique d'un modele de scoring. Tu binnes la distribution en 4 bins. Voici les pourcentages baseline (a l'entrainement) et current (cette semaine) :

| Bin | Baseline % | Current % |
|---|---|---|
| 1 | 40% | 25% |
| 2 | 30% | 30% |
| 3 | 20% | 25% |
| 4 | 10% | 20% |

**Questions :**
1. Calcule le PSI bin par bin avec la formule `PSI = sum((p_base - p_curr) * ln(p_base / p_curr))`. Donne le PSI total.
2. Interprete : pas de drift / drift modere / drift significatif (seuils 0.1 et 0.25) ?
3. Ce PSI mesure-t-il du data drift ou du concept drift ? Justifie ce que le PSI peut et ne peut pas voir.
4. Tu observes ce drift mais la performance (AUC) du modele est stable. Que conclus-tu ? Faut-il re-entrainer ?
5. Dans un cas different, la perf chute SANS data drift detectable sur les inputs. Quel type de drift est-ce, et quelle action ?
6. A quelle frequence faire tourner ce calcul en prod, et sur quel echantillon ?

### Criteres de reussite
- [ ] Le PSI total est calcule correctement (de l'ordre de 0.15)
- [ ] L'interpretation utilise les seuils 0.1 / 0.25 (drift modere "a surveiller")
- [ ] Le PSI est identifie comme mesurant le data drift (shift des inputs), pas le concept drift
- [ ] Drift sans baisse de perf -> surveiller, pas forcement re-entrainer (le modele peut rester valide)
- [ ] Perf qui chute sans data drift = concept drift -> re-entrainement requis (pas une simple recalibration)
- [ ] La frequence est quotidienne sur un echantillon (ex: 1000 events/jour)

---

## Exercice 2 : Concevoir un A/B test ML qui produit une vraie decision

### Objectif
Dimensionner un A/B test ML correct : metrique primary, guardrails, duree, et eviter les pieges classiques.

### Consigne
Ton modele de recommandation V2 bat V1 en offline (+3% de NDCG). Tu veux le valider en ligne.

**Contexte :**
- 2M sessions/jour
- Metrique business candidate : taux de clic sur les recos (CTR), actuellement 8%
- Tu veux detecter une amelioration relative de +2% (donc 8% -> 8.16%)
- Tu hesites sur : la duree du test, la metrique primary, comment splitter

**Questions :**
1. Quelle metrique primary choisis-tu et pourquoi PAS le NDCG offline ? Quelles guardrail metrics ajouter ?
2. Pourquoi un split par `user_id` hash plutot qu'aleatoire par requete ? Quel biais cela evite ?
3. Le novelty effect : decris-le et explique pourquoi il impose une duree minimale. Combien de temps laisser tourner ?
4. Tu regardes 20 metriques secondaires, 1 ressort "significative". Quel piege ? Comment l'eviter ?
5. Tu vois un signal positif des le jour 2. Pourquoi ne PAS conclure tout de suite ?
6. V2 ameliore le CTR de +2% mais degrade la latence p99 de +40%. Quelle est ta decision ? Justifie avec le role des guardrails.

### Criteres de reussite
- [ ] La metrique primary est business (CTR/engagement), choisie AVANT le test ; le NDCG offline ne prouve pas l'impact business
- [ ] Les guardrails incluent latence, error rate, cost (ne doivent pas se degrader)
- [ ] Le split par user_id hash evite le biais (un meme user verrait sinon V1 et V2 melanges = pollution)
- [ ] Le novelty effect est explique et impose 2-4 semaines de duree
- [ ] Le multiple testing est identifie (20 metriques -> 1 fausse positive) ; mitigation : metrique primary a priori ou correction de Bonferroni
- [ ] Conclure au jour 2 est rejete (novelty effect + signal pas encore stable/significatif)
- [ ] La degradation de latence (guardrail) bloque le rollout malgre le gain de CTR, OU impose une optim avant

---

## Exercice 3 : Concevoir le tracing d'un agent et le budget de cout par session

### Objectif
Concevoir la structure de spans d'un agent et un middleware d'agregation de cout par session, comme dans Langfuse/LangSmith.

### Consigne
Tu instrumentes un agent RAG (retrieve -> rerank -> 1 LLM call -> 1 tool call -> 1 LLM final). Tu veux un tracing exploitable et un controle de cout par session.

**Chiffres pour une requete type :**
- llm.call #1 (rewrite, modele mini) : 500 tok in, 120 tok out — prix mini : $0.15/1M in, $0.60/1M out
- llm.call #2 (final, modele std) : 1200 tok in, 350 tok out — prix std : $2.50/1M in, $10.00/1M out
- retrieval, rerank, tool call : pas de cout LLM (suppose negligeable)

**Questions :**
1. Decris la structure d'un span (champs essentiels) et l'arbre de spans pour cette requete (parent/enfants).
2. Calcule le cout LLM total de cette requete (somme des 2 appels).
3. Pourquoi le tracing en arbre (spans) bat-il des logs ligne-par-ligne pour debugger un agent ?
4. Concois un middleware qui agrege le cost par `session_id` : que doit-il faire a chaque appel LLM ?
5. Tu fixes un hard cap de $0.50/session. Combien de requetes-type avant de l'atteindre ? Que fait le middleware au cap ?
6. Quelles metadata attacher aux traces (au minimum) pour pouvoir filtrer/debugger en prod ?

### Criteres de reussite
- [ ] La structure de span inclut name, start/end (latence), parent_id, attributes (model, tokens, cost), status
- [ ] Le cout total de la requete est calcule correctement (somme des 2 appels)
- [ ] Le tracing en arbre est justifie (attribution parent-enfant, cout par etape, structure non-deterministe)
- [ ] Le middleware ajoute le cost delta de chaque appel au running total du session_id
- [ ] Le nombre de requetes avant le hard cap est calcule ; au cap, le middleware coupe / force un nouveau thread
- [ ] Les metadata minimales incluent user_id, session_id, prompt_version/tags (pour filtrer)
