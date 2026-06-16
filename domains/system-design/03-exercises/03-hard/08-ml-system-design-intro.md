# Exercices Hard — ML System Design Intro

---

## Exercice 1 : Concevoir la plateforme ML d'un e-commerce (feature store, serving, retraining)

### Objectif
Concevoir le systeme ML complet d'une plateforme qui sert plusieurs modeles, en resolvant skew, point-in-time correctness et le choix batch/real-time. Design d'entretien senior "ML platform".

### Consigne
Tu es responsable de la plateforme ML d'un e-commerce :
- 5 modeles : reco produit, ranking search, anti-fraude, prevision de demande, churn
- 100M predictions/jour au total, 50M users
- 6 data scientists deploient ~10 modeles/mois
- Contrainte : eliminer le training-serving skew (qui a deja cause 2 incidents)

**Livre :**

1. **Cycle de vie & composants** :
   - Dessine (ASCII) le systeme : data ingestion -> feature store -> training -> registry -> serving -> monitoring.
   - Quels composants sont partages entre les 5 modeles, lesquels sont specifiques ?

2. **Feature store** :
   - Explique pourquoi un feature store plutot qu'un cache Redis par equipe.
   - Detaille la separation offline store (training) / online store (serving) : techno, retention, latence cible.
   - Comment garantis-tu que `compute_features(event)` est IDENTIQUE au train et au serving ?
   - Point-in-time correctness : comment l'offline store l'assure pour generer un dataset sans leakage ?

3. **Batch vs real-time par modele** :
   - Pour CHACUN des 5 modeles, choisis batch / real-time / hybride avec une justification chiffree (latence, volume, fraicheur).
   - Lequel est le plus contraint en latence ? Pourquoi ?

4. **Model registry & deploiement** :
   - Que stocke le registry (artifacts, metadata, lineage, stage) ?
   - Decris la sequence shadow -> canary -> A/B -> promote, et quand tu utilises chacune.

5. **Retraining** :
   - Qu'est-ce qui declenche un retraining (drift, schedule, perf) ?
   - Comment evites-tu de promouvoir un mauvais modele (gate offline) ?

6. **Failure modes** :
   - Le online store (Redis) tombe : strategie de degradation ?
   - Une feature est calculee en Python au train et en SQL au serving : quel incident, comment l'eviter structurellement ?
   - Un modele excellent en offline (AUC 0.95) s'effondre en prod : 2 causes probables.

### Criteres de reussite
- [ ] Le systeme est dessine de bout en bout ; le feature store, le registry et le monitoring sont PARTAGES, les modeles/serving sont specifiques
- [ ] Le feature store est justifie (reutilisation cross-equipe + consistance offline/online + point-in-time), pas un simple cache
- [ ] Offline (Parquet/BQ, longue retention, point-in-time joins) vs online (Redis/DynamoDB, < 10 ms, derniere valeur) distingues
- [ ] La consistance train/serving repose sur une definition de feature UNIQUE (pas de code divergent) ; le point-in-time via as-of join sur l'historique horodate
- [ ] Choix par modele coherents : anti-fraude = real-time (< 100 ms, bloquant), reco = hybride (batch + re-rank online), demande/churn = batch, search ranking = real-time/online ; anti-fraude le plus contraint
- [ ] Registry decrit (artifacts + metadata + lineage + stage) ; sequence shadow -> canary -> A/B -> promote correcte
- [ ] Retraining declenche par drift/perf/schedule ; gate offline (>= baseline) avant promotion ; failure modes traites (fallback features par defaut/modele simple, feature store unique contre le skew, leakage/skew comme causes de l'effondrement prod)

---

## Exercice 2 : Post-mortem — le modele a 95% d'AUC qui refusait les bons clients

### Objectif
Analyser un incident de training-serving skew classique, reconstituer la chaine et concevoir la prevention structurelle.

### Consigne
Voici le rapport d'incident (resume) d'un modele de scoring de credit.

**Contexte** : modele entraine dans un notebook qui lit un dump CSV nettoye. En prod, les features sont recalculees par un service en SQL. Pas de feature store. Pas de monitoring de distribution des features. AUC offline = 0.95.

**Timeline de l'incident :**

| Heure | Evenement |
|---|---|
| J0 | Deploiement du modele (AUC offline 0.95). Tests internes verts. |
| J0 | En prod, la feature `revenu_net_mensuel` est calculee depuis le champ `salary` (brut), alors qu'au train elle venait de `monthly_income` (deja net, apres un groupby().mean()). |
| J0 | Une 2eme feature, `nb_commandes_30j`, a ses NaN remplis par la moyenne au train, mais envoyes bruts (NaN) en serving. |
| J1 | Les commerciaux remontent : le modele refuse ~40% des BONS clients. |
| J2 | L'equipe data ne reproduit pas : sur le dump CSV, le modele est toujours a 95% d'AUC. "Ca marche chez moi". |
| J3 | Les metriques techniques (latence, 5xx) sont vertes. Le modele renvoie des scores valides, juste FAUX. |
| J5 | Investigation : on decouvre que la distribution de `revenu_net_mensuel` en prod est decalee d'un facteur ~1.25 (brut vs net) par rapport au train. |
| J6 | Correction : on aligne le calcul des features. Le modele redevient correct. Cout : 6 jours de mauvaises decisions de credit. |

**Questions :**

1. **Root cause analysis** :
   - Reconstitue la chaine causale.
   - Pour chaque feature, identifie la cause exacte du skew. Classe : process, architecture, monitoring.
   - Pourquoi "ca marche sur le CSV" mais pas en prod ?

2. **Le piege "ca marche chez moi"** :
   - Pourquoi l'eval offline ne pouvait PAS attraper ce bug ?
   - Quelle propriete manquait pour que train et serving soient garantis identiques ?

3. **Pourquoi les metriques etaient vertes** :
   - Difference entre une panne d'API classique (500) et une panne de modele (200 avec un mauvais score).
   - Quelle metrique aurait alerte des J0 ? (drift / distribution des features, PSI sur les inputs)

4. **Skew par cause** :
   - Pour `revenu_net_mensuel` (code divergent) : la solution structurelle.
   - Pour `nb_commandes_30j` (NaN handling divergent) : la solution structurelle.
   - En quoi un feature store unique resout LES DEUX d'un coup ?

5. **Systeme corrige** :
   - Concois le pipeline qui aurait rendu ce skew impossible.
   - Concois un test CI qui aurait attrape le decalage AVANT le deploiement (indice : comparer les distributions train vs un echantillon de prod, ou recomputer une feature des deux cotes).

6. **Runbook** :
   - Un runbook de 6 etapes pour "skew suspecte sur un modele en prod".

### Criteres de reussite
- [ ] Chaine reconstituee : 2 features calculees differemment train vs serving -> distributions decalees -> scores faux -> 40% de bons clients refuses -> detection tardive (6 jours)
- [ ] Causes par feature classees : `revenu_net_mensuel` = code divergent (architecture), `nb_commandes_30j` = NaN handling divergent (process), pas de monitoring de distribution (monitoring)
- [ ] "Ca marche sur le CSV" explique : l'eval offline utilise les MEMES donnees biaisees que le train -> elle ne voit pas le serving
- [ ] La propriete manquante : une fonction de calcul de feature UNIQUE et partagee (feature store) garantissant train == serving
- [ ] Metriques vertes expliquees : panne de modele = 200 avec mauvais score (vs 500 d'une panne API) ; alerte attendue = PSI/drift sur la distribution des features en prod vs baseline
- [ ] Solutions structurelles : feature store unique (resout code divergent ET NaN handling d'un coup)
- [ ] Systeme corrige + test CI (comparer distributions train vs echantillon prod, ou recomputer la feature des deux cotes et assert egalite) ; runbook actionnable
