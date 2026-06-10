# Exercices Medium — ML System Design Intro

---

## Exercice 1 : Concevoir le systeme complet de detection de fraude

### Objectif
Assembler tous les composants d'un systeme ML (feature store, training, serving, monitoring) sur un cas a contrainte de latence.

### Consigne
Une fintech doit scorer chaque transaction carte en temps reel : **2 000 transactions/sec** en pic, decision (approve/review/block) en **< 50 ms au p99**.

1. **Features** : classe ces features en online (calculables a la volee) vs offline (precalculees) : montant, devise, distance geographique avec la derniere transaction, nombre de transactions du user dans la derniere heure, moyenne de depenses sur 90 jours, age du compte.
2. **Feature store** : dessine l'architecture online store / offline store. Quelle techno pour chaque (latence requise pour l'online : < 10 ms) ? Comment les deux restent-ils synchronises ?
3. **Latency budget** : decompose les 50 ms (reseau, lookup features, inference, regles metier, logging). L'inference du modele prend 15 ms : quel budget reste-t-il pour le feature lookup ?
4. **Training** : les fraudes representent 0.1% des transactions. Quel probleme pour le training et l'evaluation ? Quelle metrique d'evaluation choisis-tu (accuracy ? precision/recall ? pourquoi) ?
5. **Boucle de feedback** : les labels (fraude confirmee) arrivent avec 30-90 jours de retard (chargebacks). Comment organiser le retraining et que monitorer en attendant les labels ?

### Criteres de reussite
- [ ] Online : montant, devise ; precalcule avec mise a jour streaming : compteur 1h, distance geo ; offline batch : moyenne 90j, age du compte
- [ ] Online store = Redis/DynamoDB (< 10 ms), offline store = data warehouse/parquet, sync par pipeline commun (materialisation)
- [ ] Le budget est decompose et tient dans 50 ms (ex : 5 reseau + 10 features + 15 inference + 5 regles + marge p99)
- [ ] Accuracy rejetee (99.9% en predisant "jamais fraude") ; precision/recall ou PR-AUC retenus, avec le tradeoff cout des faux positifs vs faux negatifs
- [ ] Le monitoring sans labels repose sur le drift des features et de la distribution des scores ; retraining planifie quand les labels arrivent

---

## Exercice 2 : Plan de migration batch vers real-time

### Objectif
Evaluer rigoureusement le passage d'une inference batch a une inference temps reel.

### Consigne
Un site e-commerce calcule ses recommandations produit **chaque nuit en batch** (pour 20M users, stockees dans une table lue par le front). Le product manager veut du temps reel : "les recos doivent refleter ce que le user vient de cliquer il y a 30 secondes".

1. Liste ce que le passage au real-time change concretement sur : le serving (infra), les features (fraicheur), le monitoring, et le cout. Donne un ordre de grandeur du surcout infra (batch 1x/jour vs serving 24/7).
2. Le trafic est de 3 000 req/s en pic sur la page d'accueil. En batch, le cout est ~2h de cluster par nuit. Estime ce qu'il faut en real-time : si une inference prend 20 ms de CPU, combien de cores pour 3 000 req/s (avec utilisation cible 60%) ?
3. Propose l'architecture intermediaire **hybride** (candidats precalcules en batch + re-ranking temps reel avec les events de session). Pourquoi est-ce souvent le meilleur ratio valeur/cout ?
4. Quel test mets-tu en place pour verifier que le temps reel ameliore VRAIMENT le business (metrique, duree, groupe de controle) ?
5. Donne 2 cas ou tu refuserais la migration (signaux qui montrent que le batch suffit).

### Criteres de reussite
- [ ] Les 4 dimensions sont couvertes, avec serving 24/7 identifie comme 5-20x plus cher que le batch nocturne
- [ ] Calcul de capacite : 3000 * 0.02 s = 60 cores a 100%, / 0.6 = 100 cores en cible d'utilisation
- [ ] L'architecture hybride precalcule les candidats (batch) et ne fait en ligne que le re-ranking leger sur la session
- [ ] Le test est un A/B test avec metrique business primaire (CTR ou revenu/session), duree 2-4 semaines
- [ ] Les cas de refus sont plausibles (ex : les recos changent peu en 24h, l'uplift estime < au surcout, pas d'events de session exploitables)

---

## Exercice 3 : Post-mortem d'un training-serving skew

### Objectif
Diagnostiquer un ecart offline/online et concevoir la prevention systematique.

### Consigne
Un modele de churn affiche **AUC 0.91 offline** mais ses predictions en production sont a peine meilleures que le hasard. L'equipe a verifie : c'est bien le meme modele (memes poids) deploye.

Voici des extraits des deux pipelines :

```python
# Training (Spark, batch)
df["days_since_last_login"] = (snapshot_date - df["last_login"]).dt.days
df["avg_session_min"] = df["total_session_min"] / df["n_sessions"]  # n_sessions >= 1 garanti par le job
df["plan"] = df["plan"].fillna("free")

# Serving (Python, API)
features["days_since_last_login"] = (now() - user["last_login"]).days
features["avg_session_min"] = user["total_session_min"] / user.get("n_sessions", 0)
features["plan"] = user["plan"]  # peut etre None
```

1. Identifie au moins 3 sources de skew dans ce code (compare ligne a ligne).
2. Pour chacune, explique l'effet en production (crash ? valeur silencieusement differente ?).
3. `snapshot_date` vs `now()` : explique le probleme plus subtil de cette difference (indice : a quelle heure tournait le batch ? quel decalage de distribution ?).
4. Propose la correction structurelle : comment garantir que training et serving calculent EXACTEMENT les memes features (composant + pratique d'equipe) ?
5. Propose un test automatique qui aurait detecte le probleme avant la prod (compare quoi, sur quelles donnees, avec quel seuil ?).

### Criteres de reussite
- [ ] Les 3 skews sont identifies : division par zero possible (n_sessions=0), fillna absent au serving (None -> crash ou NaN), reference temporelle differente
- [ ] L'effet de chaque skew est decrit (exception ZeroDivisionError, valeur manquante non geree, distribution decalee)
- [ ] Le probleme snapshot vs now est explique : le batch fige a une heure fixe, le serving calcule en continu -> meme user, valeurs differentes
- [ ] La correction passe par un feature store / librairie de features partagee (une seule definition, deux materialisations)
- [ ] Le test propose compare les features online vs offline sur un echantillon (ex : 1000 users, ecart max tolere par feature)
