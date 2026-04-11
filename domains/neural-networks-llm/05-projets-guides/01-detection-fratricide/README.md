# Projet 01 — Detection de fratricide (friendly fire)

## Contexte metier

Dans une operation de haute intensite, il arrive que des unites amies se tirent dessus : confusion d'identification, erreur de marquage, tir sans coordination. C'est le **fratricide** (ou "friendly fire" / "tir fratricide"). Sur les traces SWORD, c'est un des elements les plus scrutes par les formateurs en debriefing — et un des plus difficiles a **detecter automatiquement**, parce que :

- Les unites qui ouvrent le feu ne savent **pas toujours** qu'elles tirent sur des amis (fog of war)
- Les events `FIRE` ne contiennent pas de champ `is_fratricide`, il faut le deduire du contexte
- Les traces contiennent des **milliers** de tirs par exercice, impossible d'annoter a la main

Un outil qui scanne automatiquement les traces AAR et flag les suspicions de fratricide economise des heures aux formateurs et permet des statistiques fiables (pourcentage fratricide par exercice, par type de manoeuvre, par unite).

C'est un probleme classique de **classification binaire avec forte desequilibre de classes** (typiquement 2-5% de fratricides dans les traces reelles).

## Objectif technique

Entrainer un classifieur binaire qui, pour chaque event `FIRE`, predit si c'est un fratricide (`label=1`) ou un tir conforme (`label=0`). Entree : features contextuelles construites a partir des events autour du tir. Sortie : probabilite de fratricide.

Cle pedagogique du projet : **accuracy n'est PAS la bonne metrique**. Un modele qui predit toujours "pas fratricide" atteint 97% d'accuracy et n'attrape aucun cas interessant. Il faut regarder **precision, recall, F1** et la **courbe PR**.

## Dataset synthetique

Generateur `solution/generate_dataset.py` qui simule des exercices avec ~3% de fratricides. Pour chaque event `FIRE`, on calcule les features suivantes (inspirees de ce qu'un analyste AAR regarderait a la main) :

| Feature | Signification |
|---|---|
| `friendly_units_within_500m` | nb d'unites amies dans la zone de tir (candidats fratricides) |
| `enemy_units_detected_last_60s` | nb de cibles ennemies detectees recemment par l'unite tirante |
| `time_since_last_blufor_detect_s` | derniere fois qu'un ami a ete observe dans la zone |
| `target_confidence` | confiance de la detection (0 = pas de detect, 1 = ferme) |
| `drone_marked_target` | bool, cible marquee par un drone |
| `fire_mode` | encoded : 0=ordered, 1=reactive, 2=preemptive (preemptif plus risque) |
| `fog_of_war_index` | 0-1, qualite de l'intel sur la zone au moment du tir |
| `night_op` | bool, operation de nuit (plus de risque) |
| `minutes_in_exercise` | instant dans l'exercice (fin = pression, fatigue) |

Format des events respecte le **schema canonique** decrit dans `shared/masa-context.md`. Ici on lit directement les features pre-calculees — en prod, on les extrairait du log d'events via un pipeline SD 02.

## Consigne

```python
def load_dataset(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Retourne (X, y) ou X est (n, 9) et y est (n,) binaire."""

def train_baseline_logistic(X_train, y_train):
    """Baseline : regression logistique avec class_weight='balanced'."""

def train_mlp(X_train, y_train, X_val, y_val):
    """MLP avec loss ponderee pour compenser le desequilibre."""

def evaluate(model, X_test, y_test, threshold: float = 0.5) -> dict:
    """Retourne accuracy, precision, recall, F1, AUC-PR, matrice confusion."""
```

## Etapes guidees

1. **Genere le dataset** — 5000 tirs dont ~3% de fratricides. Split 70/15/15 stratifie (important : sinon une split peut se retrouver sans positifs).
2. **Baseline majoritaire** — predit toujours "pas fratricide". Mesure l'accuracy pour bien voir pourquoi elle est trompeuse (97% sans rien faire).
3. **Regression logistique ponderee** — `class_weight='balanced'` compense automatiquement le desequilibre. Benchmark a battre.
4. **MLP** — 2 couches cachees (32/16 neurones), dropout 0.3. Loss `BCEWithLogitsLoss(pos_weight=...)` avec un `pos_weight` eleve (approximativement `n_neg / n_pos`).
5. **Threshold tuning** — par defaut on decide avec `proba > 0.5`, mais avec un dataset desequilibre c'est souvent trop conservateur. Recherche le seuil qui maximise F1 sur le val set.
6. **Metriques** — NE PAS se contenter de l'accuracy. Rapporte precision, recall, F1 sur la classe positive, et trace une courbe precision-recall.
7. **Matrice de confusion** — identifie les faux negatifs (fratricide manque, cas le plus grave) et les faux positifs (flag inutile qui pollue l'AAR).

## Criteres de reussite

- Recall classe positive > **0.70** (on rate moins de 30% des fratricides)
- Precision classe positive > **0.35** (les flags doivent etre revus par un humain de toute facon)
- F1 classe positive > **0.45** (baseline majoritaire donne 0)
- AUC-PR > **0.50** (baseline majoritaire = prevalence, soit ~0.03)
- Accuracy n'est PAS un critere — ne la regarde qu'en debug

## Points didactiques

### Pourquoi accuracy ment

Avec 3% de prevalence, un modele "tout negatif" fait 97% d'accuracy et rate **100%** des cas interessants. Utilise precision, recall, F1, et AUC-PR sur la classe positive.

### Pourquoi `class_weight` / `pos_weight`

Sans compensation, le modele converge vers "tout negatif" parce que la loss est dominee par les 97% de negatifs. On force l'optimizer a payer plus cher les faux negatifs. Typiquement `pos_weight ~= n_neg / n_pos` ~= 32 pour 3% de positifs.

### Pourquoi le seuil n'est pas 0.5

0.5 est optimal seulement si cout(faux positif) = cout(faux negatif), ce qui est rarement vrai. En fratricide, un faux negatif (rater un vrai cas) est plus grave qu'un faux positif (humain doit re-verifier). Donc on baisse le seuil pour favoriser la recall.

### Explicabilite

Pour un vrai deploiement MASA, il faut justifier **pourquoi** le modele a flag un tir. Une regression logistique donne les coefficients directement. Pour le MLP, utilise SHAP ou des permutation importance. La solution n'impose pas SHAP mais le discute dans les extensions.

## Questions de revue

- Un formateur te dit : "j'ai un modele a 97% d'accuracy, donc il marche". Qu'est-ce que tu lui reponds ?
- Comment tu gererais un dataset avec 0.1% de prevalence (tres rare) ?
- Le modele flag 10% des tirs comme fratricide possible. Le formateur doit revoir 10k tirs / exercice. Inacceptable. Comment tu redresses ?
- Un nouveau theatre d'operation introduit des ROE differentes. Ton modele reste-t-il valide ? Comment tu le sais ?
- Comment tu valides que le modele n'a pas appris un artefact du generateur (ex : "night_op=True implique fratricide" dans le dataset synthetique) ?

## Solution

Voir `solution/generate_dataset.py` et `solution/train.py` pour la correction complete : baseline logistique, MLP pondere, threshold tuning et eval.

## Pour aller plus loin

- **SHAP values** pour expliquer chaque prediction positive (quelles features ont poussees vers "fratricide")
- **Calibration** avec Platt scaling ou isotonic regression pour que `proba=0.7` veuille vraiment dire "70% de chances"
- **Apprentissage sequentiel** — remplacer les features pre-calculees par un LSTM/Transformer qui lit les events bruts
- **Active learning** — demander a un humain de labeliser les events ou le modele est incertain, iterer
- **Cross-exercise generalization** — entrainer sur exercices 1-10, tester sur 11-12 pour verifier qu'on n'overfit pas un scenario particulier
