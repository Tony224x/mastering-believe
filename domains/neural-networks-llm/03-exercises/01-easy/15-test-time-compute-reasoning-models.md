# Exercices Faciles — Jour 15 : Test-time compute & reasoning models

---

## Exercice 1 : Self-consistency a la main (majority vote)

### Objectif

Comprendre concretement pourquoi generer plusieurs reponses puis voter ameliore l'accuracy, sans entrainer quoi que ce soit.

### Consigne

On simule un "LLM faillible" qui repond a la question `7 * 8` mais se trompe parfois. On a observe les K reponses suivantes pour 3 problemes :

```
Probleme A (vraie reponse 56) : [56, 56, 49, 56, 64]
Probleme B (vraie reponse 56) : [49, 64, 56, 48, 49]
Probleme C (vraie reponse 56) : [56, 14, 56, 56, 56]
```

1. Pour chaque probleme, appliquer le **majority vote** : la reponse la plus frequente gagne. En cas d'egalite, prendre la plus petite valeur.

2. Donner l'accuracy en one-shot (on ne garde que la **premiere** reponse de chaque liste) vs l'accuracy avec majority vote (K=5).

3. Sur le probleme B, le majority vote donne-t-il la bonne reponse ? Pourquoi le vote echoue-t-il ici ?

4. Question conceptuelle : a quelle condition sur le taux d'erreur du modele le majority vote ameliore-t-il l'accuracy ? (Indice : que se passe-t-il si le modele se trompe **plus** souvent qu'il n'a raison ?)

5. **Bonus** : implementer `self_consistency(answers)` en Python avec `collections.Counter` qui reproduit la regle (majorite, tie-break sur la plus petite valeur).

### Criteres de reussite

- [ ] Majority vote : A → 56 (correct), B → 49 (incorrect), C → 56 (correct)
- [ ] Accuracy one-shot = 2/3 (A=56 ok, B=49 faux, C=56 ok) ; accuracy vote = 2/3
- [ ] L'explication sur B : les erreurs sont diverses ET la bonne reponse est minoritaire, donc le vote elit une erreur frequente
- [ ] La condition : le vote n'aide que si la bonne reponse est, en esperance, la plus probable (taux de bonne reponse > chaque mode d'erreur individuel)
- [ ] Bonus : le code reproduit exactement les 3 resultats

---

## Exercice 2 : Calculer l'advantage GRPO d'un groupe

### Objectif

Manipuler la formule centrale de GRPO (`advantage = (r - mean) / std`) qui remplace le critic de PPO par la statistique du groupe.

### Consigne

Un prompt de math a ete echantillonne G=6 fois. Les rewards (1 = reponse correcte, 0 = fausse) sont :

```
rewards = [1, 0, 1, 1, 0, 0]
```

1. Calculer `mean(rewards)` et `std(rewards)` (ecart-type de population, diviseur N et non N-1).

2. Calculer l'advantage de chaque reponse : `advantage_i = (reward_i - mean) / std`.

3. Quelles reponses ont un advantage **positif** ? Lesquelles **negatif** ? Que va faire l'update de policy a chacune (augmenter ou diminuer sa probabilite) ?

4. Cas degenere : si **toutes** les reponses du groupe ont reward = 1 (`rewards = [1,1,1,1,1,1]`), que vaut `std` ? Pourquoi divise-t-on en pratique par `max(std, 1e-6)` ? Que devient l'advantage et donc le signal d'apprentissage de ce groupe ?

5. Question conceptuelle : en quoi GRPO economise-t-il la memoire par rapport a PPO classique ?

### Criteres de reussite

- [ ] mean = 0.5, std = 0.5
- [ ] advantages = [+1, -1, +1, +1, -1, -1]
- [ ] Les reponses correctes (advantage > 0) voient leur proba augmenter, les fausses diminuer
- [ ] Cas degenere : std = 0 → division par ~0, d'ou le clip `max(std, 1e-6)` ; advantage ~0 → ce groupe n'apprend rien (tout le monde pareil, aucun signal relatif)
- [ ] L'explication : GRPO supprime le critic (value network), qui dans PPO est un 2e modele de la taille de la policy → ~2x moins de memoire

---

## Exercice 3 : Router LLM classique vs reasoning model

### Objectif

Implementer la decision la plus rentable d'un produit LLM : router la requete vers le bon modele selon la tache, le cout et la latence.

### Consigne

On dispose de 3 profils (cout par 1k tokens out, latence par 1k tokens out, qualite reasoning, qualite extraction) :

```python
PROFILES = {
    "haiku":  {"cost": 0.004, "lat": 0.2, "q_reason": 0.55, "q_extract": 0.90},
    "sonnet": {"cost": 0.015, "lat": 0.4, "q_reason": 0.72, "q_extract": 0.95},
    "opus_thinking": {"cost": 0.075, "lat": 3.0, "q_reason": 0.94, "q_extract": 0.95},
}
```

1. Ecrire `route(task_type, quality_target, latency_budget_s, output_tokens)` qui :
   - filtre les modeles dont la latence (`lat * output_tokens / 1000`) depasse le budget ;
   - filtre ceux dont la qualite (selon `task_type`) est inferieure a la cible ;
   - parmi les survivants, retourne le **moins cher**.

2. Tester sur ces scenarios :
   - `("extraction", 0.85, 3.0, 500)` → quel modele ?
   - `("reasoning", 0.90, 60.0, 2000)` → quel modele ?
   - `("reasoning", 0.92, 2.0, 1000)` → que se passe-t-il ?

3. Expliquer le 3e cas : pourquoi aucun modele ne convient, et que doit faire l'ingenieur produit a la place (cote UX ou produit) ?

4. Question conceptuelle : pourquoi router vers Haiku une tache d'extraction plutot que vers Opus thinking, meme si Opus a une qualite extraction egale ou superieure ?

### Criteres de reussite

- [ ] `route` filtre par latence PUIS par qualite, puis minimise le cout
- [ ] extraction → haiku (assez bon, le moins cher) ; reasoning 0.90/60s → opus_thinking (seul a atteindre 0.90 en reasoning)
- [ ] Cas 3 : opus_thinking depasse le budget latence (3.0s/1k > 2.0s pour 1000 tokens), les autres n'atteignent pas q_reason=0.92 → None
- [ ] L'explication cas 3 : contraintes incompatibles → changer le produit (UX async, decouper la tache) plutot que chercher un modele magique
- [ ] L'explication routing : a qualite egale, on prend le moins cher/rapide ; reasoning sur de l'extraction est du gaspillage pur (cout x10-20, latence x15)
