# Exercices — Interets composes et valeur du temps (Module 01)

> **Niveau** : Intermediaire | **Temps estime** : 60-75 min
>
> **Matiere premiere** : Theorie du Module 01 + calculateur `02-code/01-interets-composes.py`
>
> **Disclaimer** : exercices educatifs. Les taux utilises sont **illustratifs** et ne constituent **aucune promesse de rendement**. Ce contenu est educatif et **ne remplace pas un conseil financier personnalise**.

---

## Exercice 1 — Rendement nominal, rendement reel et le piege de l'inflation

### Objectif
Aller au-dela du rendement brut affiche : comprendre que c'est le **rendement reel** (net d'inflation) qui mesure le pouvoir d'achat reellement gagne, et savoir le calculer correctement.

### Consigne

Un placement affiche **6 %** de rendement nominal annuel (illustratif). L'inflation est de **3 %/an** sur la periode.

1. Beaucoup de gens calculent le rendement reel par simple soustraction : 6 % - 3 % = 3 %. Calculez le rendement reel **exact** avec la formule de Fisher : `r_reel = (1 + r_nominal) / (1 + inflation) - 1`. De combien l'approximation par soustraction s'ecarte-t-elle ?
2. Vous placez **10 000 €** pendant **30 ans**. Calculez le capital final **nominal** (a 6 %) puis le capital final **en pouvoir d'achat d'aujourd'hui** (au taux reel exact). Commentez l'ecart.
3. Un livret affiche 2 %/an alors que l'inflation est de 3 %/an. Quel est le rendement reel ? Que se passe-t-il pour le pouvoir d'achat de 10 000 € laisses 10 ans sur ce livret ?
4. En une phrase : pourquoi les projections "illustratives" du module utilisent-elles parfois un rendement reel (ex. 5 %) plutot qu'un rendement nominal (ex. 7-8 %) ?

### Criteres de reussite

- [ ] Le rendement reel exact (formule de Fisher) est calcule, et l'ecart avec la soustraction est chiffre
- [ ] Le capital final nominal et le capital en pouvoir d'achat constant sont tous deux calcules sur 30 ans
- [ ] Le cas du livret a rendement reel negatif est correctement interprete (perte de pouvoir d'achat)
- [ ] La reponse a la question 4 relie le choix du taux a la coherence avec l'inflation

---

## Exercice 2 — Versement initial vs regularite vs duree : isoler chaque levier

### Objectif
Decomposer les trois leviers de l'interet compose (capital initial, versement mensuel, duree) et mesurer lequel domine selon le contexte, en s'appuyant sur le calculateur.

### Consigne

On compare trois epargnants sur un horizon final commun, taux illustratif **7 %/an**, capitalisation mensuelle (utilisez `capital_final()` du script du Module 01) :

- **Karim** : 20 000 € de capital initial, **0 €/mois**, pendant 25 ans.
- **Lina** : 0 € de capital initial, **150 €/mois**, pendant 25 ans.
- **Sofia** : 0 € de capital initial, **150 €/mois**, mais pendant **30 ans** (elle a commence 5 ans plus tot).

1. Calculez le capital final et le total reellement verse pour chacun des trois.
2. Karim et Lina versent-ils la meme chose au total ? Qui finit avec le plus gros capital, et pourquoi ?
3. Comparez Lina et Sofia : a versement mensuel identique, quel est le gain apporte par les 5 annees supplementaires (en euros et en %) ? Combien Sofia a-t-elle verse de plus que Lina pour ce gain ?
4. Construisez un mini-tableau (epargnant / total verse / capital final / multiplicateur capital÷verse) et tirez-en la hierarchie des leviers.

### Criteres de reussite

- [ ] Les trois capitaux finaux et totaux verses sont calcules (verification possible via le script)
- [ ] La comparaison Karim/Lina est correcte et expliquee (role du temps de croissance du capital initial vs etalement des versements)
- [ ] Le gain des 5 ans de Sofia est chiffre en euros ET en pourcentage, et compare au surplus verse
- [ ] Le tableau est complet et la hierarchie des leviers est argumentee

---

## Exercice 3 — Du capital cible au versement requis (inversion de la formule)

### Objectif
Renverser le raisonnement : partir d'un objectif chiffre et trouver le versement mensuel necessaire, competence cle pour fixer un plan d'epargne realiste.

### Consigne

Noemie, 35 ans, veut disposer de **200 000 €** a 65 ans (30 ans d'horizon). Elle part de **0 €**. Taux illustratif : **6 %/an**.

1. A partir de la formule de la valeur future d'une annuite (`A = M × [((1+i)^N - 1) / i]` avec `i` le taux mensuel et `N` le nombre de mois), exprimez `M` (le versement mensuel requis) et calculez-le. *Indice : `i = 0,06/12 = 0,005`, `N = 360`.*
2. Verifiez votre resultat en injectant ce `M` dans `capital_final(0, 0.06, 30, M)` : retrouvez-vous environ 200 000 € ?
3. Refaites le calcul du versement requis si Noemie ne dispose que de **20 ans** (elle commence a 45 ans). De combien le versement mensuel doit-il augmenter ? Exprimez ce surcout en %.
4. Refaites le calcul du versement requis sur 30 ans mais avec un taux illustratif plus prudent de **4 %/an**. Que conclure sur la sensibilite du plan a l'hypothese de rendement ?

### Criteres de reussite

- [ ] La formule est correctement inversee et le versement mensuel sur 30 ans a 6 % est calcule (tolerance ±5 €)
- [ ] La verification via le script confirme l'ordre de grandeur (~200 000 €)
- [ ] L'effet de la reduction d'horizon (20 ans) sur le versement requis est chiffre en euros et en %
- [ ] L'effet du passage de 6 % a 4 % est calcule et la conclusion sur la prudence des hypotheses est formulee
