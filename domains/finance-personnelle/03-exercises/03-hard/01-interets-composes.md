# Exercices — Interets composes et valeur du temps (Module 01)

> **Niveau** : Avance | **Temps estime** : 75-90 min
>
> **Matiere premiere** : Theorie du Module 01 + calculateur `02-code/01-interets-composes.py`
>
> **Disclaimer** : exercices educatifs. Les taux et scenarios sont **illustratifs**, jamais des previsions. Ce contenu est educatif et **ne constitue pas un conseil financier**. Aucun rendement n'est garanti ; tout investissement comporte un risque de perte en capital.

---

## Exercice 1 — Arbitrage reel : prime initiale ou epargne mensuelle ?

### Objectif
Traiter un arbitrage concret et multi-contraintes ou plusieurs leviers de l'interet compose s'opposent, en integrant duree, fiscalite simplifiee et incertitude sur le rendement.

### Consigne

Hugo, 40 ans, recoit une prime exceptionnelle de **15 000 €** et peut par ailleurs degager **250 €/mois** d'epargne. Horizon : **25 ans** (jusqu'a 65 ans). Taux illustratif central : **6 %/an**, capitalisation mensuelle.

Il hesite entre trois options :
- **Option A** : investir les 15 000 € immediatement en une fois (lump sum) + 250 €/mois pendant 25 ans.
- **Option B** : garder les 15 000 € sur un livret a 2,5 % et n'investir que les 250 €/mois pendant 25 ans, "au cas ou".
- **Option C** : etaler les 15 000 € en 30 versements supplementaires de 500 €/mois pendant 30 mois (en plus des 250 €/mois), puis continuer a 250 €/mois jusqu'a 65 ans.

1. Calculez le capital final a 65 ans pour les options A et B (utilisez `capital_final()`). Chiffrez l'ecart en euros : combien coute la "prudence" de l'option B sur 25 ans ?
2. Pour l'option C, estimez le capital final (vous pouvez decomposer : les 30 versements de 500 € places puis laisses croitre, + le flux de 250 €/mois sur 25 ans). Comparez A et C : l'etalement de la prime sur 30 mois est-il couteux ? De combien ?
3. Hugo objecte : "et si le marche chute juste apres que j'ai investi mes 15 000 € ?" Repondez en distinguant ce que la theorie du module permet d'affirmer (effet du temps, horizon long) de ce qu'elle ne permet PAS d'affirmer (timing, rendement garanti). Quel role joue l'horizon de 25 ans dans l'analyse du risque de timing ?
4. Reprenez l'option A mais avec un rendement illustratif **defavorable de 3 %/an** au lieu de 6 %. De combien le capital final chute-t-il ? Que revele cet ecart sur la fragilite des projections a un seul scenario ?

### Criteres de reussite

- [ ] Capitaux finaux A et B calcules ; le cout de la prudence (B) est chiffre en euros
- [ ] L'option C est estimee et comparee a A ; l'effet de l'etalement de la prime est quantifie
- [ ] La reponse a la question 3 distingue clairement effet du temps (legitime) et timing/garantie de rendement (non affirmable) — posture honnete sur la preuve
- [ ] Le scenario defavorable a 3 % est calcule et la fragilite du scenario unique est commentee

---

## Exercice 2 — Construire et critiquer une projection multi-scenarios

### Objectif
Passer d'une projection a point unique a une **fourchette de scenarios**, competence essentielle pour communiquer honnetement une projection long terme sans surpromettre.

### Consigne

Camille investit **0 €** de capital initial et **300 €/mois** pendant **30 ans**. Plutot que d'annoncer "un chiffre", elle veut presenter une fourchette honnete.

1. Calculez le capital final a 30 ans pour trois scenarios de rendement illustratifs : **defavorable 3 %**, **central 6 %**, **favorable 8 %** (capitalisation mensuelle). Presentez-les dans un tableau (scenario / taux / capital final / multiplicateur capital÷verse).
2. Le total verse est identique dans les trois cas. Calculez-le. Quelle part du capital final, en %, provient des versements vs des interets composes dans le scenario central (6 %) ? Et dans le scenario favorable (8 %) ?
3. Un commercial presenterait volontiers uniquement le scenario a 8 %. En 3-4 phrases, expliquez pourquoi presenter une **fourchette** (et non un chiffre unique) est la posture honnete, en mobilisant la notion de "performances passees ne prejugent pas des performances futures".
4. **Sensibilite a la duree** : refaites le calcul du scenario central (6 %) pour 25 ans et 35 ans (au lieu de 30). De combien le capital final varie-t-il par tranche de 5 ans en plus ou en moins ? Pourquoi l'effet n'est-il pas symetrique (gagner 5 ans vs en perdre 5) ?

### Criteres de reussite

- [ ] Les trois scenarios de rendement sont calcules a 30 ans et presentes en tableau
- [ ] La decomposition versements vs interets est chiffree en % pour au moins le scenario central et le favorable
- [ ] L'argumentation sur la fourchette vs chiffre unique est explicite et inclut le disclaimer sur les performances passees
- [ ] La sensibilite a la duree (25/30/35 ans) est calculee et l'asymetrie de l'effet temps est expliquee (croissance exponentielle)
