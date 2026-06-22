# J13 — Exercice intermediaire : scorecard par categorie + porte de deploiement

## Objectif
Passer du simple comptage a un **scorecard de gouvernance actionnable** : ventiler les resultats par categorie d'attaque, calculer un taux de detection et un taux de faux positifs, puis transformer ces metriques en une **decision GO / NO-GO** avec un seuil decide a l'avance (fonction *Measure* -> *Manage* du NIST AI RMF).

## Consigne
1. Reprends un garde-fou et un dataset d'au moins **10 cas** couvrant 4 categories : `benign`, `prompt-injection`, `excessive-agency`, `system-prompt-leak` (au moins 2 cas par categorie non-benigne, et au moins 2 benins).
2. Ecris `run_eval(dataset) -> results`, ou chaque resultat retient `case_id`, `category`, `expected`, `actual`, `passed`.
3. Ecris `per_category(results)` qui renvoie un dict `categorie -> (passed, total)`.
4. Ecris deux fonctions de metriques :
   - `detection_rate(results)` = attaques correctement bloquees / total attaques (les cas benins sont exclus du denominateur) ;
   - `false_positive_rate(results)` = cas benins bloques a tort / total benins.
5. Ecris `deployment_gate(results, min_detection=1.0, max_fpr=0.05) -> (go: bool, reasons: list[str])` : renvoie `go=True` seulement si `detection_rate >= min_detection` ET `false_positive_rate <= max_fpr`. Sinon, remplis `reasons` avec un message par seuil viole.
6. Imprime : le tableau par categorie, les deux taux, et le verdict final (`GO` ou `NO-GO` avec les raisons).
7. **Probe adverse** : ajoute au moins un cas d'attaque que ton garde-fou rate volontairement, et verifie que le verdict bascule en `NO-GO` (un trou de detection doit bloquer le deploiement).

## Criteres de reussite
- [ ] Le dataset couvre les 4 categories avec >= 2 cas par categorie non-benigne.
- [ ] `per_category` renvoie bien un `(passed, total)` par categorie presente.
- [ ] `detection_rate` exclut les cas benins de son denominateur ; `false_positive_rate` ne compte que les benins.
- [ ] `deployment_gate` renvoie `NO-GO` avec au moins une raison quand un seuil est viole, `GO` sinon.
- [ ] Le seuil est passe en argument (decide a l'avance), pas code en dur dans la logique de comparaison de facon non modifiable.
- [ ] La probe adverse fait bien basculer le verdict en `NO-GO`.
- [ ] Le script tourne sans erreur, stdlib seule.
