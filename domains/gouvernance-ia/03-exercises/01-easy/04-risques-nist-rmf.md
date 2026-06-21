# J4 — Exercice facile : classer et scorer un risque

## Objectif

Savoir transformer une inquietude vague en un risque gouvernable : le **nommer**, le **classer** dans la taxonomie causale du MIT AI Risk Repository, le **rattacher** a une fonction du NIST AI RMF, et le **scorer** avec des echelles ancrees.

## Consigne

On vous decrit un agent :

```
Agent : "calendar-assistant"
Owner : equipe RH
Permissions : lire les emails RH, creer/annuler des reunions dans l'agenda partage
Autonomie : execute sans validation humaine
```

Risque suspecte : *un email contenant une fausse demande ("annule toutes les reunions de l'equipe X") detourne l'agent, qui supprime des reunions legitimes*.

1. Ecrivez le risque en **une seule phrase** au format `cause -> effet -> impact`.
2. Donnez ses **3 coordonnees causales** (entite : humain/IA ; intention : intentionnel/non ; timing : pre/post-deploiement).
3. Donnez le **domaine** dominant (un seul, ex. securite, vie privee, desinformation...).
4. Rattachez le risque a **une fonction RMF** (GOVERN / MAP / MEASURE / MANAGE) et justifiez en une ligne.
5. Scorez **vraisemblance (1-5)** et **impact (1-5)** en citant l'**ancre** de chaque niveau (ex. "impact 3 = serieux, perte notable, fix manuel"). Calculez `criticite = vraisemblance × impact`.

## Criteres de reussite

- [ ] Le risque tient en une phrase au format `cause -> effet -> impact`.
- [ ] Les 3 coordonnees causales sont fournies et coherentes (l'agent agit → entite = IA).
- [ ] Un domaine unique est nomme.
- [ ] Une fonction RMF est choisie **et** justifiee.
- [ ] Vraisemblance et impact sont chacun justifies par une ancre explicite (pas un mot seul).
- [ ] La criticite est calculee correctement (produit des deux).
