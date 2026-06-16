# Rationalite & Decision — Le systeme d'exploitation du jugement

## Scope

Ce domaine enseigne **comment raisonner**, pas quoi penser. C'est le "systeme d'exploitation" du jugement : un petit noyau de methodes reproductibles (estimer des probabilites, mettre a jour ses croyances avec les preuves, se calibrer, decider sous incertitude, verifier une information) applicables a n'importe quel sujet de vie ou de travail.

**Principe directeur : methode > conclusions. Exemples 100 % neutres.**
Tous les exemples utilisent des terrains neutres : probabilites, jeux, meteo, sport, depistage medical factuel, finance personnelle illustrative. Aucun sujet politique, religieux ou societalement clivant ne sera utilise comme support pedagogique.

**Pareto-first** : les 20 % de concepts couvrant 80 % des gains de qualite de jugement en quotidien sont prioritaires : probabilites conditionnelles, pensee bayesienne, calibration, esperance/risque, verification de l'information.

**Honnetete intellectuelle** : une partie de la litterature des biais cognitifs a mal replique apres 2011 (crise de la replication). Ce domaine enseigne les effets robustes *et* la prudence epistemique sur les effets fragiles.

## Prerequis

Aucun. Maths de niveau college suffisants (fractions, pourcentages, produit en croix). Aucune connaissance en psychologie ou statistiques requise.

## Planning (7 modules, ~45 min chacun)

| Module | Titre | Temps | Focus |
|--------|-------|-------|-------|
| 01 | Le systeme d'exploitation du jugement | 45 min | Systeme 1 / Systeme 2, definition d'un biais, rationalite vs intelligence |
| 02 | Probabilites utiles en 45 minutes | 45 min | Frequences, probabilite conditionnelle, taux de base, faux positifs |
| 03 | Pensee bayesienne et mise a jour | 45 min | Theoreme de Bayes intuitif + calcul, priors/posteriors |
| 04 | Heuristiques et biais (les robustes) | 45 min | Ancrage, disponibilite, cadrage, base rate neglect + nuance replication |
| 05 | Decision sous incertitude | 45 min | Esperance, utilite, aversion au risque, arbres de decision |
| 06 | Calibration et forecasting | 45 min | Score de Brier, lecons des superforecasters, journal de previsions |
| 07 | Capstone — La boite a outils de jugement | 45 min | Checklist pre-decision + journal calibre + protocole de verification |

## Structure du contenu

- `01-theory/` — 7 modules theoriques (source-of-truth)
- `02-code/` — scripts Python autonomes (taux de base, mise a jour bayesienne)
- `03-exercises/` — exercices progressifs easy/medium/hard avec solutions
- `04-projects/` — mini-projets libres lies au domaine

## Capstone (Module 07)

Construction d'une **boite a outils de jugement personnelle** :
1. **Checklist pre-decision** : 5 questions systematiques avant toute decision importante
2. **Journal de previsions calibre** : format standardise pour enregistrer et scorer ses predictions
3. **Protocole de verification** : application de SIFT + lecture laterale a un cas concret

Le capstone est intentionnellement un outil reutilisable, pas un probleme a resoudre une fois.

## Criteres de reussite

- [ ] Distinguer probabilite conditionnelle et probabilite conjointe sans hesiter
- [ ] Calculer un taux de vrais positifs a partir de la sensibilite/specificite d'un test
- [ ] Effectuer une mise a jour bayesienne simple (prior -> likelihood -> posterior)
- [ ] Identifier les 4 biais les plus robustes (ancrage, disponibilite, cadrage, base rate neglect) et le contexte ou ils s'appliquent
- [ ] Donner une prevision chiffree (probabilite en %) plutot qu'une prediction vague
- [ ] Appliquer SIFT pour verifier une source en moins de 5 minutes
- [ ] Tenir un journal de previsions pendant au moins 2 semaines et calculer son score de Brier

## Sources de reference

Voir `REFERENCES.md` pour les sources tier-1 par module (Tversky & Kahneman, Tetlock/GJP, Gigerenzer, Stanovich, Pennycook, Caulfield/SIFT).
