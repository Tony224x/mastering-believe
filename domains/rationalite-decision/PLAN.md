# PLAN — Rationalite & Decision

> **Curriculum fige.** Ne pas modifier sans raison pedagogique explicite.
> **Posture anti-clivant non negociable** : exemples 100 % neutres, methode > conclusions.

## Posture anti-clivant (rappel explicite)

Ce domaine enseigne **a raisonner, pas quoi penser**. Regles non negociables :

1. **Exemples neutres uniquement** : probabilites, jeux de hasard, paris equitables, meteo, sport, sante publique factuelle (taux de tests, faux positifs), finance personnelle illustrative. Eviter : politique partisane, religion, sujets societaux brulants.
2. **Jamais prendre parti** sur un debat sensible. Si un biais est illustre, la "bonne reponse" est une valeur calculable (probabilite, esperance), pas une opinion.
3. **Presenter les biais sans moraliser** : un biais est un raccourci adaptatif (cf. Gigerenzer), pas un defaut moral. On decrit le mecanisme et le contexte ou il echoue.
4. **Desinformation : enseigner la methode universelle** (SIFT, lateral reading, accuracy nudge). Aucun exemple ne doit cibler un bord politique.
5. **Nuance replication** : ne pas survendre les biais. Signaler quand un effet a mal replique (priming social notamment). Distinguer "robuste" de "fragile".
6. **Symetrie** : tout exercice de detection d'erreur doit aboutir a la meme conclusion quelle que soit la sensibilite de l'apprenant (car elle repose sur des chiffres ou de la logique).

## Curriculum fige — 7 modules

### Module 01 — Le systeme d'exploitation du jugement
**Objectif** : comprendre pourquoi le raisonnement peut derailler et comment le corriger.
**Concepts** : Systeme 1 (rapide, automatique) / Systeme 2 (lent, deliberatif) ; definition d'un biais cognitif ; distinction rationalite / intelligence (Stanovich : QI ne mesure pas la qualite du raisonnement) ; notions de mindware gap et dysrationalia.
**Sources** : Kahneman 2011 (avec nuance replication), Stanovich 2011.
**Exemples** : illusions de jugement sur des probabilites simples (probleme des taxis, probleme de Linda — version neutralisee).

### Module 02 — Probabilites utiles en 45 minutes
**Objectif** : maitriser les 4 outils probabilistes les plus utiles au quotidien.
**Concepts** : frequences naturelles vs probabilites ; probabilite conditionnelle P(A|B) ; taux de base (base rate) ; faux positifs / faux negatifs ; valeur predictive positive.
**Sources** : Peterson 2017 (ch. 4), Gigerenzer (frequences naturelles).
**Exemples** : depistage medical (sensibilite/specificite -> probabilite post-test) ; tests de qualite industrielle.
**Code** : `02-code/02-probabilites-utiles.py`

### Module 03 — Pensee bayesienne et mise a jour
**Objectif** : changer d'avis proportionnellement aux preuves.
**Concepts** : theoreme de Bayes intuitif et formel ; prior / likelihood / posterior ; evidence update ; odds bayesiens.
**Sources** : Peterson 2017, SEP (Stanford Encyclopedia of Philosophy).
**Exemples** : urne de billes, test medical (suite du module 02), "qui a laisse la lumiere allumee ?".
**Code** : `02-code/03-pensee-bayesienne.py`

### Module 04 — Heuristiques et biais (les robustes)
**Objectif** : connaitre les 4 biais les plus solides et le contexte ou ils s'appliquent.
**Concepts** : ancrage et ajustement ; heuristique de disponibilite ; cadrage (framing) ; base rate neglect. **Avec la nuance replication** : effets robustes vs effets fragiles (priming social).
**Sources** : Tversky & Kahneman 1974 (Science), Kahneman 2011 (avec nuance), OSC 2015 (crise de replication).
**Exemples** : estimations numeriques, probleme de Wason neutralise, scenarios de cadrage sur des jeux.
*(Note : ce module est livre par l'Agent B.)*

### Module 05 — Decision sous incertitude
**Objectif** : choisir rationnellement quand le futur est incertain.
**Concepts** : esperance mathematique ; utilite esperee ; aversion au risque ; arbres de decision ; paradoxe d'Allais.
**Sources** : Peterson 2017, SEP (utilite esperee).
**Exemples** : paris equitables, choix d'assurance illustratif, jeux de loterie.
*(Note : ce module est livre par l'Agent B.)*

### Module 06 — Calibration et forecasting
**Objectif** : penser en probabilites et mesurer la qualite de ses previsions.
**Concepts** : calibration (etre "bien calibre") ; score de Brier ; decomposition Fermi ; lecons du Good Judgment Project (Tetlock) ; tutorat entre pairs (teaming).
**Sources** : Tetlock & Gardner 2015, Mellers et al. 2014 (Psychological Science).
**Exemples** : previsions meteo et sportives, journal de previsions calibre.
*(Note : ce module est livre par l'Agent B.)*

### Module 07 — Capstone : La boite a outils de jugement
**Objectif** : assembler un kit de raisonnement personnel et l'appliquer a une decision concrete.
**Livrable de l'apprenant** :
1. Checklist pre-decision (5 questions systematiques)
2. Journal de previsions calibre (format standardise, pret a scorer)
3. Protocole de verification (SIFT applique a un cas reel neutre)
**Sources** : Munger (latticework of mental models), Caulfield 2017 (SIFT), Pennycook & Rand 2021.
*(Note : ce module est livre par l'Agent B.)*

## Sequencement pedagogique

```
01 (systeme) -> 02 (probas) -> 03 (Bayes) -> 04 (biais) -> 05 (decision) -> 06 (calibration) -> 07 (capstone)
```

Chaque module s'appuie sur le precedent : on ne peut pas comprendre Bayes sans les probabilites conditionnelles (02), ni calibrer sans savoir ce qu'est une probabilite (02-03).

## Format de chaque module

- **Theorie** : ~250-360 lignes. H1 `# Module N — Titre` ; bloc de metadonnees ; sections numerotees ; exemple chiffre en premier ; boite `> A retenir` ; 4-5 flash-cards ; Points cles ; Pour aller plus loin.
- **Exercices** : 3 exercices progressifs par module (easy). Chaque exercice : `## Exercice N`, `### Objectif`, `### Consigne`, `### Criteres de reussite` avec cases `- [ ]`.
- **Solutions** : corrige modele chiffre en `.md` (modules 02-03 peuvent renvoyer aux scripts Python).
