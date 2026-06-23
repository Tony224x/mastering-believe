# Solutions — Module 07 : Pratique délibérée

> Référence : `03-exercises/01-easy/07-pratique-deliberee.md`

---

## Solution — Exercice 1 : Classer des pratiques

**Scénario 1 — Léa (tennis, matchs libres)**
**Type : Pratique naive.**
Elle répète une activité dans sa zone de confort (matchs qu'elle gagne souvent), sans objectif précis sur une faiblesse, sans feedback structuré, sans guidance. Le plaisir de gagner masque l'absence de progression ciblée.

**Scénario 2 — Tom (guitare, passage difficile + métronome + enregistrement)**
**Type : Pratique délibérée.**
Les quatre conditions sont réunies : objectif précis (le passage qui coince), feedback immédiat (enregistrement comparé), effort hors zone de confort (60 % du tempo = à la limite), et une méthode structurée qui joue le rôle de guidance experte.

**Scénario 3 — Sofia (maths, relecture sans test)**
**Type : Pratique intentionnelle.**
Il y a un objectif défini (revoir les dérivées), mais sans feedback externe ni test actif. La relecture produit une illusion de compétence (Module 01) — Sofia croit comprendre parce que c'est fluide à relire, pas parce qu'elle peut restituer. Condition manquante principale : feedback immédiat.

**Scénario 4 — Karim (frappe clavier, textes aléatoires sans correction)**
**Type : Pratique naive.**
Aucun des quatre critères n'est rempli : pas d'objectif précis, pas de feedback (il n'observe pas ses erreurs), zone de confort maintenue, pas de guidance. Il consolide probablement ses mauvaises habitudes de frappe.

**Scénario 5 — Inès (escalade, coach avec objectif calibré)**
**Type : Pratique délibérée.**
Exemple le plus complet : objectif précis (bloc à la limite de son niveau), feedback immédiat (correction de posture en temps réel), effort calibré (juste hors zone de confort), guidance experte (coach). C'est le cas modèle d'Ericsson.

---

## Solution — Exercice 2 : Concevoir une session délibérée

*La solution ci-dessous utilise la programmation comme fil rouge. Adapte à ta compétence choisie.*

**1. Faiblesse cible**
Je bute sur la récursivité en Python : je comprends le principe mais je génère systématiquement des erreurs de cas de base, ce qui produit des stack overflows. Faiblesse précise : définir le cas de base avant le cas récursif.

**2. Objectif mesurable**
Résoudre 5 exercices de récursivité de niveau moyen sur LeetCode (factorial, fibonacci, flatten nested list, binary search récursif, sum of digits) en moins de 15 minutes chacun, sans regarder la solution, avec 0 stack overflow.

**3. Format de feedback immédiat**
Soumettre chaque solution sur LeetCode (feedback automatique : passed / failed / stack overflow). Pour chaque échec, lire le message d'erreur avant toute correction, pas simplement relire le code.

**4. Calibration de la difficulté**
J'ai déjà réussi les exercices de récursivité « easy » sans aide. Les exercices « medium » me font échouer 1 fois sur 2 — c'est la bonne zone. Si je réussis les 5 du premier coup, je monte d'un cran ; si j'échoue 4 fois sur 5, je reviens aux easy avec un angle différent.

**5. Durée et récupération**
45 minutes de pratique active (5 exercices × ~9 min). Ensuite : pause de 15 minutes, puis lecture courte (pas de code) sur un autre sujet. Ce soir : sommeil sans écran 30 min avant — la consolidation se fait la nuit.

---

## Solution — Exercice 3 : Analyser la variance Macnamara

**Exemple de réponse honnête et nuancée (~175 mots) :**

L'Affirmation A (« 10 000 heures suffisent ») est trompeuse à deux titres. D'abord, Ericsson n'a jamais énoncé de règle universelle : il a observé une moyenne sur un groupe précis dans un contexte précis. Ensuite, et surtout, elle confond volume et qualité : 10 000 heures de pratique naive peuvent laisser quelqu'un médiocre. Ce n'est pas le compteur d'heures qui construit l'expertise, c'est la pratique *délibérée*.

L'Affirmation B (« le talent seul décide ») est également fausse. Macnamara et al. (2014) montrent que la pratique délibérée explique 26 % de la variance en performance dans les jeux, 21 % en musique, 18 % en sport — des parts non négligeables, largement actionnables.

La formulation honnête : *la pratique délibérée est le levier le plus puissant sur lequel tu peux agir pour progresser dans un domaine — mais elle ne rend pas compte de tout. D'autres facteurs (capacités initiales, âge de début, accès à des experts) jouent aussi. Compte sur la qualité de ta pratique, pas sur ton compteur d'heures.*
