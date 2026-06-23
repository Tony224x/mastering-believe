# Solutions — Module 12 : Acquisition d'une compétence

---

## Exercice 1 — Décomposer une compétence en sous-compétences et identifier les goulots d'étranglement

### Corrigé modèle (compétence choisie : écrire des algorithmes de graphes en Python)

**Étape 1 — Sous-compétences listées :**

1. Représenter un graphe en mémoire (liste d'adjacence, matrice)
2. Implémenter un BFS (parcours en largeur)
3. Implémenter un DFS (parcours en profondeur, itératif et récursif)
4. Détecter un cycle dans un graphe non orienté
5. Détecter un cycle dans un graphe orienté (DFS + coloration)
6. Appliquer un tri topologique
7. Implémenter Dijkstra (chemin le plus court)
8. Lire un énoncé LeetCode type "graphe" et identifier quel algorithme appliquer

**Étape 2 — Stades diagnostiqués :**

| Sous-compétence | Stade | Justification courte |
|---|---|---|
| Représenter un graphe | [Aut] | Je le fais sans y penser |
| BFS | [A] | Je l'implémente souvent, mais j'oublie encore parfois la gestion de la file |
| DFS récursif | [A] | Ça marche, mais je confonds parfois l'ordre des appels |
| DFS itératif | [C] | Je dois relire à chaque fois — j'ai du mal à visualiser la pile |
| Détection de cycle (non orienté) | [C] | J'ai besoin d'un exemple à portée de main |
| Détection de cycle (orienté) | [C] | Je ne sais pas encore le faire seul |
| Tri topologique | [C] | Lu une fois, jamais produit sans aide |
| Identifier l'algo sur un énoncé | [C] | C'est souvent là que je bloque en premier |

**Étape 3 — Goulot d'étranglement :**

**"Identifier l'algo sur un énoncé"** est le goulot prioritaire. Si je ne reconnais pas qu'un problème est un problème de cycle, de chemin le plus court, ou de composantes connexes, je n'active même pas la sous-compétence correspondante — toutes les autres restent inutilisables. C'est un prérequis de lecture qui conditionne tout le reste.

**Étape 4 — Drill ciblé (stade cognitif) :**

**Drill "reconnaissance d'énoncé"** :
- Prendre 15 énoncés LeetCode type graphe (catégorie Graph, difficulty Easy/Medium) et, pour chacun, **avant d'écrire une seule ligne de code**, formuler à voix haute : "Ce problème est un [BFS / DFS / détection de cycle / tri topologique / Dijkstra] parce que [raison]."
- Feedback : comparer la réponse avec le tag officiel LeetCode et les solutions les plus upvotées. Comptabiliser le score de reconnaissance sur 15.
- Critère de passage au stade associatif : 11/15 ou plus deux jours de suite.

---

### Ce que cet exercice illustre

La décomposition révèle que la plupart des blocages ne sont pas où on le croit. L'étudiant pense qu'il "est mauvais en Dijkstra" — en fait il est bloqué dès la reconnaissance d'énoncé, ce qui rend Dijkstra inaccessible même si la syntaxe est connue. Travailler Dijkstra en priorité serait une erreur de diagnostic.

---

## Exercice 2 — Calibrer le feedback selon le stade

### Corrigé

**Scénario A — Lucas, code review (400 lignes, feedback "bien ou pas bien" en fin de semaine)**

- **Stade :** Cognitif. C'est sa première vraie code review.
- **Évaluation :** La stratégie est mal calibrée. Un feedback hebdomadaire binaire ("bien ou pas bien") est beaucoup trop espacé et trop peu informatif pour un apprenant en stade cognitif. Il ne peut pas relier le verdict à des actions spécifiques dans la review, et une semaine sans feedback laisse les erreurs se fossiliser.
- **Alternative :** Commencer par des worked examples : lire 3 reviews annotées par un senior, comprendre pourquoi chaque commentaire a été laissé. Puis faire une review encadrée sur un PR de 50 lignes avec feedback immédiat après (le manager commente les mêmes lignes que Lucas a commentées ou non). Introduire progressivement le volume et l'autonomie (fading).

---

**Scénario B — Amara, espagnol B1, corrections interrompantes à chaque phrase**

- **Stade :** Associatif à autonome sur les bases. B1 correspond à une fluidité partielle — les grandes erreurs ont disparu, les subtilités restent.
- **Évaluation :** Le feedback est mal calibré — trop fréquent et interruptif pour ce stade. En stade associatif/autonome, les interruptions constantes brisent le flux de la production et peuvent créer de l'anxiété, ce qui nuit à la fluence. De plus, corriger *toutes* les fautes dilue l'attention sur les erreurs vraiment fossilisées.
- **Alternative :** Cibler les corrections sur 2-3 points prioritaires (ex. le subjonctif, les prépositions de lieu). Laisser Amara finir ses phrases, noter les erreurs, et faire un debrief ciblé à la fin de chaque échange de 5 min. Espacer le feedback sur les formes déjà en cours d'automatisation.

---

**Scénario C — Thomas, débogage logique en Python, drill quotidien avec auto-correction**

- **Stade :** Cognitif sur les erreurs logiques silencieuses (il est autonome sur les erreurs de type, mais en stade cognitif sur ce nouveau type).
- **Évaluation :** La stratégie est bien calibrée. Résoudre des problèmes nouveaux, puis s'auto-corriger en comparant à une solution de référence = feedback immédiat et explicatif. Le volume quotidien (3 problèmes) est raisonnable. L'auto-confrontation à la solution permet à Thomas de détecter non seulement "faux/juste" mais "pourquoi c'est différent".
- **Ajustement possible :** À mesure que Thomas progresse vers le stade associatif, introduire de l'interleaving (mélanger erreurs logiques et autres types) pour consolider le transfert et éviter la dépendance au contexte.

---

**Scénario D — Isabelle, piano, 3 semaines, 20 répétitions par jour sans référence**

- **Stade :** Cognitif. 3 semaines de piano = stade cognitif sur pratiquement tout.
- **Évaluation :** La stratégie est mal calibrée. Répéter 20 fois sans feedback revient à de la pratique naive : Isabelle consolide les erreurs autant que les bons gestes. Sans enregistrement ni correction, elle ne peut pas détecter ce qui est faux — elle n'a pas encore les représentations mentales pour s'auto-corriger.
- **Alternative :** (1) S'enregistrer une fois par session et écouter l'enregistrement en suivant la partition. (2) Identifier un passage précis qui pose problème, le travailler en boucle lente avec feedback immédiat (écoute après chaque répétition), puis réintégrer le morceau complet. (3) Si possible, trouver un prof ou un pair pour un feedback hebdomadaire.

---

### Ce que cet exercice illustre

La même technique (répétition, feedback) n'a pas la même valeur selon le stade. Un feedback trop espacé en stade cognitif laisse l'apprenant sans boussole ; un feedback trop fréquent en stade autonome crée une dépendance. Le diagnostic de stade précède toujours le choix de la méthode.

---

## Exercice 3 — Concevoir une progression en 3 stades sur 3 semaines

### Corrigé modèle (compétence : écrire des algorithmes de graphes en Python — suite de l'Ex. 1)

---

**Semaine 1 — Stade cognitif**

*Objectif : comprendre la structure des principaux algorithmes de graphes à partir d'exemples résolus.*

- **Worked examples :** Étudier 6 algorithmes (BFS, DFS récursif, DFS itératif, cycle non orienté, cycle orienté, tri topologique) sur des implémentations annotées ligne par ligne. Pour chaque exemple : (1) lire le code sans l'exécuter, (2) annoter en français ce que fait chaque bloc, (3) fermer l'exemple et essayer de réécrire de mémoire (blank-page recall — Module 02).
- **Sous-compétence prioritaire :** reconnaissance d'énoncé (goulot identifié en Ex. 1) — drill "labellisation" sur 15 énoncés LeetCode.
- **Feedback :** immédiat après chaque blank-page recall (comparer à l'original). Fréquence élevée : feedback à chaque exercice, pas à la fin de la semaine.
- **Mesure :** score de reconnaissance d'énoncé sur 15 (cible : 11/15 avant fin S1) + capacité à réécrire BFS et DFS récursif sans aide.

---

**Semaine 2 — Stade associatif**

*Objectif : consolider les algorithmes courants, réduire la variabilité, commencer à transférer.*

- **Drills ciblés :** 4 problèmes par session — 2 sur la sous-compétence instable principale (DFS itératif + cycle orienté) + 2 sur les algorithmes déjà plus stables (BFS, DFS récursif) en interleaving. Pas de worked examples systématiques : produire d'abord, puis comparer.
- **Fading :** S1 = exemple complet disponible → S2 = "squelette" (structure du code sans le corps des boucles) disponible → milieu S2 = problème nu sans aide.
- **Interleaving :** introduit dès le début de S2 sur les sous-compétences en [A]. Pas encore sur les sous-compétences encore en [C] (cycle orienté, tri topologique) — l'interleaving prématuré surchargerait la mémoire de travail.
- **Mesure :** résoudre 5 problèmes mixtes sans aide en moins de 20 min chacun (LeetCode Easy/Medium) avec un taux de réussite ≥ 70 %.

---

**Semaine 3 — Autonome + nouveau niveau**

*Objectif : maintenir l'acquis, mesurer l'automatisation, attaquer Dijkstra.*

- **Pratique espacée (Module 03) :** réviser 2 fois dans la semaine les algorithmes acquis (BFS, DFS, cycle) sur de nouveaux problèmes jamais vus. Cible : les résoudre en moins de 10 min sans consulter de référence.
- **Nouveau défi — Dijkstra :** retour au stade cognitif. Même protocole qu'en S1 : worked examples annotés de Dijkstra (3 variantes : graphe non pondéré, pondéré, chemin le plus court multi-source). Blank-page recall après chaque exemple.
- **Mesure d'autonomie :** session de 90 min avec 8 problèmes mixtes (BFS + DFS + cycle + tri topologique), tirés au sort, sans indication du type, sans aide. Score objectif : ≥ 6/8 résolus correctement.

---

### Ce que ce plan illustre

La progression en trois semaines suit exactement le cadre Fitts & Posner : on ne passe pas à la production en S1, on n'utilise plus les worked examples en S2, on ne cherche plus le confort en S3. La mesure objective à chaque stade (pas un sentiment, mais un score ou un temps) est ce qui permet de décider du passage à l'étape suivante — ou de rester en S2 si l'objectif n'est pas atteint.

**Erreur classique à éviter :** commencer à travailler Dijkstra en S2, avant d'avoir consolidé les algorithmes précédents. La tentation est forte — Dijkstra est "plus intéressant" — mais la décomposition révèle qu'il nécessite BFS et DFS en stade associatif au minimum pour ne pas surcharger.
