# Solutions — Module 06 : Pratique deliberee & metacognition

---

## Exercice 1 — Appliquer la technique Feynman sur un concept du domaine

### Corrige modele (concept choisi : storage strength vs retrieval strength)

**Explication initiale — exemple :**

> Quand tu apprends quelque chose, ta memoire a deux dimensions differentes. La premiere, c'est la "force de stockage" : a quel point l'information est solidement ancree dans ta memoire a long terme. La deuxieme, c'est la "force de recuperation" : a quel point il est facile de l'atteindre *maintenant*.
>
> Le probleme, c'est que quand tu etudies, tu ressens surtout la force de recuperation. Si tu viens de lire quelque chose trois fois, tu peux y acceder facilement. Tu as l'impression d'avoir appris. Mais cette facilite d'acces peut disparaitre en 24 heures si tu ne revises pas.
>
> A l'inverse, quand tu espaces tes revisions, les informations semblent moins accessibles a chaque session — tu dois faire plus d'effort pour les retrouver. Mais c'est precisement cet effort qui renforce le stockage. Comme un muscle : si tu ne le sollicites pas, il ne se renforce pas.
>
> La consequence pratique : ne te fie pas a la fluidite pour savoir si tu as appris. Teste-toi. Si tu peux restituer sans aide apres deux jours, l'ancrage est plus reel que si tu peux lire facilement le cours devant toi.

**Detection des trous (exemple) :**

- "tu as l'impression d'avoir appris" → [?] : comment appelle-t-on ce phenomene precis ? (fluency illusion, JOL)
- "comme un muscle" → [?] : c'est une metaphore mais est-ce que le cerveau fonctionne vraiment comme un muscle ? (nuance : la metaphore est pedagogique, pas neurobiologique — a ne pas surdevelopper)

**Apres retour a la source :**
Le phenomene s'appelle la "fluency illusion" ou biais de JOL (Judgement of Learning) — detaille au Module 01. La metaphore du muscle est utile pedagogiquement mais ne doit pas etre prise au pied de la lettre neurobiologique.

**Retour sur la valeur de l'exercice :** meme avec un concept qu'on croit maitriser, l'explication a voix haute revele systematiquement des approximations. C'est la valeur centrale de Feynman.

---

## Exercice 2 — Diagnostiquer une strategie d'apprentissage et l'ajuster

### Corrige modele (exemple : apprendre la programmation Python)

**Description de la situation :**
- Objectif : savoir ecrire des algorithmes de tri et de recherche en Python.
- Strategie utilisee : regarder des tutos YouTube, relire des solutions sur GitHub, parfois copier-coller pour voir si ca tourne.
- Indicateur : "je comprends quand je regarde" et "ca tourne quand je copie".

**Diagnostic :**

L'indicateur "je comprends quand je regarde" est un JOL subjectif base sur la **retrieval strength** (la fluidite pendant la lecture), pas sur la **storage strength** (la capacite a reproduire sans aide). C'est le signe classique de la fluency illusion (Module 01).

La strategie est de la **pratique naive** : regarder et copier n'implique pas d'objectif precis sur une faiblesse, pas de feedback immediat structure, et reste dans la zone de confort (regarder est plus facile que produire).

Il n'y avait ni retrieval practice (jamais essaye de coder sans voir la solution), ni espacement (revisions toujours le meme jour), ni interleaving (un seul type d'algo a la fois).

**Plan de reprise sur 2 semaines :**

| Semaine | Lundi | Mercredi | Vendredi | Mesure |
|---------|-------|----------|----------|--------|
| S1 | Lire binary search (10 min), puis coder de memoire + comparer | Exercice blank-page : sliding window (20 min sans aide) | Mix interleave : 1 exercice binary search + 1 sliding window au choix de l'algo | Score sur 2 exercices sans aide |
| S2 | Revision espacee binary search (J+7) | Nouveau type : two pointers, meme protocole | Mix des 3 types interleaves | Score sur 3 exercices ; feedback : ou est-ce que je bloque encore ? |

Mesure finale : 5 problemes LeetCode easy/medium, types melanges, sans indication du type, en temps limite. Le score objectif remplace le "j'ai l'impression de comprendre".

---

## Exercice 3 — Nuancer le mythe des 10 000 heures

### Corrige modele

**Reponse attendue (exemple complet) :**

---

Tu as raison que le volume de pratique compte — enormement. Sans des milliers d'heures, personne ne devient expert dans un domaine complexe. Mais la regle des "10 000 heures" popularisee par Gladwell simplifie dangereusement la recherche originale d'Ericsson, et voila pourquoi ca risque de te freiner plutot que de t'aider.

Macnamara et al. ont fait une meta-analyse sur 88 etudes (*Psychological Science*, 2014) et mesure ce que la pratique deliberee explique vraiment en performance : **26 % de la variance** dans les jeux, **21 %** en musique, **18 %** en sport. Ca veut dire que le volume de pratique compte — mais qu'il est loin d'expliquer tout. D'autres facteurs (age de debut, qualite de l'enseignement, feedback recu, talent de depart) jouent aussi.

Et surtout, il y a pratique et pratique. Ericsson distingue la **pratique naive** — faire la meme chose encore et encore — de la **pratique deliberee** : cibler une faiblesse precise, recevoir un feedback immediat, travailler au bord de ta zone de confort. 10 000 heures de pratique naive peuvent te laisser stagnant. 1 000 heures de pratique deliberee bien structuree peuvent te faire progresser plus qu'un collegue qui cumule les heures en pilote automatique.

**Deux conseils concrets :**

1. Avant chaque session, definis une faiblesse specifique a travailler (pas "je vais faire de la programmation" mais "je vais travailler la gestion des cas limites dans les algorithmes de tri"). Et note ce que tu peux faire a la fin de la session que tu ne pouvais pas faire au debut.

2. Integre du feedback immediate a chaque cycle : code sans regarder la solution, puis compare et note l'ecart. En programmation, les tests automatiques sont un feedback immediat parfait — ne saute pas cette etape.

L'objectif n'est pas de compter tes heures. C'est de rendre chaque heure aussi deliberee que possible.

---

**Ce que ce corrige illustre :** la nuance ne consiste pas a dire "le talent est tout" ni "les 10 000 heures suffisent". Elle reconnait l'agentivite de la personne tout en donnant une image plus precise du mecanisme — et deux leviers actionnables.
