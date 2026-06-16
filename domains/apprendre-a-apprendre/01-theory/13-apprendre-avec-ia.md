# Module 13 — Apprendre avec l'IA

> **Temps estimé** : 45 min | **Prérequis** : Modules 02 (retrieval practice), 03 (spaced repetition), 08 (métacognition), 09 (mesurer son apprentissage)
> **Objectif** : Comprendre comment utiliser un LLM comme tuteur socratique, générateur de retrieval et partenaire Feynman — en sachant précisément d'où vient l'effet 2-sigma de Bloom, et en évitant les pièges qui le sabotent.

---

## 1. Le problème de départ : le 2-sigma de Bloom

En 1984, Benjamin Bloom publie *"The 2 Sigma Problem"* (*Educational Researcher*, 13(6), 4–16). Partant de plusieurs expérimentations, il constate que les étudiants qui reçoivent un **tutorat individuel combiné au mastery learning** progressent en moyenne de **deux écarts-types** par rapport à une classe conventionnelle — soit passer du 50e au 98e percentile.

Son « problème » : le tutorat 1-1 est prohibitivement coûteux à grande échelle. Peut-on reproduire cet effet autrement ?

Quarante ans plus tard, les LLMs rendent la question attaquable : un modèle peut simuler un tuteur disponible 24h/24, infiniment patient, capable de générer des questions calibrées à la demande.

> **Encart — Pseudoscience ? Non. Mais nuance obligatoire.**
>
> L'effet des 2 sigma n'est **pas la magie du tuteur humain**. Il tient en grande partie au **mastery learning** : l'exigence d'un standard de maîtrise élevé (~90 %) avant d'avancer. Ce n'est pas le canal qui explique l'effet — c'est la boucle *feedback immédiat → correction → vraie maîtrise avant de progresser*.
>
> Un LLM utilisé **passivement** (lire des explications, poser "explique-moi X") ne reproduit pas l'effet 2-sigma. Il faut que *tu* maintiennes le standard : ne pas avancer si tu ne peux pas restituer sans aide.

---

## 2. Rôle 1 : le tuteur socratique

### Le principe

Socrate ne donnait pas les réponses — il posait des questions jusqu'à ce que l'interlocuteur les trouve lui-même. Un LLM donne naturellement la réponse directe. Pour déclencher la posture socratique, **tu dois le demander explicitement**.

### Prompts utiles

**Pour apprendre un concept inconnu :**
> *"Je veux apprendre [concept]. Ne m'explique pas directement : pose-moi des questions pour tester ce que je comprends déjà, puis guide-moi par questions successives."*

**Pour vérifier une compréhension :**
> *"Je vais essayer d'expliquer [concept]. Pose-moi des questions de clarification si tu détectes des zones floues ou des confusions dans ce que je dis."*

**Pour créer de la friction active :**
> *"Je crois comprendre [concept] mais je veux le vérifier. Joue le rôle d'un étudiant naïf qui me pose des questions difficiles."*

### Pourquoi ça marche

La posture socratique force la **récupération active** (Module 02) et génère de la **difficulté désirable** (Module 04) : tu dois reformuler, préciser, trouver les connexions toi-même — au lieu de consommer passivement une explication pré-mâchée.

---

## 3. Rôle 2 : générateur de retrieval et d'espacement

### Pour la retrieval practice

> *"Génère 5 questions de rappel sur [sujet], du plus simple au plus complexe. Ne donne pas les réponses — je vais répondre d'abord, et tu corrigeras ensuite."*

> *"Crée un quiz à livre fermé sur les points suivants : [liste]. Format : question / réponse courte."*

### Pour l'espacement

> *"J'ai étudié [sujet] il y a 3 jours. Génère un mini-test de 5 questions pour consolider ma mémoire avant de passer à la suite."*

> *"Sur la base de ces thèmes que j'ai étudiés cette semaine [liste], construis-moi un plan de révision espacée pour les 2 prochaines semaines (J+3, J+7, J+14)."*

### Pour l'interleaving

> *"Crée 10 exercices qui mélangent aléatoirement les types suivants : [type A, type B, type C]. Ne regroupe pas par type."*

### Ce que ça ajoute par rapport à Anki

Anki (Module 03) est optimal pour les cartes déjà formalisées. Un LLM génère du contenu à la demande pour des sujets nouveaux, avant que tu aies formalisé tes cartes — c'est le **bootstrap de l'espacement**.

---

## 4. Rôle 3 : partenaire Feynman

### Le protocole

1. **Explique un concept au LLM** comme si tu l'enseignais — à voix haute ou par écrit.
2. **Demande une analyse critique** : *"Dans mon explication, détecte les zones floues, les sauts logiques, le jargon que je n'ai pas défini et les erreurs conceptuelles."*
3. **Revise les points soulevés**, puis recommence jusqu'à ce que l'explication soit propre.

### Pourquoi le LLM améliore le Feynman solo

Seul, tu passes à côté des trous que tu ne vois pas — parce que tu sais ce que tu *voulais* dire. Le LLM lit ce que tu as *écrit* et pointe les incohérences que ton auto-correction naturalise.

---

## 5. Les quatre pièges à éviter

| Piège | Mécanisme | Contremesure |
|---|---|---|
| **Dépendance passive** | Lire des explications = relecture déguisée (Dunlosky 2013 : utilité faible) | Chaque session LLM doit produire des questions, pas des réponses |
| **Confabulations** | Le LLM génère des informations plausibles mais fausses, surtout sur des références précises | Vérifie systématiquement les faits critiques dans les sources primaires |
| **Illusion de compréhension amplifiée** | Une bonne explication du LLM crée une fluency illusion encore plus forte que la relecture | Ferme la conversation, restitue le concept sans aide — c'est le seul test |
| **Court-circuitage de la difficulté désirable** | Poser directement "quelle est la réponse à X" supprime l'effort de récupération | Formule des tentatives avant de demander au LLM |

---

## 6. Mesurer si ton utilisation du LLM fonctionne

(Lien Module 09 — mesurer son apprentissage)

Un LLM ne doit pas remplacer une métrique de rétention. Après chaque session LLM :

- **Test pré/post** : avant la session, tente de répondre aux questions sans aide ; après, reteste les mêmes questions.
- **Test J+7** : la vraie rétention se mesure une semaine plus tard — si tu ne peux plus restituer sans aide, la session LLM a produit de la fluency, pas de la rétention.
- **Indicateur qualitatif** : est-ce que ton explication Feynman a progressé entre deux tours ? Si oui, tu avances. Sinon, tu tourne en rond.

---

> **À retenir :**
> - Le 2-sigma de Bloom vient du **mastery learning** (standard ~90 % avant d'avancer), pas du canal technologique — un LLM passif ne le reproduit pas.
> - Les trois rôles actifs : **tuteur socratique** (questions, pas réponses directes), **générateur de retrieval et d'espacement** (quiz, plans de révision), **partenaire Feynman** (détection de trous).
> - Les quatre pièges : dépendance passive, confabulations, illusion de compréhension amplifiée, court-circuitage de la difficulté désirable.
> - La mesure reste obligatoire : test pré/post et test J+7 — la fluidité pendant la session n'est pas un indicateur de rétention.

---

## 7. Ta base de connaissances portable : le pattern LLM-Wiki (et OKF)

### Le probleme humain : on abandonne nos wikis perso

Presque tout le monde a deja demarre un "second cerveau" — un Notion, un Obsidian, un carnet, un dossier de notes. Et presque tout le monde l'a abandonne. La raison est rarement le manque d'idees : c'est la **maintenance**. Relier une nouvelle note aux anciennes, la ranger au bon endroit, mettre a jour un index, garder les references coherentes — ce travail de tenue de registre est fastidieux, repetitif, et l'humain finit par lacher. Les notes s'accumulent en vrac, deviennent introuvables, et le systeme meurt.

### L'insight de Karpathy : deleguer le bookkeeping au LLM

Andrej Karpathy a propose une idee simple pour briser ce cercle (son "LLM-Wiki", voir *Pour aller plus loin*) : garder un format **simple et durable** — de simples fichiers markdown — et confier le travail penible de tenue de registre a un LLM. L'humain **curate** (decide quoi garder, corrige, oriente) ; le LLM **entretient** (relie, met a jour l'index, propage les cross-references, touche beaucoup de fichiers d'un coup).

> *"Les LLM ne s'ennuient pas, n'oublient pas de mettre a jour une cross-reference, et peuvent toucher 15 fichiers en une seule passe."* — Andrej Karpathy

Autrement dit, le bookkeeping qui pousse les humains a **abandonner** leurs wikis personnels est precisement ce que les LLM font bien. On enleve la corvee qui tuait le systeme.

### OKF : la version standardisee de l'idee

Le 12 juin 2026, Google Cloud (Sam McVeety & Amir Hormati) a formalise ce pattern en une specification ouverte, l'**Open Knowledge Format (OKF, v0.1)** — voir *Pour aller plus loin*. L'idee en une phrase : un **repertoire de fichiers markdown**, chacun avec un petit en-tete structure (frontmatter YAML, ou seul le champ `type` est obligatoire), les notes se reliant entre elles par de simples liens markdown — ce qui forme un **graphe** de connaissances. Deux fichiers optionnels organisent le tout : un `index.md` (point d'entree / sommaire) et un `log.md` (historique chronologique).

```
ma-base/
  index.md            <- sommaire / point d'entree
  log.md              <- journal chronologique de ce que j'ai appris
  retrieval-practice.md
  espacement-sm2.md
```

```
---
type: note
title: Retrieval practice
tags: [apprendre, memoire]
---
La recuperation active bat la relecture. Voir [espacement](espacement-sm2.md).
```

Le point cle est resume par la formule des auteurs : **"format, pas plateforme"**. Pas de compte, pas d'outil proprietaire requis : tes notes restent **lisibles partout**, **versionnables** avec git, et ta connaissance n'est pas **prisonniere** d'un seul logiciel.

### Application a l'apprentissage

Une telle base devient une **source vivante** pour les techniques des modules precedents. Le LLM peut lire tes notes pour **generer des quiz** de recuperation (c'est exactement le role 2 vu plus haut : generateur de retrieval) et te proposer un **plan d'espacement** a partir de ce que tu as ajoute recemment. Le `log.md`, lui, documente ta progression dans le temps — un journal honnete de ce que tu as reellement consolide.

### La nuance honnete (a ne jamais oublier)

Tout cela n'est puissant qu'a une condition : la base doit **nourrir le rappel actif**, pas le remplacer. Un wiki magnifique qu'on se contente de **relire et d'admirer** est de la **relecture passive deguisee** — la technique la moins efficace selon Dunlosky (2013). Pire, comme l'explication structuree vue dans les modules precedents (metacognition), une base bien rangee peut amplifier l'**illusion de comprehension** : elle a l'air maitrisee, donc on *croit* la maitriser. Le test ne change pas : **ferme le wiki et ressors le concept de memoire**. Si tu n'y arrives pas, la note ne t'a encore rien appris — elle t'attend pour un vrai effort de recuperation.

---

## Flash-cards

**Q1** : D'où vient l'effet 2-sigma de Bloom (1984) — et pourquoi un LLM ne le reproduit pas automatiquement ?
**R1** : L'effet vient du **mastery learning** (standard ~90 % de maîtrise avant de progresser) autant que du tutorat. Un LLM ne reproduit l'effet que si l'apprenant maintient lui-même ce standard — ne pas avancer sans restitution sans aide.

**Q2** : Comment transformer un LLM en tuteur socratique plutôt qu'en distributeur de réponses ?
**R2** : En le lui demandant explicitement : *"Ne m'explique pas directement — pose-moi des questions pour tester ce que je comprends déjà, puis guide-moi par questions successives."*

**Q3** : Quels sont les trois rôles actifs d'un LLM dans l'apprentissage ?
**R3** : (1) Tuteur socratique (questions → récupération active). (2) Générateur de retrieval et d'espacement (quiz calibrés, plans J+3/J+7/J+14). (3) Partenaire Feynman (détection de trous dans les explications de l'apprenant).

**Q4** : Qu'est-ce que l'illusion de compréhension amplifiée spécifique au LLM ?
**R4** : Une explication fluide et personnalisée du LLM crée une fluency illusion plus forte que la relecture — parce qu'elle semble "taillée pour toi". Le seul test réel : fermer la conversation et restituer sans aide.

**Q5** : Comment mesurer si une session LLM a produit de la rétention ou seulement de la fluidité ?
**R5** : Test pré/post (avant/après la session) et surtout test **J+7** — si tu ne peux plus restituer une semaine plus tard sans aide, la session a produit de la fluency, pas de la rétention.

---

## Points clés à retenir

1. Le **mastery learning** (ne pas avancer sans maîtrise réelle) est le moteur de l'effet 2-sigma — le LLM est un canal, pas une cause.
2. Un LLM n'est utile en apprentissage que s'il **génère de la friction** (questions, exercices, Feynman) — pas du confort (explications passives).
3. Les **confabulations** sont un risque réel : vérifie toujours les faits critiques dans les sources primaires.
4. La **fluency illusion amplifiée** est le piège le plus insidieux du LLM — le test de restitution sans aide reste le seul juge.
5. La mesure (Module 09) reste obligatoire : test pré/post et J+7 pour distinguer fluidité et rétention durable.

---

## Pour aller plus loin

- **Bloom, B. S. (1984)** — *The 2 Sigma Problem*, *Educational Researcher* 13(6), 4–16 : https://gwern.net/doc/psychology/1984-bloom.pdf (PDF libre)
- **Dunlosky et al. (2013)** — *Improving Students' Learning With Effective Learning Techniques*, *PSPI* 14(1) : https://journals.sagepub.com/doi/10.1177/1529100612453266
- **Brown, Roediger & McDaniel — *Make It Stick* (2014)** : https://www.hup.harvard.edu/books/9780674729018 (synthèse grand public)
- **Bjork, Dunlosky & Kornell (2013)** — *Self-Regulated Learning: Beliefs, Techniques, and Illusions*, *Annual Review of Psychology* : https://gwern.net/doc/psychology/spaced-repetition/2013-bjork.pdf
- **Karpathy, A. — *LLM-Wiki*** (gist) : le pattern d'une base de connaissance en markdown ou le LLM fait le bookkeeping (index, cross-references, log) pendant que l'humain curate : https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f
- **McVeety, S. & Hormati, A. (12 juin 2026)** — *How the Open Knowledge Format can improve data sharing*, Google Cloud Blog : la specification ouverte (OKF v0.1) qui standardise ce pattern (markdown + frontmatter, `index.md`/`log.md`, portable et vendor-neutral) : https://cloud.google.com/blog/products/data-analytics/how-the-open-knowledge-format-can-improve-data-sharing
