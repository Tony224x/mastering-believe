# Module 07 — Capstone : apprendre avec l'IA

> **Temps estime** : 60 min | **Prerequis** : Modules 01-06 (tout le domaine)
> **Objectif** : assembler les six modules en un systeme personnel d'apprentissage augmente par un LLM — en comprenant les fondements (2-sigma de Bloom), les roles concrets de l'IA (tuteur socratique, generateur de retrieval, partenaire Feynman), et en produisant un gabarit reutilisable pour n'importe quel sujet futur.

---

## 1. Point de depart : le probleme du 2-sigma de Bloom

En 1984, Benjamin Bloom publie un article devenu canonique : *"The 2 Sigma Problem: The Search for Methods of Group Instruction as Effective as One-to-One Tutoring"* (*Educational Researcher*, 13(6), 4–16).

Son constat, base sur plusieurs experimentations : les etudiants qui recoivent un **tutorat individuel avec mastery learning** (standard de maitrise eleve, 90 %, avant de passer a la suite) obtiennent en moyenne **deux ecarts-types** au-dessus des etudiants d'une classe conventionnelle. C'est enorme — ca correspond a passer du 50e au 98e percentile.

Le "probleme", c'est que le tutorat 1-1 est prohibitivement cher. Bloom se demande comment atteindre cet effet a l'echelle — sans tutorat humain pour chaque eleve.

Quarante ans plus tard, les LLMs rendent ce probleme attaquable. Un modele de langage peut simuler un tuteur disponible 24h/24, infiniment patient, capable de s'adapter au niveau de la question et de generer des exercices a la demande.

> **Nuance importante — a ne pas oublier**
>
> L'effet des 2 sigma tient en grande partie au **mastery learning** (le standard de 90 % impose avant de progresser), pas seulement au tutorat en tant que tel. Ce n'est pas le fait d'avoir un tuteur humain qui explique 1-1 qui est magique — c'est la combinaison d'un feedback immediat et personnalise avec l'exigence d'une vraie maitrise avant d'avancer. Un LLM peut aider sur les deux points, mais seulement si *tu* imposes le meme standard a toi-meme. Un LLM utilise passivement (lire des explications) ne reproduit pas l'effet 2-sigma.

---

## 2. Les trois roles concrets d'un LLM dans ton apprentissage

### Role 1 : le tuteur socratique

Platon decrivait Socrate comme un "accoucheur" d'idees — quelqu'un qui ne donne pas les reponses mais pose des questions jusqu'a ce que l'apprenant les trouve lui-meme.

Un LLM peut jouer ce role, mais **uniquement si tu le lui demandes explicitement**. Par defaut, un LLM donne la reponse directe. Pour declencher la posture socratique :

**Prompts utiles :**
- *"Je veux apprendre [concept]. Ne m'explique pas directement — pose-moi des questions pour tester ce que je comprends deja, puis guide-moi par questions successives."*
- *"Je vais essayer d'expliquer [concept]. Pose-moi des questions de clarification si tu detectes des zones floues ou des confusions."*
- *"Je crois comprendre [concept] mais je veux le verifier. Joue le role d'un etudiant naif qui me pose des questions difficiles."*

L'objectif : generer de la **difficulte desirable** en te forcant a reformuler, a preciser, a trouver toi-meme les connexions — au lieu de consommer passivement une explication pre-machee.

### Role 2 : le generateur de retrieval et d'espacement

Un LLM peut produire a la demande des exercices de retrieval calibres sur ce que tu etudies :

**Pour la retrieval practice :**
- *"Genere 5 questions de rappel sur [sujet], du plus simple au plus complexe. Ne donne pas les reponses — je vais repondre d'abord, et tu corrigeras ensuite."*
- *"Cree un quiz a livre ferme sur les points suivants : [liste]. Format question/reponse courte."*

**Pour l'espacement :**
- *"J'ai etudie [sujet] il y a 3 jours. Genere un mini-test de 5 questions pour consolider ma memoire avant de passer a la suite."*
- *"Sur la base de ces themes que j'ai etudies cette semaine [liste], construis-moi un plan de revision espacee pour les 2 prochaines semaines (J+3, J+7, J+14)."*

**Pour l'interleaving :**
- *"Cree 10 exercices qui melangent aleatoirement les types suivants : [type A, type B, type C]. Ne regroupe pas par type."*

### Role 3 : le partenaire Feynman

La technique Feynman (Module 06) est plus efficace quand il y a un interlocuteur pour detecter les trous que tu ne vois pas toi-meme.

**Protocole pratique :**
1. Explique un concept a voix haute ou par ecrit au LLM, comme si tu l'enseignais.
2. Demande-lui : *"Dans mon explication, detecte les zones floues, les sauts logiques, le jargon que je n'ai pas defini et les erreurs conceptuelles."*
3. Revise sur les points souleves, puis recommence.

Le LLM ne remplace pas la difficulte de l'explication — il la structure et identifie les manques que tu n'aurais pas detectes seul.

---

## 3. Les limites a connaitre

Un LLM utilise pour l'apprentissage presente des biais structurels qu'il faut connaitre pour ne pas tomber dans leurs pieges :

**Risque 1 : la dependance passive.** Si tu utilises le LLM pour *lire des explications*, tu es dans la relecture passive — la technique la moins efficace selon Dunlosky (2013). L'IA doit generer de la friction (questions, exercices, Feynman), pas du confort.

**Risque 2 : les confabulations.** Les LLMs peuvent generer des informations plausibles mais incorrectes — surtout sur des sujets pointus, des references bibliographiques precises, ou des faits recents. Pour l'apprentissage de faits critiques (medecine, droit, sciences), verifie systematiquement les sources primaires.

**Risque 3 : l'illusion de comprehension amplifiee.** Une bonne explication du LLM peut creer une fluency illusion encore plus forte que la relecture — parce que l'explication semble taillee pour toi. Le test reste le meme : ferme la conversation et ressors le concept sans aide.

**Risque 4 : le court-circuitage du desirable difficulty.** Poser directement la question "quelle est la reponse a X" supprime la difficulte desirable. Le vrai gain vient de l'effort pour trouver, pas de la reception de la reponse.

---

## 4. Assembler ton systeme personnel d'apprentissage augmente

Les six modules precedents t'ont donne des briques. Ce capstone les assemble en un systeme coherent :

```
ENCODER PROFONDEMENT
    Blocs deep work (M05)
    Chunking progressif (M05)
    Difficultes desirables & interleaving (M04)

ANCRER DURABLEMENT
    Retrieval practice actif (M02)
    Espacement SM-2 / Anki (M03)
    [LLM : generateur de quiz & plans de revision]

PROGRESSER VERS L'EXPERTISE
    Pratique deliberee (objectif precis, feedback) (M06)
    Representations mentales (M06)
    Metacognition : planifier / monitorer / ajuster (M06)
    Technique Feynman (M06)
    [LLM : tuteur socratique + partenaire Feynman]

DETECTER LES ILLUSIONS
    Illusion de competence (M01)
    Mauvais JOL / fluency (M01, M06)
    Neuromythes VAK, 10 000h, cerveau G/D (M01, M04, M05, M06)
```

Le systeme ne fonctionne que si les boucles de feedback sont **courtes** (on teste vite) et **honnetes** (on ne triche pas sur les indicateurs).

---

> **A retenir :**
> - L'effet 2-sigma de Bloom vient du mastery learning autant que du tutorat — un LLM ne reproduit cet effet que si *tu* maintiens le standard de maitrise (ne pas avancer avant de vraiment comprendre).
> - Les trois roles du LLM en apprentissage : **tuteur socratique** (questions, pas reponses directes), **generateur de retrieval et d'espacement** (quiz calibres, plans de revision), **partenaire Feynman** (detection de trous).
> - Les pieges a eviter : dependance passive, confabulations factuelles, illusion de comprehension amplifiee, court-circuitage de la difficulte desirable.
> - Le systeme complet combine les 6 modules : encoder (deep work + interleaving), ancrer (retrieval + espacement), progresser (pratique deliberee + metacognition + Feynman), detecter les illusions.

---

## Flash-cards

**Q1** : Que montre l'experience de Bloom (1984) sur le tutorat 1-1 ?
**R1** : Les etudiants en tutorat individuel avec mastery learning progressent en moyenne de **2 ecarts-types** par rapport a une classe conventionnelle — ce qui correspond a passer du 50e au 98e percentile.

**Q2** : A quelle condition un LLM peut-il reproduire une partie de l'effet 2-sigma ?
**R2** : Seulement si l'apprenant maintient lui-meme un **standard de maitrise eleve** (ne pas avancer sans comprendre) et utilise le LLM de facon active (generation de questions, exercices, Feynman) — pas passive (lire des explications).

**Q3** : Quels sont les trois roles actifs d'un LLM dans l'apprentissage ?
**R3** : (1) Tuteur socratique (pose des questions plutot que de donner des reponses). (2) Generateur de retrieval et d'espacement (quiz, plans de revision). (3) Partenaire Feynman (detection de trous dans les explications de l'apprenant).

**Q4** : Cite deux pieges specifiques a l'utilisation d'un LLM pour apprendre.
**R4** : (1) Dependance passive : lire des explications au lieu de s'exercer activement. (2) Illusion de comprehension amplifiee : une explication fluide du LLM cree une fausse impression de maitrise sans test reel. (Aussi : confabulations factuelles ; court-circuitage de la difficulte desirable.)

**Q5** : Ecris le prompt Feynman de base pour utiliser un LLM en partenaire d'explication.
**R5** : *"Je vais expliquer [concept] comme si je l'enseignais. Detecte dans mon explication les zones floues, les sauts logiques, le jargon non defini et les erreurs conceptuelles."*

---

## Points cles a retenir

1. Le **2-sigma de Bloom** est la boussole : l'IA-tuteur est prometteuse, mais l'effet vient du mastery learning, pas du seul canal technologique.
2. Un LLM n'est utile en apprentissage que s'il **genere de la friction** (questions, exercices, Feynman) — pas du confort (explications passives).
3. Les **confabulations** sont un risque reel : verifie toujours les faits critiques dans les sources primaires.
4. Le systeme complet combine encoder (deep work + interleaving) + ancrer (retrieval + espacement) + progresser (deliberate practice + metacognition) + detecter les illusions — c'est la somme des six modules.
5. Ce systeme est **reutilisable** pour n'importe quel domaine futur : remplace le sujet, garde la boucle.

---

## Pour aller plus loin

- **Bloom, B. S. (1984)** — *The 2 Sigma Problem*, *Educational Researcher* 13(6), 4-16 : https://gwern.net/doc/psychology/1984-bloom.pdf (PDF libre)
- **Dunlosky et al. (2013)** — *Improving Students' Learning With Effective Learning Techniques*, *PSPI* 14(1) : https://journals.sagepub.com/doi/10.1177/1529100612453266
- **Brown, Roediger & McDaniel — *Make It Stick* (2014)** : https://www.hup.harvard.edu/books/9780674729018 (synthese grand public de tout le domaine)
- **Learning How to Learn (Oakley & Sejnowski)** — MOOC Coursera : https://www.coursera.org/learn/learning-how-to-learn
- **Bjork, Dunlosky & Kornell (2013)** — *Self-Regulated Learning: Beliefs, Techniques, and Illusions*, *Annual Review of Psychology* : https://gwern.net/doc/psychology/spaced-repetition/2013-bjork.pdf
