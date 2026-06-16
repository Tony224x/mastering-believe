# Module 14 — Capstone : bâtir SON système d'apprentissage augmenté

> **Temps estimé** : 90 min (brief + auto-évaluation + livrable) | **Prérequis** : Modules 01–13 (tout le cursus)
> **Objectif** : assembler les 13 modules en un système personnel cohérent, mesurable et reproductible — appliqué à un sujet réel de ton choix — et produire un livrable portfolio qui prouve que tu sais apprendre.

---

## 1. Ce que ce capstone est (et n'est pas)

Ce n'est pas un résumé. Ce n'est pas un quiz de fin de chapitre. C'est une **mise en production** : tu prends un sujet que tu veux vraiment maîtriser, et tu construis autour de lui un système d'apprentissage complet, en réinvestissant explicitement les outils du cursus.

Le livrable attendu est dans `04-projects/README.md` (gabarit à remplir). Ce module t'en explique la logique et te donne les critères d'auto-évaluation.

> **Remarque sur le niveau de preuve**
>
> Les techniques mobilisées dans ce capstone (retrieval practice, espacement, pratique délibérée, boucle IA) ont des statuts de preuve différents. Retrieval practice et distributed practice = utilité **élevée** (Dunlosky 2013 ; Roediger & Karpicke 2006 ; Cepeda 2006). Pratique délibérée = **robuste mais à variance élevée** selon le domaine (Macnamara 2014 : 4–26 % de variance expliquée). LLM comme tuteur = **prometteur mais non répliqué à grande échelle** — l'effet dépend de l'usage actif, pas passif. Assembler ces techniques n'est pas une garantie de succès, c'est une stratégie probabiliste fondée sur les meilleures données disponibles.

---

## 2. Les trois briques que ce capstone assemble

### Brique 1 — Plan retrieval + espacement (Modules 02 et 03)

La mémoire se construit par **récupération active**, pas par relecture. Un plan retrieval, c'est répondre à deux questions concrètes avant même de commencer à étudier :

1. **À quelle fréquence et sous quelle forme vais-je me tester ?** (flashcards, blank-page, quiz, explication orale)
2. **À quels intervalles vais-je revenir sur chaque concept ?** (règle de base : doubler l'intervalle à chaque révision réussie — J+1, J+3, J+7, J+14, J+30…)

Le plan doit être écrit à l'avance, avec des **dates concrètes** et des **formats précis**. « Je réviserai quand j'en aurai besoin » n'est pas un plan — c'est un vœu.

### Brique 2 — Boucle IA (Module 13)

Un LLM n'est utile en apprentissage qu'en mode actif. Les trois rôles à spécifier dans ton système :

- **Tuteur socratique** : il ne te donne pas la réponse — il te pose des questions jusqu'à ce que tu la trouves.
- **Générateur de retrieval** : il produit des quiz calibrés sur tes thèmes, sans te donner les réponses d'emblée.
- **Partenaire Feynman** : tu lui expliques un concept, il détecte les trous, les sauts logiques, le jargon non défini.

Pour chaque rôle, spécifie **quand** (début de module ? fin de semaine ? quand tu bloques ?) et **quel prompt type** tu utiliseras. Vague = inutilisable.

### Brique 3 — Métriques de suivi (Module 09)

L'apprentissage sans mesure produit des illusions de compétence (Module 01 : fluency illusion). Les métriques que ton système doit tracker :

| Métrique | Ce qu'elle mesure | Fréquence |
|----------|------------------|-----------|
| **Taux de rappel** | % de réponses correctes sur un test à livre fermé | Chaque session de révision |
| **Rétention J+7** | % de rappel 7 jours après la première exposition | Hebdomadaire |
| **Ratio comprendre/produire** | Tu comprends quand tu lis vs tu peux produire sans aide (0–10) | Bilan hebdomadaire |
| **Delta pré/post** | Écart entre score avant et après une période d'étude | Début + fin de chaque bloc |

Ces métriques ne valent que si tu les consignes quelque part (tableau dans le gabarit, fiche papier, script — peu importe). Une métrique non notée n'existe pas.

---

## 3. La boucle complète du système

```
ENCODER (semaine 1)
    Sources sélectionnées (max 3)
    Blocs deep work planifiés — Module 06
    Interleaving si plusieurs sous-thèmes — Module 04
    Élaboration : self-explanation + dual coding modéré — Module 05
    ↓
ANCRER (semaines 2–N)
    Retrieval practice actif — Module 02
    Espacements SM-2 / Anki — Module 03
    Boucle IA : générateur de quiz — Module 13
    ↓
MONITORER (chaque semaine)
    Métriques : taux de rappel + rétention J+7 — Module 09
    Bilan métacognitif : planifier / monitorer / ajuster — Module 08
    ↓
PROGRESSER
    Pratique délibérée : objectif précis + feedback immédiat — Module 07
    Technique Feynman via LLM — Modules 08 + 13
    Stades de Fitts & Posner (cognitif → associatif → autonome) — Module 12
    ↓
DÉTECTER LES ILLUSIONS
    Neuromythes : VAK, 10 000 h, cerveau G/D — Modules 01, 04, 07
    Fluency illusion : toujours tester, jamais juste relire — Modules 01, 08
    Boucle IA : ne pas lire passivement les explications — Module 13
```

Ce cycle ne se fait pas une fois : il se répète jusqu'à l'atteinte des indicateurs de maîtrise définis au départ.

---

## 4. Comment choisir ton sujet

Le capstone fonctionne sur n'importe quel sujet, à une condition : il doit être **réel** — pas imaginaire, pas futur-conditionnel. Un sujet que tu vas commencer lundi, pas « un sujet que j'apprendrai peut-être un jour ».

Exemples de sujets qui fonctionnent bien :
- Une langue vivante (japonais, espagnol, arabe)
- Un autre domaine de ce repo (algorithmie, neural networks, finance personnelle)
- Une compétence professionnelle (prise de parole, SQL avancé, comptabilité de gestion)
- Un domaine scientifique (anatomie, physique quantitative, climatologie)

Un sujet trop vague (« je veux m'améliorer en tech ») est indétectable à la fin — l'indicateur de maîtrise doit être vérifiable par un tiers ou un test externe.

---

## 5. Grille d'auto-évaluation

Avant de valider ton livrable, passe chaque critère. Un critère non coché = retravail, pas passation.

### 5.1 — Sujet et objectif

- [ ] Le sujet est nommé précisément (pas « la programmation » mais « Python — manipulation de fichiers et APIs »)
- [ ] L'objectif est formulé en une phrase avec un verbe d'action mesurable (« être capable de ___ »)
- [ ] L'horizon temporel est fixé (nombre de semaines)
- [ ] Au moins deux indicateurs de maîtrise sont objectifs — vérifiables sans auto-proclamation (test, production évaluée, certification partielle, démo pour un pair)

### 5.2 — Plan retrieval + espacement

- [ ] Le format de retrieval est spécifié pour chaque session (flashcards / blank-page / quiz / explication orale — pas juste « révision »)
- [ ] Les dates de révision sont inscrites à l'avance avec intervalles croissants (le premier intervalle ne dépasse pas J+3)
- [ ] L'interleaving est planifié si le sujet a plusieurs sous-thèmes (pas un thème par semaine — un mix par session)
- [ ] Un score cible est défini pour chaque révision (permet de décider si l'intervalle peut s'allonger ou doit raccourcir)

### 5.3 — Boucle IA

- [ ] Les trois rôles (socratique, retrieval, Feynman) sont tous utilisés ou explicitement décartés avec justification
- [ ] Pour chaque rôle retenu : le **timing** est précisé (début de module / fin de semaine / quand bloqué)
- [ ] Pour chaque rôle retenu : un **prompt type** est rédigé (pas juste « je vais utiliser le mode socratique »)
- [ ] Les limites sont actées : pas de lecture passive d'explications sans retrieval actif juste après

### 5.4 — Métriques de suivi

- [ ] Le taux de rappel est mesuré à chaque session de révision (pas « j'ai l'impression d'avoir bien révisé »)
- [ ] La rétention J+7 est planifiée (un test 7 jours après chaque première exposition)
- [ ] Le ratio comprendre/produire est estimé chaque semaine
- [ ] Les métriques sont consignées quelque part (tableau, fichier, carnet)

### 5.5 — Boucle métacognitive

- [ ] Un bilan hebdomadaire est planifié (jour + durée fixés à l'avance)
- [ ] Au moins deux déclencheurs d'ajustement sont définis avec seuil explicite (ex : « si taux de rappel < 60 % deux semaines de suite → réduire l'intervalle + session Feynman »)
- [ ] Un critère de « ça ne marche pas » est défini pour pivoter de stratégie (pas juste ajuster les intervalles)

### 5.6 — Qualité globale du livrable

- [ ] Le gabarit `04-projects/README.md` est rempli complètement (aucune section laissée vide ou en « à compléter »)
- [ ] Aucun élément n'est un vœu pieux (tout est planifié, daté ou mesuré)
- [ ] Le système est auto-suffisant : quelqu'un d'autre pourrait l'exécuter à ta place avec les mêmes informations

---

## 6. Qu'est-ce qu'un bon livrable ressemble ?

Un bon livrable n'est pas long. Il est **précis**. Il répond à :

- Sujet : dans 6 semaines, je saurai faire X (testable par Y).
- Encoding : 3 sources, 35 min/jour, interleaving à partir de S2.
- Retrieval : blank-page à J+1, Anki quotidien, quiz LLM à J+7, J+14, J+28.
- IA : tuteur socratique le lundi (début de semaine), générateur de quiz le jeudi, Feynman le dimanche.
- Métriques : taux de rappel noté après chaque quiz, rétention J+7 testée, bilan dimanche soir.
- Ajustement : si rappel < 60 % → raccourcir. Si ratio comprendre/produire < 5/10 à S3 → ajouter sessions de production.

Un mauvais livrable contient des formules génériques : « je vais utiliser Anki », « j'utiliserai le LLM pour réviser », « je ferai un bilan régulier ». Ces formules ne permettent pas de savoir si le système a été suivi ou non — et donc de l'évaluer.

---

> **À retenir :**
> - Le capstone n'est pas un résumé — c'est une mise en production sur un sujet réel, avec plan, métriques et boucle IA.
> - Les trois briques : plan retrieval + espacement (Modules 02/03) + boucle IA (Module 13) + métriques de suivi (Module 09).
> - La grille d'auto-évaluation (section 5) est l'outil de validation — un critère non coché = à retravail.
> - Un bon système est précis, daté, et auto-suffisant — pas une liste de bonnes intentions.

---

## Flash-cards

**Q1** : Quelles sont les trois briques que ce capstone assemble, et quel module chacune réinvestit ?
**R1** : (1) Plan retrieval + espacement (Modules 02 et 03). (2) Boucle IA — trois rôles du LLM (Module 13). (3) Métriques de suivi — taux de rappel, rétention J+7, ratio comprendre/produire (Module 09).

**Q2** : Quelle est la règle de base pour planifier les intervalles de révision ?
**R2** : Doubler l'intervalle à chaque révision réussie : J+1, J+3, J+7, J+14, J+30… Si la révision est ratée (score < seuil), raccourcir et revenir plus tôt.

**Q3** : Quels sont les trois rôles actifs d'un LLM dans un système d'apprentissage, et comment les activer ?
**R3** : (1) Tuteur socratique : lui demander de poser des questions plutôt que d'expliquer. (2) Générateur de retrieval : lui demander un quiz calibré sans réponses d'emblée. (3) Partenaire Feynman : lui expliquer un concept et lui demander de détecter les trous.

**Q4** : Cite les quatre métriques de suivi clés du Module 09.
**R4** : Taux de rappel (% correct à livre fermé par session) ; rétention J+7 (% de rappel 7 jours après exposition) ; ratio comprendre/produire (0–10 hebdomadaire) ; delta pré/post (score avant vs après un bloc d'étude).

**Q5** : Qu'est-ce qui distingue un bon livrable capstone d'un mauvais ?
**R5** : Un bon livrable est précis, daté et auto-suffisant (quelqu'un d'autre pourrait l'exécuter). Un mauvais contient des formules génériques (« je vais utiliser Anki », « je ferai des bilans réguliers ») sans timing ni format ni seuil d'ajustement.

---

## Points clés à retenir

1. Un système d'apprentissage n'est pas une liste de techniques — c'est un **cycle avec des boucles de feedback courtes et honnêtes**.
2. Les trois briques du capstone (retrieval + espacement, boucle IA, métriques) sont interdépendantes : les métriques pilotent les ajustements de l'espacement, et la boucle IA alimente le retrieval.
3. L'indicateur de maîtrise doit être **objectif** avant de commencer — pas redéfini en cours de route pour rationaliser l'arrêt.
4. Le LLM n'est un accélérateur que s'il génère de la **friction** (questions, exercices, Feynman). Utilisé passivement, il amplifie l'illusion de compétence.
5. Ce système est **transférable** : remplace le sujet, garde la boucle. C'est le vrai livrable du cursus.

---

## Pour aller plus loin

- **Dunlosky, J. et al. (2013)** — *Improving Students' Learning With Effective Learning Techniques*, *PSPI* 14(1) : https://journals.sagepub.com/doi/10.1177/1529100612453266 *(retrieval practice et distributed practice = utilité élevée)*
- **Roediger, H. L., & Karpicke, J. D. (2006)** — *Test-Enhanced Learning*, *Psychological Science* 17(3) : https://journals.sagepub.com/doi/10.1111/j.1467-9280.2006.01693.x
- **Cepeda, N. J. et al. (2006)** — *Distributed Practice in Verbal Recall Tasks*, *Psychological Bulletin* 132(3) : https://augmentingcognition.com/assets/Cepeda2006.pdf
- **Bloom, B. S. (1984)** — *The 2 Sigma Problem*, *Educational Researcher* 13(6) : https://gwern.net/doc/psychology/1984-bloom.pdf *(boussole du tutorat augmenté)*
- **Macnamara, B. N. et al. (2014)** — *Deliberate Practice and Performance: A Meta-Analysis*, *Psychological Science* 25(8) : https://journals.sagepub.com/doi/10.1177/0956797614535810 *(nuance sur la pratique délibérée)*
- **Brown, Roediger & McDaniel — *Make It Stick* (2014)** : https://www.hup.harvard.edu/books/9780674729018
