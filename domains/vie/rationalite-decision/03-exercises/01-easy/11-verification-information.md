# Exercices — Module 11 : Vérification de l'information à l'ère de l'IA

> **Prérequis** : `01-theory/11-verification-information.md`
> **Durée estimée** : 35-45 min (environ 12-15 min par exercice)
> **Mode** : réflexion écrite, pas de code

---

## Exercice 1 — Analyser une fausse citation (niveau facile)

### Objectif

Appliquer le mouvement **T (Trace)** de SIFT pour évaluer l'authenticité d'une citation attribuée à une figure connue.

### Consigne

On vous soumet le passage suivant, trouvé dans un article en ligne :

> *"Albert Einstein aurait déclaré lors d'une conférence à Berlin en 1926 : 'L'imagination est plus importante que la connaissance, car la connaissance est limitée, tandis que l'imagination embrasse le monde entier.' Source : Berliner Tagblatt, 28 octobre 1926, p. 4."*

La citation elle-même est réelle et vérifiable (elle apparaît dans un entretien de 1929). En revanche, la source citée ici (le Berliner Tagblatt du 28 octobre 1926) est **incorrecte** — la vraie source est *The Saturday Evening Post*, 26 octobre 1929.

1. Sans regarder la solution, listez les **3 à 5 vérifications concrètes** que vous feriez pour évaluer cette citation (quels outils, quelles requêtes).
2. Notez ce qui vous semble **plausible** (rend la citation crédible) et ce qui devrait **alerter** (rend la source douteuse).
3. Formulez un verdict en une phrase : accepteriez-vous cette citation telle quelle ? Pourquoi ?

### Critères de réussite

- [ ] Au moins 3 vérifications concrètes identifiées (Google Scholar, Einstein Archives, recherche du titre exact, etc.)
- [ ] Au moins un élément "plausible" et un élément "alerte" distincts nommés
- [ ] Le verdict final mentionne explicitement la nécessité de **remonter à la source primaire** avant d'utiliser la citation

---

## Exercice 2 — Détecter une image hors contexte (niveau intermédiaire)

### Objectif

Appliquer les étapes **S + F + T** de SIFT pour évaluer une image accompagnant une affirmation.

### Consigne

Un post sur un réseau social montre une photographie spectaculaire d'un chantier effondré, avec la légende :

> *"Catastrophe hier soir à [ville fictive] — un immeuble en construction s'est écroulé faisant plusieurs blessés. Les travaux de sécurité étaient inexistants."*

Le post a 12 000 partages en 2 heures.

1. **Avant de partager**, décrivez le raisonnement SIFT que vous appliqueriez, mouvement par mouvement (S, I, F, T).
2. Pour l'étape T, précisez **quel outil** vous utiliseriez, **quelle requête** vous tapperiez, et **quel résultat** vous permettrait de conclure que l'image est hors contexte.
3. Imaginez que la recherche inversée montre que cette image est parue dans un article de presse étranger, daté de 4 ans auparavant. Rédigez en 2-3 phrases le commentaire que vous posteriez pour signaler l'erreur — en restant factuel et sans attaquer l'auteur du post.

### Critères de réussite

- [ ] Les 4 mouvements SIFT sont décrits avec une action concrète chacun (pas juste nommés)
- [ ] L'outil de recherche inversée est nommé (TinEye ou Google Images) avec une démarche précise
- [ ] Le commentaire de signalement est factuel, non accusatoire, et cite la source trouvée

---

## Exercice 3 — Évaluer un faux remède "miracle" et une hallucination de LLM (niveau avancé)

### Objectif

Combiner la détection de fausse promesse thérapeutique (SIFT complet) et la vérification d'une citation produite par un LLM.

### Consigne

Vous lisez l'article suivant sur un blog de bien-être :

> *"Une étude publiée en 2023 dans le Journal of Cellular Longevity a montré que le supplément AlphaCèle® augmente la production mitochondriale de 47 % en 30 jours chez l'adulte sain. Les chercheurs, dirigés par le Pr. Hana Novak (Université de Prague), concluent que cette molécule représente 'une avancée sans précédent pour la longévité cellulaire'. Disponible ici : [lien boutique]."*

*(Note : "AlphaCèle®", le "Journal of Cellular Longevity" et le "Pr. Hana Novak" dans ce contexte sont fictifs — ils ne doivent pas être cherchés en dehors de cet exercice.)*

**Partie A — Analyse SIFT**

1. Listez les **signaux d'alerte** présents dans cet extrait (au moins 4, en justifiant chacun).
2. Déroule le mouvement **F (Find better coverage)** : quelles requêtes précises tapez-vous pour trouver des sources indépendantes sur ce supplément ?

**Partie B — Vérification de la citation LLM**

Un LLM vous fournit la référence suivante pour appuyer un propos sur la cognition :

> *"Novak, H., & Petersen, R. (2022). 'Mitochondrial synthesis enhancement through polyphenol supplementation: a randomized controlled trial.' Cellular Metabolism, 31(7), 112-128. DOI: 10.1016/j.cmet.2022.00317."*

3. Décrivez **exactement** les 4 étapes du protocole de vérification d'une citation (tel que vu dans le module) appliquées à cette référence.
4. Si le DOI renvoie sur une page d'erreur et que Google Scholar ne trouve pas ce titre exact, quelle est votre conclusion ? Comment formuleriez-vous cela dans un document de travail ?

### Critères de réussite

- [ ] Au moins 4 signaux d'alerte identifiés avec une justification pour chacun (pas juste listés)
- [ ] Les requêtes de l'étape F sont précises (termes exacts, opérateurs de recherche si pertinent)
- [ ] Les 4 étapes du protocole de vérification de citation sont appliquées dans l'ordre correct
- [ ] La conclusion sur la citation non-vérifiable est formulée avec le degré de certitude approprié (probable, non confirmée) — sans affirmer péremptoirement que le chercheur n'existe pas
