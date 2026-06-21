# Solutions — Module 13 : Apprendre avec l'IA

---

## Exercice 1 — Générer un jeu de retrieval avec un LLM et le critiquer

### Corrigé modèle (concept choisi : l'interleaving)

**Exemple de prompt soumis au LLM :**
> *"Génère 6 questions de retrieval practice sur l'interleaving en apprentissage, du plus simple au plus complexe. Format : question seule, sans réponse."*

**Exemple de questions générées :**
1. Qu'est-ce que l'interleaving en apprentissage ?
2. Quelle est la différence entre la pratique bloquée et la pratique entrelacée ?
3. Rohrer & Taylor (2007) ont comparé interleaving et blocked practice en mathématiques — quel résultat ont-ils trouvé ?
4. Pourquoi l'interleaving semble-t-il moins efficace sur le moment mais produit de meilleurs résultats à long terme ?
5. Donne un exemple d'interleaving dans l'apprentissage d'une langue étrangère.
6. Quand l'interleaving est-il contre-indiqué ou moins utile — et pourquoi ?

**Réponses de mémoire (exemple attendu pour Q3) :**
> "Rohrer & Taylor ont trouvé que la pratique entrelacée donnait environ 72 % de réussite contre 38 % pour la pratique bloquée, mais je ne suis pas sûr du contexte exact (mathématiques ou autre)."

**Auto-correction attendue sur Q3 :**
Le LLM devrait confirmer les chiffres (72 % vs 38 %, contexte mathématiques) ou alerter si la source est approximative. Si le LLM cite ces chiffres sans mention de la source Rohrer & Taylor, c'est un signal de confabulation à vérifier dans le module théorique.

---

**Critique du jeu de questions — exemple structuré :**

**Forces :**
- La progression existe : Q1 est purement définitionnelle, Q4 demande de comprendre le mécanisme (pas juste mémoriser), Q6 demande une nuance.
- Q5 force un transfert (application à un domaine non cité dans le cours) — c'est une vraie question de compréhension.

**Faiblesses :**
- Q1 ("Qu'est-ce que l'interleaving ?") est une question de reconnaissance très facile — elle teste si on a lu le mot, pas si on comprend le concept.
- Q3 cite une étude précise (Rohrer & Taylor) : si les chiffres sont incorrects dans la réponse du LLM, c'est une confabulation. Toujours vérifier dans la source ou le module.
- Le quiz ne mélange pas l'interleaving avec d'autres concepts du domaine — il reste "bloqué" sur un thème, ce qui est paradoxal pour un quiz sur l'interleaving.

**Ce que ce corrigé illustre :** un LLM génère des questions utiles mais souvent trop lisses en bas de la progression et potentiellement factuellement fragiles sur les références. La critique active transforme un outil passif en outil d'apprentissage.

---

## Exercice 2 — Protocole Feynman augmenté par LLM

### Corrigé modèle (concept choisi : l'effet 2-sigma de Bloom)

**Explication initiale (exemple) :**
> Bloom a montré en 1984 que les étudiants qui reçoivent un tutorat individuel apprennent beaucoup mieux — environ deux fois mieux — que ceux dans une classe normale. Le tutorat permet au tuteur d'adapter ce qu'il enseigne à l'élève en temps réel, de donner du feedback immédiat, et de ne pas passer à la suite avant que l'élève ait vraiment compris. Aujourd'hui, on peut utiliser un LLM comme tuteur pour reproduire cet effet à moindre coût.

**Points que le LLM devrait soulever :**
1. **Saut logique** : "deux fois mieux" est imprécis — l'effet est de 2 écarts-types, ce qui correspond à passer du 50e au 98e percentile. Ce n'est pas "deux fois mieux".
2. **Zone floue** : l'explication attribue l'effet au seul tutorat adaptatif. Elle omet la composante critique : le **mastery learning** (standard de maîtrise ~90 % avant de progresser) — qui est le vrai moteur de l'effet selon Bloom lui-même.
3. **Glissement factuel** : "un LLM peut reproduire cet effet" est sur-vendu sans la condition — le LLM ne reproduit l'effet que si l'apprenant maintient lui-même le standard de maîtrise.

**Version corrigée des passages problématiques :**
> Bloom a montré en 1984 que les étudiants en tutorat individuel **avec mastery learning** (standard de maîtrise ~90 % avant de progresser) progressent en moyenne de **deux écarts-types** par rapport à une classe conventionnelle — ce qui correspond à passer du 50e au 98e percentile. L'effet ne vient pas seulement du tutorat adaptatif : il vient surtout de l'exigence de vraie maîtrise avant d'avancer. Un LLM peut aider à reproduire cet effet, mais uniquement si l'apprenant maintient lui-même ce standard — ne pas avancer sans pouvoir restituer sans aide.

**Ce que ce corrigé illustre :** le glissement le plus courant est d'attribuer l'effet 2-sigma au "tutorat" seul — en oubliant la composante mastery learning. C'est précisément le genre de trou que le LLM-partenaire détecte, parce qu'il lit ce qu'on a écrit, pas ce qu'on voulait écrire.

---

## Exercice 3 — Construire et évaluer un plan d'espacement généré par LLM

### Corrigé modèle

**Exemple de plan généré par un LLM (avec défauts typiques) :**

| Jalon | Thèmes | Activité suggérée |
|---|---|---|
| J+3 | Retrieval practice, spaced repetition | Relire les modules et faire un résumé |
| J+7 | Difficultes désirables, attention | Regarder une vidéo récapitulative |
| J+14 | Métacognition, mesure | Faire un quiz sur ces thèmes |
| J+21 | Tous les thèmes | Révision générale — relire les notes |

---

**Analyse critique attendue :**

**1. Intervalles croissants :**
Les intervalles J+3, J+7, J+14, J+21 sont croissants — c'est correct. Le plan respecte la logique de la courbe d'oubli (réviser avant d'oublier complètement, avec des gaps croissants).

**2. Qualité des activités :**
- J+3 : "relire les modules et faire un résumé" = **relecture** = utilité **faible** (Dunlosky 2013). La relecture crée une fluency illusion sans renforcer la rétention.
- J+7 : "regarder une vidéo récapitulative" = **visionnage passif** = utilité faible. Pire que la relecture.
- J+14 : "faire un quiz" = **retrieval practice** = utilité **élevée**. C'est la seule activité correcte du plan.
- J+21 : "relire les notes" = retour à la relecture passive.

Résultat : 3 activités sur 4 sont à utilité faible selon Dunlosky. Le LLM par défaut reproduit les intuitions communes (relecture, résumés) — pas les techniques validées.

**3. Mesure de rétention :**
Le plan ne prévoit aucun indicateur de rétention. Il planifie des sessions mais ne dit pas comment savoir si la rétention a eu lieu. Manque critique : absence de test pré/post à chaque jalon (Module 09).

**4. Version corrigée de deux jalons :**

| Jalon | Thèmes | Activité corrigée | Mesure |
|---|---|---|---|
| J+3 | Retrieval practice, spaced repetition | Blank-page recall : fermer le cours, écrire de mémoire les 5 points clés de chaque module sur une feuille blanche | Score sur 10 points clés : combien restitués sans aide ? Seuil cible : ≥ 7/10 avant de passer à J+7 |
| J+7 | Difficultés désirables, attention | Quiz entrelacé : 8 questions mélangées sur les thèmes J+3 + les nouveaux (difficultés désirables, attention) — sans indiquer le thème de chaque question | Corriger avec le module théorique ; noter les questions ratées pour cibler la révision J+14 |

**Ce que ce corrigé illustre :** un LLM par défaut génère des plans qui *ressemblent* à de l'organisation mais reproduisent les techniques à faible utilité. La valeur de l'exercice est dans l'évaluation critique, pas dans le plan généré.

---

**Rappel transversal :** dans les trois exercices, la valeur pédagogique ne vient pas du LLM — elle vient de ton effort critique (analyser les questions, détecter les trous, évaluer le plan). C'est exactement la logique de l'effet 2-sigma : le canal (LLM) ne compte pas, c'est le standard de maîtrise que tu t'imposes qui fait la différence.
