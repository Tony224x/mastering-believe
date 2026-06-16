# Module 12 — Acquisition d'une compétence (opérationnel)

> **Temps estimé** : 45 min | **Prérequis** : Modules 01–11, et en particulier Module 07 (pratique délibérée)
> **Objectif** : disposer d'un cadre opérationnel unique — les trois stades de Fitts & Posner — pour structurer l'acquisition de n'importe quelle compétence, choisir les bons drills à chaque stade, et calibrer le feedback en conséquence.

---

## 1. Pourquoi ce module existe (et ce qu'il ne refait pas)

Au Module 07, tu as vu *pourquoi* la pratique délibérée est plus efficace que la simple répétition : objectif précis, feedback immédiat, travail au bord de la zone de confort, représentations mentales. Ici, on répond à la question suivante : **comment organiser concrètement ce processus de A à Z quand tu attaques une nouvelle compétence ?**

La réponse la plus transferable que la recherche en psychologie motrice et cognitive ait produite est le modèle de **Fitts & Posner (1967)**. Il date, il est un peu linéaire — on le nuancera — mais il reste la carte mentale la plus utile pour un apprenant autonome.

---

## 2. Le cadre : les trois stades de Fitts & Posner

En 1967, Paul Fitts et Michael Posner publient *Human Performance*, une synthèse sur l'acquisition des habiletés motrices et cognitives. Leur modèle identifie trois stades successifs.

### Stade 1 — Cognitif

**Ce qui se passe :** tu comprends la tâche pour la première fois. Tu t'appuies massivement sur l'attention consciente. Chaque micro-étape exige un effort délibéré — rien n'est automatique. Les erreurs sont nombreuses, variées, et souvent difficiles à diagnostiquer soi-même.

**Exemple fil rouge — apprendre à déboguer du code :**
Tu lis le message d'erreur et tu ne sais pas par où commencer. Tu relances en tâtonnant. Tu modifies une ligne, tu espères, tu relances. Il n'y a pas encore de stratégie — juste de l'exploration brute.

**Encart — langue :**
> En espagnol, tu construis chaque phrase mot par mot, tu cherches la conjugaison, tu comptes mentalement les accords. Parler prend trois fois plus de temps qu'en français.

**Encart — musique :**
> Devant une nouvelle partition, tu déchiffres note par note, tu regardes les doigts, tu regardes la partition, tu regardes à nouveau les doigts. Tu ne peux pas encore écouter ce que tu joues.

**Ce qu'il faut faire à ce stade :**
- Étudier des **exemples résolus** (worked examples — Sweller et al., 2019) avant de tenter de produire seul. Pour le novice, résoudre à froid surcharge la mémoire de travail ; commencer par comprendre des solutions déjà construites est plus efficace.
- Fractionner la compétence en sous-composantes (voir section 3).
- Accepter un **taux d'erreur élevé** — c'est normal et informatif.
- Chercher un **feedback externe** fréquent : tu n'as pas encore les représentations mentales pour t'auto-corriger.

---

### Stade 2 — Associatif

**Ce qui se passe :** les grandes erreurs disparaissent. Les sous-étapes commencent à se connecter. L'attention consciente diminue — tu n'as plus besoin de penser à chaque micro-geste. La progression est visible, mais la performance reste variable.

**Exemple fil rouge :**
Tu as maintenant un protocole de débogage : lire le traceback de bas en haut, isoler la ligne incriminée, vérifier les types, tester avec un print. Tu l'appliques encore consciemment, mais les étapes se sont soudées.

**Encart — langue :**
> Tu construis des phrases sans chercher la conjugaison des verbes réguliers. Les irréguliers prennent encore de l'effort.

**Encart — musique :**
> Tes mains connaissent les accords de base. Tu peux parfois écouter ce que tu joues pendant que tu joues — pas toujours.

**Ce qu'il faut faire à ce stade :**
- Passer des **exemples résolus aux problèmes avec échafaudage réduit** (*fading* — Sweller) : d'abord l'exemple complet, puis l'exemple à trous, puis le problème sans aide.
- Cibler les sous-compétences encore instables avec des **drills répétés et ciblés** (pas des exercices généralistes).
- Commencer à introduire de l'**interleaving** (Module 04) — alterner types de problèmes — pour consolider le transfert.
- Le feedback peut s'espacer légèrement : tu commences à pouvoir détecter certaines erreurs toi-même.

---

### Stade 3 — Autonome

**Ce qui se passe :** la compétence est automatisée. Elle consomme très peu de ressources attentionnelles. Tu peux faire autre chose en parallèle — écouter la musique pendant que tu joues, penser au problème suivant pendant que tu débogues la ligne courante.

**Exemple fil rouge :**
Lire un traceback Python est devenu un réflexe. Tu n'y penses plus — tu vois immédiatement où regarder. L'attention libérée se tourne vers des questions de design ou de performance.

**Ce qu'il faut faire à ce stade :**
- **Ne pas stagner dans le confort.** L'automatisation est utile mais elle arrête la progression si tu ne te challenges pas sur des compétences adjacentes plus difficiles.
- Maintenir la maîtrise par de la **pratique espacée** (Module 03) — la compétence automatisée se dégrade si elle n'est pas exercée.
- Si tu veux continuer à progresser, revenir en stade cognitif sur une sous-compétence de niveau supérieur (ex : debugging de race conditions dans du code concurrent).

---

> **Nuance sur le modèle :**
> Fitts & Posner est un modèle descriptif, pas une loi anatomique. Dans la pratique, le passage d'un stade à l'autre n'est pas un saut franc : certaines sous-composantes d'une compétence sont en stade 3 pendant que d'autres restent en stade 1. Et on peut régresser (contexte nouveau, stress, longue pause). Utilise ce modèle comme une **boussole**, pas comme un protocole rigide.

---

## 3. Décomposer une compétence avant de commencer

La première erreur face à une compétence nouvelle : s'y attaquer en bloc. Le module 07 t'a dit de cibler une *faiblesse précise* — mais encore faut-il savoir quelles sont les sous-composantes.

**Protocole de décomposition en 3 étapes :**

1. **Lister les sous-compétences.** Une compétence complexe est toujours un assemblage. Le debugging Python, c'est : lire un traceback, identifier le type d'erreur, isoler la cause, vérifier les hypothèses, corriger, valider. Chacune de ces étapes peut être travaillée séparément.

2. **Identifier les goulots d'étranglement.** Dans ton assemblage, quelle sous-compétence bloque le reste ? Travaille d'abord celle-là. En débogage, si tu ne sais pas lire un traceback, toutes les autres étapes sont bloquées.

3. **Ordonner l'apprentissage.** Certaines sous-compétences sont des prérequis stricts (lire un traceback avant d'identifier la cause), d'autres peuvent être parallélisées ou interleaved (types d'erreurs différents).

**Encart — transfert :**
> Ce protocole vient directement de la recherche sur le transfert analogique (Gick & Holyoak, 1983 — cf. les sources) : la structure abstraite d'une compétence se transfère mieux si on l'a explicitée. Nommer "ce que je suis en train d'apprendre" aide à réutiliser le schème dans un nouveau contexte.

---

## 4. Drills ciblés : s'entraîner sur la faiblesse, pas sur ce qu'on sait déjà

Ces drills opérationnalisent les quatre conditions de la pratique délibérée vues au Module 07 (Ericsson) — objectif précis, feedback immédiat, sortie de zone de confort, guidance — appliquées ici par stade de Fitts.

Un drill ciblé est un exercice délibérément construit pour isoler une sous-compétence spécifique et la pousser au bord de la zone de confort.

**Exemple fil rouge — drill stade 1 :**
Tu n'arrives pas à lire les tracebacks. Drill : prendre 10 tracebacks différents (erreurs de type, d'indice, d'import, de syntaxe) et, pour chacun, formuler à voix haute *avant* de regarder le code : "L'erreur est de type X, elle vient de la ligne Y, la cause probable est Z." Puis vérifier.

**Exemple fil rouge — drill stade 2 :**
Tu débogues bien les erreurs de type mais pas les erreurs logiques silencieuses. Drill : 8 fonctions volontairement buguées (résultats faux sans exception levée). Les déboguer en moins de 3 minutes chacune avec uniquement des prints stratégiques — pas un debugger.

**Règle générale :**
- Drill stade 1 → exhaustivité et compréhension (tous les cas de figure).
- Drill stade 2 → rapidité et fiabilité sur les cas courants + exploration des cas difficiles.
- Drill stade 3 → maintien, transfert à des contextes nouveaux.

---

## 5. Calibrer le feedback selon le stade

Le feedback n'a pas la même forme ni la même fréquence optimale selon le stade.

| Stade | Type de feedback idéal | Fréquence | Source |
|-------|------------------------|-----------|--------|
| Cognitif | Immédiat, explicatif, pas uniquement "faux/juste" | Très élevée | Enseignant, pair, solution commentée, IA socratique |
| Associatif | Différé possible sur les aspects déjà acquis, immédiat sur les nouvelles cibles | Élevée → modérée | Tests, auto-confrontation à une solution de référence |
| Autonome | Espacé, principalement via les résultats dans des contextes réels | Faible, ciblée | Projets réels, code review, performance mesurée |

**Attention au feedback trop fréquent au stade autonome :** il peut créer une dépendance et ralentir l'automatisation. Le stade autonome se renforce précisément parce que le circuit fonctionne sans supervision constante.

---

## 6. L'immersion avec feedback : cas particulier des compétences ouvertes

Pour les compétences "ouvertes" — langue vivante, improvisation musicale, communication orale — le volume d'exposition (immersion) joue un rôle majeur. Mais l'immersion passive n'est pas équivalente à de la pratique délibérée.

**Ce que l'immersion apporte :** volume, contexte signifiant, expositions variées.
**Ce qu'elle ne remplace pas :** la correction ciblée des erreurs fossilisées, les drills sur les points faibles.

La formule efficace : **immersion + boucle de feedback active**.
- Regarder une série en espagnol (immersion) *et* noter les constructions incompréhensibles pour les travailler après (feedback ciblé).
- Jouer de la guitare avec un groupe (immersion) *et* s'enregistrer pour écouter les passages instables (feedback différé).

---

> **À retenir :**
> - Les trois stades de Fitts & Posner — cognitif, associatif, autonome — décrivent la progression naturelle d'une compétence et dictent la stratégie de pratique à chaque moment.
> - Au stade cognitif : exemples résolus, décomposition, feedback fréquent. Au stade associatif : fading, drills ciblés, interleaving. Au stade autonome : pratique espacée, nouveau challenge.
> - Décomposer une compétence avant de commencer (sous-composantes + goulots d'étranglement) est plus efficace que s'y attaquer en bloc.
> - Le feedback doit être calibré selon le stade : très fréquent au stade cognitif, espacé au stade autonome.
> - L'immersion sans feedback ciblé est du volume, pas de la pratique délibérée.

---

## Flash-cards

**Q1** : Quels sont les trois stades de Fitts & Posner et en quoi chacun se distingue-t-il ?
**R1** : (1) Cognitif : attention maximale, erreurs fréquentes, chaque sous-étape est consciente. (2) Associatif : sous-étapes qui se soudent, erreurs qui diminuent, moins de charge attentionnelle. (3) Autonome : automatisation, faible charge cognitive, performance stable.

**Q2** : Qu'est-ce qu'un "worked example" et à quel stade est-il le plus utile ?
**R2** : Un exemple entièrement résolu, commenté, que l'apprenant étudie avant de tenter de produire seul. Utile principalement au stade cognitif — résoudre à froid surcharge le novice (Sweller et al., 2019).

**Q3** : Qu'est-ce que le "fading" et à quel moment l'introduire ?
**R3** : Retrait progressif de l'aide (exemple complet → exemple à trous → problème sans aide). À introduire au passage du stade cognitif au stade associatif, quand les grandes erreurs ont disparu.

**Q4** : Pourquoi un feedback trop fréquent peut-il nuire au stade autonome ?
**R4** : Parce que l'automatisation se construit précisément quand le circuit fonctionne sans supervision externe constante. Un feedback trop dense crée une dépendance et ralentit l'ancrage du pattern automatique.

**Q5** : Quelle est la différence entre immersion et pratique délibérée dans une compétence ouverte (ex. langue) ?
**R5** : L'immersion fournit du volume et du contexte signifiant, mais ne corrige pas les erreurs fossilisées ni ne travaille les points faibles précis. La pratique délibérée cible ce que l'immersion ne corrige pas seule — d'où la formule : immersion + boucle de feedback actif.

---

## Points clés à retenir

1. **Fitts & Posner comme boussole :** diagnostiquer à quel stade tu te trouves sur chaque sous-compétence change radicalement la stratégie à adopter — pas de recette unique.
2. **Décomposer avant de commencer :** lister les sous-composantes et identifier le goulot d'étranglement évite de pratiquer ce qu'on sait déjà au lieu de travailler ce qui bloque.
3. **Worked examples au départ, fading ensuite :** l'erreur classique du débutant est de vouloir produire seul trop tôt — commencer par les exemples résolus réduit la surcharge cognitive.
4. **Drills ciblés sur la faiblesse, pas sur la zone de confort :** un drill n'est utile que s'il pousse au-delà de ce que tu maîtrises déjà.
5. **L'automatisation n'est pas la fin :** au stade autonome, la progression reprend en attaquant une sous-compétence de niveau supérieur — et en revenant temporairement au stade cognitif.

---

## Pour aller plus loin

- **Fitts, P. M., & Posner, M. I. (1967).** *Human Performance*. Brooks/Cole. Référence & synthèse : https://www.oxfordreference.com/display/10.1093/oi/authority.20110803095821507
- **Sweller, J., van Merriënboer, J. J. G., & Paas, F. (2019).** *Cognitive Architecture and Instructional Design: 20 Years Later*, *Educational Psychology Review*, 31, 261–292. https://link.springer.com/article/10.1007/s10648-019-09465-5 — la synthèse sur les worked examples et le fading.
- **Gick, M. L., & Holyoak, K. J. (1983).** *Schema Induction and Analogical Transfer*, *Cognitive Psychology*, 15(1), 1–38. https://www.sciencedirect.com/science/article/abs/pii/0010028583900026 — pourquoi décomposer et nommer la structure aide le transfert.
- **Module 07 de ce domaine** — la pratique délibérée (Ericsson) : les quatre conditions de base dont ce module est l'application opérationnelle.
- **Module 04 de ce domaine** — interleaving & difficultés désirables : à combiner avec les drills du stade associatif.
