# Module 05 — Heuristiques & Biais

> **Temps estimé** : 45 min | **Prérequis** : Modules 01-04
>
> **Objectif** : Identifier les quatre heuristiques/biais les mieux établis (ancrage, disponibilité, cadrage, biais de confirmation), comprendre leurs mécanismes et leurs contre-mesures, et calibrer son niveau de confiance sur l'état réel de la réplication en psychologie.

---

## 1. Heuristiques : outils, pas défauts

Une **heuristique** est une règle de décision rapide et frugale qui exploite les régularités de l'environnement. Tversky et Kahneman (1974) ont montré qu'elles produisent des erreurs *systématiques et prévisibles*. Gigerenzer et ses collègues ont montré l'autre face : dans des environnements bruités, une règle simple peut **prédire mieux** qu'un modèle complexe, parce qu'elle évite l'*overfitting*.

Ces deux cadres — "heuristiques comme sources d'erreurs" (Kahneman) et "heuristiques comme outils adaptatifs" (Gigerenzer) — sont **complémentaires**, pas opposés. Ce module couvre les quatre biais dont la robustesse est bien établie.

> **Honnêteté sur la preuve** : les effets décrits ci-dessous ont été répliqués dans de nombreuses études indépendantes. **Ne pas les confondre avec les effets de priming social** (café chaud, amorçage « vieillesse »), qui ont largement échoué à la réplication après 2011 (voir encadré §2).

---

## 2. Encadré — La crise de réplication en psychologie

> **À lire avec soin** : cet encadré vous aide à calibrer votre confiance sur les résultats psychologiques.

En 2015, l'Open Science Collaboration (OSC) a tenté de répliquer 100 études publiées dans des revues phares. Résultat : seulement **~36 % des réplications** étaient statistiquement significatives, et les tailles d'effet répliquées étaient en moyenne **la moitié** de l'effet original.

**Ce qui a résisté** : ancrage numérique, biais de cadrage, biais de confirmation — effets larges, répliqués dans de nombreux laboratoires et conditions.

**Ce qui a échoué** : effets de *priming social* (ex. : tenir une tasse chaude → plus d'empathie ; lire des mots liés à la vieillesse → marcher plus lentement). Kahneman lui-même a reconnu en 2012 que ce domaine avait un « problème de train ». Ces effets ne sont **pas enseignés ici comme acquis**.

**Leçon méthodologique** : un seul article, même publié dans *Science*, n'établit pas un fait — la convergence de réplications indépendantes avec tailles d'effet documentées compte. C'est exactement l'attitude que ce cours vous entraîne à adopter.

*Source : Open Science Collaboration, 2015. Science 349(6251):aac4716. https://osf.io/ezcuj/overview*

---

## 3. Ancrage numérique

**Mécanisme** : lorsqu'un chiffre est présenté en premier, il tire les estimations suivantes vers lui — même si ce chiffre est manifestement arbitraire et sans rapport avec la question posée.

**Expérience classique (neutre)** — Tversky & Kahneman (1974) :
Participants invités à faire tourner une roue de loterie truquée (10 ou 65), puis à estimer le **nombre de pays membres de l'ONU**. Groupe roue=10 : médiane **25**. Groupe roue=65 : médiane **45**. Écart de 20 pays pour un chiffre généré aléatoirement, sans lien avec la question.

**Quand l'ancrage aide** : en négociation, annoncer sa position en premier ancre la fourchette de discussion.

**Quand il trompe** : toute estimation chiffrée — budget, durée d'un projet, valeur d'un bien — risque d'être contaminée par le premier chiffre aperçu, même fortuit.

**Contre-mesure** :
1. Générer sa propre estimation *avant* de consulter des chiffres de référence.
2. Chercher des arguments dans la direction *opposée* à l'ancre.
3. Multiplier les perspectives indépendantes (agrégation de prévisions non-ancrées).

---

## 4. Heuristique de disponibilité

**Mécanisme** : on estime la fréquence ou la probabilité d'un événement par la **facilité avec laquelle des exemples viennent à l'esprit** — et non par les statistiques de base.

**Exemple neutre** : après une couverture médiatique intensive d'accidents d'avion, la majorité des gens surestiment le risque de mort en avion et sous-estiment celui en voiture, alors que les chiffres de mortalité par kilomètre parcouru sont inverses (voiture : ~7 × plus mortel par km, données NHTSA/IATA).

La saillance émotionnelle et médiatique d'un événement n'est pas un indicateur fiable de sa fréquence réelle.

**Quand la disponibilité aide** : si les événements fréquents sont aussi saillants (bruits quotidiens dans l'environnement), la heuristique fonctionne bien.

**Contre-mesures** :
1. Chercher les statistiques agrégées (*base rates*) avant de juger.
2. Se demander : « Cet exemple me vient-il facilement parce qu'il est fréquent, ou parce qu'il est spectaculaire ? »

---

## 5. Cadrage (*Framing Effect*)

**Mécanisme** : la présentation d'une même information change les décisions, même quand les options sont **mathématiquement identiques**.

**Expérience classique — problème des 600 patients** — Tversky & Kahneman (1981) :

Scénario : une épidémie menace 600 personnes. Deux programmes de réponse.

| Cadrage | Programme A / C | Programme B / D |
|---------|----------------|----------------|
| **Gain** | A : 200 personnes **sauvées** avec certitude | B : 1/3 de chance de sauver les 600, 2/3 que personne ne soit sauvé |
| **Perte** | C : 400 personnes **mourront** avec certitude | D : 1/3 de chance que personne ne meure, 2/3 que 600 meurent |

A = C et B = D en espérance. Pourtant : en cadrage gain, la majorité choisit A (option certaine) ; en cadrage perte, la majorité choisit D (option risquée). Les préférences **s'inversent**.

Ce résultat est robuste et répliqué dans de nombreuses cultures et contextes.

**Contre-mesure** : avant toute décision importante, reformuler l'option dans les **deux cadrages** (gain *et* perte) et vérifier si votre choix reste cohérent.

---

## 6. Biais de confirmation

**Mécanisme** : on cherche, interprète et mémorise préférentiellement les informations qui **confirment** nos croyances préexistantes, tout en accordant moins de poids à celles qui les infirment.

**Tâche de Wason (version abstraite)** — Wason (1968) :

On vous montre 4 cartes. Chaque carte a une lettre d'un côté et un chiffre de l'autre.

```
[ E ]   [ K ]   [ 4 ]   [ 7 ]
```

Règle à vérifier : *« Si une carte a une voyelle d'un côté, alors elle a un chiffre pair de l'autre. »*

**Quelles cartes faut-il retourner pour tester cette règle ?**

La majorité des gens retourne **E et 4**. Correct : **E et 7**.

- **E** : il faut vérifier si l'autre côté est pair (logique).
- **4** : la règle ne dit rien sur ce qu'il y a de l'autre côté d'un chiffre pair — inutile.
- **7** : si l'autre côté était une voyelle, la règle serait réfutée — indispensable, mais peu retourné.
- **K** : consonne, hors scope de la règle.

On cherche à confirmer (E et 4 valident la règle) plutôt qu'à réfuter (7 peut l'infirmer).

**Contre-mesures** :
1. Formuler explicitement **l'hypothèse réfutante** : « Qu'est-ce qui devrait être vrai si j'ai *tort* ? »
2. Chercher activement la preuve contraire avant de conclure.
3. Appliquer la méthode bayésienne (Module 04) : forcer la mise à jour même sur des données défavorables.

---

## 7. Récapitulatif pratique

| Biais | Signal d'alerte | Contre-mesure rapide |
|-------|----------------|----------------------|
| **Ancrage** | Un chiffre est présent avant l'estimation | Estimer indépendamment, puis chercher des références |
| **Disponibilité** | Des exemples frappants dominent mon jugement | Chercher les statistiques agrégées (base rates) |
| **Cadrage** | Ma décision change si on reformule | Reformuler dans les deux sens (gain/perte) |
| **Confirmation** | Je cherche ce qui valide ma position | Chercher l'argument réfutant le plus fort |

---

> **À retenir** :
> - Ancrage, disponibilité, cadrage et biais de confirmation sont des effets robustes et répliqués.
> - Les effets de priming social (OSC 2015) sont fragiles — ne pas les confondre avec ces quatre biais.
> - Une heuristique n'est pas une erreur : son efficacité dépend de l'appariement à l'environnement (Gigerenzer).
> - Contre-mesure universelle : formuler l'hypothèse réfutante *avant* de chercher des confirmations.

---

## Flash-cards (Module 05)

**Q1 : Qu'est-ce que l'ancrage numérique ? Donnez un exemple neutre chiffré.**
> R : Tendance à tirer ses estimations vers un chiffre présenté en premier, même arbitraire. Ex : une roue de loterie affichant 65 fait estimer le nombre de pays membres de l'ONU à ~45 en médiane, contre ~25 pour une roue affichant 10 (Tversky & Kahneman, 1974).

**Q2 : Pourquoi l'heuristique de disponibilité peut-elle induire en erreur dans l'évaluation des risques ?**
> R : Parce que la saillance médiatique ou émotionnelle d'un événement ne corrèle pas avec sa fréquence réelle. Un accident d'avion très couvert peut sembler plus probable qu'un accident de voiture, alors que c'est l'inverse par km parcouru.

**Q3 : Dans le problème des 600 patients, pourquoi les préférences s'inversent-elles selon le cadrage ?**
> R : En cadrage gain, on est averse au risque (on préfère la certitude). En cadrage perte, on devient preneur de risque pour éviter une perte certaine. Pourtant A=C et B=D en espérance.

**Q4 : Dans la tâche de Wason abstraite (E, K, 4, 7 — règle « voyelle → chiffre pair »), quelles cartes faut-il retourner et pourquoi ?**
> R : E (vérifier si l'autre côté est pair) et 7 (si l'autre côté était une voyelle, la règle serait réfutée). La carte 4 est inutile car la règle ne contraint pas ce qu'on trouve derrière un chiffre pair.

**Q5 : Qu'enseigne la crise de réplication de 2015 (OSC) sur les biais psychologiques ?**
> R : Seulement ~36 % des réplications étaient significatives. Les biais les mieux répliqués (ancrage, cadrage) restent solides. Les effets de priming social sont fragiles et ne doivent pas être présentés comme acquis.

---

## Points clés à retenir

1. Ancrage, disponibilité, cadrage, biais de confirmation : robustes et répliqués de manière convergente.
2. Priming social (café chaud, marche lente) : fragile — ne pas l'enseigner comme un fait établi (OSC 2015).
3. Les heuristiques ne sont pas des défauts moraux : leur efficacité dépend de l'environnement (Gigerenzer).
4. Contre-mesure centrale : expliciter l'hypothèse réfutante et chercher à se réfuter soi-même.
5. Reformuler dans les deux cadrages (gain/perte) avant toute décision importante.

---

## Pour aller plus loin

- **Article fondateur** : Tversky, A. & Kahneman, D. (1974). *Judgment under Uncertainty: Heuristics and Biases.* Science, 185(4157), 1124-1131. https://www.science.org/doi/10.1126/science.185.4157.1124
- **Monographie** : Kahneman, D. (2011). *Thinking, Fast and Slow.* Farrar, Straus and Giroux. *(À lire avec le recul sur le priming : chapitre 4 est fragile.)*
- **Contrepoint essentiel** : Gigerenzer, G., Todd, P. M. & ABC Research Group (1999). *Simple Heuristics That Make Us Smart.* Oxford University Press. https://global.oup.com/academic/product/simple-heuristics-that-make-us-smart-9780195143812
- **Crise de réplication** : Open Science Collaboration (2015). *Estimating the Reproducibility of Psychological Science.* Science, 349(6251). https://osf.io/ezcuj/overview
- **Tâche de Wason originale** : Wason, P. C. (1968). *Reasoning about a rule.* Quarterly Journal of Experimental Psychology, 20(3), 273-281.
