# Module 09 — Pensée causale

> **Temps estimé** : 45 min | **Prérequis** : Modules 01–08
> **Objectif** : Distinguer corrélation et causalité, identifier les confondants, comprendre les contrefactuels, et saisir pourquoi la randomisation (RCT) est le standard de référence pour établir une cause.

---

## 1. Corrélation ≠ causalité : le point de départ

### 1.1 Un exemple frappant : glaces et noyades

Chaque été, deux phénomènes augmentent simultanément dans une ville balnéaire :

| Mois | Ventes de glaces (milliers) | Noyades |
|------|-----------------------------|---------|
| Janvier | 12 | 2 |
| Avril | 28 | 5 |
| Juillet | 95 | 18 |
| Octobre | 31 | 6 |

Corrélation ? **Oui, forte et positive.** Conclusion logique ? Les glaces causent les noyades ?

Évidemment non. Les deux variables montent ensemble parce qu'elles ont un **ancêtre commun** : la chaleur estivale pousse les gens à manger des glaces *et* à se baigner. La chaleur est un **confondant**.

> **À retenir** : Une corrélation mesure une co-variation statistique. Elle ne dit rien sur la direction ni sur l'existence d'une relation causale.

### 1.2 Autres exemples canoniques

- **Cigognes et naissances** : Les pays européens avec plus de cigognes ont plus de naissances. Confondant : la taille du pays (grandes zones rurales → plus d'habitat pour les cigognes *et* plus de naissances en valeur absolue).
- **Engrais et rendement** : Dans un champ hétérogène, les zones qui reçoivent plus d'engrais peuvent avoir été choisies par l'agriculteur car elles sont déjà plus fertiles. Confondant : la qualité initiale du sol.

---

## 2. Les confondants : qui se cache derrière la corrélation ?

### 2.1 Définition formelle

Un **confondant** (ou variable confusionnelle, *confounder* en anglais) est une variable Z qui :

1. **Cause** (ou prédit) X (la variable supposée explicative)
2. **Cause** (ou prédit) Y (la variable supposée expliquée)
3. **N'est pas** sur le chemin causal entre X et Y

```
    Z (chaleur)
   ↙          ↘
X (glaces)    Y (noyades)
```

X et Y se corrèlent, *non pas* parce que l'un cause l'autre, mais parce qu'ils partagent la cause commune Z.

### 2.2 Contrôler un confondant

Pour briser la corrélation spurieuse, on **conditionne sur Z** : on compare les observations où Z est fixé à une même valeur.

Exemple : si l'on regarde *uniquement les journées à 35 °C*, est-ce que les ventes de glaces prédisent encore les noyades ? Non — à température constante, la relation disparaît (ou s'effondre fortement).

> **À retenir** : Conditionner sur un confondant (le tenir fixe statistiquement) permet de tester si la corrélation X-Y survit une fois Z contrôlé.

---

## 3. Le raisonnement contrefactuel

### 3.1 La question causale fondamentale

Judea Pearl formule la causalité autour d'une question : *"Qu'est-ce qui se serait passé si X avait été différent ?"*

> Si cet agriculteur n'avait **pas** utilisé d'engrais, quel aurait été son rendement ?

Ce monde alternatif — où X est changé mais tout le reste est tenu constant — est un **contrefactuel**. On ne peut jamais l'observer directement pour un individu donné (Pearl appelle cela le *fundamental problem of causal inference*).

### 3.2 L'effet causal moyen

Puisqu'on ne peut observer le contrefactuel individuel, on l'**estime en groupe** :

- Effet causal moyen = (rendement moyen avec engrais) − (rendement moyen sans engrais)

Le défi : le groupe "avec engrais" et le groupe "sans engrais" doivent être **comparables sur tout le reste** pour que cette soustraction ait un sens causal.

---

## 4. L'essai contrôlé randomisé (RCT) : la randomisation comme solution

### 4.1 Pourquoi randomiser ?

Dans une étude **observationnelle**, l'agriculteur choisit *lui-même* où mettre l'engrais — et il le met sur ses meilleures parcelles. Les groupes "traité" et "contrôle" ne sont pas comparables : le sol de départ diffère.

Dans un **RCT (Randomized Controlled Trial)**, on **tire au sort** qui reçoit le traitement. La randomisation garantit que :

- Les confondants connus *et* inconnus sont distribués également entre les groupes
- Toute différence de résultat observée après le traitement est causalement attribuable au traitement

```
Population
    ↓  tirage au sort
 ┌──────────┬──────────┐
 │  Groupe  │  Groupe  │
 │  traité  │ contrôle │
 └──────────┴──────────┘
   reçoit X   ne reçoit pas X
        ↓
   mesurer Y dans les deux groupes
   → effet = Y_traité − Y_contrôle
```

> **À retenir** : La randomisation "équilibre" tous les confondants (même ceux qu'on n'a pas mesurés). C'est pour ça que le RCT est appelé l'**étalon-or** (gold standard) de la causalité.

### 4.2 Exemple : A/B test produit

Un site web veut savoir si un nouveau bouton "Commander" (plus visible) augmente les ventes. Il **assigne aléatoirement** les visiteurs :
- 50 % voient l'ancien bouton
- 50 % voient le nouveau bouton

Les deux groupes sont comparables (même période, même trafic). Si le taux de conversion est de 4,2 % vs 3,8 %, la différence est causalement attribuable au bouton — pas à des différences de profil entre les visiteurs.

### 4.3 Limites du RCT

Le RCT ne s'applique pas toujours :
- **Éthique** : on ne peut pas exposer un groupe à un risque avéré.
- **Faisabilité** : impossible de randomiser l'altitude à laquelle vit quelqu'un.
- **Coût et durée** : un essai en médecine peut prendre 10 ans.

Quand le RCT est impossible, les méthodes observationnelles rigoureuses (avec appariement, variables instrumentales, etc.) permettent d'**approcher** la causalité — mais avec plus d'hypothèses à justifier (Module 10).

---

## 5. L'échelle de la causalité (Pearl)

Pearl propose trois niveaux de raisonnement causal :

| Niveau | Question | Exemple |
|--------|----------|---------|
| **Observation** | Que s'est-il passé ? | Corrélation glaces ↔ noyades |
| **Intervention** | Que se passerait-il si je faisais X ? | Si je donne l'engrais à cette parcelle ? |
| **Contrefactuel** | Qu'aurais-je obtenu si j'avais fait autrement ? | Cette parcelle qui a reçu l'engrais aurait-elle eu autant de rendement sans lui ? |

La statistique classique opère au niveau 1. Le RCT nous donne accès au niveau 2. Le niveau 3 est le plus difficile — il nécessite un modèle causal complet (ou des hypothèses fortes).

---

## 6. Synthèse visuelle : comment diagnostiquer une corrélation

```
Corrélation observée entre X et Y
         ↓
         ┌──────────────────────────────────────────┐
         │ Y a-t-il un confondant Z plausible ?     │
         └──────────────────────────────────────────┘
            Oui ↓                    Non ↓
   Contrôler Z (ou randomiser)   Chercher la direction :
   → corrélation persistante ?    X→Y ? ou Y→X ? (causalité inverse)
            Oui ↓
   Lien causal plausible
   (toujours provisoire : d'autres confondants ?)
```

> **À retenir** : Tout lien causal affirmé sans randomisation ou contrôle rigoureux des confondants reste une **hypothèse**, non une démonstration.

---

## Flash-cards

**Q1 : Qu'est-ce qu'un confondant ?**
> R : Une variable Z qui cause à la fois X et Y, créant une corrélation spurieuse entre X et Y sans lien causal direct entre eux.

**Q2 : Pourquoi les ventes de glaces et les noyades sont-elles corrélées ?**
> R : Parce qu'un confondant commun — la chaleur estivale — augmente les deux simultanément. La corrélation est réelle, la causalité entre glaces et noyades est nulle.

**Q3 : Que signifie "conditionner sur un confondant" ?**
> R : Fixer la valeur de Z (ex. : comparer uniquement les jours à 35 °C) pour tester si la corrélation X-Y subsiste une fois Z neutralisé.

**Q4 : Pourquoi la randomisation résout-elle le problème des confondants ?**
> R : En assignant aléatoirement le traitement, elle équilibre en moyenne tous les confondants (connus et inconnus) entre les groupes, rendant ceux-ci comparables.

**Q5 : Qu'est-ce qu'un contrefactuel en inférence causale ?**
> R : Le scénario alternatif hypothétique : "qu'aurait-il observé si X avait été différent ?" — non observable directement, estimé en moyenne via des groupes comparables.

---

## Points clés à retenir

1. **Corrélation ≠ causalité** : deux variables peuvent co-varier sans que l'une cause l'autre.
2. **Les confondants** sont la cause la plus fréquente de corrélations trompeuses : ils partagent un ancêtre commun avec les deux variables observées.
3. **Conditionner sur Z** (tenir Z fixe) permet de tester si la corrélation X-Y est spurieuse.
4. **La randomisation (RCT)** équilibre tous les confondants, connus ou non — c'est pourquoi elle est l'étalon-or pour établir la causalité.
5. **Les contrefactuels** formalisent la question causale ("qu'aurait-on obtenu sans X ?") et expliquent pourquoi comparer des groupes non randomisés est délicat.
6. Quand un RCT est impossible, des méthodes observationnelles rigoureuses peuvent approcher la causalité, mais avec des hypothèses supplémentaires à justifier.

---

## Pour aller plus loin

- **The Book of Why** — Judea Pearl & Dana Mackenzie, 2018 (Basic Books). Vulgarisation accessible du raisonnement causal, des DAG et de l'opérateur *do*. https://en.wikipedia.org/wiki/The_Book_of_Why
- **Causal Inference: What If** — Miguel A. Hernán & James M. Robins, 2020 (Chapman & Hall/CRC) — **PDF gratuit en ligne**. Manuel rigoureux : confondants, contrefactuels, DAG, RCT vs observationnel. https://miguelhernan.org/whatifbook
- **Causal Inference in Statistics: A Primer** — Judea Pearl, Madelyn Glymour & Nicholas P. Jewell, 2016 (Wiley). Introduction formelle accessible (~160 p., questions d'étude incluses). https://web.cs.ucla.edu/~kaoru/primer-complete-2019.pdf
