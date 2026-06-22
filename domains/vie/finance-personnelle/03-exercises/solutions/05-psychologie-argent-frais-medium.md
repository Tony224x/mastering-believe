# Solutions — Module 05 : Psychologie de l'argent et impact des frais (niveau intermédiaire)

> Ces corrigés sont des réponses modèles. Les exercices de réflexion (protocole personnel) n'ont pas de solution unique — évaluez la cohérence et l'ancrage dans les concepts du module.
>
> **Disclaimer** : contenu éducatif, rendements/frais **illustratifs**, aucun rendement garanti. **Pas un conseil financier personnalisé.**

---

## Solution Exercice 1 — Chiffrer le behaviour gap (Dara vs Paul)

### Question 1 : capitaux finaux (50 000 € + 300 €/mois, 10 ans)

- Dara (6 %/an) : `capital_final_mensuel(50000, 300, 0.06, 10)` ≈ **138 284 €**
- Paul (3 %/an effectif) : `capital_final_mensuel(50000, 300, 0.03, 10)` ≈ **109 030 €**

### Question 2 : behaviour gap

Écart : 138 284 − 109 030 = **~29 254 €**, soit **~21 %** de moins pour Paul — alors qu'il a investi **exactement la même chose** que Dara (même capital, mêmes versements). La seule différence est le **comportement** : avoir vendu au creux et manqué une partie du rebond.

### Question 3 : biais et "différence comportementale"

Biais de Paul : **aversion à la perte** (vendre pour "stopper la douleur" du −35 %) + **biais de récence** (extrapoler la baisse récente, "ça va continuer à baisser") + une touche de **tentative de market timing** (croire pouvoir revenir "au bon moment"). Le module insiste : ni Dara ni Paul n'ont eu plus de "talent" ou de chance — ils détenaient le même fonds. C'est la **gestion de soi** (rester investi vs paniquer) qui a créé l'écart, d'où "la différence n'est pas technique, elle est comportementale".

### Question 4 : valeur au creux et cristallisation

Au creux (−35 % sur les 50 000 € de départ) : 50 000 × 0,65 = **32 500 €**. Tant que Paul ne vend pas, cette perte est "sur le papier" : si le marché rebondit, la valeur remonte. En **vendant** à 32 500 €, il transforme la perte latente en **perte réelle et définitive** (il a encaissé 32 500 € au lieu de 50 000 €) et se prive du rebond. C'est ce qu'on appelle "cristalliser" la perte.

### Question 5 : dispositif anti-biais

L'**investissement automatique régulier** (Module 05 §4) : un virement programmé qui achète la même somme chaque mois, quels que soient les marchés. Mécanisme : il **retire la décision émotionnelle de l'équation**. Paul n'aurait pas eu à "décider" pendant le krach ; mieux, il aurait continué d'acheter pendant la baisse (plus de parts pour le même budget -> prix moyen plus bas). L'automatisation exploite l'inertie en notre faveur.

---

## Solution Exercice 2 — Frais composés sur trois horizons

### Partie A : tableau (30 000 € initial, 7 % brut, capitalisation annuelle)

`Capital = 30 000 × (1 + 0,07 − frais)^n`

| Frais annuels | Capital à 10 ans | Capital à 20 ans | Capital à 30 ans |
|---|---|---|---|
| 0,05 % (net 6,95 %) | ≈ 58 739 € | ≈ 115 010 € | ≈ **225 188 €** |
| 0,75 % (net 6,25 %) | ≈ 55 006 € | ≈ 100 856 € | ≈ 184 922 € |
| 1,75 % (net 5,25 %) | ≈ 50 043 € | ≈ 83 476 € | ≈ **139 247 €** |

### Partie B : perte due aux frais à 30 ans

Référence (0,05 %) : 225 188 €. Ligne 1,75 % : 139 247 €.
Perte = 225 188 − 139 247 = **~85 941 €**, soit **~38,2 %** du capital de référence. Un écart de 1,7 point de frais détruit plus du tiers du capital final.

### Partie C : pourquoi l'écart en % grandit avec l'horizon

Les frais sont prélevés **chaque année sur tout le capital**, et le manque à gagner qu'ils créent ne peut plus composer les années suivantes. Comme les intérêts composés du Module 01, mais **à l'envers** : chaque euro perdu en frais est aussi un euro qui ne produira jamais d'intérêts. Plus l'horizon est long, plus ce "manque qui ne compose pas" s'accumule — d'où un écart en % qui s'élargit (de quelques % à 10 ans à ~38 % à 30 ans).

### Partie D : perception trompeuse et réflexe

"1,7 point" paraît négligeable parce que notre cerveau raisonne **linéairement** et sous-estime la composition. Mais composé sur 30 ans, c'est ~38 % du capital final perdu. Le réflexe : **chercher le TER (Total Expense Ratio, ou frais courants)** — le coût annuel total du fonds — **avant** de choisir, et le calculer sur l'horizon réel. C'est une décision prise une seule fois, à l'ouverture, mais dont l'effet se compose pendant des décennies.

---

## Solution Exercice 3 — Protocole anti-panique (exemple modèle)

### Partie A : protocole en 5 règles

1. **Allocation cible** : ex. 70 % actions monde (indiciel, TER < 0,20 %) / 30 % obligations. Décidée selon l'horizon et la tolérance au risque, écrite.
2. **Automatisation** : virement de 300 €/mois le jour de paie, sans intervention manuelle.
3. **Consultation** : portefeuille consulté **1 fois par trimestre maximum** ; notifications de cours désactivées.
4. **Rééquilibrage** : 1 fois/an, uniquement si une classe s'écarte de > 5 points de la cible (via les flux entrants en priorité).
5. **Conduite en krach > 20 %** : ne rien vendre, maintenir les versements, relire ce protocole. Aucune décision dans l'émotion.

### Partie B : stress test

| Scénario | Réaction émotionnelle probable | Biais | Ce que le protocole impose |
|---|---|---|---|
| **1 — +30 %, tout le monde en parle** | "Je vais charger, ça monte !" | FOMO + biais de récence | Garder le versement automatique inchangé. Le prix est déjà haut ; on n'augmente pas sur l'euphorie. Éventuel rééquilibrage qui **allège** les actions montées. |
| **2 — krach −30 %, "pire crise depuis 2008"** | "Je vends pour limiter la casse." | Aversion à la perte + biais de récence | Ne rien vendre. Continuer les versements (achat "en solde"). Relire le protocole. |
| **3 — collègue triple sa mise sur un actif spéculatif** | "Je rate l'occasion du siècle." | FOMO + excès de confiance | S'en tenir à l'allocation. Au plus, une "poche spéculative" plafonnée à un montant qu'on peut perdre intégralement, jamais le cœur du portefeuille. |

### Partie C : pourquoi écrire à l'avance

Une bonne résolution prise **dans l'instant** (pendant un krach ou une euphorie) est prise par le "système 1" émotionnel de Kahneman — exactement quand on est le moins rationnel. Un protocole **écrit à froid** transfère la décision au "système 2" réfléchi, et l'automatisation supprime carrément la décision. On suit une règle décidée quand on était lucide, au lieu d'improviser quand on est paniqué ou grisé. C'est le principe central du module : retirer l'émotion de l'équation.

---

## Résumé des enseignements clés (medium)

1. Le **behaviour gap** est chiffrable : paniquer et manquer le rebond peut coûter ~20 % du capital, à investissement identique.
2. Vendre au creux **cristallise** une perte latente en perte réelle.
3. Les **frais composent à l'envers** : un écart de 1,7 point peut détruire ~38 % du capital sur 30 ans -> vérifier le **TER avant** de choisir.
4. Un **protocole écrit à froid** + l'automatisation battent les bonnes résolutions prises dans l'émotion.
5. Rendements et frais **illustratifs** ; aucun rendement garanti.
