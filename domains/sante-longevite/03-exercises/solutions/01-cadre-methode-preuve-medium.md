# Solutions (medium) — Module 01 : Cadre, methode & niveaux de preuve

> Ces corriges sont des modeles de reference, pas des reponses uniques. La methode de calcul et la calibration du jugement comptent plus que la formulation exacte.
>
> **⚠️ Disclaimer medical.** Contenu educatif. Ne remplace pas un avis medical. Les exemples chiffres sont fictifs ou simplifies a des fins pedagogiques.

---

## Exercice 1 — Convertir un risque relatif en risque absolu

### Corrige modele

**Cas A — Depistage (RRR = 50 %, base = 2/1000 sur 10 ans)**
- Risque absolu groupe expose : 2/1000 × (1 - 0,50) = **1/1000**
- Reduction absolue du risque (RAR) : 2/1000 - 1/1000 = 1/1000 = **0,001 (0,1 %)**
- NNT = 1 / 0,001 = **1000** → il faut "traiter"/depister 1000 personnes pour eviter 1 cas sur 10 ans
- Conclusion : le "-50 %" est vrai mais trompeur. L'impact absolu est minuscule (0,1 %) parce que la maladie est rare. NNT de 1000 = faible benefice individuel, a mettre en balance avec couts et faux positifs du depistage.

**Cas B — Comportement / DPP-like (RRR = 58 %, base = 290/1000 sur 3 ans)**
- Risque absolu groupe intervention : 290/1000 × (1 - 0,58) = 290 × 0,42 ≈ **122/1000**
- RAR : 290 - 122 = **168/1000 = 0,168 (16,8 %)**
- NNT = 1 / 0,168 ≈ **6** → environ 6 personnes a accompagner pour eviter 1 cas de diabete sur 3 ans
- Conclusion : ici le "-58 %" ET l'absolu sont impressionnants, parce que le risque de base est eleve (population prediabetique). NNT ≈ 6 = effet cliniquement majeur. C'est exactement pourquoi le DPP fait reference.

**Cas C — Aliment (RR = 2,0, base = 1/100 000 sur 10 ans)**
- Risque absolu groupe expose : 1/100 000 × 2,0 = **2/100 000**
- Augmentation absolue : 2/100 000 - 1/100 000 = 1/100 000 = **0,00001 (0,001 %)**
- NNH = 1 / 0,00001 = **100 000** → il faut 100 000 personnes exposees pour 1 cas supplementaire
- Conclusion : "double le risque" sonne effrayant, mais doubler un risque infime donne un risque toujours infime. Impact absolu negligeable pour un individu.

**Enseignement cle** : un meme verbe ("reduit de 50 %", "double") cache des realites cliniques opposees selon le risque de base. Toujours exiger : risque de base + risque absolu + NNT/NNH. Le RR seul ne permet aucune decision rationnelle.

---

## Exercice 2 — Trier un dossier de preuves contradictoires

### Corrige modele

**1. Classement par niveau de preuve**
- **S4** (meta-analyse de 9 ECR) — niveau 1, le plus fort : agrege plusieurs ECR, augmente la puissance, reduit le hasard.
- **S3** (ECR double aveugle) — niveau 2 : design causal, mais petit (60 sujets) donc puissance limitee.
- **S2** (transversale) — niveau 4 : une seule mesure, association sans temporalite, fortement confondue.
- **S1** (temoignages d'influenceur) — niveau 5 : anecdotique, biais d'auto-selection et de placebo, possible conflit d'interet commercial.

**2. "Pas de difference" prouve-t-il l'inefficacite ?**
Non. S3 est un petit ECR (60 sujets). Avec si peu de sujets, l'etude peut manquer de **puissance statistique** : meme si un petit effet reel existait, l'etude pourrait ne pas le detecter (IC large incluant 0). C'est la distinction cle :
- **Absence de preuve d'effet** = on n'a pas reussi a montrer un effet (peut venir d'un manque de puissance).
- **Preuve d'absence d'effet** = on a montre, avec une etude bien dimensionnee, que l'effet est nul ou trivial (IC etroit autour de 0).
S3 seul releve plutot du premier cas.

**3. Pourquoi S4 pese plus que S3 ?**
La meta-analyse S4 combine 1200 sujets issus de 9 ECR. Elle a donc une **puissance bien superieure** et un **IC etroit autour de 0** — ce qui se rapproche d'une vraie "preuve d'absence d'effet cliniquement pertinent". Elle integre aussi plusieurs contextes, ce qui renforce la generalisabilite et reduit le risque qu'un resultat soit du a un seul echantillon atypique.

**4. Synthese calibree (exemple)**
"Les meilleures preuves disponibles — une meta-analyse de 9 essais randomises (1200 personnes) — ne montrent pas d'effet significatif du complement Z sur la fatigue ; l'effet est au mieux tres faible. Les temoignages enthousiastes relevent probablement de l'effet placebo ou d'autres changements concomitants. Sur cette base, Z ne semble pas un levier efficace, meme si on ne peut jamais exclure un benefice marginal chez certains sous-groupes."

**Enseignement cle** : face a des sources contradictoires, on ne "fait pas la moyenne" — on pondere par le niveau de preuve et la puissance. Un ECR negatif de petite taille n'enterre pas une question ; une meta-analyse d'ECR la tranche bien mieux.

---

## Exercice 3 — Auditer un article de presse reel

### Corrige modele

Cet exercice depend de l'article choisi — il n'y a pas de corrige unique. Voici un **exemple de grille bien remplie** (article fictif type : "Le yaourt reduit le risque de depression de 20 %").

| Question | Reponse modele |
|----------|----------------|
| 1. Source | Etude observationnelle (cohorte), citee en fin d'article |
| 2. Niveau de preuve | 3 (cohorte prospective) |
| 3. Taille d'effet | RR = 0,80 donne ; **risque absolu non fourni** par l'article (signal de prudence) |
| 4. Population | ~30 000 adultes, age moyen 50 ans, un seul pays — generalisabilite limitee |
| 5. Beneficiaires / contre-indications | S'applique a des adultes d'age moyen ; rien sur les jeunes, les personnes deja depressives, intolerants au lactose |

**Ecart titre/etude** : le titre suggere une causalite ("reduit"), alors que le design n'autorise qu'une association. Les consommateurs reguliers de yaourt ont probablement une alimentation et un mode de vie globalement plus sains (confondants).

**Titre reformule** : "Une etude observationnelle associe la consommation reguliere de yaourt a un risque de depression plus faible — un lien qui ne demontre pas de causalite et pourrait refleter un mode de vie plus sain."

**Criteres d'un bon audit** :
- L'article et l'etude sont identifies (pas juste "j'ai lu un truc")
- "Non disponible" est utilise honnêtement quand l'info manque (ex. risque absolu) — c'est un constat, pas un echec
- L'ecart le plus grave est nomme precisement (causalite surestimee est le plus frequent)
- La reformulation degonfle la causalite sans nier l'interet du signal

**Enseignement cle** : la grille des 5 questions transforme une lecture passive en audit actif. Le manque d'information (risque absolu absent, population non precisee) est lui-meme une donnee : un article serieux donne ces elements.
