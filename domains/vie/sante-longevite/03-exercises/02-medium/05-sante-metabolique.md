# Exercices (medium) — Module 05 : Sante metabolique

> **Prerequis** : avoir lu `01-theory/05-sante-metabolique.md` et complete `01-easy/05-sante-metabolique.md`
>
> **⚠️ Disclaimer medical.** Ces exercices sont a visee educative uniquement. Ils ne constituent pas un outil de diagnostic et ne remplacent pas la consultation d'un medecin. L'interpretation d'un bilan biologique reel appartient toujours a un professionnel de sante. Les biomarqueurs et valeurs cites sont des reperes generaux, pas un seuil personnel. La metformine et autres molecules sont des objets de recherche, pas des recommandations.

---

## Exercice 1 — Interpreter une tendance, pas une photo (3 bilans dans le temps)

### Objectif

Depasser la lecture d'un bilan ponctuel (vu en easy) pour raisonner sur la **trajectoire** d'un profil metabolique — la lecon "la tendance, c'est le film ; le bilan, c'est la photo".

### Consigne

> **⚠️ Rappel** : exercice educatif, valeurs fictives. Ne l'utilisez pas pour vous auto-interpreter.

Bilans fictifs de "Kim" sur 3 ans (1 par an), apres une periode de prise de poids puis de changement de mode de vie en annee 3 :

| Biomarqueur | An 1 | An 2 | An 3 | Reference indicative |
|-------------|------|------|------|----------------------|
| Glycemie a jeun (g/L) | 0,98 | 1,06 | 1,01 | < 1,00 ; 1,00-1,25 prediabete |
| HbA1c (%) | 5,5 | 5,9 | 5,7 | < 5,7 ; 5,7-6,4 prediabete |
| Triglycerides (g/L) | 1,30 | 1,75 | 1,45 | < 1,50 souhaitable |
| Tour de taille (cm) | 86 | 94 | 89 | (selon profil/origine) |

1. Decrivez la **trajectoire** de chaque marqueur (an 1 → an 2 → an 3) en mots.
2. Que raconte cette trajectoire sur la sante metabolique de Kim, et sur l'effet du changement de mode de vie en annee 3 ?
3. Pourquoi un seul bilan (ex. an 2) serait-il trompeur, dans un sens comme dans l'autre ?
4. Quels leviers du mode de vie expliquent le plus plausiblement l'amelioration de l'an 3 (mecanismes) ?

### Criteres de reussite

- [ ] La trajectoire de chaque marqueur est decrite (degradation an 2, amelioration an 3)
- [ ] L'interpretation relie la trajectoire au mode de vie (degradation puis intervention) sans poser de diagnostic
- [ ] L'argument "un bilan isole trompe" est explicite (an 2 surestime le risque, an 1/an 3 pourraient rassurer a tort selon le contexte)
- [ ] Au moins 2 leviers (activite → sensibilite a l'insuline ; perte de poids/tour de taille ; alimentation/fibres → triglycerides) sont relies a un mecanisme
- [ ] Le champ lexical reste educatif (pas de diagnostic, renvoi au medecin pour l'interpretation reelle)

---

## Exercice 2 — Du relatif a l'absolu : raisonner le DPP comme un clinicien

### Objectif

Approfondir le DPP (vu en easy) en manipulant incidence absolue, reduction absolue et NNT — pour comprendre pourquoi le -58 % est cliniquement majeur, au-dela du chiffre relatif.

### Consigne

Donnees reelles du module : dans le DPP, incidence annuelle ~**11 %** (controle) vs ~**4,8 %** (mode de vie intensif) ; -58 % relatif ; metformine -31 %.

1. Calculez la **reduction absolue d'incidence annuelle** entre controle et mode de vie. Calculez le **NNT annuel** (combien de personnes accompagnees 1 an pour eviter 1 cas).
2. Comparez : le "-58 % relatif" et la "reduction absolue annuelle" racontent-ils la meme histoire ? Lequel est le plus parlant cliniquement, et pourquoi ?
3. Estimez grossierement l'incidence du groupe metformine (-31 % relatif applique a 11 %) et comparez son NNT a celui du mode de vie. Que conclure sur l'efficacite comparee ?
4. Pourquoi cet ECR (avec randomisation) permet-il une inference causale que les cohortes du module 03/06 ne permettent pas ?

> Rappel : NNT = 1 / reduction absolue (en proportion).

### Criteres de reussite

- [ ] La reduction absolue annuelle (~6,2 points) et le NNT annuel (~16) sont calcules correctement
- [ ] La difference relatif/absolu est expliquee, avec une preference argumentee pour l'absolu en clinique
- [ ] L'incidence metformine (~7,6 %) et son NNT (~29) sont estimes, et la comparaison conclut a la superiorite du mode de vie
- [ ] L'argument "ECR randomise → inference causale" est correctement oppose aux cohortes observationnelles
- [ ] Le statut de la metformine reste celui de l'ECR (bras compare), sans glissement vers une reco de longevite

---

## Exercice 3 — Cartographier les boucles de retroaction metaboliques

### Objectif

Approfondir la carte mentale (vue en easy) en y ajoutant les **boucles de retroaction** (cercles vicieux/vertueux) qui relient sommeil, stress, activite, nutrition et glycemie.

### Consigne

A partir des mecanismes du module (et des modules 02, 03, 06), construisez **2 boucles de retroaction** :

1. **Un cercle vicieux** : decrivez une chaine d'au moins 4 etapes ou la degradation d'un levier en aggrave d'autres, qui reviennent aggraver le premier. (ex. : mauvais sommeil → cortisol → glycemie → ... → mauvais sommeil)
2. **Un cercle vertueux** : decrivez une chaine d'au moins 4 etapes ou l'amelioration d'un levier en ameliore d'autres en retour.
3. Identifiez, dans chaque boucle, **le point d'entree le plus actionnable** (ou intervenir pour casser/amorcer la boucle).
4. Expliquez pourquoi un "plan en silo" (n'agir que sur un levier isole) est moins efficace que d'exploiter ces boucles.

### Criteres de reussite

- [ ] Le cercle vicieux comporte ≥ 4 etapes avec des mecanismes corrects (cortisol, insulinoresistance, ghreline, etc.) et boucle reellement sur lui-meme
- [ ] Le cercle vertueux comporte ≥ 4 etapes avec mecanismes corrects
- [ ] Un point d'entree actionnable est identifie pour chaque boucle, avec justification
- [ ] L'argument anti-silo (interdependance des leviers) est explique avec reference aux boucles construites
