# Exercices (hard) — Module 05 : Attention & deep work

> **Niveau** : Avance | **Temps estime** : 60-75 min
> Conception d'un systeme de deep work sur deux semaines, calcul du budget d'attention reel, et analyse critique d'un mythe sur l'attention.

---

## Exercice 1 — Concevoir ton systeme de deep work sur 2 semaines

### Objectif
Concevoir, instrumenter et rendre mesurable un systeme de deep work que tu vas reellement executer, avec une boucle d'amelioration.

### Consigne
Concois un systeme de deep work documente pour un objectif reel sur 2 semaines.

1. **Cible et budget** : combien d'heures de deep work par semaine vises-tu, reparties comment ? (Sois realiste : un debutant ne tient pas 4 h/jour.)
2. **Architecture des blocs** : duree, nombre/jour, pauses, et progression eventuelle de la duree au fil des jours (montee en charge progressive).
3. **Protection** : regles d'environnement + protocole de gestion des interruptions internes/externes.
4. **Couplage avec l'encodage** : comment chaque bloc se termine-t-il par du retrieval, et comment l'espacement organise les revisions de ce qui est appris en deep work ?
5. **Instrumentation** : comment mesures-tu le **ratio de deep work effectif** (temps reellement concentre / temps assis) chaque jour, et quel seuil declenche un ajustement ?
6. **Boucle** : bilan a J+7, 1-2 ajustements possibles.

### Criteres de reussite
- [ ] Le budget hebdo est realiste et reparti (pas un ideal intenable)
- [ ] L'architecture des blocs inclut une montee en charge progressive
- [ ] La protection couvre interruptions internes ET externes
- [ ] Chaque bloc est couple a du retrieval, et l'espacement organise les revisions
- [ ] Un ratio de deep work effectif est mesure, avec un seuil d'ajustement
- [ ] Une boucle d'amelioration a J+7 est definie

---

## Exercice 2 — Calculer ton budget d'attention reel

### Objectif
Quantifier l'ecart entre temps "passe a etudier" et temps reellement productif, pour piloter par le ratio plutot que par les heures affichees.

### Consigne
On modelise une session avec interruptions.

1. Tu etudies **90 minutes**. Tu es interrompu **6 fois**. Chaque interruption dure 2 minutes ET coute en plus ~3 minutes de "re-chargement du contexte" (le temps de revenir au niveau de concentration anterieur — ordre de grandeur illustratif, pas une constante universelle).
2. Calcule : temps perdu direct, temps de re-chargement total, temps de deep work effectif restant, et le **ratio deep work effectif / 90 min**.
3. Refais le calcul si tu reduis a **2 interruptions**. Compare les deux ratios.
4. Montre qu'eliminer 4 interruptions "ne fait pas gagner 4 × 2 = 8 min" mais bien plus — quantifie le gain reel et explique pourquoi (le cout de re-chargement domine).
5. **Honnetete sur la preuve** : precise que le "3 min de re-chargement" est un ordre de grandeur pour raisonner, pas une mesure exacte ; ce qui est solide, c'est l'*existence* d'un cout de reprise non negligeable, pas sa valeur precise.

### Criteres de reussite
- [ ] Les 4 quantites du point 2 sont calculees correctement
- [ ] Le ratio est recalcule pour 2 interruptions et compare
- [ ] Le gain reel d'eliminer 4 interruptions est chiffre et explique (cout de re-chargement, pas seulement 2 min/interruption)
- [ ] Le paragraphe d'honnetete distingue l'ordre de grandeur illustratif de ce qui est reellement etabli
- [ ] Aucune valeur n'est presentee comme un resultat d'etude precis

---

## Exercice 3 — Analyser un mythe : "les jeunes sont nes multitaskers / le cerveau peut s'entrainer a tout faire en meme temps"

### Objectif
Demonter le mythe du multitasking efficace (et du brain-training generalisant) avec le mecanisme et la preuve, sans nier ce qui est vrai.

### Consigne
Une croyance repandue : *"Les jeunes generations, nees avec les ecrans, sont devenues de vrais multitaskers ; et de toute facon, avec des jeux d'entrainement cerebral, on peut muscler son attention pour tout gerer a la fois."*

Redige une analyse structuree :
1. **La part de vrai** : qu'est-ce qui peut s'ameliorer reellement avec l'entrainement (une tache *specifique*, l'automatisation) ?
2. **L'erreur 'multitasking'** : pourquoi le cerveau ne traite pas deux taches attentionnelles en parallele mais alterne avec un cout de commutation ? Quel role joue la memoire de travail limitee (~4 chunks, Cowan 2001) ?
3. **L'erreur 'brain-training'** : restitue ce qu'etablit Simons et al. (2016) (>130 etudes) — amelioration sur la tache entrainee, **pas** de transfert generalise. Relie au fait qu'aucun jeu ne "muscle l'attention en general".
4. **Pourquoi le mythe survit** : 2 raisons (familiarite avec les outils confondue avec aptitude ; promesse d'un raccourci).
5. **Garde-fou anti-sur-correction** : que serait-il faux d'affirmer en sens inverse ? (Ex : "on ne peut jamais rien faire en parallele" — distingue taches automatisees vs taches attentionnelles.)
6. **Regle pratique** en 2 phrases.

### Criteres de reussite
- [ ] La part de vrai (gain sur tache specifique, automatisation) est reconnue
- [ ] Le mecanisme du switching cost + memoire de travail limitee est explique
- [ ] Simons et al. (2016) est cite correctement (pas de transfert generalise, >130 etudes)
- [ ] Au moins 2 raisons de survie du mythe sont donnees
- [ ] La sur-correction est nuancee (taches automatisees vs attentionnelles)
- [ ] Ni le "multitasking efficace" ni le brain-training generalisant ne sont valides

---

*Solutions disponibles dans `03-exercises/solutions/05-attention-deep-work-hard.md`*
