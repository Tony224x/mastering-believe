# Exercices (medium) — Module 04 : Nutrition fondee sur les preuves

> **Prerequis** : avoir lu `01-theory/04-nutrition.md` et complete `01-easy/04-nutrition.md`
>
> **⚠️ Disclaimer medical.** Ces exercices sont a visee educative uniquement. Ils ne constituent pas un conseil nutritionnel individualise et ne remplacent pas l'avis d'un medecin ou d'un(e) dieteticien(ne). Ce module ne prescrit aucun regime. Les choix alimentaires dependent de votre etat de sante, vos antecedents et vos preferences.

---

## Exercice 1 — Auditer un regime "etiquete" avec la grille des invariants

### Objectif

Depasser le decryptage de titres (vu en easy) pour appliquer la regle du module — "la structure compte plus que l'etiquette" — a des regimes reels, en evaluant s'ils respectent les invariants independamment de leur nom.

### Consigne

Pour chacun des 3 menus "etiquetes" ci-dessous, evaluez le respect des **invariants** (peu transforme, fibres, proteines suffisantes, sucres ajoutes/trans limites) — **sans juger l'etiquette**.

**Menu "keto" A** : oeufs + bacon industriel le matin ; saucisses + fromage fondu industriel le midi ; steak + beurre le soir. Quasi pas de legumes.

**Menu "keto" B** : oeufs + avocat + epinards ; saumon + grande salade + huile d'olive ; tofu/poulet + brocolis + noix. Glucides tres bas mais beaucoup de legumes.

**Menu "vegan" C** : cereales sucrees au lait vegetal le matin ; "nuggets" vegetaux industriels + frites le midi ; pizza vegetale industrielle le soir.

Pour chaque menu :
1. Cochez quels invariants sont respectes / violes.
2. Concluez : l'etiquette ("keto", "vegan") predit-elle la qualite ? Justifiez avec le contraste A vs B et le cas C.
3. Reformulez chaque menu en gardant l'etiquette mais en le rendant conforme aux invariants (1-2 substitutions).

### Criteres de reussite

- [ ] Les invariants respectes/violes sont identifies correctement pour les 3 menus
- [ ] Le contraste "keto A (mauvais) vs keto B (bon)" est explicite : meme etiquette, qualite opposee
- [ ] Le cas "vegan C ultra-transforme" montre qu'une etiquette "saine" peut etre malsaine
- [ ] La conclusion enonce clairement "structure > etiquette" avec preuve par les exemples
- [ ] Les substitutions proposees respectent les invariants sans changer l'etiquette

---

## Exercice 2 — Lire PREDIMED avec ses reserves (generalisabilite & effet absolu)

### Objectif

Approfondir la lecture critique d'un ECR nutritionnel de reference (PREDIMED) en raisonnant sur la generalisabilite et la difference relatif/absolu — au-dela de "c'est un ECR donc c'est causal".

### Consigne

Rappel du module : PREDIMED (Estruch, NEJM 2018, version corrigee) — ~7 447 sujets **a haut risque cardiovasculaire**, ~30 % de reduction du critere composite CV. La version 2013 a ete retractee (randomisation par menage dans certains centres), republiee en 2018 avec effets comparables.

Repondez :
1. **Population** : les sujets sont "a haut risque CV". Peut-on transferer directement le -30 % a une personne jeune et en bonne sante ? Expliquez le concept de generalisabilite (validite externe).
2. **Relatif vs absolu** : pourquoi un meme "-30 % relatif" produit-il un benefice absolu plus grand chez une population a haut risque que chez une population a bas risque ? (lien avec l'exercice du module 01)
3. **Statut de la preuve** : comment presenter PREDIMED honnetement, en integrant a la fois sa force (ECR, critere dur) et ses reserves (retraction/correction, Cochrane "modere") ?
4. **Invariants vs schema** : qu'est-ce que PREDIMED autorise a dire sur "le regime mediterraneen", et qu'est-ce qu'il **n'autorise pas** a dire (ex. que c'est LE seul regime valide) ?

### Criteres de reussite

- [ ] La generalisabilite (validite externe) est expliquee : effet etabli sur haut-risque, transfert incertain a bas-risque
- [ ] Le lien risque de base → benefice absolu est correct (haut risque = plus de cas evitables)
- [ ] La presentation de PREDIMED integre force ET reserves (correction, "preuve moderee" Cochrane)
- [ ] La distinction "ce que PREDIMED prouve" vs "ce qu'il ne prouve pas" est claire (un exemple soutenu, pas un dogme exclusif)

---

## Exercice 3 — Construire une assiette Harvard pour 3 contextes et budgets

### Objectif

Appliquer la Harvard Healthy Eating Plate (50 % legumes/fruits, 25 % cereales completes, 25 % proteines) a des contraintes reelles (budget, temps, gout), en preservant les invariants.

### Consigne

Construisez **3 versions** d'une journee respectant les proportions Harvard, pour 3 contextes :
- **Contexte 1 — petit budget** : minimiser le cout par portion (legumineuses, surgeles, cereales completes economiques).
- **Contexte 2 — peu de temps** : maximiser la rapidite sans tomber dans l'ultra-transforme (batch cooking, surgeles non panes, conserves rincees).
- **Contexte 3 — sans viande** : proteines vegetales variees couvrant les acides amines.

Pour chaque contexte :
1. Proposez 3 repas respectant ~50/25/25.
2. Indiquez ou se trouvent les fibres et les proteines.
3. Nommez **un piege ultra-transforme** typique de ce contexte et son alternative.

### Criteres de reussite

- [ ] Les 3 journees respectent approximativement les proportions Harvard (50 % legumes/fruits, 25 % cereales completes, 25 % proteines)
- [ ] Chaque repas a une source identifiable de fibres et de proteines
- [ ] Les contraintes (budget / temps / sans viande) sont reellement prises en compte (pas 3 menus identiques)
- [ ] Un piege ultra-transforme par contexte est nomme avec une alternative concrete
- [ ] Aucune etiquette de regime imposee ; le surgele/conserve "brut" est correctement distingue de l'ultra-transforme
