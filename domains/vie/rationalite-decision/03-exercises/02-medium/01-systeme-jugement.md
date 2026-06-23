# Exercices Medium — Module 01 : Le systeme d'exploitation du jugement

> **Niveau** : Medium | **Temps estime** : ~30 min

---

## Exercice 1 : Cas limites Systeme 1 / Systeme 2

### Objectif

Classer des situations ambigues ou hybrides ou la frontiere S1/S2 n'est pas evidente, et justifier le classement en citant une propriete precise du systeme (vitesse, effort, automaticite, basculement par expertise).

### Consigne

Pour chacune des 6 situations, indiquez le systeme dominant (**S1**, **S2**, ou **S1 -> S2** si la situation commence en S1 puis force un basculement vers S2) et justifiez en une phrase qui cite une propriete du systeme. Attention : certaines situations sont des pieges ou l'apparence (facile/difficile) ne correspond pas au systeme reel.

1. Un chef de projet experimente estime "au feeling" la duree d'une tache qu'il a deja realisee des dizaines de fois ; le chiffre lui vient en 2 secondes.
2. Le meme chef de projet estime une tache d'un type totalement nouveau et se force a la decouper en sous-taches chiffrees.
3. Un controleur qualite voit defiler des pieces sur une chaine ; il "sent" qu'une piece est defectueuse, puis sort la piece du flux pour la mesurer au pied a coulisse afin de verifier.
4. Vous lisez le mot "BLEU" ecrit en encre rouge et on vous demande de nommer la couleur de l'encre (pas de lire le mot). Vous devez ralentir pour ne pas dire "bleu".
5. Un joueur d'echecs amateur (niveau debutant) calcule sa prochaine coup en explorant mentalement plusieurs variantes.
6. Un grand maitre d'echecs identifie en une fraction de seconde le "bon coup" dans une position de milieu de partie classique qu'il a deja rencontree des milliers de fois.

### Criteres de reussite

- [ ] Situation 1 = **S1** (expertise consolidee, estimation automatique et rapide)
- [ ] Situation 2 = **S2** (tache nouvelle, decoupage deliberatif et effortful)
- [ ] Situation 3 = **S1 -> S2** (intuition initiale S1, puis verification mesuree S2 declenchee volontairement)
- [ ] Situation 4 = **S1 -> S2** (lecture automatique S1 = effet Stroop, puis inhibition deliberative S2 pour nommer la couleur)
- [ ] Situation 5 = **S2** (debutant : exploration sequentielle effortful, pas d'automatisation)
- [ ] Situation 6 = **S1 acquis** (meme tache que 5 mais expertise = reconnaissance de patterns automatisee)
- [ ] La paire 5 vs 6 est explicitement identifiee comme "meme tache, systeme different selon l'expertise"
- [ ] La paire 1 vs 2 est explicitement identifiee comme "meme personne, systeme different selon la nouveaute de la tache"

---

## Exercice 2 : Quel etage de Stanovich fait defaut ?

### Objectif

Identifier precisement quel composant du modele tripartite de Stanovich echoue dans chaque scenario : esprit autonome (S1), esprit algorithmique (puissance S2 brute), ou esprit reflexif (disposition a interroger/corriger les sorties). Distinguer en particulier le **mindware gap** d'un **echec de l'esprit reflexif** (l'outil existe mais n'est pas active).

### Consigne

Pour chacun des 4 scenarios, indiquez : (a) quel composant de Stanovich est en cause, (b) s'il s'agit d'un mindware gap (outil conceptuel manquant) ou d'un echec de l'esprit reflexif (outil disponible mais non active). Justifiez en une phrase.

**Scenario A** : Lea connait parfaitement la probabilite conditionnelle et le theoreme de Bayes (elle les a enseignes). Pressee, elle resout le probleme batte-et-balle a la volee et repond "0,10 €" sans verifier. Quand on lui demande de reprendre, elle trouve immediatement 0,05 €.

**Scenario B** : Sofiane est excellent en arithmetique mentale. On lui pose un probleme de probabilite conditionnelle (taux de base). Il n'a jamais appris la notion de taux de base et confond P(test positif | malade) avec P(malade | test positif). Meme en prenant son temps, il ne sait pas par ou commencer.

**Scenario C** : Un analyste tres brillant doit comparer deux fournisseurs de pieces. Il dispose de toutes les donnees de taux de defaut, mais s'arrete a la premiere donnee qui confirme son intuition de depart (le fournisseur qu'il preferait deja) et ne cherche pas la donnee qui pourrait l'infirmer. Il sait pourtant qu'il "faut chercher la preuve contraire".

**Scenario D** : Un stagiaire doit additionner une longue colonne de couts de production. Il connait l'addition, mais sa memoire de travail sature et il fait des erreurs de report sur les grands nombres.

### Criteres de reussite

- [ ] Scenario A = **esprit reflexif** qui n'a pas ete active (mindware present : elle connait Bayes) ; **pas** un mindware gap
- [ ] Scenario B = **mindware gap** (outil conceptuel "taux de base" absent) ; l'esprit algorithmique est bon mais inutilisable sans l'outil
- [ ] Scenario C = **esprit reflexif** defaillant (biais de confirmation : l'outil "chercher la preuve contraire" existe mais n'est pas applique) ; **pas** un mindware gap
- [ ] Scenario D = **esprit algorithmique** (limite de puissance brute : saturation de la memoire de travail), ni mindware gap ni echec reflexif
- [ ] La distinction "outil manquant (B)" vs "outil present mais non active (A, C)" est explicite
- [ ] Le scenario D est correctement distingue des autres comme une limite de capacite, pas une erreur de rationalite

---

## Exercice 3 : Pieges de reflexion cognitive (calcul + leurre S1)

### Objectif

Resoudre trois pieges quantitatifs neutres de type "test de reflexion cognitive", en donnant la reponse correcte calculee ET en nommant explicitement la reponse-leurre que le Systeme 1 propose.

### Consigne

Pour chaque piege : (1) donnez la reponse intuitive (leurre S1) que la plupart des gens lancent, (2) calculez la reponse correcte en montrant l'etape de calcul, (3) expliquez en une phrase pourquoi le S1 se trompe.

**Piege A — Machines et widgets** : Si 5 machines fabriquent 5 widgets en 5 minutes, combien de temps faut-il a 100 machines pour fabriquer 100 widgets ?

**Piege B — Nenuphars** : Sur un etang, une zone de nenuphars double de surface chaque jour. Il faut 48 jours pour que les nenuphars couvrent tout l'etang. Combien de jours faut-il pour qu'ils couvrent la moitie de l'etang ?

**Piege C — Production en serie** : Une chaine produit 200 pieces en 4 heures a cadence constante. Combien de pieces produit-elle en 10 heures a la meme cadence ? (Piege plus simple, sert de controle : verifiez que vous ne sur-corrigez pas — ici la reponse intuitive proportionnelle est correcte.)

### Criteres de reussite

- [ ] Piege A : leurre S1 = "100 minutes" ; reponse correcte = **5 minutes** (1 machine fait 1 widget en 5 min ; 100 machines font 100 widgets en parallele en 5 min)
- [ ] Piege B : leurre S1 = "24 jours" ; reponse correcte = **47 jours** (doublement : la veille du jour ou c'est plein, c'est moitie plein)
- [ ] Piege C : reponse = **500 pieces** (cadence = 50 pieces/h ; 50 x 10 = 500) ; ici l'intuition proportionnelle est correcte
- [ ] L'apprenant identifie que A et B ont un leurre S1, mais que C n'en a pas (controle anti-sur-correction)
- [ ] Chaque calcul est montre etape par etape et le resultat est verifiable
