# Solutions Medium — Module 01 : Le systeme d'exploitation du jugement

---

## Exercice 1 — Cas limites Systeme 1 / Systeme 2

| Situation | Systeme | Justification |
|---|---|---|
| 1. Chef de projet experimente, estimation "au feeling" en 2 s d'une tache deja faite | **S1** | Expertise consolidee : l'estimation est automatisee par la repetition, rapide et sans effort |
| 2. Meme chef, tache nouvelle, decoupage en sous-taches chiffrees | **S2** | Tache nouvelle : decomposition sequentielle, deliberative et effortful, pas d'automatisation possible |
| 3. Controleur qualite : intuition "piece defectueuse" puis mesure au pied a coulisse | **S1 -> S2** | L'alerte initiale est une reconnaissance de pattern S1 ; la verification mesuree est un acte S2 declenche volontairement |
| 4. Lire "BLEU" en encre rouge et nommer la couleur de l'encre | **S1 -> S2** | La lecture du mot est automatique (S1, effet Stroop) ; nommer la couleur exige une inhibition deliberative (S2) |
| 5. Joueur d'echecs **debutant** calculant ses variantes | **S2** | Sans expertise, l'exploration est sequentielle, lente et effortful |
| 6. **Grand maitre** identifiant le bon coup en une fraction de seconde | **S1 acquis** | Meme tache que 5, mais des milliers d'heures ont automatise la reconnaissance de patterns -> bascule en S1 |

**Les deux paires-pieges** :

```
Paire 5 vs 6 : MEME TACHE (trouver le bon coup), systeme different selon l'EXPERTISE.
  Debutant -> S2 (effort) ; Grand maitre -> S1 (reconnaissance automatique).
  L'expertise convertit des taches S2 en taches S1.

Paire 1 vs 2 : MEME PERSONNE, systeme different selon la NOUVEAUTE de la tache.
  Tache connue -> S1 ; tache nouvelle -> S2.
  Ce n'est pas la personne qui "est S1" ou "S2" : c'est l'adequation tache/experience.
```

**Pourquoi ce sont des cas limites** : l'apparence (facile/difficile, simple/complexe) ne predit PAS le systeme. Une tache objectivement complexe (diagnostic d'expert, coup de grand maitre) peut etre traitee en S1 si elle est suffisamment automatisee ; une tache objectivement simple (nommer une couleur) peut exiger du S2 quand le S1 produit une reponse concurrente a inhiber.

---

## Exercice 2 — Quel etage de Stanovich fait defaut ?

Rappel du modele tripartite :
- **Esprit autonome** = S1 (associations automatiques).
- **Esprit algorithmique** = S2 brut (puissance de calcul, memoire de travail) — ce que mesure le QI.
- **Esprit reflexif** = disposition a interroger et corriger les sorties des deux premiers.

Et la distinction cle :
- **Mindware gap** = l'outil conceptuel n'existe pas dans la tete de la personne.
- **Echec de l'esprit reflexif** = l'outil existe, mais n'a pas ete active (paresse cognitive, vitesse, surconfiance).

### Scenario A — Lea, batte-et-balle a la volee

```
Composant en cause : ESPRIT REFLEXIF (non active).
Mindware gap ?     : NON. Lea connait Bayes/proba conditionnelle ; le mindware est present.
```
Lea a tous les outils. Sous la pression de la vitesse, elle n'a simplement pas active son esprit reflexif (pas verifie la sortie du S1). Preuve : des qu'on l'invite a reprendre, elle corrige seule et trouve 0,05 €. L'outil etait la ; l'activation a manque.

### Scenario B — Sofiane, taux de base inconnu

```
Composant en cause : MINDWARE GAP (outil conceptuel "taux de base" absent).
Esprit algorithmique : bon (excellent en arithmetique), mais inutilisable sans l'outil.
```
Confondre P(test positif | malade) et P(malade | test positif) sans connaitre la notion de taux de base est un manque d'outil, pas un manque d'intelligence. Meme en prenant son temps, il ne sait pas par ou commencer : c'est la signature d'un mindware gap (et non d'un esprit reflexif paresseux). Le remede est d'installer le mindware : probabilite conditionnelle, theoreme de Bayes, frequences naturelles.

### Scenario C — Analyste qui s'arrete a la donnee confirmante

```
Composant en cause : ESPRIT REFLEXIF (defaillant -> biais de confirmation).
Mindware gap ?     : NON. Il sait qu'il "faut chercher la preuve contraire".
```
La regle existe dans sa tete ("chercher la preuve contraire") mais n'est pas appliquee : c'est un echec d'activation de l'esprit reflexif, pas une absence d'outil. Remede : transformer la regle en routine forcee (chercher activement la donnee infirmante avant de conclure).

### Scenario D — Stagiaire, saturation de la memoire de travail

```
Composant en cause : ESPRIT ALGORITHMIQUE (limite de puissance brute).
Mindware gap ?     : NON. Echec reflexif ? NON.
```
Le stagiaire connait l'addition (pas de mindware gap) et n'a aucune intuition fausse a corriger (pas d'echec reflexif). Il bute sur une **limite de capacite** : sa memoire de travail sature sur les grands nombres. Ce n'est pas une erreur de rationalite, c'est une contrainte de ressource algorithmique. Remede : externaliser la charge (poser l'operation par ecrit, utiliser une calculatrice) — un outil pour soulager la capacite, pas pour corriger un raisonnement.

**Synthese** :

```
A : outil PRESENT, non active   -> echec reflexif
B : outil ABSENT                -> mindware gap
C : outil PRESENT, non active   -> echec reflexif (biais de confirmation)
D : ni outil manquant ni intuition fausse -> limite d'esprit algorithmique
```

---

## Exercice 3 — Pieges de reflexion cognitive

### Piege A — Machines et widgets

```
Leurre S1        : "100 minutes" (on apparie betement 5->5->5 puis 100->100->100).
Calcul correct   :
  5 machines font 5 widgets en 5 minutes
  => 1 machine fait 1 widget en 5 minutes (chaque machine travaille en parallele)
  => 100 machines font 100 widgets en 5 MINUTES (toujours en parallele, 1 widget par machine)
Reponse correcte : 5 minutes.
```
**Pourquoi le S1 se trompe** : il detecte le motif "5, 5, 5" puis "100, 100, ?" et complete par analogie de surface ("100") sans modeliser le fait que les machines travaillent en parallele.

### Piege B — Nenuphars

```
Leurre S1        : "24 jours" (on divise betement 48 par 2).
Calcul correct   :
  La surface DOUBLE chaque jour.
  Au jour 48 : etang plein (100 %).
  La veille (jour 47) : la surface etait deux fois plus petite => 50 %.
Reponse correcte : 47 jours.
```
**Pourquoi le S1 se trompe** : il raisonne lineairement ("moitie du temps = moitie de la surface") alors que la croissance est exponentielle. Sous un doublement, la moitie est atteinte juste un pas avant la fin.

### Piege C — Production en serie (controle anti-sur-correction)

```
Pas de leurre S1 piegeux ici : l'intuition proportionnelle est CORRECTE.
Calcul :
  200 pieces en 4 heures => cadence = 200 / 4 = 50 pieces/heure
  En 10 heures : 50 x 10 = 500 pieces
Reponse correcte : 500 pieces.
```
**Lecon du controle** : tous les enonces "qui ressemblent a un piege" ne sont pas des pieges. La cadence etant constante, la proportionnalite est exacte ici. Le but : ne pas **sur-corriger** par reflexe. L'esprit reflexif sert a verifier, pas a rejeter systematiquement la premiere reponse — parfois l'intuition est juste.

**A retenir** : A et B ont un leurre S1 (analogie de surface ; linearisation d'un processus exponentiel). C n'en a pas. Activer le S2, c'est verifier — et accepter l'intuition quand elle resiste a la verification.
