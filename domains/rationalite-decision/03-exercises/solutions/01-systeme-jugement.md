# Solutions — Module 01 : Le systeme d'exploitation du jugement

---

## Exercice 1 : Systeme 1 ou Systeme 2 ?

| Situation | Systeme | Justification |
|---|---|---|
| 1. Joueur de tennis retournant un smash | **S1** | Expertise consolidee : le mouvement est automatise par des milliers d'heures de pratique |
| 2. Etudiant calculant 37 x 48 | **S2** | Tache numerique nouvelle, requiert concentration et calcul pas a pas |
| 3. Conducteur freinant au feu rouge | **S1** | Reponse automatisee apres apprentissage : le stimulus (rouge) declenche la reponse sans deliberation |
| 4. Comptable verifiant une declaration complexe | **S2** | Nouveau client, regles nombreuses, verification minutieuse : traitement deliberatif |
| 5. Reconnaitre l'odeur du cafe | **S1** | Reconnaissance sensorielle immediate, inconsciente |
| 6. Traduire une phrase dans une langue recente | **S2** | Langue non automatisee : chercher les mots, verifier la grammaire = effort deliberatif |
| 7. Medecin expert reconnaissant un diagnostic | **S1 acquis** | Expertise tres poussee : la reconnaissance de patterns est automatisee. Un novice ferait la meme tache en S2. |
| 8. Taper son mot de passe habituel | **S1** | Sequence motorique automatisee par la repetition |

**Note sur la situation 7** : c'est le cas le plus interessant. La meme tache peut etre S1 ou S2 selon le niveau d'expertise. L'expertise convertit des taches S2 en taches S1 par automatisation.

---

## Exercice 2 : Reconnaitre un biais

### Scenario A — Heuristique de disponibilite

**Reponse correcte** : Les mots avec "R" en troisieme position sont bien plus nombreux en francais que les mots commencant par "R". Mais les mots commencant par R sont plus *faciles a evoquer* (on les cherche par leur initiale).

**Mecanisme** : heuristique de disponibilite (Tversky & Kahneman, 1973). Le cerveau juge la frequence d'une categorie par la facilite avec laquelle des exemples viennent a l'esprit. Facilite d'evocation ≠ frequence reelle.

**Direction du biais** : systematiquement en faveur des categories ou les exemples sont plus accessibles en memoire.

### Scenario B — Deux sequences equiprobables

Chaque lancer est independant. P(rouge) = 1/2 pour chaque lancer.

```
P(RRRRRR) = (1/2)^6 = 1/64 ≈ 1,56 %
P(RRBRRB) = (1/2)^6 = 1/64 ≈ 1,56 %
```

Les deux sequences ont exactement la meme probabilite. L'intuition juge RRRRRR comme "moins probable" car elle semble non-aleatoire (biais de representativite : on confond "aleatoire" avec "qui ressemble a ce qu'on imagine de l'aleatoire"). En realite, toutes les sequences de 6 lancers ont la probabilite identique de (1/2)^6.

### Scenario C — Batte et balle

**Calcul correct** :
- Notons b = prix de la balle
- Batte = b + 1,00 €
- Batte + balle = b + 1,00 + b = 1,10 €
- 2b = 0,10 €
- b = **0,05 €**, batte = 1,05 €

**Erreur du Systeme 1** : le S1 decompose rapidement "1,10 € en deux parties dont une vaut 1,00 € de plus" et sort "0,10 €" par analogie rapide. Il ne verifie pas algebriquement. L'activation du S2 (ecrire l'equation) corrige l'erreur immediatement.

---

## Exercice 3 : Rationalite vs intelligence

**Question 1** :
- Marie a plus de **puissance algorithmique** (S2 brut) : meilleure memoire de travail, calcul rapide, algorithmie.
- Thomas raisonne **mieux sur les probabilites conditionnelles dans son domaine** : 10 ans d'experience ont calibre son S1 sur ce domaine specifique.

**Question 2** : L'erreur de Marie est un **mindware gap** (Stanovich). Elle manque d'un outil conceptuel : la formalisation de P(A|B) et du theoreme de Bayes. Ce n'est pas un manque d'intelligence, c'est un manque d'outil.

**Question 3** : Marie doit **combler son mindware gap** : apprendre explicitement la probabilite conditionnelle et le theoreme de Bayes. Une fois le mindware installe, son esprit algorithmique peut l'utiliser correctement.

**Question 4** : Non. L'expertise de Thomas est **domaine-specifique**. Il est calibre sur les probabilites des matchs sportifs grace a 10 ans d'immersion dans ce contexte. Cela ne lui donne pas automatiquement une bonne calibration sur les probabilites medicales, juridiques ou financieres.
