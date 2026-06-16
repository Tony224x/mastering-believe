# Exercices — Module 07 (Capstone) : Apprendre avec l'IA

---

## Exercice 1 — Tester les trois roles du LLM en une session

### Objectif
Experimenter concretement les trois modes d'utilisation d'un LLM (tuteur socratique, generateur de retrieval, partenaire Feynman) et evaluer ce qui produit le plus d'apprentissage reel.

### Consigne
Choisis un concept issu de ce domaine que tu veux consolider (ou un concept de n'importe quel autre sujet que tu etudies actuellement).

**Partie 1 — Mode passif (temoin) :**
Demande au LLM de t'expliquer le concept directement. Lis l'explication. Dur : 5-7 minutes. Ensuite, ferme la conversation et ecris de memoire ce que tu as retenu (blank-page, 2 min max).

**Partie 2 — Mode tuteur socratique :**
Ouvre une nouvelle conversation. Dis au LLM : *"Je veux apprendre [concept]. Ne m'explique pas directement. Pose-moi des questions pour tester ce que je sais deja, puis guide-moi par questions successives."* Dur : 10-12 minutes. Ensuite, blank-page identique (2 min).

**Partie 3 — Mode partenaire Feynman :**
Nouvelle conversation. Explique le concept au LLM comme si tu l'enseignais, et demande-lui de detecter les zones floues. Revise sur les points souleves. Dur : 10-12 minutes. Blank-page final (2 min).

**Comparaison :**
- Lequel des trois blank-page etait le plus complet et precis ?
- Dans quel mode as-tu ressenti le plus de difficulte / inconfort ?
- Que t'a revele le mode Feynman que les deux autres n'ont pas revele ?

### Criteres de reussite
- [ ] Les trois modes sont testes separement (pas en un seul fil de conversation)
- [ ] Les trois blank-page sont ecrits sans revenir a la conversation avant
- [ ] La comparaison inclut une observation honnete sur la qualite differente des restitutions
- [ ] Le compte-rendu mentionne un trou specifique detecte par le mode Feynman

---

## Exercice 2 — Generer un plan de retrieval et d'espacement avec un LLM

### Objectif
Utiliser un LLM comme generateur de systeme d'espacement et evaluer la qualite du plan produit par rapport aux criteres scientifiques du domaine.

### Consigne
Tu viens d'etudier ce domaine (les sept modules d'"apprendre-a-apprendre") sur les 7 derniers jours.

**Etape 1 :** Donne au LLM la liste des concepts cles de chaque module (tu peux utiliser les flash-cards de fin de chapitre comme base). Demande-lui : *"Sur la base de ces concepts, construis-moi un plan de revision espacee pour les 4 prochaines semaines. Applique la regle des 10-20 % du delai avant test. Pour chaque session, indique quels concepts revisiter et sous quelle forme (flashcard, quiz, exercice, explication orale)."*

**Etape 2 :** Evalue le plan produit selon ces criteres :
- [ ] Les intervalles augmentent-ils progressivement (ne pas revisiter tous les concepts a J+1 puis J+2) ?
- [ ] La regle des 10-20 % du delai est-elle respectee (ou approchee) ?
- [ ] Le plan inclut-il de l'interleaving (pas un concept par session — plusieurs) ?
- [ ] Les formats varies (flashcard, quiz, explication orale) sont-ils presentes ?

**Etape 3 :** Si le plan ne satisfait pas un ou plusieurs criteres, reformule ton prompt et demande une version corrigee. Note quelle reformulation a ete necessaire.

### Criteres de reussite
- [ ] Le prompt initial contient la liste des concepts (pas juste "fais un plan pour le domaine")
- [ ] Les 4 criteres d'evaluation sont renseignes explicitement (check ou non)
- [ ] Si le plan initial etait defaillant, la reformulation est notee

---

## Exercice 3 — Concevoir son systeme d'apprentissage personnel (gabarit capstone)

### Objectif
Produire le livrable central du domaine : ton propre systeme d'apprentissage augmente, applicable a n'importe quel futur sujet.

### Consigne
Choisis un sujet que tu vas apprendre dans les prochaines semaines (peut etre un autre domaine de ce repo, une langue, une competence technique, un domaine scientifique — peu importe).

**Remplis le gabarit suivant (le gabarit complet est dans `04-projects/README.md`) :**

**1. Sujet cible :**
Nomme le sujet et formule l'objectif en une phrase : "Dans 4 semaines, je serai capable de ___."

**2. Indicateurs de maitrise :**
Quels tests externes (pas subjectifs) te diront que tu as atteint l'objectif ? (ex : passer un quiz specifique, expliquer X sans notes a quelqu'un, resoudre Y type de problemes en autonomie)

**3. Plan d'encodage (semaine 1) :**
Quels materiaux ? Quels blocs deep work ? Quel interleaving si plusieurs sous-themes ?

**4. Plan de retrieval et d'espacement (semaines 2-4) :**
A quelles dates vais-je me retester ? Sur quels concepts en priorite ? Quel format (flashcards Anki, exercices, blank-page, quiz LLM) ?

**5. Role de l'IA :**
A quels moments et dans quel role exact vais-je utiliser un LLM ? (tuteur socratique, generateur de quiz, partenaire Feynman — ou les trois)

**6. Boucle metacognitive :**
Comment vais-je monitorer et ajuster ? Quelle frequence de bilan ? Quel seuil de "ce n'est pas en train de marcher" m'amene a changer de strategie ?

### Criteres de reussite
- [ ] L'objectif est formule en une phrase verifiable (pas "je veux m'ameliorer en X")
- [ ] Les indicateurs de maitrise sont objectifs (test, production, evaluation externe)
- [ ] Le plan de retrieval contient des dates concretes et des intervalles croissants
- [ ] Le role du LLM est specifie (mode et timing) — pas juste "je vais l'utiliser"
- [ ] La boucle metacognitive definit un declencheur explicite pour ajuster la strategie
