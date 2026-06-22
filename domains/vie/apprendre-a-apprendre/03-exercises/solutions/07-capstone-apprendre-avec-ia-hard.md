# Solutions (hard) — Module 07 (Capstone) : Apprendre avec l'IA

> Corrige de reference pour des productions personnelles etendues. On evalue la coherence avec tout le domaine et le respect des garde-fous.

---

## Exercice 1 — Plan complet et instrumente

**Ce qu'un capstone etendu de qualite contient (les 8 sections) :**
1. **Decomposition + Pareto** : le sujet est decoupe ET le 20 % a fort levier est identifie comme point de depart (concret before abstract, Pareto-first).
2. **Encodage deep work** : blocs dimensionnes a la tolerance reelle, montee en charge, chaque bloc finissant par un retrieval de cloture (l'attention se transforme en encodage teste).
3. **Espacement chiffre** : un vrai calendrier (intervalles croissants conformes a Cepeda 10-20 %) ou une politique Anki/SM-2, avec gestion explicite des cartes ratees (reset court).
4. **Interleaving** : les sous-themes confondables sont melanges, avec une phase "identifier le type avant de resoudre".
5. **Pratique deliberee** : pour les *competences* (pas juste les faits), des cycles avec feedback immediat cibles sur les faiblesses prioritaires — distincte du simple retrieval.
6. **Role de l'IA** : 3 modes avec prompts, timing, ET garde-fous (pas de lecture passive ; verification des faits ; blank-page avant re-test).
7. **Tableau de bord** : indicateurs objectifs hebdo, seuils, declencheurs d'ajustement et de pivot.
8. **Livrable final** : un capstone du sujet verifiable (projet, examen, demo, production).

**Marqueurs d'excellence :** la coherence — chaque element est relie a une technique a preuve (retrieval/espacement = eleve ; interleaving/self-explanation = modere) ; le plan privilegie le retrieval sur la relecture apres l'encodage initial ; rien ne repose sur un ressenti.

**Garde-fou :** un plan, meme tres detaille, qui repose sur "relire jusqu'a se sentir pret" ou sur un "style d'apprentissage" est invalide.

---

## Exercice 2 — Planning d'espacement chiffre (objectif J+28)

**Regle Cepeda : ~10-20 % de 28 j -> premier intervalle ~3-6 jours.**

**Calendrier modele :**

| Session | Jour | Intervalle depuis precedent | Concepts (interleave) | Format | Duree |
|---------|------|-----------------------------|------------------------|--------|-------|
| S1 | J+1 | — | M01 + M02 + M03 | Blank-page recall | 20 min |
| S2 | J+5 | 4 j | M02 + M04 + M05 | Flashcards (toutes) | 20 min |
| S3 | J+12 | 7 j | M01 + M03 + M06 + lacunes S2 | Quiz LLM a froid | 25 min |
| S4 | J+22 | 10 j | Tous modules, focus lacunes | Blank-page + explication orale (Feynman) | 30 min |
| Test | J+28 | 6 j | Ensemble du domaine | Test formate | — |

**Politique de cartes ratees (logique SM-2) :** une carte/concept rate -> intervalle reset court (re-injection a J+1/J+2), puis re-espacement si reussie ensuite. Charge estimee : si ~20 % des concepts echouent par session, chaque session ajoute ~1/5 du corpus en re-injection rapide -> la charge se concentre sur le difficile (effet voulu).

**Declencheurs d'ajustement (chiffres) :**
1. Quiz d'un module < 60 % deux sessions de suite -> reduire son intervalle + ajouter une session Feynman ciblee.
2. Ratio "comprendre / produire" tres asymetrique a S3 -> ajouter de la production active (ecrire/expliquer).

**Honnetete sur la preuve :** la regle 10-20 % (Cepeda et al. 2008) est une regularite empirique robuste mais c'est une **moyenne** ; l'optimum exact varie selon l'individu et le materiel. Un SRS (Anki/SM-2) affine ensuite **par carte** via le feedback de notation. Le calendrier ci-dessus est un point de depart raisonnable, pas une verite figee.

**Reference :** Cepeda, N. J. et al. (2008). *Psychological Science*, 19(11), 1095–1102.

---

## Exercice 3 — Audit anti-mythe de son propre systeme

**Trame d'audit attendue :**

1. **Neuromythes :** verifier qu'aucune partie du plan ne suppose un "style d'apprentissage" (ex. "comme je suis visuel, je ne fais que des schemas"), un effet de brain-training, ou un "10 000 h" naif. Correction type : adapter le format au **contenu** (pas au style) ; supprimer tout objectif d'heures-seuil au profit d'objectifs de maitrise.
2. **Illusions de competence :** reperer les indicateurs subjectifs ("relire jusqu'a me sentir pret", "j'ai l'impression de comprendre") et les remplacer par des mesures objectives (test a froid, production sans aide).
3. **Verification IA :** identifier ou le plan fait confiance au LLM sur des faits ; ajouter une etape "verifier chiffres/citations en source primaire" (le LLM peut confabuler).
4. **Honnetete sur la preuve :** pour 2 techniques du plan, indiquer le niveau de preuve sans sur-vente — ex. retrieval practice et espacement = preuve **elevee** (Dunlosky 2013) ; interleaving et self-explanation = preuve **moderee** ; deep work = cadre utile mais issu d'un essai grand public (Newport), moins formalise que les meta-analyses.
5. **Robustesse / rattrapage :** une regle "si je rate une semaine, je reprends le calendrier sans tout bachoter le meme jour" (ne pas re-creer de massed practice).

**Version corrigee :** la sortie attendue est un plan ou (a) aucun neuromythe ne subsiste, (b) tous les indicateurs cles sont objectifs, (c) les faits IA sont verifies, (d) les niveaux de preuve sont annonces honnetement, (e) une regle de rattrapage anti-bachotage existe.

**Garde-fou final :** cet exercice EST le controle qualite anti-clivant/anti-mythe du domaine. A la fin, aucun style d'apprentissage, aucun brain-training et aucune lecture naive des 10 000 heures ne doivent rester valides dans le plan.

**Reference :** Dunlosky, J. et al. (2013). *PSPI*, 14(1), 4–58. ; Pashler, H. et al. (2008). *PSPI*, 9(3), 105–119.
