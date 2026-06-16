# Solutions — Module 07 (Capstone) : Apprendre avec l'IA

---

## Exercice 1 — Tester les trois roles du LLM en une session

### Corrige modele

**Resultat attendu (pattern observe par la grande majorite des apprenants) :**

Le blank-page apres le **mode passif** est generalement le plus pauvre : on retient des fragments, des formulations du LLM, mais on est rarement capable de reconstituer la structure ou les connexions.

Le blank-page apres le **mode socratique** est plus dense : les questions du LLM ont force a activer les connaissances preexistantes, a chercher des exemples, a faire des liens. L'effort cognitif pendant la session est plus eleve — et c'est precisement ce qui ancre mieux.

Le blank-page apres le **mode Feynman** est souvent le plus precis sur les points couverts, mais revele aussi les zones que tu n'as pas reussies a expliquer (et que tu n'aurais peut-etre pas detectees autrement). C'est le feedback le plus direct sur les lacunes reelles.

**Ce que revele le mode Feynman que les autres ne revelent pas :**
Les sauts logiques implicites — les endroits ou tu crois faire une connexion mais ou tu sautes en realite une etape. Par exemple, expliquer "l'espacement" sans etre capable de dire *pourquoi* oublier partiellement aide a ancrer (le mecanisme de reconstruction active). La fluidite du mode passif masque ces sauts ; le mode Feynman les expose.

**Conclusion de l'exercice :**
Les trois modes ne sont pas equivalents. Le mode passif consomme du temps avec un retour d'apprentissage faible. Les modes socratique et Feynman exigent plus — et produisent plus. Idalement, on commence par le mode socratique pour construire, et on finit par Feynman pour verifier.

---

## Exercice 2 — Generer un plan de retrieval et d'espacement avec un LLM

### Corrige modele

**Prompt initial bien formule :**

> *"J'ai etudie les concepts suivants au cours des 7 derniers jours : [liste des flash-cards / concepts cles de chaque module]. Construis-moi un plan de revision espacee pour les 4 prochaines semaines. Applique approximativement la regle des 10-20 % du delai avant test (Cepeda et al. 2008). Pour chaque session, precise quels concepts revisiter, dans quel ordre (interleave, pas un concept par session), et sous quelle forme (flashcards Anki, quiz a reponse courte, blank-page recall, explication orale a voix haute)."*

**Evaluation du plan produit :**

Un bon plan present ces caracteristiques :
- J+1 ou J+2 : revision rapide sur l'ensemble (intervalles courts au debut)
- J+3-4 : mix de concepts de plusieurs modules
- J+7-8 : tests interleaves, format plus exigeant (blank-page plutot que flashcard)
- J+14 : session de consolidation, formats varies
- J+21-28 : test final de l'ensemble du domaine

**Ce que le LLM fait souvent mal sans instruction precise :**
- Il propose des sessions mono-thematique (M01 lundi, M02 mardi…) au lieu d'interleaver.
- Il sous-estime les intervalles en conservant des revisions trop rapprochees.
- Il ne specifie pas de format de retrieval actif — il dit "revise" sans dire comment.

**Reformulation qui corrige ces defauts :**
*"Dans le plan que tu viens de generer, assure-toi que : (1) chaque session melange au moins 3 modules differents, (2) les intervalles entre deux revisions du meme concept doublent a chaque fois (J+2, J+5, J+12, J+26...), (3) chaque concept est revise sous au moins deux formats differents sur la periode."*

---

## Exercice 3 — Concevoir son systeme d'apprentissage personnel (gabarit capstone)

### Corrige modele (exemple : apprendre le japonais niveau A2)

---

**1. Sujet cible :**
Japonais — niveau survie. Objectif : dans 6 semaines, je serai capable de tenir une conversation de 5 minutes sur des sujets courants (presentations, directions, commandes au restaurant) et de lire les hiragana/katakana sans aide.

**2. Indicateurs de maitrise :**
- Lire 50 mots en hiragana/katakana en moins de 2 minutes sans erreur (test chronometrique)
- Repondre correctement a 80 % des questions dans l'application JLPT N5 niveau zero
- Tenir une conversation de 5 min avec un partenaire de pratique ou un LLM joue le role d'un locuteur natif simulant le niveau de comprehension A1

**3. Plan d'encodage (semaine 1) :**
- Materiaux : Genki I (chapitre 1-3), Anki deck hiragana/katakana (existant)
- Blocs deep work : 35 min/jour (5 jours sur 7), telephone en mode avion
- Interleaving : chaque session alterne ecriture des caracteres (moteur), lecture de mots, exercices d'ecoute — jamais le meme type deux blocs consecutifs

**4. Plan de retrieval et d'espacement (semaines 2-6) :**
- Anki quotidien : 10-15 cartes hiragana/katakana en due (SM-2 automatique)
- J+3 : blank-page recall sur le vocabulaire de S1, format ecrit
- J+7 : quiz de 20 questions (mix vocab, grammaire, lecture) — score cible : 70 %
- J+14 : revision interleave + 5 min de conversation simulee avec LLM
- J+28 : test cronometrique hiragana/katakana + quiz JLPT N5

**5. Role de l'IA :**
- **Generateur de quiz** : 2 fois/semaine, prompt "genere 10 questions de japonais niveau N5 sur [themes de la semaine], du plus facile au plus dur, sans donner les reponses"
- **Partenaire Feynman** : 1 fois/semaine, "je vais t'expliquer la structure de la phrase japonaise (SOV, particules) — detecte mes erreurs et zones floues"
- **Simul partenaire de conversation** : 1 fois/semaine, "joue le role d'un passant japonais ne parlant pas anglais. Je vais te demander mon chemin en japonais."

**6. Boucle metacognitive :**
- Bilan hebdomadaire le dimanche soir (5 min) : score du quiz de la semaine + estimation du ratio "je comprends / je peux produire"
- Declencheur d'ajustement : si score < 60 % deux semaines de suite sur le meme type → reduire l'intervalle (revenir plus tot sur ce type) et ajouter une session Feynman supplementaire
- Declencheur de pivot de strategie : si au bout de 3 semaines le ratio "comprendre / produire" reste tres asymetrique → ajouter des sessions de production orale (shadow repetition, dictee) et reduire la lecture passive

---

**Ce que ce gabarit illustre :**
Le systeme n'est pas une liste de bonnes intentions. C'est un cycle avec des indicateurs mesurables, des dates concretes, des seuils d'ajustement explicites et un role precis pour l'IA — pas "je vais utiliser ChatGPT pour apprendre". Chaque element est lie a une ou plusieurs techniques du domaine.
