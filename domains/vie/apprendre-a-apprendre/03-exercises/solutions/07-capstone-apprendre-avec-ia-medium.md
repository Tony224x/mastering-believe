# Solutions (medium) — Module 07 (Capstone) : Apprendre avec l'IA

> Corrige de reference. Ces exercices sont des productions personnelles ; voici les criteres de qualite et les pieges.

---

## Exercice 1 — Auditer un plan genere par IA

**Pattern observe :** un prompt vague ("apprends X en 4 semaines") donne souvent un plan a dominante **passive** (lire le chapitre 1, regarder la video 2...), mono-thematique (un theme par bloc, pas d'interleaving), et sans indicateur objectif ("a la fin tu comprendras X").

**Audit attendu :**
1. **Retrieval** : souvent absent ou faible -> reformuler : *"Pour chaque module, ajoute une activite de recuperation active (quiz a froid, blank-page) plutot que de la lecture."*
2. **Espacement** : souvent tout est vu une fois -> *"Ajoute des revisions espacees a intervalles croissants (regle ~10-20 % du delai) des concepts deja vus."*
3. **Interleaving** : souvent un theme par jour -> *"Melange plusieurs sous-themes confondables par session au lieu d'un seul."*
4. **Indicateurs** : souvent "comprendre" -> *"Definis un indicateur objectif de maitrise par semaine (test, production, seuil chiffre)."*
5. **Neuromythes** : verifier qu'il n'introduit pas de "trouve ton style d'apprentissage", de jeu de brain-training, ou un "10 000 h" naif. **S'il le fait, le signaler et le retirer.**

**Conclusion attendue :** les 2 corrections les plus importantes sont en general (a) passer du passif au retrieval et (b) ajouter l'espacement — ce sont les deux techniques a preuve "elevee" (Dunlosky 2013).

**Garde-fou :** ne jamais laisser passer un neuromythe glisse par le LLM ; l'audit sert precisement a ca.

---

## Exercice 2 — Indicateurs de maitrise objectifs

**Exemples de transformation reussie :**

| Objectif vague | Indicateur objectif | Niveau | Obtention |
|----------------|--------------------|--------|-----------|
| "Comprendre les fonctions Python" | Resoudre 8/10 exercices 'fonctions' en autonomie en < 30 min | Cible | Jeu d'exercices auto-corrige |
| "Etre a l'aise en conversation espagnole" | Tenir 5 min de conversation A2 sans bloquer > 5 s, evalue par un partenaire/LLM | Stretch | Partenaire ou LLM simulant un locuteur |
| "Maitriser les bases de la stats" | Repondre correctement a 7/10 questions d'un quiz donne, sans notes | Minimum viable | Quiz a froid |

**Criteres de qualite :**
- Chaque indicateur est **observable** (on peut dire oui/non) et a un **seuil**.
- Les 3 niveaux (minimum / cible / stretch) creent une echelle, pas un tout-ou-rien.
- Le mode d'obtention est precis.

**Piege :** "je saurai que je maitrise quand je me sentirai a l'aise" — c'est un JOL, pas un indicateur. Interdit.

---

## Exercice 3 — Cadrer l'usage de l'IA

**Les 3 roles (prompts + timing) :**
- **Tuteur socratique** (au demarrage d'un concept / quand on bloque) : *"Je veux apprendre [X]. Ne m'explique pas directement — pose-moi des questions pour tester ce que je sais, puis guide-moi par questions."*
- **Generateur de retrieval** (avant chaque revision) : *"Genere 10 questions sur [themes], du simple au complexe. Ne donne pas les reponses — je reponds d'abord, tu corriges ensuite."*
- **Partenaire Feynman** (fin de module) : *"Je vais t'expliquer [X] comme si je l'enseignais. Detecte mes zones floues, sauts logiques, jargon non defini et erreurs."*

**3 garde-fous :**
1. "Je ne lis pas d'explication passive sans faire un blank-page juste apres."
2. "Je verifie les faits critiques (chiffres, citations) dans une source primaire — le LLM peut confabuler."
3. "Je ferme la conversation avant de me re-tester, pour ne pas avoir la reponse sous les yeux."

**Mode passif et illusion de competence :** demander une explication et la lire reproduit l'illusion du Module 01 — *en pire*, car le texte du LLM est tres fluide et personnalise, donc encore plus susceptible de produire un faux sentiment de maitrise. La fluidite n'est pas l'apprentissage.

**Nuance 2-sigma de Bloom (1984) :** l'effet de ~2 ecarts-types tient en grande partie au **mastery learning** (exiger ~90 % de maitrise avant d'avancer) combine au feedback personnalise — pas a l'ecoute passive d'un tuteur. Un LLM peut aider sur les deux, mais seulement si *toi* imposes le standard de maitrise et fais le travail de recuperation. Un LLM consomme passivement ne reproduit pas l'effet 2-sigma.

**Garde-fou :** le LLM n'est jamais un raccourci magique ; c'est un amplificateur conditionnel a un usage actif.

**Reference :** Bloom, B. S. (1984). *Educational Researcher*, 13(6), 4–16.
