# Solutions — Module 14 (Capstone) : Bâtir SON système d'apprentissage augmenté

> Ce corrigé illustre un système complet sur un sujet concret : **maîtriser les bases de la comptabilité de gestion pour un non-comptable**. Il n'existe pas de « bonne réponse unique » — ton livrable doit être adapté à ton sujet. Ce qui compte, c'est la rigueur de la structure, pas le sujet choisi.

---

## Exercice 1 — Choisir le bon sujet et formuler un objectif mesurable

### Corrigé modèle

**Sujet choisi** : Comptabilité de gestion — lecture de comptes de résultat, notion de marge, coûts fixes vs variables, seuil de rentabilité.

**Objectif formulé :**
> « Dans 5 semaines, je serai capable d'analyser un compte de résultat simplifié, de calculer le seuil de rentabilité et d'expliquer la différence entre coûts fixes et variables — vérifiable par l'examen blanc du MOOC "Introduction à la comptabilité de gestion" (Coursera) et une session d'explication à un pair non-comptable. »

**Indicateur 1 (minimum viable)** : Obtenir 70 % ou plus à l'examen blanc du MOOC après 5 semaines.

**Indicateur 2 (cible)** : Expliquer en 10 minutes à quelqu'un sans formation en comptabilité ce qu'est un seuil de rentabilité, avec un exemple chiffré inventé, et répondre à ses questions sans notes — évalué par le pair (oui/non : « tu as répondu à mes questions sans lire un document »).

**Filtre « demi-atteinte »** : si j'ai révisé pendant 5 semaines mais pas passé l'examen blanc, on ne sait pas si j'ai appris. L'objectif est détectable à moitié (ex : je passe l'examen à 55 % mais l'explication au pair échoue → je sais ce qui manque).

---

## Exercice 2 — Construire un plan retrieval + espacement sur 4 semaines

### Corrigé modèle

**Concepts/compétences clés (8 éléments) :**
1. Structure d'un compte de résultat (charges / produits / résultat)
2. Marge brute et marge nette — définition et calcul
3. Coûts fixes vs coûts variables — distinction et exemples
4. Contribution unitaire et taux de marge sur coûts variables
5. Seuil de rentabilité — calcul et interprétation
6. Levier opérationnel — concept et implication
7. Lecture d'un bilan simplifié — actif / passif / capitaux propres
8. Lien entre rentabilité et trésorerie (piège fréquent)

**Formats de retrieval retenus :**
- Blank-page recall (feuille vierge, sans aide, 10 min)
- Quiz à réponse courte généré par LLM (5–10 questions, format Q/R)
- Explication orale à voix haute (seul ou à un pair)

**Plan sur 5 semaines :**

| Semaine | Dates | Concepts ciblés | Format de retrieval | Score cible |
|---------|-------|----------------|---------------------|-------------|
| S1 | J+0 à J+7 | Concepts 1–4 (encodage) | Blank-page à J+1 et J+3 sur chaque concept | Référence (noter sans juger) |
| S2 | J+8 à J+14 | Mix 1–4 + nouveaux 5–6 | Quiz LLM interleave (mélange 4 concepts) + Anki quotidien | > 60 % au quiz |
| S3 | J+15 à J+21 | Tous (1–8) | Explication orale de 2 concepts/session + quiz LLM | > 70 % au quiz |
| S4 | J+22 à J+28 | Révision ciblée sur < 70 % | Blank-page + quiz LLM sur les concepts faibles | > 75 % |
| S5 | J+29 à J+35 | Test complet (indicateurs) | Examen blanc MOOC + session Feynman avec pair | Indicateur 1 atteint |

**Règle d'ajustement** : score < 60 % sur un type de concept deux sessions de suite → réduire l'intervalle de 50 % et ajouter une session Feynman ciblée sur ce concept avant de repartir.

**Interleaving** : à partir de S2, chaque session de quiz mélange des concepts de catégories différentes — pas « une session sur les coûts fixes, une session sur le seuil ». Ex : Q1 sur la marge brute, Q2 sur le coût fixe, Q3 sur le seuil, Q4 sur la marge nette, Q5 sur le levier — dans cet ordre non groupé.

---

## Exercice 3 — Gabarit portfolio complété (exemple commenté)

### Corrigé modèle

Voici le gabarit `04-projects/README.md` rempli pour ce sujet. Les commentaires entre crochets expliquent les choix — dans ton livrable, tu les supprimes.

---

**Sujet cible :**
Comptabilité de gestion pour non-comptable.

Objectif : dans 5 semaines, je serai capable d'analyser un compte de résultat simplifié, de calculer le seuil de rentabilité et d'expliquer la différence entre coûts fixes et variables — vérifiable par l'examen blanc du MOOC et une session Feynman avec un pair.

Prérequis : calculs arithmétiques de base (%, multiplication, division).

---

**Indicateurs de maîtrise :**
- Indicateur 1 (minimum viable) : 70 % à l'examen blanc du MOOC.
- Indicateur 2 (cible) : explication réussie à un pair — validé par le pair (oui/non).
- Comment mesurer : résultat de l'examen blanc (objectif) + validation du pair (binaire externe).

---

**Plan d'encodage initial :**

| Source | Type | Durée estimée | Priorité |
|--------|------|--------------|---------|
| MOOC « Introduction à la comptabilité de gestion » (Coursera) | Vidéos + quiz intégrés | ~8 h | 1 |
| *La comptabilité de gestion* (Langlois & Bonnier, édition courante) | Manuel — chapitres 1–4 seulement | ~4 h | 2 |
| Exercices corrigés du MOOC | Pratique | ~3 h | 3 |

Structure des sessions de deep work :
- Durée par session : 40 minutes
- Fréquence : 5 fois/semaine (lundi au vendredi)
- Distractions coupées : téléphone en mode avion, notifications ordinateur désactivées
- Interleaving prévu : à partir de S2, chaque session alterne vidéo + blank-page + exercice (jamais deux vidéos consécutives)

---

**Plan de retrieval et d'espacement :**
*(Voir tableau Exercice 2 ci-dessus — repris tel quel dans le gabarit)*

Outil SRS choisi : Anki — deck à créer sur les 8 concepts clés (une carte par formule / définition).

---

**Rôle de l'IA :**

**Mode tuteur socratique :**
- Quand : lundi de chaque semaine, en début de session, avant de revoir les notes.
- Prompt type : *« Je veux consolider [concept de la semaine, ex : seuil de rentabilité]. Ne m'explique pas directement — pose-moi 5 questions de difficulté croissante pour tester ce que je sais déjà, puis guide-moi sur les points où je bloque. »*

**Mode générateur de retrieval :**
- Quand : jeudi de chaque semaine, avant la session de révision espacée.
- Prompt type : *« Génère 8 questions de comptabilité de gestion niveau débutant sur les thèmes suivants : [liste des 3–4 concepts de la semaine]. Mélange les types de questions (calcul, définition, interprétation). Ne donne pas les réponses — je vais répondre d'abord, puis tu corriges et commentes. »*

**Mode partenaire Feynman :**
- Quand : dimanche de chaque semaine, en fin de session (15 min).
- Prompt type : *« Je vais t'expliquer [concept, ex : la différence entre coûts fixes et coûts variables] comme si je l'enseignais à quelqu'un qui n'a jamais fait de comptabilité. Écoute mon explication et détecte : (1) les zones floues, (2) les sauts logiques, (3) le jargon non défini, (4) les erreurs conceptuelles. »*

**Limites actées :**
- Je ne lirai pas les explications du LLM passivement — après chaque explication, je ferme la conversation et fais un blank-page de 3 minutes.
- Je vérifierai les formules critiques dans le manuel (le LLM peut confabuler sur des calculs précis).

---

**Métriques de suivi :**

| Métrique | Quand | Comment noter |
|----------|-------|---------------|
| Taux de rappel | Après chaque quiz LLM ou Anki | % correct dans un carnet (ou colonne Notion) |
| Rétention J+7 | 7 jours après chaque première exposition à un concept | Re-quiz sur le même concept sans avoir révisé entre les deux |
| Ratio comprendre/produire | Bilan du dimanche (0–10) | Note subjective mais consignée chaque semaine |
| Delta pré/post | J+0 (avant S1) + J+35 (après S5) | Même quiz de 10 questions pris au début et à la fin |

---

**Boucle métacognitive :**

Bilan hebdomadaire : dimanche, 10 minutes, après la session Feynman.
- Score du quiz de la semaine : ___
- Ratio comprendre/produire : ___/10
- Ce qui a bien fonctionné : ___
- Ce qui n'a pas fonctionné : ___
- Ajustement pour la semaine suivante : ___

Déclencheurs d'ajustement :
1. Taux de rappel < 60 % sur le même type de concept deux semaines consécutives → réduire l'intervalle de moitié + ajouter une session Feynman supplémentaire sur ce concept.
2. Ratio comprendre/produire < 5/10 à la fin de S3 → remplacer 50 % du temps de lecture par des exercices de production active (calculs chronométrés, explication orale enregistrée).
3. Score de l'examen blanc < 60 % en fin de S4 → retravailler les deux thèmes les plus faibles avec le tuteur socratique avant S5, reporter la date de la session Feynman avec pair de 1 semaine.

---

**Capstone du projet :**
Livrable final : passer l'examen blanc du MOOC et faire la session d'explication à un pair.
Date cible : J+35 (fin de S5).
Critère de « done » : 70 % à l'examen blanc ET validation du pair (« tu as répondu à mes questions sans lire un document »). Les deux conditions doivent être remplies.

---

### Ce que ce corrigé illustre

**Précision vs vœux pieux.** Chaque élément du système a un timing, un format, un seuil. « Je vais utiliser Anki » n'est pas une décision — « Je vais créer un deck de 8 cartes sur les concepts clés et les revoir en due quotidiennement à partir de J+3 » l'est.

**Interdépendance des briques.** Les métriques (taux de rappel < 60 %) alimentent directement les déclencheurs d'ajustement de l'espacement. La boucle IA (générateur de quiz le jeudi) alimente les métriques de la semaine. Le bilan du dimanche (Feynman + métacognition) alimente le planning de S+1. Ce n'est pas trois outils indépendants — c'est une boucle.

**Le LLM amplifie, il ne remplace pas.** Le plan fonctionnerait sans LLM (avec Anki + blank-page + examen blanc). Le LLM accélère le retrieval calibré (générateur de quiz) et le feedback sur les lacunes (Feynman). Il ne se substitue pas au test externe final.

**L'indicateur de maîtrise prime sur le sentiment.** Le critère de « done » n'est pas « je me sens prêt ». C'est 70 % à l'examen blanc ET validation du pair. Si l'un des deux manque, le système n'est pas terminé — il est ajusté.
