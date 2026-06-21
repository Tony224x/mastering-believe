# Exercices — Module 14 (Capstone) : Bâtir SON système d'apprentissage augmenté

---

## Exercice 1 — Choisir le bon sujet et formuler un objectif mesurable

### Objectif
Apprendre à formuler un objectif d'apprentissage qui peut être évalué objectivement — base indispensable avant de construire un système.

### Consigne

**Étape 1 — Choisir ton sujet.**
Choisis un sujet que tu vas commencer à étudier dans les deux prochaines semaines. Il doit être réel — pas hypothétique. Note-le en une phrase.

**Étape 2 — Formuler l'objectif.**
Reformule ton sujet en objectif selon ce template :

> « Dans ___ semaines, je serai capable de ___ [action concrète] — vérifiable par ___ [test, production, évaluation externe]. »

Exemples corrects :
- « Dans 4 semaines, je serai capable de résoudre les exercices de niveau N5 JLPT avec 80 % de réussite — vérifiable par le quiz officiel JLPT N5 niveau débutant. »
- « Dans 6 semaines, je serai capable d'expliquer sans notes le fonctionnement d'un réseau de neurones convolutif à quelqu'un qui ne connaît pas le ML — vérifiable par une session Feynman enregistrée évaluée par un pair. »

Exemples incorrects (à éviter) :
- « Je veux m'améliorer en japonais. »
- « Je veux mieux comprendre les réseaux de neurones. »
- « Je veux maîtriser Python. »

**Étape 3 — Définir deux indicateurs de maîtrise.**
Formule deux indicateurs objectifs (un minimum viable, un cible). Ils doivent répondre à : « Comment quelqu'un d'autre pourrait-il vérifier que j'ai atteint cet objectif ? »

**Étape 4 — Vérification.**
Soumets ton objectif à ce filtre : si tu l'atteignais à moitié, le saurait-on ? Si non, reformule.

### Critères de réussite
- [ ] Le sujet est nommé en une phrase précise (pas un champ générique)
- [ ] L'objectif contient une action concrète et un mode de vérification externe
- [ ] L'horizon temporel est fixé (nombre de semaines)
- [ ] Les deux indicateurs de maîtrise sont objectifs — sans auto-proclamation
- [ ] Le filtre « demi-atteinte détectable » est passé explicitement

---

## Exercice 2 — Construire un plan retrieval + espacement sur 4 semaines

### Objectif
Planifier à l'avance les sessions de retrieval practice et les intervalles d'espacement pour le sujet choisi en Exercice 1 — en appliquant concrètement les principes des Modules 02 et 03.

### Consigne

**Étape 1 — Lister les 5 à 10 concepts/compétences clés** à acquérir sur ce sujet (ce que tu dois vraiment savoir, pas tout ce qui existe sur le sujet).

**Étape 2 — Choisir tes formats de retrieval** parmi :
- Flashcards Anki (SM-2 automatique)
- Blank-page recall (feuille vierge, sans aide, 5-10 min)
- Quiz à réponse courte (auto-généré ou via LLM)
- Explication orale à voix haute (sans notes)
- Production active (résoudre un problème, écrire, coder)

Tu dois utiliser au moins deux formats différents sur les 4 semaines.

**Étape 3 — Planifier les sessions.** Remplis ce tableau (ou adapte-le à ta durée) :

| Semaine | Dates | Concepts ciblés | Format de retrieval | Score cible |
|---------|-------|----------------|---------------------|-------------|
| S1 | J+0 à J+7 | Encodage initial des concepts 1–5 | Blank-page J+1, J+3 | — (prise de référence) |
| S2 | J+8 à J+14 | Mix concepts 1–5 + nouveaux 6–8 | Quiz interleave + Anki quotidien | > 60 % |
| S3 | J+15 à J+21 | Consolidation tous concepts | Explication orale + quiz | > 70 % |
| S4 | J+22 à J+28 | Test complet (indicateurs §2) | Format du test final | Indicateur 1 atteint |

**Règle d'ajustement** : si le score d'une session est inférieur au score cible, réduire l'intervalle suivant de moitié avant de repartir.

**Étape 4 — Planifier l'interleaving.** Si ton sujet a plusieurs sous-thèmes (ex : japonais = hiragana + vocabulaire + grammaire), assure-toi que chaque session de S2 et S3 mélange au moins deux sous-thèmes. Note explicitement comment tu vas interleaver.

### Critères de réussite
- [ ] Les 5–10 concepts/compétences clés sont listés
- [ ] Au moins deux formats de retrieval différents sont planifiés
- [ ] Le tableau est rempli avec des dates concrètes (pas « semaine 2 » flottant)
- [ ] Les intervalles augmentent progressivement (pas de révision quotidienne de tous les concepts sur 4 semaines)
- [ ] L'interleaving est décrit si le sujet a plusieurs sous-thèmes
- [ ] Un score cible et une règle d'ajustement sont définis

---

## Exercice 3 — Remplir le gabarit portfolio complet

### Objectif
Produire le livrable final du cursus : un système d'apprentissage personnel augmenté, appliqué à ton sujet réel, en réinvestissant les trois briques (retrieval + espacement, boucle IA, métriques) et la boucle métacognitive.

### Consigne

Ouvre le fichier `04-projects/README.md`. C'est ton gabarit portfolio. Remplis-le intégralement pour ton sujet (celui des Exercices 1 et 2).

**Section par section :**

**1. Sujet cible** : reprend ton objectif formulé en Exercice 1 + les indicateurs de maîtrise.

**2. Indicateurs de maîtrise** : reprend les deux indicateurs + ajoute comment tu les mesureras concrètement.

**3. Plan d'encodage initial** : 3 sources maximum (évite la paralysie du choix) + structure des sessions de deep work (durée, fréquence, ce que tu vas couper comme distraction, interleaving prévu).

**4. Plan de retrieval et d'espacement** : reprend le tableau de l'Exercice 2 + l'outil SRS choisi (Anki / Notion / plan manuel).

**5. Rôle de l'IA** : pour chaque rôle (socratique, retrieval, Feynman), spécifie :
- Le **timing** exact (« début de chaque module », « tous les jeudis avant la session de révision », « quand je bloque sur un concept »)
- Un **prompt type** rédigé — pas juste le mode en label

**6. Boucle métacognitive** : bilan hebdomadaire (jour + durée fixés) + au moins deux déclencheurs d'ajustement avec seuil numérique explicite.

**7. Capstone du projet** : quel est ton livrable final pour prouver la maîtrise ? Date cible + critère de « done ».

**Après avoir rempli le gabarit :**
Passe la grille d'auto-évaluation du Module 14 (section 5 du brief théorique). Note les critères non cochés et retravaille ces sections avant de considérer le livrable terminé.

### Critères de réussite
- [ ] Toutes les sections du gabarit `04-projects/README.md` sont remplies (aucune vide ou en « à compléter »)
- [ ] Le plan de retrieval reprend le tableau de l'Exercice 2 avec dates concrètes
- [ ] Les trois rôles du LLM ont chacun un timing précis et un prompt type rédigé
- [ ] Les métriques incluent taux de rappel + rétention J+7 + ratio comprendre/produire
- [ ] Au moins deux déclencheurs d'ajustement sont définis avec seuil numérique
- [ ] La grille d'auto-évaluation du Module 14 a été passée — les critères non cochés ont été retraités
- [ ] Le livrable final du projet est nommé, daté, avec un critère de « done » vérifiable
