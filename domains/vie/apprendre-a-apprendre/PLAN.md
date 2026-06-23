# PLAN figé — `apprendre-a-apprendre` (cursus complet J1–J14, mode full)

> Contrat lu par les builders de Phase 4. **Ne lisez PAS les autres jours.**
> Sources : `REFERENCES.md` du domaine + `shared/track-vie/apprendre-a-apprendre-references.md`,
> `-curricula-evidence.md`, `-j14-sources.md`.

## Règles transverses (OBLIGATOIRES sur CHAQUE jour)
- **Anti-pseudoscience** : debunk explicite des neuromythes là où c'est pertinent, avec un **encart « Pseudoscience ? »** identifiable. Le mythe des **styles d'apprentissage (VAK)** doit être tué dès J1 (Pashler 2008, Dekker 2012). Autres : brain-training (Simons 2016), cerveau gauche/droit, « 10% du cerveau », « 10 000 heures » (Macnamara).
- **Honnêteté sur la preuve** : bloc « statut de preuve / nuance » en fin des modules concernés, avec taille d'effet ou réplication. Ne PAS survendre : dual coding = utilité **modérée** ; sommeil-consolidation = poids causal **débattu** ; manuscrit>clavier = **réplication échouée 2019** (c'est le traitement, pas l'outil) ; 2-sigma = effet du **mastery learning 90%**, pas magie du tuteur ; pratique délibérée = **variance Macnamara** (pas « 10 000 h »).
- Langue FR, **concret AVANT le principe**. ~45 min/module.
- Format théorie : H1 `# Module N — Titre` ; `> **Temps estimé** : 45 min | **Prérequis** : Modules 01-0(N-1)` + `> **Objectif** :` ; sections ; `> **À retenir** :` ; **4-5 flash-cards** ; `## Points clés à retenir` ; `## Pour aller plus loin`.
- Exercices : `03-exercises/01-easy/NN-slug.md` = **3 exercices gradués easy→hard** (Objectif/Consigne/Critères `- [ ]`). Solution : `03-exercises/solutions/NN-slug.md` (ou `.py` si code).
- Code (seulement où indiqué) : `02-code/NN-slug.py`, stdlib pur, exit 0.

## Carte de réutilisation
| Nouveau | Réutilise l'existant |
|---------|----------------------|
| 01-pourquoi-tu-oublies | `01-theory/01-pourquoi-tu-oublies.md` (upgrade + ajouter styles d'apprentissage) |
| 02-retrieval-practice | `02-retrieval-practice.md` |
| 03-spaced-repetition | `03-spaced-repetition.md` + `02-code/03-spaced-repetition.py` |
| 04-difficultes-desirables | `04-difficultes-desirables.md` |
| 06-attention-deep-work | `05-attention-deep-work.md` |
| 07-pratique-deliberee | `06-pratique-deliberee-metacognition.md` (partie pratique délibérée) |
| 08-metacognition | `06-pratique-deliberee-metacognition.md` (partie métacognition) |
| 13-apprendre-avec-ia | `07-capstone-apprendre-avec-ia.md` (devient module d'enseignement, plus le capstone) |

---

## J1 — Pourquoi tu oublies (+ mythe des styles d'apprentissage)
- Courbe d'oubli (Ebbinghaus) ; mémoire = récupération (pas stockage passif) ; illusion de compétence (fluency). **Encart Pseudoscience : styles d'apprentissage = mythe** (Pashler 2008, Dekker 2012). Mention sommeil = consolidation (nuance Schmid 2022).
- Slug : `01-pourquoi-tu-oublies`. Pas de code.

## J2 — Retrieval practice
- Active recall, flashcards, blank-page recall ; effet test. Variante : **enseigner à un pair** (retrieval génératif). Source : Roediger & Karpicke 2006.
- Slug : `02-retrieval-practice`. Pas de code.

## J3 — Spaced repetition & Anki
- Distributed practice (Cepeda 2006) ; intervalles croissants ; SM-2 ; mise en place d'Anki.
- Code : `03-spaced-repetition.py` (planificateur SM-2) — réutiliser l'existant.
- Slug : `03-spaced-repetition`.

## J4 — Difficultés désirables
- Interleaving & variation (Bjork ; Rohrer & Taylor : interleaved 72% vs blocked 38%) ; pourquoi « plus dur sur le moment = mieux ancré ».
- Slug : `04-difficultes-desirables`. Pas de code.

## J5 — Élaboration & encodage profond (🆕)
- Self-explanation (Chi) ; elaborative interrogation ; dual coding (Paivio) ; levels of processing (Craik & Lockhart — en encart, en posant sa **circularité**).
- **Disclaimer obligatoire** : élaboration = utilité **modérée**, dual coding = utilité **faible→modérée** (Dunlosky 2013) — complément, PAS technique #1.
- Slug : `05-elaboration-encodage`. Pas de code.

## J6 — Attention, charge cognitive & deep work
- Mémoire de travail (~4 chunks, Cowan) ; charge cognitive ; chunking ; focus sans distraction pendant la session (Newport). **= capacité cognitive** (le « se mettre au travail/durer » est en J10).
- Slug : `06-attention-deep-work`. Pas de code.

## J7 — Pratique délibérée
- Objectifs précis + feedback immédiat + représentations mentales (Ericsson). **Nuance variance Macnamara** (la pratique explique ~26%/21%/18%… selon le domaine — dégonfler « 10 000 h »). *Théorie ; l'opérationnel par stades est en J12.*
- Slug : `07-pratique-deliberee`. Pas de code.

## J8 — Métacognition
- Planifier / monitorer / ajuster ; technique Feynman ; calibrer ses jugements ; éviter l'illusion de fluidité.
- Slug : `08-metacognition`. Pas de code.

## J9 — Mesurer son apprentissage (🆕)
- Feedback formatif ; test pré/post ; suivre la rétention dans le temps ; calibration (lien J8) ; définir des **métriques** d'apprentissage. *Prérequis caché du capstone J14.*
- Code : `09-mesurer-apprentissage.py` (suivi de rétention + delta pré/post + courbe d'oubli mesurée).
- Slug : `09-mesurer-apprentissage`.

## J10 — Motivation, habitudes, énergie & apprendre sous pression
- Boucles d'habitude (Wood & Rünger 2016) ; procrastination (Steel 2007) ; sommeil comme **énergie** ; **apprendre sous pression** : anxiété de test, valeur de l'erreur, jour J.
- Slug : `10-motivation-habitudes`. Code optionnel (tracker d'habitudes léger).

## J11 — Lecture & prise de notes efficaces
- SQ3R (Robinson 1946), méthode Cornell (Pauk 1962) ; **debunk surlignage/relecture** (utilité faible, Dunlosky) ; manuscrit vs clavier : **réplication échouée Morehead 2019** → c'est le **traitement** (reformuler), pas l'outil.
- Slug : `11-lecture-prise-notes`. Pas de code.

## J12 — Acquisition d'une compétence (🆕, opérationnel)
- **UN seul cadre transférable** : stades de Fitts & Posner (cognitif → associatif → autonome), décomposition d'une compétence, drills ciblés, immersion-avec-feedback, calibrage du feedback selon le stade. **Ne PAS multiplier les domaines** : 1 fil rouge (au choix, ex. le code) + 2-3 encarts courts pour la généralité. Renvoyer la théorie Ericsson à J7 (ne pas la re-faire).
- Slug : `12-acquisition-competence`. Pas de code.

## J13 — Apprendre avec l'IA
- LLM comme tuteur socratique ; générateur de retrieval practice & d'espacement ; partenaire Feynman. **2-sigma de Bloom = effet du mastery learning (standard 90%), pas magie du tutorat** (disclaimer). Dépend de J9 (mesure) et J2/J3/J8.
- Slug : `13-apprendre-avec-ia`. Pas de code (LLM mocké/illustratif ; pas de clé API requise).

## J14 — Capstone : système d'apprentissage augmenté
- Livrable portfolio : choisir un sujet réel, bâtir un **plan retrieval + espacement** (J2/J3), une **boucle IA** (J13), et des **métriques de suivi** (J9) ; auto-évaluer la rétention.
- Livrables : `01-theory/14-capstone-systeme-apprentissage.md` (brief + grille) + exercices + solution + `04-projects/README.md` (gabarit). 
- Slug : `14-capstone-systeme-apprentissage`.
