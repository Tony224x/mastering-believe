# Module 11 — Vérification de l'information à l'ère de l'IA

> **Temps estimé** : 45 min | **Prérequis** : Modules 01-10
>
> **Objectif** : Appliquer la méthode SIFT et la lecture latérale pour vérifier une information, détecter une image hors-contexte, une fausse citation ou une hallucination de LLM — quelle que soit la source.

---

## 1. Pourquoi la vérification est une compétence de raisonnement

Les informations incorrectes circulent depuis toujours. Ce qui change aujourd'hui : la vitesse de diffusion, le volume, et l'arrivée de générateurs de texte fluides qui produisent des erreurs **avec le même ton confiant que les vérités**.

Résultat empirique clé (Pennycook & Rand, 2019) : la sensibilité aux fausses informations s'explique **davantage par un manque de réflexion analytique** que par des motivations partisanes. Cela signifie que la vérification est un **levier cognitif universel** — elle s'améliore par la pratique, indépendamment des opinions.

> **À retenir** : Le problème central n'est pas la malveillance des sources — c'est le **mode automatique de lecture** (Système 1). La vérification active le Système 2.

---

## 2. La méthode SIFT (Caulfield, 2017)

SIFT est le protocole standard des fact-checkers professionnels. Quatre mouvements, dans cet ordre :

### S — Stop (Pause)

Avant de partager, de citer ou de croire, **faites une pause**. La simple pause réduit la transmission d'informations non vérifiées (Pennycook et al., 2021 : un "accuracy nudge" de 3 secondes améliore significativement la qualité des contenus partagés).

**Signal d'alerte** : fort impact émotionnel (surprise, indignation, enthousiasme) → risque de court-circuit de la vérification.

---

### I — Investigate the source (Examinez la source)

Avant de lire le contenu, renseignez-vous **sur** la source — pas dedans.

Questions clés :
- Qui est l'auteur ou l'organisation ? Quel historique éditorial ?
- Ce site a-t-il une réputation connue pour ce type de contenu ?
- Y a-t-il un conflit d'intérêts déclaré (ou non déclaré) ?

**Lecture latérale** (voir section 3) : quitter la page et ouvrir d'autres onglets pour chercher ce que des tiers disent *de* cette source — c'est plus rapide et plus fiable que de lire tout le document.

---

### F — Find better coverage (Trouvez d'autres sources)

Une seule source, même fiable, ne suffit pas à confirmer une affirmation inhabituellement forte.

Démarche :
1. Chercher le même fait sur **3 sources indépendantes**.
2. Vérifier si une agence de fact-checking reconnue (AFP Factuel, Snopes, Reuters Fact Check) a déjà traité la question.
3. Si toutes les sources citent la même source initiale, remonter à celle-ci directement (étape T).

**Piège fréquent** : 10 articles peuvent tous reprendre le même dépêche d'origine — la multiplicité n'est pas l'indépendance.

---

### T — Trace claims, quotes, and media to their original context (Remontez à la source)

Les erreurs les plus communes ne sont pas des inventions totales — ce sont des **vraies informations sorties de leur contexte** : une citation tronquée, une image réelle mais datant d'un autre événement, un chiffre exact mais dont le dénominateur a été omis.

Démarche :
1. Localiser la **source primaire** (article original, communiqué, donnée brute).
2. Vérifier que la citation est complète et non tronquée.
3. Pour les images : vérifier la date et le lieu d'origine (voir section 5).

---

## 3. Lecture latérale vs lecture verticale

| | Lecture verticale | Lecture latérale |
|---|---|---|
| **Définition** | Lire le document de bout en bout | Ouvrir d'autres onglets pour voir ce que des tiers disent *de* la source |
| **Analogie** | Plonger en apnée | Regarder depuis la surface plusieurs zones |
| **Avantage** | Profondeur sur un texte connu | Rapidité pour évaluer une source inconnue |
| **Quand l'utiliser** | Source déjà évaluée comme fiable | Première rencontre avec une source ou une affirmation |

**Exemple pratique** : un article affirme "le café réduit la fatigue cognitive de 40 % selon une étude récente". Lecture latérale : ouvrir un onglet, taper `café cognition méta-analyse` + chercher l'auteur cité. En 2-3 minutes : les méta-analyses disponibles montrent des effets bien plus modestes et fortement variables selon la tolérance à la caféine. L'article a vraisemblablement sur-interprété une seule étude.

---

## 4. Hallucinations de LLM et fausses citations

Les grands modèles de langage (LLMs) génèrent des textes fluides et confiants même lorsqu'ils **inventent** des faits, des citations, des URLs ou des DOIs. Ce phénomène s'appelle **hallucination**.

### Signes d'alerte d'une citation hallucinée

1. L'URL renvoie sur une page 404 ou inexistante.
2. Le DOI est invalide (vérifiable sur doi.org en 10 secondes).
3. L'auteur existe bien mais n'a pas écrit ce titre.
4. La date ou la revue ne correspondent pas à l'œuvre connue de l'auteur.
5. La formulation est suspicieusement parfaite — trop bien adaptée à l'argument en cours.

### Protocole de vérification d'une citation (4 étapes)

1. **Copier le titre exact** entre guillemets dans Google Scholar : `"titre exact de l'article"`.
2. Si trouvé → vérifier auteur, revue, année correspondent.
3. Si non trouvé via le titre → tester le DOI sur [doi.org](https://doi.org).
4. Si toujours introuvable → la citation est **très probablement hallucinée** ; ne pas l'utiliser.

> **À retenir** : Une source inventée avec confiance n'est pas plus vraie qu'une source inventée avec hésitation. Le style ne remplace pas la vérification.

**Exemple concret — fausse citation type** :

Un LLM produit : *"Einstein, A. (1930). 'The measure of intelligence is the ability to change.' Annals of Physics, 12(4), 88."* — Ce titre n'existe pas dans les œuvres d'Einstein. La formule lui est attribuée sur internet depuis des décennies sans aucune source primaire. Vérification : le titre ne donne aucun résultat sur Google Scholar ; le DOI est fictif ; les archives d'Einstein (einstein.caltech.edu) ne contiennent pas cette phrase. Verdict : citation apocryphe, non utilisable.

---

## 5. Images hors contexte et deepfakes

### Images hors contexte

Une image authentique peut induire en erreur si elle est présentée avec un lieu ou une date incorrects. Méthode de vérification par **recherche inversée d'image** :

| Outil | URL | Usage |
|-------|-----|-------|
| Google Images | images.google.com | Clic droit → "Rechercher cette image" |
| TinEye | tineye.com | Cherche les occurrences datées plus anciennes |
| InVID / WeVerify | weverify.eu | Extension navigateur, analyse vidéo aussi |

**Exemple concret** : une photographie montre des dommages importants sur un bâtiment présentés comme "récents". Recherche TinEye → la même image apparaît sur un article de presse daté de plusieurs années auparavant, dans un autre pays. L'image est réelle mais recyclée hors contexte.

**Signal d'alerte** : photo sans métadonnées visibles, angle inhabituel, qualité incohérente avec la date supposée.

### Deepfakes vidéo et audio

Les deepfakes sont des vidéos ou audios générés par IA, imitant le visage ou la voix d'une personne réelle. Exemples non-politiques de cas documentés : acteurs de cinéma dont la voix ou le visage est copié sans consentement dans des publicités ou contenus viraux.

**Signaux visuels d'alerte** :
- Flou ou artefacts aux bords du visage (oreilles, cheveux, lunettes).
- Clignements des yeux irréguliers ou absents.
- Décalage léger entre les mouvements des lèvres et le son.
- Éclairage incohérent entre le visage et le fond.

**Démarche de vérification** :
1. Chercher la vidéo originale sur le compte officiel/vérifié de la personne.
2. Chercher `[nom de la personne] + deepfake + [titre de la vidéo]` pour voir si la manipulation a déjà été signalée.
3. Outils de détection (indicatifs, pas définitifs) : FakeCatcher (Intel), Deepware Scanner.

> **Limite importante** : les outils de détection de deepfakes ont un taux d'erreur non négligeable. La détection humaine reste le premier filtre — douter et vérifier la source reste la méthode la plus fiable.

---

## 6. Le cas du faux remède "miracle"

**Schéma classique d'une fausse promesse thérapeutique** :

1. Une affirmation extraordinaire ("efface 30 ans en 3 jours", "guérit X sans effets secondaires").
2. Une source opaque (pas d'auteur nommé, pas de revue, pas de DOI).
3. Un témoignage unique présenté comme preuve suffisante.
4. Un mécanisme qui sonne scientifique sans l'être ("booste l'énergie cellulaire").

**Protocole SIFT appliqué** :

- **S** : l'affirmation est extraordinaire → pause obligatoire.
- **I** : qui vend ce produit ? Conflit d'intérêts évident.
- **F** : chercher `[nom du produit] + essai clinique` et `[nom du produit] + arnaque` ou `+ hoax`.
- **T** : l'étude citée existe-t-elle ? Remontez à la publication primaire.

> **À retenir** : Une affirmation extraordinaire exige des preuves extraordinaires. Un témoignage unique, même sincère, ne remplace pas un essai contrôlé randomisé.

---

## 7. Synthèse — Quand utiliser quelle stratégie

| Situation | Stratégie prioritaire |
|-----------|----------------------|
| Source inconnue | Lecture latérale (étape I de SIFT) |
| Affirmation forte sans source | Find better coverage (étape F) |
| Citation ou chiffre précis | Trace to origin (étape T) |
| Citation générée par un LLM | Google Scholar + doi.org |
| Image présentée comme récente | TinEye ou Google Images inversée |
| Vidéo d'une personnalité publique | Chercher la source officielle + termes "deepfake" |
| Promesse extraordinaire | SIFT complet + essai clinique |

---

## Flash-cards (Module 11)

**Q1 : Qu'est-ce que SIFT et que signifie chaque lettre ?**
> R : **S**top (pause), **I**nvestigate the source, **F**ind better coverage, **T**race claims to original context. Protocole de vérification en 4 mouvements (Caulfield, 2017).

**Q2 : Quelle est la différence entre lecture verticale et lecture latérale ?**
> R : Verticale = lire le document lui-même. Latérale = quitter la page et chercher ce que d'autres sources disent *de* cette source. La latérale est plus rapide pour évaluer une source inconnue.

**Q3 : Comment vérifier qu'une citation générée par un LLM est authentique ?**
> R : 1) Titre exact entre guillemets sur Google Scholar. 2) DOI sur doi.org. 3) Confirmer auteur + revue + année. Si introuvable → probablement hallucinée.

**Q4 : Quel outil utilise-t-on pour vérifier si une image a été publiée à une date antérieure ?**
> R : TinEye (tineye.com) ou Google Images (recherche inversée). Ils montrent les occurrences datées plus anciennes de la même image.

**Q5 : Selon Pennycook & Rand (2019), quel est le principal facteur qui explique la sensibilité aux fausses informations ?**
> R : Le **manque de réflexion analytique** (pensée Système 1 non freinée), pas uniquement les motivations partisanes. Le levier est cognitif et universel.

---

## Points clés à retenir

1. **SIFT** (Stop, Investigate, Find, Trace) est le protocole standard des fact-checkers professionnels.
2. La **lecture latérale** — quitter la page pour chercher ce que des tiers disent *de* la source — est plus efficace que lire en profondeur une source inconnue.
3. Les **LLMs hallucinent** avec confiance : vérifier toute citation par Google Scholar + doi.org avant usage.
4. Une image authentique peut tromper si elle est **hors contexte** : vérifier via TinEye ou Google Images.
5. Les **deepfakes** ne se détectent pas à coup sûr par l'œil nu — chercher la source officielle reste le filtre principal.
6. Une affirmation extraordinaire (remède miracle, chiffre spectaculaire) exige une vérification SIFT complète et une preuve de niveau essai contrôlé.

---

## Pour aller plus loin

- **SIFT** : Caulfield, M. (2017). *Web Literacy for Student Fact-Checkers.* CC BY 4.0. https://pressbooks.pub/webliteracy/
- **Lazy, not biased** : Pennycook, G. & Rand, D. G. (2019). *Cognition*, 188, 39-50. https://www.sciencedirect.com/science/article/abs/pii/S001002771830163X
- **Accuracy nudge** : Pennycook, G. et al. (2021). *Nature*, 592, 590-595. https://www.nature.com/articles/s41586-021-03344-2
- **Fact-checking** : AFP Factuel. https://factuel.afp.com/ | Reuters Fact Check. https://www.reuters.com/fact-check/
- **Recherche inversée d'image** : TinEye. https://tineye.com | InVID/WeVerify. https://weverify.eu
- **Vérification de DOI** : doi.org. https://doi.org
- **Citations d'Einstein** (cas d'usage de vérification) : Einstein Archives Online. https://www.einstein.caltech.edu/
