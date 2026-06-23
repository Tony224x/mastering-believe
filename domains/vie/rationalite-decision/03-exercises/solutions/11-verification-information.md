# Solutions — Module 11 : Vérification de l'information à l'ère de l'IA

> Corrigé modèle. Lire après avoir tenté les exercices.

---

## Solution — Exercice 1 : Analyser une fausse citation

### Vérifications concrètes (étape T de SIFT)

1. **Google Scholar** — taper `"L'imagination est plus importante que la connaissance" Einstein` + `"Berliner Tagblatt"`. Résultat attendu : aucune occurrence de la source citée dans la littérature académique.
2. **Einstein Archives Online** (einstein.caltech.edu) — chercher par date (octobre 1926) et par titre de journal.
3. **Quote Investigator** (quoteinvestigator.com) — site spécialisé dans la traçabilité des citations célèbres. Une recherche sur cette phrase révèle que la source authentique est *The Saturday Evening Post*, 26 octobre 1929 (entretien "What Life Means to Einstein" par George Sylvester Viereck), et non le Berliner Tagblatt en 1926.
4. **Recherche de l'article original** — chercher `"Saturday Evening Post" Einstein 1929 imagination knowledge`.
5. **Vérification secondaire** — comparer la traduction française avec le texte anglais original pour s'assurer de la fidélité.

### Éléments plausibles vs signaux d'alerte

| Plausible | Signal d'alerte |
|-----------|-----------------|
| La phrase est attribuée à Einstein depuis des décennies — vraisemblable dans l'esprit | La source primaire est une référence précise (journal, date, page) qui doit être vérifiable |
| Le contenu est cohérent avec d'autres déclarations connues d'Einstein | Le Berliner Tagblatt de 1926 antérieur de 3 ans à la vraie date — incohérence factuelle |
| La formulation est fluide et convaincante | Aucun hyperlien, aucun DOI, aucune archive numérique citée |

### Verdict

Non utilisable telle quelle : la source citée (Berliner Tagblatt, 1926) ne correspond pas à la source vérifiable. La citation elle-même existe, mais sa vrai source est *The Saturday Evening Post*, 1929. **Règle : même une citation réelle doit être tracée jusqu'à sa source primaire avant usage.**

---

## Solution — Exercice 2 : Détecter une image hors contexte

### Application SIFT mouvement par mouvement

**S — Stop** : 12 000 partages en 2 heures + forte charge émotionnelle (accident, blessés, accusation d'insécurité) = signal d'alerte typique. Ne pas partager avant vérification.

**I — Investigate the source** : Qui a posté ce message ? Compte récent ? Peu d'abonnés ? Pas de liens vers un article de presse structuré ? Ces indices affaiblissent la fiabilité. (Lecture latérale : chercher le nom du compte + "fiable" ou "fake".)

**F — Find better coverage** : Chercher `chantier effondré [ville fictive] site:lemonde.fr OR site:leparisien.fr` et `chantier effondré [date supposée] [ville fictive]`. Si aucun grand média ne couvre l'événement d'hier, c'est un signal fort d'inexactitude.

**T — Trace to origin** : outil = **TinEye** (tineye.com) ou **Google Images** (clic droit → "Rechercher cette image"). Requête : uploader ou coller l'URL de l'image. Résultat permettant de conclure : TinEye affiche des occurrences datées d'il y a 4 ans, sur des articles de presse d'un autre pays. L'image est réelle mais son contexte est différent.

### Commentaire de signalement (exemple modèle)

> "Attention, cette image semble hors contexte : une recherche inversée sur TinEye montre qu'elle a été publiée il y a 4 ans dans [insérer la source trouvée], dans un contexte différent. Je n'ai pas trouvé de confirmation de l'incident décrit dans les médias locaux de [ville fictive]. Il vaut mieux vérifier avant de partager."

*(Ton factuel, source citée, pas d'accusation sur l'intention de l'auteur du post.)*

---

## Solution — Exercice 3 : Faux remède et citation LLM

### Partie A — Signaux d'alerte (au moins 4)

1. **Chiffre spectaculaire non sourcé dans le corps du texte** : "+47 % de production mitochondriale" — un résultat aussi précis doit pointer vers une étude avec DOI accessible.
2. **Revue inconnue** : "Journal of Cellular Longevity" — une recherche rapide sur PubMed ne retourne pas cette revue dans l'index MEDLINE ; cela suffit à alerter.
3. **Mécanisme pseudo-scientifique vague** : "augmente la production mitochondriale" sans préciser la molécule active, la voie métabolique, la taille d'échantillon ou le groupe contrôle.
4. **Lien direct vers une boutique** : le contenu est publié par le vendeur du produit — conflit d'intérêts évident et non déclaré.
5. **Citation directe du chercheur dans un article de blog** sans lien vers l'article original — impossible de vérifier si la citation est complète ou sortie de son contexte.
6. *(Bonus)* **Étude à participant unique implicitement généralisée** : "chez l'adulte sain" ne précise pas N ; un N très faible peut produire des résultats spectaculaires non réplicables.

### Requêtes pour l'étape F

- `AlphaCèle supplément essai clinique PubMed`
- `AlphaCèle® avis OR arnaque OR hoax`
- `"Journal of Cellular Longevity" facteur d'impact site:scimagojr.com`
- `mitochondrie supplément méta-analyse 2022 2023`
- `Hana Novak Prague mitochondria researcher`

Si aucun résultat structuré sur PubMed ou une base scientifique reconnue ne ressort, l'affirmation n'est pas étayée par une preuve publiée et évaluée par les pairs.

### Partie B — Protocole de vérification en 4 étapes

**Étape 1 — Titre exact sur Google Scholar**
Taper : `"Mitochondrial synthesis enhancement through polyphenol supplementation"`. Si aucun résultat ou résultat non correspondant → passer à l'étape 2.

**Étape 2 — Vérification du DOI**
Ouvrir [doi.org](https://doi.org) et taper : `10.1016/j.cmet.2022.00317`. Si la page retourne une erreur 404 ou "DOI not found" → citation non vérifiable.

**Étape 3 — Vérification croisée auteur + revue**
Chercher `Novak Petersen Cellular Metabolism 2022` pour voir si des travaux de ces auteurs dans cette revue existent. *Cellular Metabolism* est une revue réelle et indexée ; si ce titre n'y figure pas, c'est décisif.

**Étape 4 — Conclusion et formulation**
Le DOI renvoie une erreur et Google Scholar ne trouve pas le titre exact.

**Formulation correcte dans un document de travail** :

> "La référence 'Novak & Petersen (2022), Cellular Metabolism' n'a pas pu être vérifiée : le titre exact n'apparaît pas sur Google Scholar et le DOI fourni (10.1016/j.cmet.2022.00317) est invalide. Cette citation est **probablement hallucinée** par le LLM et ne doit pas être utilisée sans vérification complémentaire directe (contact de l'auteur supposé, ou recherche dans l'archive de la revue)."

**Pourquoi "probablement" et non "certainement"** : il est théoriquement possible que la référence existe sous un titre légèrement différent ou avec une erreur de DOI. La porte reste ouverte à une vérification manuelle supplémentaire — mais en l'absence de confirmation, la citation est **inutilisable**.

---

## Récapitulatif des patterns à retenir

| Type d'erreur | Signal principal | Outil de vérification |
|---------------|------------------|-----------------------|
| Fausse citation de personnalité | Source précise mais non vérifiable | Google Scholar + Quote Investigator |
| Image hors contexte | Émotion forte + absence de couverture médiatique | TinEye, Google Images |
| Faux remède miracle | Chiffre spectaculaire + vendeur = auteur | PubMed, recherche `hoax` + méta-analyse |
| Citation hallucinée par LLM | DOI invalide ou titre introuvable | doi.org + Google Scholar |
