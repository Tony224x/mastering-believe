# Solutions (hard) — Module 03 : Spaced repetition

> Verifie l'exercice 1 avec `02-code/03-spaced-repetition.py`. Exercice 2 personnel (criteres). Exercice 3 : trame attendue.

---

## Exercice 1 — Planning SM-2 complet (notes 5, 4, 3, 5, 5)

**Convention : intervalle = `round(intervalle_precedent × EF apres mise a jour)`, comme dans le script.**

| Session | Note | EF avant | EF apres | Intervalle precedent | Prochain intervalle | Date (J+cumul) |
|---------|------|----------|----------|---------------------|---------------------|----------------|
| 1 | 5 | 2.50 | 2.60 | 0 | 1 (1re reussie) | J+1 |
| 2 | 4 | 2.60 | 2.60 | 1 | 6 (2e reussie) | J+7 |
| 3 | 3 | 2.60 | 2.46 | 6 | round(6 × 2.46) = 15 | J+22 |
| 4 | 5 | 2.46 | 2.56 | 15 | round(15 × 2.56) = 38 | J+60 |
| 5 | 5 | 2.56 | 2.66 | 38 | round(38 × 2.66) = 101 | J+161 |

**Detail du calcul EF, session 3 (note = 3) :**
- `EF = 2.60 + (0.1 - (5-3) × (0.08 + (5-3) × 0.02))`
- `EF = 2.60 + (0.1 - 2 × (0.08 + 0.04))`
- `EF = 2.60 + (0.1 - 2 × 0.12)`
- `EF = 2.60 + (0.1 - 0.24) = 2.60 - 0.14 = 2.46`

**Pourquoi l'intervalle augmente malgre la note 3 :** une note de 3 est consideree comme un **succes** par SM-2 (le seuil est note >= 3). L'EF baisse legerement (la carte etait un peu plus dure que prevu), mais comme c'est un succes, le compteur de repetitions continue et l'intervalle = intervalle × EF augmente quand meme. Seule une note < 3 (echec reel de recuperation) declenche le reset.

**Variante echec (session 3 = note 2) :**

| Session | Note | EF apres | Intervalle | Date (J+cumul) |
|---------|------|----------|-----------|----------------|
| 1 | 5 | 2.60 | 1 | J+1 |
| 2 | 4 | 2.60 | 6 | J+7 |
| 3 | **2** | 2.28 | **1 (reset)** | J+8 |
| 4 | 5 | 2.38 | 1 (1re reussie apres reset) | J+9 |
| 5 | 5 | 2.48 | 6 (2e reussie) | J+15 |

**Effet du reset :** une seule note < 3 remet l'intervalle a 1 et le compteur de repetitions a 0. La carte redemarre le cycle (1 jour, puis 6 jours...). Resultat : la ou la sequence reussie atteignait J+161 a la session 5, la variante echec n'est qu'a J+15 — la maitrise a long terme est repoussee de plusieurs paliers. **Lecon :** une seule mauvaise recuperation coute cher en planning, ce qui est voulu — SM-2 force a re-consolider ce qui n'est pas solide.

**Verification :** adapte `test_notes` dans le script et compare ; EF a ± 0.01.

---

## Exercice 2 — Systeme d'espacement personnel

**Elements d'un systeme solide :**
1. **Outil** : pour un gros volume (centaines/milliers de cartes) sur un horizon long -> Anki (SM-2/FSRS), car la gestion manuelle des intervalles individuels est impraticable. Pour un petit corpus a horizon court -> plan manuel suffit.
2. **Creation de cartes** : une idee par carte ; regle "ca merite une carte si je veux m'en souvenir dans 6 mois et que ca ne se rededuit pas trivialement" ; **limiter le flux de nouvelles cartes** (ex. 10-20/jour) pour eviter l'avalanche de revisions qui mene a l'abandon.
3. **Notation honnete** : noter selon la *recuperation reelle*, pas selon "je l'aurais su". Se sur-noter gonfle les intervalles -> la carte revient trop tard -> oubli. La discipline de notation est ce qui fait marcher l'algorithme.
4. **Retard** : apres une pause, traiter les cartes en retard progressivement, sans tout faire le meme jour (sinon on re-cree de la massed practice). Accepter un leger lissage sur quelques jours.
5. **Garde-fou anti-illusion** : verifier que les cartes testent comprehension/application (mecanisme, "pourquoi", cas d'usage) et pas seulement la reconnaissance d'un mot.
6. **Maturite/sortie** : une carte dont l'intervalle depasse un seuil (ex. plusieurs mois) et qui est reussie de facon stable est "mature" ; elle reste dans le deck mais ne pese plus sur le quotidien (l'espacement la gere). On ne supprime pas une carte reussie — on la laisse s'espacer.

**Garde-fou :** un systeme qui consiste a relire les cartes "pour reviser" sans tenter de repondre d'abord n'est pas un systeme d'espacement valide — c'est de la relecture deguisee.

---

## Exercice 3 — Mythe "l'espacement n'apporte rien a temps egal"

**1. Part de vrai :** le bachotage produit un vrai pic de performance a court terme et peut suffire si l'echeance est immediate et le contenu jetable. La sensation d'efficacite (fluidite) est reelle sur le moment.

**2. Erreur de fond :** *a temps total egal*, l'espacement bat la pratique massee. Mecanisme : reviser apres un debut d'oubli rend la recuperation plus *effortful*, et c'est cet effort qui consolide la trace (difficulte desirable). Bachoter en continu maintient une fluidite artificielle qui ne demande pas d'effort de recuperation — donc consolide peu. Ce n'est pas une question de quantite de temps mais de *distribution* du temps.

**3. Preuve :** Cepeda et al. (2006), meta-analyse de **839 mesures sur 317 experiences**, etablit que la pratique espacee bat massivement la pratique massee. Cepeda et al. (2008, >1350 sujets) donne la regle chiffree : intervalle optimal ≈ 10-20 % du delai avant le test. Crucial : ces comparaisons sont **a temps egal** — c'est la repartition, pas le volume, qui fait la difference.

**4. Honnetete sur la preuve :** le bachotage peut etre rationnel quand (a) l'echeance est demain et (b) le contenu n'a pas besoin d'etre retenu apres. Dans ce cas precis, optimiser le pic immediat est defendable. L'espacement n'est donc pas "toujours superieur pour tout objectif" — il est superieur pour la **retention durable**.

**5. Regle pratique :** pour tout ce que tu veux retenir au-dela de quelques jours, repartis le meme temps total en sessions espacees a intervalles croissants ; reserve le bachotage aux contenus jetables a echeance immediate.

**Garde-fou :** le corrige reconnait le cas legitime du bachotage et ne presente pas l'espacement comme une solution universelle. Pas de sur-vente.

**Reference :** Cepeda, N. J. et al. (2006). *Psychological Bulletin*, 132(3), 354–380. ; Cepeda, N. J. et al. (2008). *Psychological Science*, 19(11), 1095–1102.
