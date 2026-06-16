# Solutions — Module 05 : Elaboration & encodage profond

---

## Exercice 1 — Self-explanation sur un exemple resolu

### Corrige modele (exemple avec un probleme algorithmique : "trouver tous les sous-tableaux de somme egale a k")

**Exemple resolu choisi :**

```python
def subarray_sum(nums, k):
    count = 0
    prefix_sum = 0
    freq = {0: 1}           # Etape 1 : initialisation
    for n in nums:
        prefix_sum += n     # Etape 2 : somme prefixe courante
        diff = prefix_sum - k  # Etape 3 : complement
        count += freq.get(diff, 0)  # Etape 4 : compter
        freq[prefix_sum] = freq.get(prefix_sum, 0) + 1  # Etape 5 : enregistrer
    return count
```

**Self-explanation attendue (modele) :**

*Etape 1 — initialisation {0: 1} :*
> "Ce qui se passe ici : on cree un dictionnaire qui dit 'la somme 0 est apparue 1 fois'. La raison : si la somme prefixe courante est exactement k, on a besoin que `prefix_sum - k = 0` soit dans le dictionnaire pour compter ce sous-tableau. Sans cette initialisation, on raterait les sous-tableaux qui commencent a l'indice 0. Le principe sous-jacent : toute solution a ce type de probleme doit traiter le 'cas vide' — sous-tableau commencant au debut du tableau."

*Etape 3 — calcul de diff :*
> "Ce qui se passe : on cherche si une somme prefixe anterieure valait `prefix_sum - k`. La raison : si `somme[i..j] = k`, alors `prefix[j] - prefix[i-1] = k`, donc `prefix[i-1] = prefix[j] - k`. On cherche dans le passe. Le principe sous-jacent : deux-pass transforms en one-pass en stockant l'historique."

*Etape 5 — enregistrement :*
> "Ce qui se passe : on enregistre la somme prefixe courante apres l'avoir utilisee. La raison (ordre important) : si on enregistrait avant d'utiliser, on compterait les sous-tableaux vides (k=0) deux fois. Le principe : dans les algos qui lisent leur propre historique, l'ordre read-then-write evite l'auto-contamination."

**Points souvent manques lors de la verification :**
- L'initialisation {0: 1} est souvent compris mais la raison profonde (sous-tableaux partant de l'indice 0) est rarement formulee precisement.
- L'ordre etape 4 → etape 5 (compter AVANT d'enregistrer) est une subtilite que la relecture passive ne capture pas.

**Connexions spontanees a valoriser :**
- Lien avec le pattern "two-pass" vu en algo (ici reduit a un one-pass grace au dictionnaire).
- Lien avec "prefix sum" vu dans d'autres problemes de sous-tableaux.

---

## Exercice 2 — Dual coding : schema maison a partir d'un paragraphe de theorie

### Corrige modele (concept : dual coding lui-meme)

**Schema attendu (description textuelle du schema modele) :**

```
[Information a encoder]
         |
    +---------+
    |         |
[Traitement  [Traitement
  verbal]      imagerie]
(logogens)   (imagens)
    |         |
    v         v
[Trace      [Trace
 verbale]    visuelle]
    |         |
    +---------+
         |
  [Recup. possible
   par l'une OU l'autre
   des 2 voies]
```

**Annotations textuelles minimales attendues :**
1. Fleche "echec d'une voie" → l'autre voie reste disponible (redondance).
2. Note "logogens = mots/phrases" sur le cote verbal.
3. Note "imagens = images mentales, schemas" sur le cote imagerie.
4. Annotation "auto-genere > copie" pres du bloc imagerie (generer son schema vaut mieux que regarder celui du cours).

**Narration modele :**
> "Quand j'apprends quelque chose, mon cerveau peut traiter l'information de deux manieres : en mots (systeme verbal, logogens) ou en images mentales (systeme imagerie, imagens). Si j'utilise les deux — en lisant ET en dessinant — je cree deux traces separees en memoire. Ca signifie que plus tard, quand j'essaie de me souvenir, j'ai deux portes d'entree au lieu d'une. Meme si la porte verbale est bloquee (je n'arrive pas a me rappeler la phrase), la porte visuelle peut s'ouvrir. L'avantage est maximal quand les deux representations sont complementaires — une image qui dit la meme chose que le texte mot pour mot n'apporte rien."

**Erreurs frequentes a corriger :**
- Dessiner le schema du livre au lieu de le construire de zero (ne compte pas).
- Confondre dual coding (deux systemes cognitifs) avec "mettre du texte et des images sur la meme diapo" (la disposition ne suffit pas — le traitement actif est necessaire).

---

## Exercice 3 — Elaborative interrogation en chaine : construire un reseau de "pourquoi"

### Corrige modele (exemple avec 5 faits de finance personnelle)

**Les 5 faits :**
1. Les frais annuels d'un fonds actif (1,5 %) effacent ~40 % du gain sur 30 ans compare a un fonds indiciel (0,2 %) a performance identique.
2. Le biais de confirmation pousse les investisseurs a sur-ponderer les infos qui confirment leurs positions.
3. Diversifier entre actifs decorreles reduit la volatilite du portefeuille sans reduire le rendement attendu.
4. L'effet de l'interet compose s'accelere de facon non lineaire dans le temps.
5. Vendre en panique lors d'un krach est une des principales causes de sous-performance des particuliers.

**Score immediat typique (memorisation passive, 2 min d'attente) :** 3–4/5 faits mais avec peu de details.

**Elaborative interrogation modele pour le fait 1 (frais) :**

1. *Pourquoi est-ce vrai ? Quel mecanisme l'explique ?*
> "Parce que les frais se retranchemnt chaque annee sur la valeur totale du portefeuille, pas seulement sur les gains. Sur 30 ans, 1,3 point de frais supplementaires jouent sur une base qui croit geometriquement — l'effet compose amplifie l'ecart, pas lineairement mais exponentiellement. Sur 100 000€ a 7 % brut : avec 0,2 % de frais → ~730 000€ ; avec 1,5 % de frais → ~430 000€. La difference est de 300 000€ — c'est 40 % du gain, pas 40 % du capital final."

2. *Dans quelles conditions est-ce faux ou nuance ?*
> "Si le fonds actif surperforme regulierement de plus de 1,3 % net, les frais sont justifies. Mais les donnees SPIVA montrent que ~80-90 % des fonds actifs sous-performent leur indice sur 15 ans. La surperformance durable nette de frais est rare."

3. *A quoi cela me fait-il penser que je connais deja ?*
> "A la courbe d'oubli — un cout qui s'applique periodiquement sur une base croissante. Et a la notion de 'charge' en cognitive load theory : une friction legere mais constante finit par peser enormement sur le systeme entier."

**Score differe typique (24h, apres elaboration) :** 5/5 avec explication causale, parfois meme les chiffres.

**Analyse attendue (modele) :**

> "L'elaborative interrogation a produit un reseau de connexions autour de chaque fait : causes mecaniques, limites, analogies avec des concepts connus. Lors du test differe, je n'ai pas simplement retrouve les 5 phrases — j'ai reconstruit les faits depuis leur logique sous-jacente. Ca correspond exactement a ce que Chi et al. (1989) observent avec le self-explanation : les apprenants qui relient les informations a des principes se souviennent mieux et transferent mieux, parce qu'ils n'ont pas juste memorise la surface. En termes de levels of processing (Craik & Lockhart 1972), le traitement semantique profond (sens, causes, analogies) produit une trace plus durable que la lecture superficielle. Dunlosky (2013) reste prudent (utilite moderee) — et en effet, ce n'est pas 5/5 garanti a coup sur : si les connexions generees sont incorrectes, elles peuvent induire des faux souvenirs. L'etape de verification est donc critique."

**Point d'honnetete a valoriser :**
Si le score differe ne s'ameliore pas, les explications valides incluent :
- Les connexions generees etaient fausses ou peu pertinentes.
- L'intervalle de 24h est trop court pour mesurer l'avantage de l'elaboration sur la memoire a long terme (la difference se creuse sur plusieurs semaines).
- La charge cognitive elevee de l'exercice 3 a pu creer de la fatigue qui masque le gain.

Un apprenant honnete qui rapporte ca demontre une meilleure metacognition qu'un apprenant qui force une conclusion positive.
