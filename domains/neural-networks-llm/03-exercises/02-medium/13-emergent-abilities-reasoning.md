# Exercices Medium — Jour 13 : Emergent abilities & reasoning

---

## Exercice 4 : Self-consistency — simulation vs theorie binomiale

### Objectif

Quantifier exactement ce que rapporte le vote majoritaire (self-consistency), et decouvrir son talon d'Achille : il AMPLIFIE aussi les erreurs.

### Consigne

Modele simplifie : chaque chaine de raisonnement donne la bonne reponse avec probabilite p, sinon une mauvaise reponse tiree uniformement parmi 4 distracteurs. On vote a la majorite sur k chaines (tie-break aleatoire).

1. Implementer :
   - `analytical_majority(p, k)` : probabilite que la bonne reponse soit majoritaire, calculable par enumeration des repartitions (ou par simulation de reference tres large). Pour simplifier, une version "bonne reponse vs meilleur distracteur" par multinomiale enumeree est acceptee pour k <= 21 — documenter l'approche
   - `simulate_majority(p, k, n_trials, rng)` : version Monte Carlo (20 000 essais)

2. Verifier l'accord simulation/analytique a ± 1 pt pour p ∈ {0.4, 0.55, 0.7} et k ∈ {1, 5, 21}. (Si l'analytique exact est trop lourd, comparer 20k essais a une simulation de reference 200k essais.)

3. Construire le tableau `p \ k` et verifier les 3 enseignements :
   - p=0.7, k=21 : accuracy > 0.95 (le vote transforme un modele moyen en modele fort)
   - p=0.55, k: 1→21 : le gain existe mais est plus lent
   - **p=0.4 (sous le seuil face aux distracteurs ?)** : avec 4 distracteurs uniformes a 0.15 chacun, la bonne reponse reste la PLUS probable → le vote l'amplifie quand meme ! Verifier que l'accuracy MONTE avec k. Puis refaire avec un distracteur systematique (le modele fait toujours la MEME erreur avec proba 0.45 > 0.4) : cette fois l'accuracy DESCEND avec k. C'est le vrai message : self-consistency amplifie la reponse modale, pas la bonne reponse.

4. Cout : ajouter au tableau le cout en "appels modele" (k) et calculer le gain marginal de k=21→41 pour p=0.7 — conclure sur les rendements decroissants.

### Criteres de reussite

- [ ] Simulation et reference concordent a ± 1 pt sur les 9 cases
- [ ] Le cas "erreurs dispersees" monte avec k MEME a p=0.4, et le cas "erreur systematique a 0.45" descend avec k — les deux sont demontres
- [ ] La conclusion "le vote amplifie la reponse modale" est ecrite et justifiee par les chiffres
- [ ] Les rendements decroissants sont chiffres (gain k=1→5 vs k=21→41)
- [ ] Le tie-break est gere proprement (documente)

---

## Exercice 5 : L'emergence est-elle un artefact de la metrique ?

### Objectif

Reproduire l'argument du papier "Are Emergent Abilities a Mirage?" : une capacite qui progresse de maniere LISSE peut sembler "emerger" brutalement selon la metrique choisie.

### Consigne

Modele jouet : la probabilite qu'un modele produise chaque token correctement est une fonction lisse de l'echelle s (en log-params) : `p_token(s) = sigmoid((s - 22) / 1.5)` pour s ∈ [16, 28] (s = log10 des FLOPs, peu importe l'unite).

1. Une "tache" = produire une reponse de L tokens, reussie seulement si TOUS les tokens sont corrects : `p_exact(s, L) = p_token(s)^L`.

2. Calculer et afficher (table + courbe ASCII) pour s ∈ [16, 28] par pas de 0.5 :
   - metrique lisse : `p_token(s)`
   - metrique exact-match : `p_exact(s, L)` pour L ∈ {1, 4, 16, 64}

3. Quantifier l'"emergence apparente" de chaque courbe (calculs exacts possibles : `p_exact` franchit le seuil a quand `p_token = a^(1/L)`, soit `s = 22 + 1.5 * logit(a^(1/L))`) :
   - `s_10` : echelle ou la metrique franchit 10%. Verifier que s_10 se decale fortement vers la droite avec L : `s_10(L=64) - s_10(L=1) > 8` unites — le modele "ne sait rien faire" sur quasi toute la plage
   - fraction "morte" de la plage [16, 28] ou la metrique reste < 10% : verifier < 25% pour L=1 mais > 85% pour L=64
   - largeur de transition `w = s_90 - s_10` : verifier qu'elle diminue avec L (w(64) < w(1)), meme si c'est le DECALAGE de s_10 plus que la compression qui cree l'effet "emergence"

4. Verifier le second argument du papier : avec une metrique CONTINUE sur la meme capacite — ici `edit_credit(s, L) = p_token(s)` (credit partiel par token, i.e. l'esperance de la fraction de tokens corrects) — il n'y a AUCUNE emergence : la courbe est la meme sigmoid lisse pour tout L.

5. Conclure en commentaire : qu'est-ce que ca implique (et n'implique PAS) pour les vraies emergent abilities ? (Ca montre que certaines "emergences" sont des effets de seuil de la metrique ; ca ne prouve pas que TOUTES le sont.)

### Criteres de reussite

- [ ] Les courbes p_token et p_exact sont calculees et affichees pour les 4 valeurs de L
- [ ] Le decalage s_10(64) - s_10(1) > 8 et la fraction morte (< 25% vs > 85%) sont verifies ; w(64) < w(1)
- [ ] La metrique a credit partiel supprime l'emergence (demontre numeriquement : meme s_10/s_90 pour tout L)
- [ ] Le point cle est formule : la capacite sous-jacente est lisse, c'est la METRIQUE exact-match qui cree la discontinuite apparente
- [ ] La nuance finale (ce que l'experience ne prouve pas) est presente

---

## Exercice 6 : Implementer un induction head algorithmique

### Objectif

Implementer A LA MAIN l'algorithme que les transformers decouvrent pour faire de l'in-context learning : l'induction head ([A][B]...[A] → predire [B]).

### Consigne

1. Implementer deux predicteurs de token suivant :
   - `bigram_predictor(corpus)` : statistiques de bigrammes apprises sur un corpus FIXE ("in-weights learning") — predit `argmax count(prev, .)`
   - `induction_predictor(context)` : SANS aucune statistique pre-apprise — cherche la derniere occurrence PRECEDENTE du token courant dans le contexte et predit le token qui la suivait ("in-context learning"). Fallback : token le plus frequent du contexte.

2. **Benchmark 1 — sequences repetees** (le test classique des induction heads) : sequences `S + S` ou S = 20 tokens aleatoires d'un vocab de 50 (100 sequences, seed fixe). Predire chaque token de la 2e moitie a partir du prefixe :
   - induction : accuracy >= 95% sur la 2e moitie (des la 2e occurrence, tout est predictible)
   - bigram (entraine sur un AUTRE corpus du meme vocab) : accuracy proche du hasard (< 10%)

3. **Benchmark 2 — tokens jamais vus** : sequences repetees sur un vocab DISJOINT de celui du corpus d'entrainement du bigram. L'induction head fonctionne a l'identique (>= 95%) ; le bigram est a 0. C'est la signature de l'ICL : la capacite est dans l'ALGORITHME, pas dans les poids.

4. **Benchmark 3 — texte naturel jouet** : sur un texte avec des repetitions naturelles (ex : "le chat dort. le chien dort. le chat mange. le chien mange."), comparer les deux et verifier que le bigram gagne cette fois sur les transitions frequentes quand le contexte ne contient pas encore la repetition. Discuter : les vrais LLMs combinent les deux.

5. Decomposer l'induction head en deux "tetes" comme dans les transformers reels (commentaire + code structure en 2 etapes) : (1) "previous token head" : construire la table `token a la position i-1` ; (2) "induction head" : matcher le token courant contre cette table et copier le successeur. Le code doit suivre explicitement ces 2 etapes.

### Criteres de reussite

- [ ] Induction >= 95% et bigram < 10% sur le benchmark 1
- [ ] Le benchmark vocab-disjoint donne induction >= 95% / bigram == 0 (la dissociation in-context vs in-weights est demontree)
- [ ] Le code est structure en 2 etapes (previous-token puis match-and-copy) avec le parallele transformers commente
- [ ] Le benchmark 3 nuance le resultat (chaque mecanisme a son regime)
- [ ] Les fallbacks (premiere occurrence d'un token) sont geres sans crash
