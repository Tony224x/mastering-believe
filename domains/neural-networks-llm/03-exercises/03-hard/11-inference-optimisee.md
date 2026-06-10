# Exercices Hard — Jour 11 : Inference optimisee

---

## Exercice 7 : Speculative decoding greedy de bout en bout

### Objectif

Implementer le pipeline complet draft-then-verify en mode greedy avec deux vrais modeles de langage (n-grammes), et prouver l'equivalence exacte avec le decodage du gros modele seul.

### Consigne

1. **Les deux modeles** : sur un petit corpus de texte (quelques phrases repetitives, ex : du texte type "le chat dort. le chien dort. le chat mange..."), construire :
   - modele CIBLE : trigramme caractere avec lissage add-0.1
   - modele DRAFT : bigramme caractere avec lissage add-1.0 (plus faible, mais correle)
   Les deux exposent `next_logprobs(context) -> (vocab,)`.

2. **Reference** : `generate_greedy(target, prompt, n_tokens)` — argmax token par token avec le modele cible seul. C'est la verite terrain.

3. **Speculatif greedy** : `generate_speculative(target, draft, prompt, n_tokens, k=4)` :
   - le draft propose k tokens en greedy
   - le cible "verifie" : calcule ses argmax sur les k positions (en un seul passage conceptuel — ici une boucle, mais compter UNE verification par cycle)
   - accepter le plus long prefixe ou `argmax_cible == token_draft` ; au premier desaccord, emettre l'argmax du cible et jeter le reste
   - si tout est accepte, emettre en bonus l'argmax du cible a la position k+1

4. **Tests** :
   - **equivalence exacte** : sur 5 prompts et 80 tokens generes, la sortie speculative est IDENTIQUE caractere par caractere a la reference (c'est la propriete fondamentale du verify greedy)
   - compter les appels de verification du cible vs les appels token-par-token de la reference : speedup theorique = `n_tokens / n_cycles` ; l'afficher avec le taux d'acceptation moyen
   - cas defavorable : remplacer le draft par un modele uniforme (aleatoire) → l'equivalence doit TENIR quand meme, seul le speedup s'effondre (~1 token/cycle). Verifier les deux points.

5. Tableau final : `draft | taux d'acceptation | tokens/cycle | appels cible (spec) | appels cible (ref)` pour draft=bigramme et draft=uniforme, k ∈ {2, 4, 8}.

### Criteres de reussite

- [ ] L'equivalence exacte est verifiee sur les 5 prompts pour TOUTES les configs (bons et mauvais drafts, tous les k)
- [ ] Le comptage des appels est honnete (1 verification de cycle = 1 appel cible, +1 pour le token bonus documente)
- [ ] Le bon draft donne > 1.5 tokens/cycle sur ce corpus, le draft uniforme ~1
- [ ] Le tableau montre l'effet de k et la conclusion (k optimal depend du taux d'acceptation) est ecrite
- [ ] Execution < 30 s

---

## Exercice 8 : Simulateur de continuous batching

### Objectif

Construire un simulateur discret de serving LLM et quantifier pourquoi le continuous batching (vLLM-style) ecrase le batching statique en throughput ET en latence.

### Consigne

1. **Charge de travail** (seed fixe) : 48 requetes, temps d'arrivee ~ Poisson (espacement exponentiel moyen 2 steps), longueur de sortie tiree uniformement dans [10, 120] tokens. Un step de simulation = un forward de decode du GPU qui produit 1 token pour CHAQUE sequence active (capacite max : 8 slots). On ignore le prefill (le documenter).

2. **Scheduler statique** : les requetes attendent en file ; quand 8 sont disponibles (ou que la file est vide et qu'au moins une attend depuis le debut — choisir : batch complet uniquement, sauf fin de trace), le batch demarre et le GPU est occupe jusqu'a ce que la sequence LA PLUS LONGUE du batch finisse. Les slots des sequences finies restent occupes mais inutiles (c'est le gachis a mesurer). Aucune admission en cours de batch.

3. **Scheduler continu** : a CHAQUE step, les slots liberes par les sequences terminees sont immediatement reattribues aux requetes en attente.

4. **Metriques** (les memes pour les deux) :
   - makespan (steps pour tout finir), throughput = tokens totaux / makespan
   - latence par requete = step de fin - step d'arrivee ; moyenne et p95
   - utilisation = tokens utiles produits / (steps_GPU_actifs * 8 slots)

5. **Verifications** :
   - les deux schedulers produisent EXACTEMENT le meme nombre total de tokens (somme des longueurs)
   - aucune requete ne demarre avant son arrivee, aucune sequence ne depasse sa longueur tiree (asserts dans la boucle)
   - throughput continu / throughput statique >= 1.3 sur ce scenario, et utilisation continue > utilisation statique
   - p95 de latence : le continu doit faire mieux (pas de "convoy effect" derriere une sequence de 120 tokens)

6. Afficher une mini-timeline ASCII (slots en lignes, temps en colonnes, un caractere par requete) sur les 60 premiers steps pour visualiser les trous du statique.

### Criteres de reussite

- [ ] La conservation des tokens (meme total pour les deux schedulers) est verifiee par assert
- [ ] Les contraintes de validite (arrivee, longueur, capacite <= 8) sont testees dans la boucle
- [ ] Throughput continu >= 1.3x statique ET utilisation continue superieure, valeurs affichees
- [ ] Latences moyenne et p95 calculees ; le convoy effect du statique est visible et commente
- [ ] La timeline ASCII est lisible et montre les slots morts du batching statique
- [ ] Execution < 10 s
