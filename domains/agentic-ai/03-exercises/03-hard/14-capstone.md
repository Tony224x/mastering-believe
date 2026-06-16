# Exercices Hard — Capstone (J14)

> Ces exercices DURCISSENT et ETENDENT le capstone `AcmeResearcher` au niveau
> production. Les solutions embarquent un mini-AcmeResearcher autonome (offline).

---

## Exercice 1 : Self-critique loop avec reflexion bornee et garde-fous de cout

### Objectif
Ajouter une capacite majeure au capstone : une boucle writer → critic → reviser, bornee par un budget de reflexion ET un budget de cout, qui ameliore le rapport sans jamais boucler indefiniment ni exploser la facture (combine Reflexion + budget enforcement).

### Consigne
Etends `AcmeResearcher` avec une boucle de self-critique :

1. **Critic agent** : apres le writer, un `CriticAgent` (mock LLM trace) evalue le draft et retourne `{issues: list[str], severity: float, revised: str | None}`
   - Issues heuristiques : rapport trop court, pas de chiffre, pas de citation, conclusion absente
2. **Boucle bornee** : tant qu'il reste des issues bloquantes ET `revision_count < max_revisions` (ex: 2) ET le budget n'est pas epuise → on revise et on re-critique
3. **Double garde-fou** :
   - `max_revisions` : borne dure sur le nombre d'iterations (anti-boucle infinie)
   - `budget` : chaque revision coute des tokens ; si le budget va etre depasse, on s'arrete avec le meilleur draft obtenu (pas d'exception non geree, degradation gracieuse)
4. **Convergence trackee** : la severite des issues doit DECROITRE a chaque revision (sinon on stoppe — le critic ne fait pas mieux). Le `state` expose `revision_history: list[dict]`
5. Teste 3 scenarios : (a) draft deja bon (0 revision), (b) draft mauvais qui converge en 2 revisions, (c) budget serre qui force un arret anticipe avec le meilleur draft disponible

### Criteres de reussite
- [ ] La boucle writer→critic→reviser est implementee et tracee (spans)
- [ ] `max_revisions` borne le nombre d'iterations (anti-boucle infinie)
- [ ] Le budget force un arret gracieux (jamais de crash, retourne le meilleur draft)
- [ ] La severite des issues decroit a chaque revision (convergence verifiee)
- [ ] `revision_history` documente chaque iteration
- [ ] Les 3 scenarios donnent le bon nombre de revisions et un draft final coherent

---

## Exercice 2 : Harnais d'eval avec adversarial cases, ablations et matrice de robustesse

### Objectif
Construire l'eval de production complete du capstone : pas seulement des cas heureux, mais des cas adverses, des ablations de composants (retire une defense → l'attaque doit reussir), et une matrice qui prouve que CHAQUE garde-fou est necessaire.

### Consigne
Construis un `CapstoneEvalSuite` qui pousse l'eval du capstone bien au-dela des 3 cas du code :

1. **Cas standard** (>= 3) : requetes factuelles ou le verdict doit etre `ok` et les keywords presents
2. **Cas adverses** (>= 4) : injection directe, injection indirecte via un finding empoisonne, requete hors-domaine (doit admettre l'ignorance, pas halluciner), requete geante (DoS)
3. **Ablation testing** : pour chaque garde-fou (input_guard, output_guard, budget, hitl), construis une variante du `AcmeResearcher` ou CE garde-fou est DESACTIVE, puis verifie que l'attaque correspondante REUSSIT (prouve que le garde-fou etait necessaire). Avec le garde-fou actif, l'attaque echoue
4. **Matrice de robustesse** : un tableau `{garde-fou x attaque}` indiquant si l'attaque est bloquee. La diagonale (garde-fou actif vs son attaque) doit etre "bloquee", la version ablatee "passee"
5. Produit un rapport : `{standard_pass_rate, adversarial_block_rate, ablation_matrix, all_guards_necessary}`

### Criteres de reussite
- [ ] >= 3 cas standard et >= 4 cas adverses
- [ ] L'injection indirecte via finding empoisonne est testee (pas seulement l'injection directe d'input)
- [ ] Le cas hors-domaine verifie l'absence d'hallucination
- [ ] L'ablation de chaque garde-fou prouve sa necessite (attaque reussit sans lui)
- [ ] La matrice de robustesse montre la diagonale active=bloquee / ablatee=passee
- [ ] `adversarial_block_rate` == 100% avec tous les garde-fous actifs
- [ ] Le rapport conclut `all_guards_necessary == True`
