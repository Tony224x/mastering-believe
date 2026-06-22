# Solutions — Module 04 : Nutrition fondee sur les preuves

> **Note pedagogique :** Ces solutions sont des modeles, pas des reponses uniques. La nutrition est un domaine ou les situations individuelles varient. L'objectif est de verifier que votre raisonnement est ancre dans les invariants et dans la methode critique du module — pas de correspondre mot pour mot.

---

## Solution — Exercice 1 : Planifier une journee de repas "invariants-first"

**Exemple de journee (modele mediterraneen-inspire, mais pas obligatoire) :**

| Repas | Menu exemple | Invariants respectes |
|-------|-------------|----------------------|
| Petit-dejeuner | Flocons d'avoine (cereales completes) + fruits frais + poignee de noix + cafe sans sucre | Fibres (avoine, fruits), proteines (noix), peu ou pas de sucres ajoutes |
| Dejeuner | Salade de lentilles (legumineuses), legumes de saison rotes, filet de sardine ou oeuf dur, huile d'olive | Fibres (lentilles, legumes), proteines completes, peu transforme, graisses saines |
| Collation (optionnelle) | Pomme + quelques amandes | Fibres, proteines vegetales, peu transforme |
| Diner | Riz complet + poele de legumes + tofu ou poulet + herbes | Cereales completes, proteines, fibres, peu transforme |

**Exemple de substitution d'ultra-transforme :**
- Ultra-transforme habituel : biscuits industriels (collation)
- Alternative : fruit frais + oleagineux (meme satiation, sans additifs, avec fibres)

**Ce qui n'est pas prescrit ici :** les quantites, les horaires, le nombre de repas. Ces variables dependent de chaque individu.

---

## Solution — Exercice 2 : Decrypter un titre nutritionnel

**Titre A** — "Manger du chocolat noir reduit le risque cardiaque de 22 % (etude sur 100 000 personnes)"

- Type d'etude probable : **observationnelle** (cohorte). 100 000 personnes, pas de randomisation mentionnee.
- Relation causale ? **Non**. Les amateurs de chocolat noir pourraient etre en meilleure situation socioeconomique (acces a des produits de qualite), avoir d'autres habitudes saines.
- Facteur de confusion : niveau de revenu, education alimentaire, alimentation generale. Les consommateurs de chocolat noir de qualite ne ressemblent probablement pas a ceux qui consomment surtout du chocolat au lait industriel.

**Titre B** — "Un ECR confirme que les omega-3 abaissent les triglycerides"

- Type d'etude : **ECR** (mentionne explicitement).
- Relation causale ? **Oui, plus legitime.** Un ECR isole l'effet de l'intervention (omega-3) du reste. C'est le niveau de preuve le plus robuste pour etablir la causalite.
- Nuance : verifier la taille d'effet, la population cible, le produit teste, la duree. "Abaissent les triglycerides" ne dit rien sur la reduction de la mortalite cardiovasculaire.

**Titre C** — "Sauter le petit-dejeuner augmente le risque de diabete de 30 %"

- Type d'etude probable : **observationnelle**. Le chiffre et le design suggèrent une cohorte.
- Relation causale ? **Non** directement. Les personnes qui sautent le petit-dejeuner pourraient etre en situation d'insecurite alimentaire, avoir un sommeil irregulier, travailler en horaires decales — autant de facteurs qui degradent independamment la sante metabolique.
- Facteur de confusion principal : style de vie global. Sauter le petit-dejeuner est souvent un signal d'un mode de vie plus chaotique, pas la cause directe.

---

## Solution — Exercice 3 : Evaluer ses croyances sur les regimes

**Exemple de traitement d'une croyance :**

*Croyance* : "Les graisses saturees sont mauvaises et on devrait les eviter completement."

*Invariant sous-jacent* : La recherche sur les graisses saturees a evolue. La revue Cochrane (Hooper et al., 2020) montre une reduction modeste du risque cardiovasculaire en remplacant les graisses saturees par des graisses insaturees. Mais les acides gras trans industriels (partiellement hydrogenes) ont un effet nettement plus negatif que les graisses saturees naturelles (beurre, fromage). L'ultra-transformation de l'aliment compte autant que le type de graisse.

*Reformulation* : "Reduire les graisses trans industrielles est soutenu par la preuve. Remplacer les graisses saturees par des graisses insaturees (huile d'olive, oleagineux) peut avoir un benefice modere selon le contexte. Eliminer completement les graisses saturees naturelles (beurre, fromage) n'est pas justifie par la preuve actuelle."

**Ce que cela illustre :** la croyance originale contenait un noyau valide (reduire les graisses trans) mais le generalisait de facon excessive (toutes les graisses saturees). L'invariant permet de garder le noyau utile et d'abandonner le generalisme.
