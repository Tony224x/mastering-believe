# Projet guide 02 — Faut-il s'endetter pour la voiture ?

> **Disclaimer.** Projet **purement éducatif**. Rien ici n'est un conseil
> financier ou en crédit personnalisé. Les taux sont illustratifs ; tout crédit
> engage l'emprunteur. Vérifiez le TAEG réel, l'assurance et les frais avant de
> signer quoi que ce soit. Contexte métier partagé :
> [`shared/logistics-context.md`](../../../logistics-context.md).

**Niveau** : moyen · **Modules mobilisés** : 03 (dette et crédit), 01 (coût d'opportunité).

## 1. Contexte metier

Sofia est **technicienne terrain** pour un intégrateur FleetSim : elle se
déplace d'un entrepôt client à l'autre pour installer et calibrer les robots.
Elle a besoin d'un véhicule fiable (8 000 €). Deux questions la taraudent :

- Le concessionnaire lui propose **deux offres de crédit**. Laquelle, et
  pourquoi celle à la « petite mensualité » n'est pas forcément la moins chère ?
- Elle a **15 000 € d'épargne** placés. Faut-il payer comptant, ou financer et
  garder l'argent investi ?

Et en toile de fond, elle traîne déjà **deux ou trois petites dettes** : dans
quel ordre les rembourser pour s'en sortir au mieux ?

## 2. Objectif technique

Construire un **analyseur de crédit** qui répond chiffres en main :
1. mensualité, coût total et **coût du crédit** (intérêts) d'une offre ;
2. **tableau d'amortissement** (part intérêts / part capital mois par mois) ;
3. comparaison **comptant vs crédit + placement de la différence** ;
4. simulation **avalanche vs boule de neige** sur plusieurs dettes à budget fixe.

## 3. Consigne

```python
def mensualite(capital: float, taux_annuel: float, mois: int) -> float:
    """M = P·r / (1 − (1+r)^−n), avec r = taux mensuel. Gère r == 0."""

def rembourser(dettes: list[Dette], budget_total: float, strategie: str) -> dict:
    """Simule le remboursement multi-dettes. strategie ∈ {'avalanche','boule'}.
    Retourne {'mois': float, 'total_interets': float}."""
```

Contraintes :
- **stdlib uniquement**, déterministe.
- `mensualite` doit gérer le **taux 0 %** (sinon division par zéro).
- `rembourser` doit avoir un **garde-fou anti-boucle-infinie** (si le budget ne
  couvre même pas les intérêts, la dette ne se rembourse jamais).
- Chaque sortie doit être **interprétée** (un nombre nu n'aide pas à décider).

## 4. Etapes guidees

1. **Mensualité** — implémente la formule fermée du prêt amortissable. Teste-la sur un cas à la main (ex. 1 200 € à 0 % sur 12 mois → 100 €/mois).
2. **Coût du crédit** — `mensualité × mois − capital`. C'est le vrai prix du crédit, indépendant de la mensualité affichée.
3. **Amortissement** — boucle : `intérêt = solde × r` ; `capital_remboursé = mensualité − intérêt` ; `solde −= capital_remboursé`. Observe que la 1re mensualité est surtout des intérêts.
4. **Comptant vs crédit** — à *effort égal* : la mensualité sort du salaire dans les deux cas. La seule différence est : payer comptant (et placer la mensualité économisée) **ou** garder l'épargne placée. Le verdict dépend du signe de `(rendement_placement − taux_crédit)`.
5. **Avalanche vs boule** — chaque mois : applique les intérêts, paie les minimums, dirige le surplus vers la dette cible (taux max = avalanche ; plus petit solde = boule de neige). Compte les mois et le total d'intérêts.

## 5. Criteres de reussite

- [ ] `python solution/credit_analyzer.py` tourne sans dépendance externe
- [ ] `mensualite(1200, 0.0, 12) == 100.0` (cas taux zéro géré)
- [ ] L'offre à **mensualité plus faible mais durée plus longue** apparaît **plus chère au total** (le piège est démontré, pas juste affirmé)
- [ ] Le verdict comptant/crédit **bascule** correctement selon le rendement : comptant gagne si le placement rapporte moins que le taux du crédit, crédit gagne sinon
- [ ] L'avalanche paie **moins d'intérêts** que la boule de neige sur le même jeu de dettes
- [ ] Budget insuffisant → le garde-fou stoppe (pas de boucle infinie)

> **Piège à comprendre** : une « petite mensualité » rassure le budget mensuel
> mais peut coûter **bien plus cher au total**. Dans la solution, l'offre B
> (4,5 % / 72 mois) a une mensualité presque deux fois plus faible que l'offre A
> (4,9 % / 36 mois)… pour un coût du crédit **~2× supérieur** (≈ 1 143 € contre
> ≈ 619 €), *malgré un taux affiché plus bas*. La durée mange tout.

## 6. Corrige

Voir [`solution/credit_analyzer.py`](./solution/credit_analyzer.py) (commenté)
et [`solution/analyse.md`](./solution/analyse.md) pour la lecture des résultats
et les nuances honnêtes (risque, fiscalité, comportement réel).

## 7. Pour aller plus loin

- **Remboursement anticipé** — ajoute un versement exceptionnel au mois 6 et mesure les intérêts économisés (gros sur un crédit jeune).
- **Assurance emprunteur** — intègre une prime mensuelle ; elle change souvent le classement des offres.
- **TAEG vs taux nominal** — modélise des frais de dossier et calcule le vrai TAEG.
- **Boule de neige avec relance** — ajoute le minimum d'une dette soldée au surplus suivant (effet « boule » accéléré) et compare à l'avalanche.
