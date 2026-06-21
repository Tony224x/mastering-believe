# Exercice difficile — EU AI Act (J5)

## Exercice : Mini-registre de gouvernance des tiers avec scoring de due diligence

### Objectif
Assembler les briques du jour en un **outil de gouvernance des tiers** : un registre de composants IA achetes qui, pour chaque ligne, classe l'usage, evalue la due diligence fournisseur, calcule un score de risque de conformite, et signale la deadline la plus proche au niveau du portefeuille.

### Consigne
1. Reutilise `Tier`, `classify`, `due_diligence`, `SupplierComponent`, `DEADLINES` de `02-code/05-eu-ai-act.py`.
2. Definis un `dataclass RegisteredComponent` qui combine un `SystemProfile` (l'usage) et un `SupplierComponent` (le fournisseur), plus un `component_id` et un `owner` (rappel des 4 piliers : un owner nomme).
3. Ecris `assess(component)` qui retourne un `dict` avec :
   - la classification de l'usage (tier, deadline) via `classify()` ;
   - le resultat de la due diligence (`passes`, liste de `gaps`) via `due_diligence()` ;
   - un `compliance_risk_score` entier que tu definis comme : `poids_du_tier * (1 + nb_gaps)`, ou `poids_du_tier` vaut 4/3/2/1 selon `Tier.value` (un haut risque avec des trous score haut).
4. Ecris `portfolio_report(components, today)` qui :
   - assess chaque composant ;
   - trie par `compliance_risk_score` decroissant ;
   - **refuse** (statut `BLOCKED`) tout composant dont l'usage est `UNACCEPTABLE`, quel que soit le score ;
   - calcule la **deadline la plus proche** parmi tous les composants encore deployables (ignorer `"n/a"`) ;
   - retourne un rapport (dict ou texte) board-ready.
5. Construis un portefeuille d'au moins 4 composants couvrant : un haut risque conforme, un haut risque avec des gaps, un risque limite, et un usage inacceptable. Affiche le rapport.
6. **Probe adversariale** : ajoute un composant dont la due diligence echoue ET l'usage est haut risque, et verifie qu'il remonte en tete du tri (score le plus eleve) sans planter.

### Criteres de reussite
- [ ] `assess` renvoie tier, deadline, `passes`, `gaps` et un `compliance_risk_score` coherent (haut risque + gaps = score le plus eleve)
- [ ] Le composant a usage inacceptable est marque `BLOCKED` et n'entre pas dans le calcul de la deadline la plus proche
- [ ] La deadline la plus proche du portefeuille est correcte (ex. `2026-08-02` si un haut risque Annexe III est present) et ignore les `"n/a"`
- [ ] Le tri par risque place le haut risque non conforme en premier
- [ ] La probe adversariale tourne sans exception et le composant fautif est bien en tete
- [ ] Tout en stdlib (`dataclasses`, `datetime`, `enum`), aucune dependance externe
