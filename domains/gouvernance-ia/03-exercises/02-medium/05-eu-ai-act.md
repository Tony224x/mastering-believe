# Exercice moyen — EU AI Act (J5)

## Exercice : Du tier aux obligations + jours restants avant deadline

### Objectif
Aller au-dela de la classification : produire, pour un systeme, la **liste de ses obligations** et le **nombre de jours restants** avant que sa deadline ne morde — une sortie directement exploitable dans un point de gouvernance.

### Consigne
1. Reutilise `classify()` et `Classification` de `02-code/05-eu-ai-act.py`.
2. Ecris une fonction `days_until_deadline(classification, today)` qui :
   - parse la deadline (`"YYYY-MM-DD"`) avec `datetime.date.fromisoformat` (stdlib uniquement) ;
   - retourne le nombre de jours entre `today` et la deadline ;
   - gere proprement le cas `deadline == "n/a"` (tier minimal) en renvoyant `None`.
3. Ecris une fonction `compliance_brief(profile, today)` qui appelle `classify()` puis renvoie un `dict` avec : `name`, `tier` (le nom du tier), `deadline`, `days_left`, et `obligations`.
4. Genere le brief pour au moins trois systemes de tiers differents (dont un haut risque Annexe III et un haut risque Annexe I) avec `today = date(2026, 6, 21)`.
5. Affiche les briefs tries du plus urgent (moins de jours restants) au moins urgent ; les systemes sans deadline (`None`) en dernier.

### Criteres de reussite
- [ ] `days_until_deadline` renvoie un entier pour les tiers a deadline et `None` pour le tier minimal
- [ ] Pour un haut risque Annexe III avec `today = 2026-06-21`, `days_left` est positif et coherent avec le `2026-08-02` (de l'ordre de ~42 jours)
- [ ] Le haut risque Annexe I affiche une deadline `2027-08-02` (donc plus de jours restants que l'Annexe III)
- [ ] Le tri place le systeme le plus urgent en premier et les `None` a la fin
- [ ] Aucune dependance externe (stdlib `datetime` seulement)
