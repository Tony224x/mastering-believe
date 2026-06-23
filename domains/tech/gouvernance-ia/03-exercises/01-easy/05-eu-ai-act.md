# Exercice facile — EU AI Act (J5)

## Exercice : Classer cinq usages dans le bon tier

### Objectif
Savoir associer un usage IA a l'un des 4 tiers de l'EU AI Act et nommer la deadline applicable, en reutilisant le classifieur de `02-code/05-eu-ai-act.py`.

### Consigne
1. Importe (ou recopie) `SystemProfile`, `classify` et `DEADLINES` depuis `02-code/05-eu-ai-act.py`.
2. Cree cinq `SystemProfile` correspondant a ces usages :
   - a) Un filtre anti-spam interne.
   - b) Un assistant conversationnel qui se presente comme une IA.
   - c) Un systeme de tri automatique de CV pour le recrutement (Annexe III).
   - d) Un dispositif de notation sociale generalisee opere par une autorite publique.
   - e) Une IA composant de securite d'un dispositif medical (Annexe I).
3. Pour chacun, appelle `classify()` et affiche : le nom, le tier obtenu, la deadline.
4. Verifie a la main que chaque tier est celui attendu (minimal / limite / haut / inacceptable / haut).

### Criteres de reussite
- [ ] Les 5 systemes sont classes sans erreur d'execution
- [ ] (a) ressort `MINIMAL`, (b) `LIMITED`, (c) `HIGH` (Annexe III), (d) `UNACCEPTABLE`, (e) `HIGH` (Annexe I)
- [ ] La deadline affichee pour (c) est `2026-08-02` et pour (e) `2027-08-02`
- [ ] Tu peux expliquer en une phrase pourquoi (c) et (e) sont tous deux « haut risque » mais ont des deadlines differentes
