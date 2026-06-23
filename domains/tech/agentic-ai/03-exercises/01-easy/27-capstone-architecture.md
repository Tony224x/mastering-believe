# Exercices — Capstone architecture (J27)

---

## Exercice 1 : Compaction du scratchpad VirtualFS

### Objectif

Appliquer le context engineering (J15) aux briques du capstone : eviter qu'un scratchpad grossisse sans limite.

### Consigne

En partant de `02-code/27-capstone-architecture.py` :

1. Cree une sous-classe `CompactingVirtualFS(VirtualFS)`.
2. Ajoute `append(name, line)` qui ajoute une ligne a un fichier (le cree si absent).
3. Ajoute un seuil `max_lines` (defaut 20) : quand un fichier depasse `max_lines` lignes, `append` doit **compacter** en gardant les 5 premieres lignes + une ligne `... [compacted N lines] ...` + les 5 dernieres.
4. Teste : append 50 lignes dans `log.md`, verifie que le fichier final a au plus `max_lines` lignes, contient le marqueur de compaction, et conserve la premiere et la derniere ligne.

### Criteres de reussite

- [ ] `append` cree puis etend un fichier
- [ ] La compaction se declenche au-dela de `max_lines` (juste apres : 11 lignes)
- [ ] Le marqueur `[compacted N lines]` est present
- [ ] Le fichier compacte conserve debut (`event 0`) et fin (`event 49`)
- [ ] Le fichier final fait au plus `max_lines` lignes ; le test tourne sans erreur

---

## Exercice 2 : Journal d'audit du DurableEngine

### Objectif

Rendre le DurableEngine observable : tracer chaque decision execute/skip.

### Consigne

1. Cree `AuditedDurableEngine(DurableEngine)` qui surcharge `run`.
2. A chaque etape, enregistre un evenement dans une liste `self.audit` : `{"step": name, "action": "executed"|"skipped", "ts": <compteur logique>}`.
3. Reutilise la logique de reprise du parent (tu peux re-implementer la boucle en t'inspirant du parent, ou appeler des helpers).
4. Teste le scenario crash/reprise du module (crash avant `report`, puis reprise) et verifie que l'audit montre 2 `skipped` puis 1 `executed` au second run.

### Criteres de reussite

- [ ] `self.audit` contient un evenement par etape traitee
- [ ] Les actions `executed`/`skipped` sont correctes au premier et au second run
- [ ] Le timestamp logique est croissant
- [ ] Le scenario crash/reprise est reproduit
- [ ] Le code tourne sans erreur

---

## Exercice 3 : Routeur a 3 niveaux avec budget

### Objectif

Etendre le `ModelRouter` (J24) avec un palier intermediaire et un plafond de budget.

### Consigne

1. Cree `BudgetedRouter` avec 3 tiers : `nano` (0.2), `weak` (1.0), `strong` (8.0).
2. Routage par longueur : `< 6` mots → nano ; `< 12` → weak ; sinon → strong.
3. Ajoute un `budget: float` : si router un appel ferait depasser le budget, **degrade** d'un cran (strong→weak→nano) jusqu'a rentrer dans le budget ; si meme nano ne rentre pas, leve `BudgetExceeded`.
4. Teste : un budget serre force des degradations observables, et un budget nul sur une requete complexe leve l'exception.

### Criteres de reussite

- [ ] Les 3 tiers sont routes correctement selon la longueur
- [ ] La degradation se declenche quand le budget est insuffisant
- [ ] `BudgetExceeded` est levee quand meme nano ne rentre pas
- [ ] `total_cost` reste <= budget
- [ ] Le test montre au moins une degradation et une exception
