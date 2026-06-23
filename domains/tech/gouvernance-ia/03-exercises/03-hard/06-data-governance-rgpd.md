# Exercice (hard) — Décideur de DPIA + effacement propageable

## Objectif

Compléter l'assessor avec les deux mécanismes de gouvernance les plus spécifiques au paradigme agentique : (1) un **décideur de DPIA / AIPD** (Art. 35 + heuristique CNIL « deux critères ») qui rend un verdict motivé, et (2) un **effacement propageable** (Art. 17) qui prouve qu'un agent peut réellement « oublier » une personne dans *tous* ses stores (mémoire + logs), pas seulement dans une base principale.

## Consigne

1. Repars de ton code medium.
2. Écris `decide_dpia(profile) -> (verdict, reasons)` où `verdict ∈ {"REQUIRED", "RECOMMENDED", "NOT_REQUIRED"}` :
   - **Triggers durs (Art. 35(3))** : grande échelle + données sensibles → REQUIRED ; décision automatisée + profilage → REQUIRED (un seul suffit).
   - **Heuristique CNIL** : compter les critères parmi `{sensitive, automated_decision, large_scale, profiling, vulnerable_subjects, innovative_use}`. ≥ 2 → REQUIRED ; exactement 1 → RECOMMENDED ; 0 → NOT_REQUIRED.
   - Le verdict doit s'accompagner d'une liste `reasons` lisible.
3. Implémente une classe `AgentStores` qui simule trois stockages : `main_db`, `agent_memory`, `audit_logs` (chacun une liste de dicts avec une clé `subject`). Écris `forget(stores, subject) -> dict` qui supprime *toutes* les entrées de ce sujet dans les trois stores et renvoie un décompte par store.
4. Ajoute une **vérification adversariale** : après `forget`, une fonction `assert_erased(stores, subject)` qui lève (ou retourne False) si une trace du sujet subsiste **où que ce soit**.
5. Smoke test : un agent « innovant + grande échelle » → DPIA REQUIRED ; un effacement qui laisse une entrée résiduelle dans `audit_logs` doit être détecté par `assert_erased`.

## Criteres de reussite

- [ ] Le script tourne avec `python <fichier>` sans erreur (stdlib seule).
- [ ] Un trigger dur (grande échelle + sensible, OU automated_decision + profiling) renvoie `REQUIRED` même si un seul est présent.
- [ ] L'heuristique CNIL distingue correctement REQUIRED (≥2), RECOMMENDED (=1), NOT_REQUIRED (0).
- [ ] `forget` retourne un décompte non nul pour un sujet présent dans les trois stores, et zéro pour un sujet absent.
- [ ] `assert_erased` détecte une trace résiduelle laissée volontairement dans un seul store (probe adversariale) et confirme l'effacement complet sinon.
