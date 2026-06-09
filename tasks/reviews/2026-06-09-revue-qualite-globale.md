# Revue qualité globale — 9 juin 2026

Revue menée par 7 agents en parallèle : 1 par domaine (robotics-ai scindé en théorie+code / exercices vu sa taille) + 1 revue transverse repo. Chaque agent a vérifié : conformité aux conventions CLAUDE.md, exactitude technique (lecture du code et des solutions), cohérence théorie ↔ code ↔ exercices, qualité pédagogique. En complément, les **193 fichiers Python du repo compilent tous sans erreur** (`py_compile`).

## Verdicts par périmètre

| Périmètre | Note | Erreurs techniques | Principal écart |
|---|---|---|---|
| algorithmie-python | **8,5/10** | Aucune (toutes les solutions exécutées avec succès) | Medium/hard absents pour les sujets 04-14 |
| agentic-ai | **8,2/10** | Aucune (API LangGraph/MCP correctes) | Medium/hard absents pour les sujets 04-14 ; `03-memory-state.qd` placeholder |
| neural-networks-llm | **7,8/10** | Aucune erreur math/PyTorch | 9 modules sans flash-cards (J7, J15-J22) ; capstone code ≠ exercice |
| system-design | **7,5/10** | 1 nuance (MongoDB CAP configurable) | Medium/hard absents pour les modules 04-14 |
| robotics-ai (théorie+code) | **7,5/10** | Aucune erreur scientifique ; 1 bug moyen (FiLM `cond_dim` impair) | `04-projects/` et `05-projets-guides/` inexistants ; J11 PPO = squelette |
| robotics-ai (exercices) | **7,2/10** | Aucun bug math sur 25+ solutions | En-têtes non normalisés ; 9 exos aux critères non testables |
| Transverse (repo) | **6,5/10** | — | CLAUDE.md/README désynchronisés de la réalité |

**Note globale estimée : ~7,6/10.** Fond technique très sain (zéro erreur scientifique ou bug critique sur l'ensemble du repo), mais conformité structurelle inégale.

## Constats systémiques (présents dans plusieurs domaines)

### 1. Exercices medium/hard limités aux 3 premiers sujets — pattern sur 4 domaines (HAUTE)
Dans algorithmie-python, system-design, agentic-ai et neural-networks-llm : seuls les sujets 01-03 ont des exercices medium (3 fichiers) et hard (3 fichiers). Les sujets 04-14 n'ont qu'un exercice easy chacun. La « progressive overload » promise par CLAUDE.md s'arrête donc au jour 3. Robotics-ai est l'exception vertueuse : 28 easy + 28 medium + 28 hard avec correspondance 1:1.

### 2. Documentation repo désynchronisée (HAUTE, correctif rapide)
- `CLAUDE.md` affirme « chaque domaine a un dossier `05-projets-guides/` avec 3 projets » → faux pour robotics-ai (dossier inexistant, mais référencé par `domains/robotics-ai/README.md` ligne ~80 → lien cassé).
- Robotics-ai absent du tableau « Domaines disponibles » du `README.md` racine.
- Le dossier `tasks/` documenté dans CLAUDE.md n'existait pas (recréé par cette revue).
- `shared/templates/` ne couvre que `03-exercises/workspace/` (~30 % de la structure d'un domaine) alors que CLAUDE.md dit « Copy shared/templates/ structure ».

### 3. Quarkdown (`01-theory-qd/`) incomplet là où il existe (MOYENNE)
- agentic-ai : `03-memory-state.qd` est un placeholder de 35 lignes (vs 500+ dans le .md).
- neural-networks-llm : `11-inference-optimisee.qd` manquant (21/22 fichiers).
- Les README de domaines ne mentionnent jamais l'existence du site Quarkdown.

### 4. Conventions de formatage hétérogènes (BASSE)
- robotics-ai exercices : `## Criteres de reussite` (19 fichiers) vs `## Critères de réussite` (9 fichiers) ; un `## Consigne détaillée` isolé.
- system-design : `## Flash Cards` (3 fichiers) vs `## Flash cards` (11 fichiers).
- robotics-ai théorie : format des Q&A flash-cards hétérogène (parsing automatisé impossible).

### 5. `04-projects/` vide partout (BASSE-MOYENNE)
4 domaines ont un `.gitkeep` seul, robotics-ai n'a pas le dossier. Aucun README n'explique que c'est un espace de projets libres.

## Constats par domaine

### algorithmie-python — 8,5/10
Le plus solide. 14/14 théories avec 5 flash-cards, 14/14 codes et solutions **exécutés avec succès**, Big-O vérifiés exacts, 3 projets guidés excellents (A*, Bresenham, event queue) tous fonctionnels. À faire : medium/hard pour sujets 04-14.

### agentic-ai — 8,2/10
Zéro erreur technique : `StateGraph`, `Command(goto=...)`, reducers, MCP JSON-RPC tous corrects. Projet phare `02-supervisor-swarm-multi-tier` jugé production-grade. À faire : medium/hard 04-14, compléter `03-memory-state.qd`, ajouter des tests au projet phare.

### neural-networks-llm — 7,8/10
Maths précises (backprop, attention, RoPE), shapes PyTorch tracées, aucune erreur détectée. À faire : **9 modules sans flash-cards** (`01-theory/07, 15-22`) — violation directe de CLAUDE.md ; aligner `02-code/14-capstone.py` (mini-GPT char-level) avec `03-exercises/01-easy/14-capstone.md` (parameter counting mini-LLaMA) ; créer `11-inference-optimisee.qd`.

### system-design — 7,5/10
Théorie à jour (Valkey/Dragonfly post-licence Redis, MLA, etc.), 14/14 codes compilent, 14/14 flash-cards. Une seule imprécision : MongoDB classé CP sans mentionner que c'est configurable (`01-theory/01-principes-fondamentaux.md:83`). À faire : medium/hard 04-14, remplir `04-projects/`.

### robotics-ai — 7,4/10 (moyenne des deux revues)
Zéro erreur scientifique sur SE(3), jacobiennes, DDPM/DDIM, RSSM, VLA. Correspondance théorie↔code 1:1 sur 28 jours, capstone Diffusion Policy (J24-J28) cohérent et exécutable CPU. À faire :
- Créer `05-projets-guides/` (3 projets LogiSim) + `04-projects/` — ou corriger la doc et le lien cassé du README.
- Étoffer `02-code/11-policy-gradients-ppo.py` (actuellement squelette ~50 lignes renvoyant à CleanRL, rupture dans la progression RL J9-J12).
- Normaliser les en-têtes d'exercices (accents) et rendre testables les critères de 9 exos (J01, J15-17, J19, J21, J23-24, J27).
- Bug moyen : FiLM dans `02-code/16-diffusion-policy.py:118-120` suppose `cond_dim` pair sans assert.
- Documenter les dépendances optionnelles par plage de jours (mujoco, torch).

## Plan d'action priorisé

| # | Action | Effort | Impact |
|---|--------|--------|--------|
| 1 | Corriger CLAUDE.md + README racine (robotics-ai au tableau, retirer l'affirmation « chaque domaine a 05-projets-guides », corriger le lien cassé du README robotics-ai) | ~15 min | Élimine les incohérences doc↔réalité |
| 2 | Ajouter les flash-cards manquantes aux 9 modules neural-networks-llm | ~2 h | Restaure la conformité spaced-repetition |
| 3 | Normaliser les en-têtes (`Criteres de reussite`, `Flash Cards`, format Q&A) via find/replace | ~30 min | Cohérence + tooling possible |
| 4 | Compléter les 2 fichiers Quarkdown manquants/placeholder (agentic-ai 03, nn-llm 11) | ~1 h | Sites QD complets |
| 5 | Créer les exercices medium/hard pour les sujets 04-14 des 4 domaines concernés (~22 fichiers/domaine) | plusieurs jours | Restaure la progression pédagogique complète — chantier principal |
| 6 | Robotics-ai : 05-projets-guides (3 projets LogiSim) + étoffer J11 PPO | 1-2 jours | Aligne le 5e domaine sur le standard |
| 7 | Compléter `shared/templates/` (01-theory, 02-code, 04-projects, 05-projets-guides) | ~30 min | Onboarding contributeurs |
| 8 | Script CI `test_all_solutions` (exécution de toutes les solutions, matplotlib en mode headless `Agg`) | ~3 h | Détection de régressions |
