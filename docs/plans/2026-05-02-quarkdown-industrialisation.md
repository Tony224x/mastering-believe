# Quarkdown Industrialisation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restructurer la production de cours Quarkdown du repo `mastering-believe` pour scaler à 4+ domaines × 10-20 chapitres, sans casser l'invariant `.md` source-of-truth + GitHub-readable.

**Architecture:** Migration des `.qd` vers un dossier `01-theory-qd/` sibling de `01-theory/` dans chaque domaine, avec un `main.qd` par domaine (1 site auto-suffisant par domaine). Scaffolder Python pour générer la structure, build wrapper PowerShell pour itérer sur tous les domaines. Skill `quarkdown-course-author` repositionné en "enrich" (le scaffolder fait le "convert").

**Tech Stack:** Quarkdown v2.0 (Kotlin/JVM), Python 3.x (scaffolder + post-process), PowerShell 5.1 (build wrapper).

**Spec source:** `docs/superpowers/specs/2026-05-02-quarkdown-industrialisation-design.md`

---

## Tâches

### Tâche 1 — Migration des `.qd` existants vers `01-theory-qd/`

**Fichiers :**
- Move : `domains/agentic-ai/01-theory/01-anatomie-agent.qd` → `domains/agentic-ai/01-theory-qd/01-anatomie-agent.qd`
- Move : `domains/agentic-ai/01-theory/02-tool-use-function-calling.qd` → `domains/agentic-ai/01-theory-qd/02-tool-use-function-calling.qd`
- Move : `quarkdown/site/03-memory-state.qd` → `domains/agentic-ai/01-theory-qd/03-memory-state.qd`
- Create : `domains/agentic-ai/01-theory-qd/main.qd` (rapatrié depuis `quarkdown/site/main.qd`, chemins corrigés)
- Edit : sidebar nav dans les 3 `.qd` (chemins relatifs locaux)
- Delete : `quarkdown/site/main.qd` (plus de besoin), puis le dossier `quarkdown/site/`

**Étape 1.1** — `git mv` les 3 `.qd` vers `01-theory-qd/`

**Étape 1.2** — Créer `domains/agentic-ai/01-theory-qd/main.qd` avec :
- Préambule identique à l'ancien `quarkdown/site/main.qd`
- Sidebar nav avec chemins locaux : `01-anatomie-agent.qd`, `02-tool-use-function-calling.qd`, `03-memory-state.qd`
- Liens dans le corps du document recalibrés en chemins locaux

**Étape 1.3** — Update sidebar nav dans `01-anatomie-agent.qd`, `02-tool-use-function-calling.qd`, `03-memory-state.qd` :
- `[Accueil](main.qd)` au lieu de `[Accueil](../../../quarkdown/site/main.qd)`
- Chemins frères : `01-anatomie-agent.qd` au lieu de `../../../domains/agentic-ai/01-theory/01-anatomie-agent.qd`

**Étape 1.4** — Supprimer `quarkdown/site/`

**Étape 1.5** — Test build :
```powershell
$env:JAVA_HOME = "C:\Program Files\Eclipse Adoptium\jdk-21.0.11.10-hotspot"
$env:Path = "$env:JAVA_HOME\bin;C:\Users\antho\.tools\quarkdown\bin;$env:Path"
quarkdown c domains/agentic-ai/01-theory-qd/main.qd --out quarkdown/output-site/agentic-ai
python quarkdown/post-build-fix-links.py quarkdown/output-site/agentic-ai
```
Expected: `[Success]` sans warning `Unresolved reference`, sans besoin de `--allow global-read` (chemins locaux).

**Étape 1.6** — Commit : `chore(quarkdown): migrer .qd existants vers 01-theory-qd/ par domaine`

---

### Tâche 2 — Scaffolder Python `scaffold-domain.py`

**Fichiers :**
- Create : `quarkdown/scripts/scaffold-domain.py`

**Étape 2.1** — Créer le script avec API :
```
python quarkdown/scripts/scaffold-domain.py <domain> [--force]
```

Comportement :
1. Vérifie que `domains/<domain>/01-theory/` existe (sinon erreur explicite)
2. Crée `domains/<domain>/01-theory-qd/` si absent
3. Liste les `01-xxx.md` dans `01-theory/` (triés par préfixe numérique)
4. Lit `domains/<domain>/README.md` si présent pour extraire le titre du domaine (fallback : nom du dossier capitalized)
5. Génère `01-theory-qd/main.qd` avec :
   - Préambule standard
   - Sidebar nav listant tous les chapitres détectés
   - Plan du module dans le corps
6. Pour chaque `.md` détecté, génère un `.qd` placeholder (sauf si déjà existant et pas `--force`) :
   - Préambule standard
   - Sidebar nav identique à `main.qd`
   - Copie brute du contenu du `.md` source

**Étape 2.2** — Test sur `agentic-ai` (déjà migré) avec `--force` :
- Doit régénérer un `main.qd` cohérent
- Doit régénérer les `.qd` placeholders pour les chapitres NON enrichis (4-14)
- Les 3 `.qd` enrichis (1, 2, 3) ne sont écrasés QUE si `--force`

⚠️ Pour ce test, NE PAS utiliser `--force` afin de préserver les `.qd` enrichis. Vérifier juste que les 11 placeholders manquants sont créés.

**Étape 2.3** — Test build d'agentic-ai après scaffold :
```powershell
quarkdown c domains/agentic-ai/01-theory-qd/main.qd --out quarkdown/output-site/agentic-ai
python quarkdown/post-build-fix-links.py quarkdown/output-site/agentic-ai
```
Expected: 14 pages générées (3 enrichies + 11 placeholders), sidebar complète sur toutes.

**Étape 2.4** — Commit : `feat(quarkdown): scaffolder de domaine pour generer main.qd + placeholders`

---

### Tâche 3 — Build wrapper `build-all.ps1`

**Fichiers :**
- Create : `quarkdown/scripts/build-all.ps1`

**Étape 3.1** — Créer le script avec API :
```powershell
pwsh quarkdown/scripts/build-all.ps1 [-Domain <name>] [-Watch]
```

Comportement :
1. Configure JAVA_HOME et PATH si non déjà set
2. Itère sur `domains/*/01-theory-qd/main.qd` (auto-détection)
3. Si `-Domain <name>` : ne build que ce domaine, échoue clairement si `01-theory-qd/main.qd` absent
4. Pour chaque domaine :
   - `quarkdown c <main.qd> --out quarkdown/output-site/<domain>`
   - `python quarkdown/post-build-fix-links.py quarkdown/output-site/<domain>`
5. Si `-Watch` : exclusif avec mode multi-domaine, nécessite `-Domain`, ajoute `-p -w` au quarkdown
6. Récap final : domains buildés OK, domains skippés (no `01-theory-qd/`), erreurs

**Étape 3.2** — Test sans flag :
- Doit builder agentic-ai (seul domaine avec `01-theory-qd/`)
- Doit skipper proprement les 3 autres

**Étape 3.3** — Test avec `-Domain neural-networks-llm` :
- Doit échouer proprement avec message "no 01-theory-qd/ found, run scaffold-domain.py first"

**Étape 3.4** — Commit : `feat(quarkdown): build wrapper PowerShell pour iterer sur les domaines`

---

### Tâche 4 — Validation E2E sur neural-networks-llm

**Étape 4.1** — Scaffold neural-networks-llm :
```powershell
python quarkdown/scripts/scaffold-domain.py neural-networks-llm
```
Expected: 20 `.qd` placeholders + `main.qd` créés dans `domains/neural-networks-llm/01-theory-qd/`.

**Étape 4.2** — Build :
```powershell
pwsh quarkdown/scripts/build-all.ps1 -Domain neural-networks-llm
```
Expected: `[Success]`, 20 pages dans `quarkdown/output-site/neural-networks-llm/`, post-process OK.

**Étape 4.3** — Build-all sans flag :
```powershell
pwsh quarkdown/scripts/build-all.ps1
```
Expected: 2 domaines buildés (agentic-ai + neural-networks-llm), 2 skippés (algorithmie-python + system-design).

**Étape 4.4** — Commit : `feat(quarkdown): scaffold neural-networks-llm (20 placeholders)`

---

### Tâche 5 — Mise à jour du skill `quarkdown-course-author`

**Fichiers :**
- Edit : `C:/Users/antho/.claude/skills/quarkdown-course-author/SKILL.md`

**Étape 5.1** — Repositionner le skill de "convert" vers "enrich" :
- Préciser dès le début que le `.qd` est créé par `scaffold-domain.py` (pas par le skill)
- Le skill bosse sur un `.qd` placeholder existant dans `domains/<domain>/01-theory-qd/<chapter>.qd`
- Workflow modifié : Phase 1 = vérifier que `01-theory-qd/<chapter>.qd` existe (sinon demander à l'user de scaffold), Phase 2 = enrichir le contenu, Phase 3 = build + verify

**Étape 5.2** — Adapter les chemins :
- Sidebar nav : chemins frères (`01-xxx.qd`) au lieu de `../../../quarkdown/site/`
- Build : `quarkdown c domains/<domain>/01-theory-qd/main.qd --out quarkdown/output-site/<domain>` au lieu de `quarkdown/site/main.qd`
- Plus besoin de `--allow global-read` (chemins locaux)

**Étape 5.3** — Référencer le scaffolder et le build wrapper dans les "outils du skill"

**Étape 5.4** — Commit (dans le repo skills si versionné, sinon juste la modif locale)

---

### Tâche 6 — Mise à jour de la documentation

**Fichiers :**
- Edit : `quarkdown/README.md` (build commands, structure)
- Edit : `CLAUDE.md` (mention `01-theory-qd/` dans l'arbo)
- Edit : `tasks/lessons/quarkdown.md` (ajouter une leçon sur le nouveau layout si pertinent)

**Étape 6.1** — `quarkdown/README.md` :
- Mettre à jour la section Build avec les nouvelles commandes (`build-all.ps1`)
- Documenter le scaffolder
- Mettre à jour la section "Architecture (cible)" avec le layout `01-theory-qd/`
- Supprimer la mention de `quarkdown/site/` (n'existe plus)

**Étape 6.2** — `CLAUDE.md` (racine du repo) :
- Mettre à jour l'arborescence pour mentionner `01-theory-qd/`
- Préciser que le skill `quarkdown-course-author` enrichit, le scaffolder convertit

**Étape 6.3** — Commit : `docs(quarkdown): documenter le nouveau layout 01-theory-qd et les scripts`

---

### Tâche 7 — Vérification finale

**Étape 7.1** — Inspection visuelle de l'output :
- Lancer un serveur HTTP local sur `quarkdown/output-site/` ou ouvrir directement le `index.html`
- Vérifier que la sidebar liste les 14 chapitres agentic-ai et 20 chapitres neural-networks-llm
- Vérifier que les 3 `.qd` enrichis d'agentic-ai conservent leur rendu riche (mermaid, callouts)
- Vérifier que les placeholders affichent le contenu `.md` brut (pas vides, pas cassés)

**Étape 7.2** — Test de portabilité KaView :
- Vérifier que `output-site/agentic-ai/01-anatomie-agent/index.html` ouvre correctement en `file://`
- Vérifier que les liens inter-chapitres marchent (post-process passé)

**Étape 7.3** — Récap pour le user : verdict PASS/PARTIAL/FAIL avec preuves concrètes (commandes + output)

---

## Critères de succès (rappel)

- [ ] `domains/agentic-ai/01-theory-qd/` contient `main.qd` + 14 chapitres `.qd` (3 enrichis + 11 placeholders)
- [ ] `domains/neural-networks-llm/01-theory-qd/` contient `main.qd` + 20 chapitres `.qd` placeholders
- [ ] `quarkdown/site/` n'existe plus
- [ ] `pwsh quarkdown/scripts/build-all.ps1` build les 2 domaines sans warning et sans `--allow global-read`
- [ ] `pwsh quarkdown/scripts/build-all.ps1 -Domain algorithmie-python` échoue proprement
- [ ] Le skill `quarkdown-course-author` reflète le nouveau workflow
- [ ] Les `.md` originaux sont intacts (vérifier `git status` ne montre AUCUN changement dans `01-theory/*.md`)
