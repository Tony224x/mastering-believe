# Industrialisation Quarkdown — Design

**Date** : 2026-05-02
**Statut** : Validé, prêt pour implémentation
**Scope** : Restructurer la production de cours Quarkdown du repo
`mastering-believe` pour scaler à 4+ domaines × 10-20 chapitres, sans
casser l'invariant `.md` source-of-truth + GitHub-readable.

## Contexte

État actuel :
- 4 domaines actifs (`agentic-ai`, `neural-networks-llm`,
  `algorithmie-python`, `system-design`)
- 2 chapitres seulement enrichis en `.qd` :
  `domains/agentic-ai/01-theory/01-anatomie-agent.qd` et
  `02-tool-use-function-calling.qd`, juxtaposés aux `.md` dans le même
  dossier
- 1 site Quarkdown unique assemblé via `quarkdown/site/main.qd` qui
  référence des `.qd` cross-dossiers (`../../domains/...`)
- Le pattern d'adjacence `.md`/`.qd` ne scale pas : 4 domaines × 15
  chapitres = 60 `.qd` dispersés, dossiers `01-theory/` qui doublent
  de taille, peu de lisibilité dans l'arborescence

Cible KaView : chaque domaine doit pouvoir être zippé indépendamment
et servi à la WKWebView via `loadFileURL:`. Le post-process
`quarkdown/post-build-fix-links.py` est déjà en place pour rendre les
hrefs portables.

## Décisions

5 décisions structurantes prises lors du brainstorming.

### D1 — Layout : `01-theory-qd/` sibling de `01-theory/`

Chaque domaine gagne un dossier numéroté parallèle qui héberge la
version enrichie Quarkdown :

```
domains/<domain>/
├── README.md
├── 01-theory/                    ← reading path GitHub (.md), inchangé
│   ├── 01-xxx.md
│   └── ...
├── 01-theory-qd/                 ← NOUVEAU : reading path Quarkdown
│   ├── main.qd                   ← assembly du site domaine
│   ├── 01-xxx.qd                 ← même nom de base que le .md
│   └── ...
├── 02-code/
├── 03-exercises/
└── 04-projects/
```

**Pourquoi sibling et non sous-dossier** : symétrie avec les autres
dossiers numérotés (`02-code/`, `03-exercises/`), découverte
immédiate à la racine du domaine, scale propre.

**Convention de nommage** : un `.qd` partage exactement le nom de base
de son `.md` source (`01-anatomie-agent.md` ↔ `01-anatomie-agent.qd`).
Pairing évident pour un lecteur.

### D2 — Migration immédiate des 2 `.qd` existants

Un seul commit qui :
- `git mv` les 2 `.qd` existants vers
  `domains/agentic-ai/01-theory-qd/`
- `git mv` le placeholder `quarkdown/site/03-memory-state.qd` vers
  `domains/agentic-ai/01-theory-qd/03-memory-state.qd`
- Crée `domains/agentic-ai/01-theory-qd/main.qd` (rapatrié depuis
  `quarkdown/site/main.qd` avec chemins corrigés en relatif local)
- Supprime le dossier `quarkdown/site/` devenu vide
- Met à jour les sidebar nav internes des 3 `.qd` pour utiliser des
  chemins relatifs locaux (plus de `../../`, donc plus besoin de
  `--allow global-read`)

**Pourquoi immédiate et non progressive** : 2 fichiers, migration
triviale, évite une dette permanente où la convention "post-décision
= `01-theory-qd/`, pré-décision = juxtaposé" rendrait le repo confus.

### D3 — 1 site par domaine, `main.qd` dans le dossier domaine

Chaque `01-theory-qd/main.qd` est l'assembly d'UN site auto-suffisant.
Build :

```powershell
quarkdown c domains/agentic-ai/01-theory-qd/main.qd `
  --out quarkdown/output-site/agentic-ai
```

Output : `quarkdown/output-site/agentic-ai/<doc-name>/`. Un domaine =
un bundle = un zip distribuable indépendamment vers KaView.

**Bénéfices** :
- Découplage : publier un domaine ne nécessite pas de rebuilder les
  autres
- Pas de chemin `../../` dans les nav → `--allow global-read` plus
  nécessaire
- Sidebar de taille raisonnable (10-20 entrées par site, pas 70+)
- Matche le principe "un domaine = un module self-contained" du
  CLAUDE.md

### D4 — Scaffolding script + enrichissement progressif

Nouveau script `quarkdown/scripts/scaffold-domain.py <domain>` qui,
pour un domaine donné :
- Crée `domains/<domain>/01-theory-qd/` si absent
- Génère `main.qd` avec sidebar nav listant tous les `.md` détectés
  dans `01-theory/`
- Pour chaque `01-xxx.md`, crée un `01-xxx.qd` placeholder avec :
  - Préambule standard (`.doctype docs`, `.theme paperwhite`,
    `layout:hyperlegible`)
  - Sidebar nav identique à `main.qd`
  - Copie brute du contenu du `.md` source

Le site est **buildable dès le jour 1** — toutes les pages existent
en rendu basique mais propre. Pas de liens cassés dans la sidebar
pendant des semaines.

L'enrichissement progressif (mermaid, callouts colorés, paragraphes
denses, exemples chiffrés) se fait ensuite chapitre par chapitre via
le skill `quarkdown-course-author` mis à jour (cf. D5), sans bloquer
la mise en ligne.

**Idempotence** : le script ne réécrit pas un `.qd` existant sauf
flag `--force`. Sécurise l'enrichi déjà fait.

### D5 — Mise à jour du skill `quarkdown-course-author`

Le skill bascule sémantiquement de "convert" vers "enrich" :
- Ne crée plus de `.qd` from scratch (c'est le scaffolder qui le
  fait)
- Travaille sur un `.qd` placeholder existant dans
  `domains/<domain>/01-theory-qd/`, le densifie selon les règles
  éditoriales déjà documentées dans
  `references/editorial-style.md`
- Adapte les chemins dans le SKILL.md : nouveau dossier cible
  `domains/<domain>/01-theory-qd/<chapter>.qd`, sidebar nav locale
  (plus de `../../../quarkdown/site/`)

Les references/`editorial-style.md` et `quarkdown-cheatsheet.md`
restent inchangées (les règles éditoriales et de syntaxe Quarkdown
sont indépendantes du layout).

### D6 — Build pipeline : script wrapper unique

Nouveau script `quarkdown/scripts/build-all.ps1` :
- Itère sur `domains/*/01-theory-qd/main.qd` (auto-détection : un
  domaine sans `01-theory-qd/` est ignoré)
- Pour chaque domaine :
  1. `quarkdown c <main.qd> --out quarkdown/output-site/<domain>`
  2. `python quarkdown/post-build-fix-links.py
     quarkdown/output-site/<domain>`
- Flag `-Domain <name>` pour cibler 1 seul domaine
- Flag `-Watch` pour live-preview sur 1 domaine (`-p -w` Quarkdown)

Une seule commande pour tout builder : `pwsh
quarkdown/scripts/build-all.ps1`.

**Bénéfices** :
- Plus d'oubli du post-process (intégré au script, donc plus de
  bundles cassés en KaView)
- Reproductible, scriptable côté CI plus tard sans réécriture
- Préserve la possibilité d'invocation manuelle pour debug
  (`quarkdown c ...` reste utilisable directement)

## Co-existence des 2 reading paths

| Audience | Voie de lecture | Source |
|----------|-----------------|--------|
| Lecteur GitHub (browser sur le repo) | `domains/<dom>/01-theory/*.md` | rendu natif GitHub |
| Lecteur KaView (iPad/iPhone WKWebView) | `quarkdown/output-site/<dom>/<doc>/index.html` | bundle compilé Quarkdown |
| Site web public (à venir) | idem KaView via GitHub Pages | idem |

Le `.md` reste le **source-of-truth pédagogique**. Le `.qd` est une
version **dérivée enrichie** pour rendu HTML — jamais le canonique.
Si une correction de fond doit être faite, elle est faite dans le
`.md` puis répercutée dans le `.qd` (le scaffolder peut être rejoué
avec `--force` sur un seul fichier pour récupérer le diff brut, à
re-enrichir ensuite).

## Trade-offs assumés

- **Maintenance double** : un chapitre enrichi vit dans 2 fichiers.
  Mitigation : invariant strict (`.md` source, `.qd` dérivé),
  documenté dans le CLAUDE.md du repo et le skill.
- **Pas de génération automatique du `.qd` à partir du `.md`** : le
  scaffolding ne fait que copier le contenu, l'enrichissement est
  toujours manuel via le skill. C'est volontaire — l'enrichissement
  est l'acte pédagogique de valeur, le scaffolder ne sert qu'à
  garantir que le site est buildable.

## Hors scope explicite

- Sync du bundle vers KaView (couvert par
  `kaview/kalira-code/docs/kaview/quarkdown-integration.md`)
- CI/GitHub Actions automatisée (planté pour plus tard, le script
  wrapper sera promu)
- Thème dark mode Quarkdown (planté pour plus tard)
- Recherche full-text cross-domaines (chaque site a sa search bar
  Quarkdown native, pas de hub central pour le moment)

## Critères de succès

- [ ] `domains/agentic-ai/01-theory-qd/` contient `main.qd` + 3
  chapitres `.qd` (les 2 enrichis migrés + le placeholder
  `03-memory-state`)
- [ ] `quarkdown/site/` n'existe plus
- [ ] `pwsh quarkdown/scripts/build-all.ps1` build le site
  agentic-ai sans warning et sans `--allow global-read`
- [ ] `pwsh quarkdown/scripts/build-all.ps1 -Domain neural-networks-llm`
  échoue proprement avec un message "no 01-theory-qd/ found, run
  scaffold-domain.py first"
- [ ] `python quarkdown/scripts/scaffold-domain.py neural-networks-llm`
  produit 20 `.qd` placeholders + `main.qd`, le site build
  immédiatement
- [ ] Le skill `quarkdown-course-author` enrichit un `.qd` placeholder
  existant sans erreur
- [ ] Les 2 voies de lecture restent fonctionnelles : `01-theory/*.md`
  toujours lisibles sur GitHub, bundle Quarkdown servable via
  WKWebView KaView
