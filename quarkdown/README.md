# Quarkdown — pipeline de rendu pour les cours

Ce dossier contient le pipeline qui transforme les sources `.qd`
(versions enrichies des cours `.md`) en **HTML statique** consommable :

- Site web (GitHub Pages, navigation, recherche, math, mermaid)
- App **KaView** (HTML bundle affiche dans WKWebView, offline-first)
- Slides Reveal.js et PDF a partir de la meme source

> **Decision d'architecture** : les `.md` restent source-of-truth
> lisibles sur GitHub dans `domains/<domain>/01-theory/`. Les `.qd`
> sont des versions enrichies (math LaTeX, callouts, mermaid,
> variables, fonctions) regroupees dans
> `domains/<domain>/01-theory-qd/` — un site Quarkdown par domaine.

## Statut

- **agentic-ai** : 3 chapitres enrichis (`01-anatomie-agent`,
  `02-tool-use-function-calling`, `05-langgraph-fondamentaux`)
  + 11 placeholders. Site buildable de bout en bout.
- **neural-networks-llm** : 22 placeholders, scaffold initial.
- Les autres domaines n'ont pas encore de `01-theory-qd/` — utiliser
  le scaffolder pour en creer un (cf. ci-dessous).

## Prerequis

- **Java 17+** (Quarkdown est ecrit en Kotlin/JVM)
- **Quarkdown CLI** (v2.x)
- **Python 3.x** (post-process des liens + scaffolder)
- **Node + Puppeteer** uniquement si export PDF souhaite

## Installation (Windows)

```powershell
# 1. Java 17+ via Scoop (recommande, pas d'admin)
scoop install temurin21-jdk

# 2. Quarkdown via Scoop
scoop bucket add extras
scoop install quarkdown

# Verifier
java -version
quarkdown --version
```

Alternative : telecharger le JAR depuis
https://github.com/iamgio/quarkdown/releases et le wrapper soi-meme.

## Build

Toutes les commandes sont a lancer depuis la racine du repo. Le
wrapper `build-all.ps1` applique le post-process des liens et gere
le mode watch. Il assume que `java` et `quarkdown` sont dans le
PATH (cas par defaut apres `scoop install`).

> **Overrides locaux optionnels** si l'install n'est pas dans le
> PATH (ex. JDK ou Quarkdown bin custom) :
>
> ```powershell
> $env:QUARKDOWN_JAVA_HOME = "C:\path\to\jdk-21"
> $env:QUARKDOWN_BIN_DIR   = "C:\path\to\quarkdown\bin"
> ```
>
> Le wrapper les detecte et les ajoute au PATH pour l'invocation.

```powershell
# Build de tous les domaines qui ont un 01-theory-qd/
powershell -NoProfile -ExecutionPolicy Bypass -File quarkdown/scripts/build-all.ps1

# Build d'un seul domaine
powershell -NoProfile -ExecutionPolicy Bypass -File quarkdown/scripts/build-all.ps1 -Domain agentic-ai

# Live preview (watch + auto-reload, 1 seul domaine a la fois)
powershell -NoProfile -ExecutionPolicy Bypass -File quarkdown/scripts/build-all.ps1 -Domain agentic-ai -Watch
```

Output : `quarkdown/output-site/<domain>/` par domaine. Gitignored.

> **Note** : un domaine sans `01-theory-qd/` est silencieusement
> ignore. Pour scaffolder un nouveau domaine, voir la section
> suivante.

## Lire les sites compiles (apres build)

Apres un build, les sites vivent dans
`quarkdown/output-site/<domain>/Mastering-Believe-<Titre-Du-Domaine>/`.
Il y a 3 facons de les ouvrir, par ordre de simplicite :

### 1. Ouverture directe `file://` (le plus simple)

Le post-process `post-build-fix-links.py` (auto-applique par
`build-all.ps1`) reecrit les liens en `<chap>/index.html` explicites,
ce qui rend le bundle portable hors serveur HTTP. Donc :

```powershell
# Ouvre le site agentic-ai dans le navigateur par defaut
start quarkdown/output-site/agentic-ai/Mastering-Believe-Systemes-IA-Agentiques/index.html
```

Marche en `file://`, en double-clic, dans WKWebView (KaView). Seule
limitation : la **search bar globale** de Quarkdown (en haut) peut
necessiter un serveur HTTP pour charger `search-index.json` selon le
navigateur (politique CORS sur `file://`).

### 2. Live preview avec auto-reload (en cours d'enrichissement)

Pour iterer sur un chapitre, le mode watch rebuild + reload le
navigateur a chaque sauvegarde du `.qd` :

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  quarkdown/scripts/build-all.ps1 -Domain agentic-ai -Watch
```

Quarkdown sert le site sur `http://localhost:8089` (port par defaut)
et ouvre le browser. Ctrl+C pour arreter.

### 3. Serveur HTTP local (search bar 100% fonctionnelle)

Si on veut juste re-lire un build deja fait (sans watch) avec la
search bar globale qui marche partout :

```powershell
cd quarkdown/output-site/agentic-ai/Mastering-Believe-Systemes-IA-Agentiques
python -m http.server 8765
# puis ouvrir http://localhost:8765/
```

### 4. Embarquer le bundle dans une app native (iOS / Android)

Les bundles produits sont **portables** : grace au post-process,
les liens inter-chapitres pointent en `<chap>/index.html` explicite,
ce qui les rend chargeables dans une `WebView` native via
`file://` (sans serveur HTTP embarque).

Workflow type :

1. Builder le domaine : `build-all.ps1 -Domain <domain>`.
2. Zipper le dossier `output-site/<domain>/Mastering-...` produit.
3. Embarquer le zip dans les `Resources/` de l'app, le dezipper au
   premier lancement dans un dossier writable (ex.
   `Application Support/courses/<domain>/`).
4. Charger `index.html` dans la WebView en pointant le 2eme argument
   d'autorisation de lecture sur la **racine du bundle** (sinon
   CSS/JS/sous-pages refusent silencieusement de charger).

Pour automatiser : un workflow GitHub Actions sur tag `v*` qui build
tous les domaines + zip + upload comme asset de GitHub Release est
l'approche standard. L'app fetch alors
`releases/latest/download/<domain>-bundle.tar.gz` et cache localement
(ETag pour eviter les re-DL inutiles). Avantage : maj des cours sans
passer par les stores.

**Limitation connue** : la search bar globale Quarkdown ne marche pas
en `file://` a cause de la politique CORS sur `search-index.json`.
Sidebar + prev/next + nav inter-chapitres fonctionnent normalement.
Pour la search, soit demarrer un mini-serveur HTTP local dans l'app
(overkill), soit exposer l'index dans une UI native qui parse
`search-index.json` directement.

## Scaffolder un nouveau domaine

Le scaffolder lit le reading path source-of-truth
(`domains/<domain>/01-theory/NN-*.md`) et genere :

- `01-theory-qd/main.qd` — index du site avec sidebar nav
- `01-theory-qd/NN-*.qd` — un placeholder par chapitre (preambule
  standard + sidebar nav + copie brute du `.md`)

Le site devient buildable des le jour 1. L'enrichissement
(mermaid, callouts denses, exemples chiffres, etc.) se fait ensuite
chapitre par chapitre via le skill `quarkdown-course-author`.

```powershell
# Scaffold initial d'un domaine
python quarkdown/scripts/scaffold-domain.py agentic-ai

# Re-scaffold apres ajout/retrait de chapitres .md
# (regenere main.qd, sync les sidebars dans les .qd existants,
#  cree les nouveaux placeholders, n'ecrase rien)
python quarkdown/scripts/scaffold-domain.py agentic-ai

# Forcer la regeneration de TOUS les .qd (destructif)
python quarkdown/scripts/scaffold-domain.py agentic-ai --force
```

Idempotent par defaut : `main.qd` est toujours regenere (c'est
l'index), les `.qd` chapitres existants conservent leur contenu
mais leur sidebar est synchronisee avec la liste courante.

## Pattern site multi-page (sidebar nav)

La sidebar laterale gauche est generee par la combinaison :

```
.doctype {docs}
.theme {paperwhite} layout:{hyperlegible}

.pagemargin {lefttop}
    .navigation role:{pagelist}
        - [Accueil](main.qd)
        - [1. Chapitre 1](01-chapitre-un.qd)
        - [2. Chapitre 2](02-chapitre-deux.qd)
```

Ce bloc est **duplique dans chaque `.qd`** du site (le main + chaque
chapitre) pour que la nav soit visible partout. Le scaffolder s'en
charge automatiquement et synchronise tous les `.qd` quand la liste
de chapitres change. Quarkdown ajoute en bonus :

- une **search bar** globale en haut
- des liens **prev/next** automatiques en bas de chaque page

Comme tous les `.qd` d'un site vivent dans le meme dossier
(`01-theory-qd/`), les chemins de la sidebar sont locaux (pas de
`../`), ce qui evite le besoin de `--allow global-read` au build.

## Convention `.md` <-> `.qd`

| Cas | Ou ca vit |
|-----|-----------|
| Source-of-truth lisible GitHub | `domains/<domain>/01-theory/NN-*.md` |
| Version enrichie pour le site | `domains/<domain>/01-theory-qd/NN-*.qd` |
| Index du site Quarkdown | `domains/<domain>/01-theory-qd/main.qd` |
| Build artifacts | `quarkdown/output-site/<domain>/` (gitignored) |

Le `.md` n'est jamais modifie par le pipeline `.qd`. Les deux
co-existent : GitHub affiche le `.md`, le site Quarkdown affiche le
`.qd` enrichi.

## Architecture

```
mastering-believe (repo, source-of-truth)
  domains/<domain>/
    01-theory/      ← .md source-of-truth, lisibles GitHub
      01-...md
      02-...md
    01-theory-qd/   ← site Quarkdown enrichi (1 par domaine)
      main.qd
      01-...qd
      02-...qd
  quarkdown/
    README.md (ce fichier)
    scripts/
      scaffold-domain.py   ← scaffold + sync sidebar
      build-all.ps1        ← build wrapper (JAVA_HOME, post-process, watch)
    post-build-fix-links.py ← rewrite liens pour bundle portable
    output-site/<domain>/  ← build artifacts (gitignored)
       │
       └─→ deployable :
            - GitHub Pages (site web public)
            - bundle KaView (zip d'assets HTML embarque dans l'app iOS)
```

## Workflow type pour ajouter / enrichir un chapitre

1. Le `.md` source existe deja dans `01-theory/` (sinon le creer).
2. Si le domaine n'a pas encore de `01-theory-qd/`, lancer le
   scaffolder : `python quarkdown/scripts/scaffold-domain.py <domain>`.
3. Si le `.md` vient d'etre ajoute, re-lancer le scaffolder pour
   creer le placeholder `.qd` correspondant et synchroniser la
   sidebar de tout le domaine.
4. Enrichir le `.qd` placeholder (cf. skill `quarkdown-course-author`).
5. Builder : `powershell -File quarkdown/scripts/build-all.ps1 -Domain <domain>`.
6. Verifier le rendu (sidebar, callouts, mermaid) dans
   `quarkdown/output-site/<domain>/`.

## Integration KaView (option preferee : build-time)

KaView affiche les cours dans `WKWebView` (renderer HTML natif iOS).
Le pipeline cible :

1. CI GitHub Actions sur `mastering-believe` : compile tous les `.qd`
   via `build-all.ps1` (ou son equivalent bash).
2. Tag de release : zip `quarkdown/output-site/` -> GitHub Release.
3. KaView fetch le zip au build (ou on-demand dans l'onglet Cours).
4. WKWebView affiche `output-site/<domain>/<chapter>/index.html`.

Le post-process `post-build-fix-links.py` rewrite les liens pour que
le bundle soit portable (file:// + WKWebView), c'est applique
automatiquement par `build-all.ps1`.

Avantages : KaView reste offline-first, pas de runtime Java embarque,
zero dependance reseau pour lire un cours apres install.

## Patterns Quarkdown a standardiser pour le repo

- `.box type:{note|tip|warning|callout|error}` — callouts standard
  (seuls types valides du stdlib). Pas de `.callout` custom.
- `.mermaid caption:{...}` — diagrammes avec numerotation auto.
- Tables Markdown standard pour les comparaisons.
- Code Python en blocs ``` ``` standard (coloration auto).

Les patterns de flashcards / exercices restent a designer apres
plus de chapitres enrichis (besoin reel a valider avant
d'abstraire).
