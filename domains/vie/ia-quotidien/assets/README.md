# Assets visuels — ia-quotidien

**Standard qualité** (SSOT skill) :  
[`.claude/skills/mastering-domain-creator/references/svg-pedagogique.md`](../../../../.claude/skills/mastering-domain-creator/references/svg-pedagogique.md)

Références croisées : skill `imagine` (code vs image), skill `pptx` (palette / QA visuelle), skill `diagram` (SVG net).

## Design system (Teal Trust)

| Rôle | Hex |
|------|-----|
| Primary | `#0F766E` |
| Accent | `#F59E0B` |
| Danger | `#DC2626` |
| Surface | `#F8FAFC` |
| Ink | `#0F172A` |

Chaque SVG : **1200×680**, ombre système, barre latérale teal, `title`+`desc`, accents FR, 1 idée.


## Affichage dans GitHub / apps (PNG)

Les **schémas pédagogiques** sont fournis en double :

| Rôle | Format | Usage |
|------|--------|-------|
| Source éditable | `*.svg` | édition, qualité vectorielle |
| **Affichage cours** | `*.png` (1200×680) | Markdown des modules — compatible GitHub, mobile, previews |

Les fichiers `01-theory/*.md` et le README pointent vers les **PNG** pour éviter les images « cassées » (SVG mal supporté dans plusieurs apps). Les SVG restent la source de vérité visuelle.

## Inventaire + revue visuelle (2026-08-09)

| Fichier | Module | Revue |
|---------|--------|-------|
| `01-llm-vs-knowledge.svg` | J1 | OK — deux colonnes contraste fort |
| `02-rccfc-prompt.svg` | J2 | OK — RCCFC + version simple 3 blocs |
| `03-vair-checklist.svg` | J3 | OK — 4 cartes V-A-I-R |
| `03b-donnees-feu.svg` | J3 | OK — labels VERT/ORANGE/ROUGE + avant/après |
| `04-socratique.svg` | J4 | OK — cyclé 1-2-3 |
| `05-avant-apres-texte.svg` | J5 | OK — pipeline notes→plan→final |
| `06-excel-flow.svg` | J6 | OK — 5 étapes + résultat 600 |
| `07-formule-expliquee.svg` | J7 | OK — SOMME.SI en 3 morceaux |
| `08-nettoyage-donnees.svg` | J8 | OK — sale→propre→graphique |
| `09-classeur-onglets.svg` | J9 | OK — 4 onglets |
| `10-pitch-story.svg` | J10 | OK — arc 5 temps |
| `11-slide-avant-apres.svg` | J11 | OK — mur vs slide |
| `12-chrono-oral.svg` | J12 | OK — 4 blocs notes |
| `13-deck-8-slides.svg` | J13 | OK — grille 8 slides |
| `14-check-final.svg` | J14 | OK — vérifier / répéter / livrer |
| `parcours-14j.svg` | README | OK — 4 blocs + 2 livrables |

Pipeline QA : raster `assets/preview/*.png` (gitignore) → inspection → fix.

## Captures d'écran

Voir [`screens/README.md`](./screens/README.md).

## Ludique

Voir `../PROGRESS.md` (badges sans dark patterns).
