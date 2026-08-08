# SVG pedagogiques haute qualite — standard Mastering Believe

> **Quand** : tout schema place dans `domains/<track>/<slug>/assets/*.svg`  
> **Pourquoi** : public visuel (track vie en particulier) ; un SVG bas de gamme mine la confiance plus vite qu'un paragraphe moyen.  
> **Regle d'or** (skill `imagine`) : pour texte exact, chiffres, fleches et structure → **construire en code (SVG/HTML)**, jamais generer le schema via un modele image.

## References de design (a lire avant de dessiner)

| Source | Ce qu'on en tire |
|--------|------------------|
| Skill **imagine** (`~/.grok/skills/imagine/SKILL.md`) | Texte/nombres/structure → code, pas Imagine. Verif visuelle en boucle. |
| Skill **pptx** — Design Ideas + QA visuelle | Palette a dominance (60–70 %), contraste, marges, pas d'egalite de couleurs, boucle *render → inspect → fix*. |
| Skill **diagram** (gstack) | SVG net pour docs ; labels courts ; 5–15 nœuds max par schema. |
| Skill **quarkdown-course-author** — checklist qualite | Callouts / densite / relecture visuelle avant PASS. |
| Best-in-class pedagogie visuelle | Duolingo / Brilliant / Khan : une idee par ecran, gros contraste, feedback couleur **labelle en texte** (accessibilite). |

## Design system obligatoire (domaine `ia-quotidien` et suivants)

### Palette « Teal Trust » (inspiree pptx, theme IA/confiance)

| Role | Hex | Usage |
|------|-----|--------|
| Primary | `#0F766E` | Cartes OK, accents, titres secondaires |
| Primary deep | `#115E59` | Headers sombres, ombres portees |
| Accent | `#F59E0B` | Attention / etape active |
| Danger | `#DC2626` | Interdits, hallucinations |
| Surface | `#F8FAFC` | Fond page |
| Card | `#FFFFFF` | Cartes |
| Ink | `#0F172A` | Texte principal |
| Muted | `#64748B` | Captions, footer |
| Line | `#E2E8F0` | Bordures |

**Dominance** : ~65 % surface claire, ~25 % teal, ~10 % accent/danger. Ne jamais donner le meme poids a 5 pastels.

### Typographie

- Famille : `Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif`
- Titre : 22–28 px, weight 700, ink
- Corps carte : 14–16 px, weight 500–600
- Caption / footer : 12 px, muted
- **Accents francais corrects** (é, è, à, ô…) — ASCII seul = FAIL qualite
- Pas de Helvetica seul en fallback primaire si Inter est dispo ; garder stack systeme robuste

### Geometrie & rythme

- ViewBox cible : **1200 × 680** (16:9 pedagogique) sauf bannieres etroites
- Marges exterieures ≥ **32 px**
- Gap entre cartes ≥ **20 px**
- Rayon cartes : **16–20 px**
- Ombre : filtre SVG `feDropShadow` leger (dx=0 dy=4 std=8 opacity 0.08) — une seule ombre systeme
- Fleches : trait 2.5 px + marker-end defini dans `<defs>`
- Alignement : grille 8 px mentale ; centres verticaux partages

### Accessibilite (non negociable)

```xml
<svg role="img" aria-labelledby="t d" ...>
  <title id="t">…</title>
  <desc id="d">…</desc>
```

- Couleur **jamais** seule porteuse de sens : toujours un label texte (VERT / ORANGE / ROUGE, AVANT / APRES)
- Contraste texte ≥ WCAG AA sur fond de carte
- Le module markdown doit repeter **« En une phrase : »** sous le SVG

### Contenu pedagogique

1. **Une idee** par SVG (pas un wiki en 12 boites)
2. Titre en haut centré ou left-aligned avec barre laterale teal de 6 px
3. Footer discret : `Module NN · domaine` (pas de branding invente)
4. Exemples chiffrés coherents avec les exercices (ex. total entrees **600**)

## Pipeline de production (obligatoire)

```
1. Spec 1 phrase (ce que l'apprenant doit retenir en regardant)
2. Dessiner le SVG en code (Python string / template design system)
3. Rasteriser (browse ou rsvg/cairo) → PNG 2x
4. Lire le PNG (outil Read / inspection visuelle)
5. Noter les defauts : overflow, bas contraste, gaps inegaux, texte coupe
6. Corriger → re-rasteriser → re-lire
7. STOP seulement apres un cycle fix-and-verify propre
```

Inspire de la **QA visuelle pptx** : « Assume there are problems. Your first render is almost never correct. »

## Anti-patterns (FAIL automatique)

| Interdit | Pourquoi |
|----------|----------|
| Rectangles colores bruts sans ombre/hierarchie | Aspect "slides PowerPoint 2003" |
| 5+ couleurs pastels a poids egal | Pas de dominance (pptx) |
| Texte sans accents en FR | Illisible / non pro |
| Labels qui debordent du rect | Overflow |
| Fleches qui ne touchent rien | Schema trompeur (imagine) |
| SVG sans title/desc | A11y |
| Generer le schema via Imagine/DALL·E | Texte garble (imagine) |
| Camembert 12 parts / 3D | Few / sobriete pedagogique |

## Checklist par fichier (cocher avant commit)

- [ ] ViewBox 1200×680 (ou justifie)
- [ ] Palette Teal Trust respectee
- [ ] title + desc + role=img
- [ ] Accents FR corrects
- [ ] 1 idee claire
- [ ] Ombre systeme + rayon 16+
- [ ] Marges ≥ 32
- [ ] PNG inspecte a l'œil (cycle fix)
- [ ] Lien module : `![…](../assets/xx.svg)` + *En une phrase*

## Emplacement

```
domains/<track>/<slug>/assets/
  01-….svg
  …
  README.md          # inventaire + ce standard pointe ici
.claude/skills/mastering-domain-creator/references/svg-pedagogique.md  # SSOT
```
