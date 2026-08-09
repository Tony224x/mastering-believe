"""Injection des schémas dans les .md (style KaView / ia-quotidien)."""

from __future__ import annotations

import re
from pathlib import Path

from .captions import CAPTIONS, HERO_SLUGS
from .registry import SPECS

# assets_gen/ → scripts/ → agentic-ai/
ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets"
THEORY = ROOT / "01-theory"

# Image + toutes les blockquotes qui suivent (légendes), jusqu'à une ligne non-quote.
_ASSET_BLOCK = re.compile(
    r"\n*!\[(?P<alt>[^\]]*)\]\((?P<path>(?:\.\./)?assets/)(?P<slug>[^)/]+)\.svg\)"
    r"(?:\n\s*)*"
    r"(?P<body>(?:>[^\n]*(?:\n|$))+)",
    re.MULTILINE,
)

# Légendes orphelines d'une passe précédente (sans image juste au-dessus).
_ORPHAN_NEGATIVE = re.compile(
    r"\n*> \*\*Si le schéma ne s'affiche pas :\*\*[^\n]*\n"
    r"(?:>[^\n]*\n)*",
    re.MULTILINE,
)

_META_REPS: list[tuple[str, str]] = [
    ("Temps estime", "Temps estimé"),
    ("Prerequis", "Prérequis"),
    ("maitriser", "maîtriser"),
    ("reellement", "réellement"),
    ("modele", "modèle"),
    ("implementer", "implémenter"),
    ("implemente", "implémente"),
    ("evaluation", "évaluation"),
    ("securite", "sécurité"),
    ("memoire", "mémoire"),
    ("patterns avance", "patterns avancé"),
    ("HITL avance", "HITL avancé"),
    ("agent autonome avance", "agent autonome avancé"),
]


def image_block(slug: str, asset_rel: str) -> str:
    """Bloc image + légende positive (lisible même sans le rendu)."""
    meta = CAPTIONS[slug]
    return (
        f"\n\n![{meta['alt']}]({asset_rel})\n\n"
        f"> **En une phrase :** {meta['phrase']}\n"
        f">\n"
        f"> **Visuel :** {meta['legend']}\n"
    )


def _strip_existing_asset_block(text: str, slug: str) -> str:
    """Retire tous les blocs image+légende pour ce slug."""

    def repl(m: re.Match[str]) -> str:
        if m.group("slug") == slug:
            return "\n"
        return m.group(0)

    text = _ASSET_BLOCK.sub(repl, text)
    text = _ORPHAN_NEGATIVE.sub("\n", text)
    return text


def polish_module_header(text: str) -> str:
    """Style KaView : accents FR + meta blockquote aéré (comme ia-quotidien)."""
    lines = text.splitlines(keepends=True)
    if not lines:
        return text

    out: list[str] = []
    in_meta = False
    meta_done = False

    for i, line in enumerate(lines):
        if meta_done:
            out.append(line)
            continue

        if not in_meta and line.startswith(">") and i > 0:
            in_meta = True

        if not in_meta:
            out.append(line)
            continue

        if line.startswith(">"):
            body = line
            for old, new in _META_REPS:
                body = body.replace(old, new)
            if "**Objectif**" in body and out:
                prev = out[-1]
                if (
                    prev.startswith(">")
                    and prev.strip() != ">"
                    and "**Objectif**" not in prev
                ):
                    out.append(">\n")
            out.append(body)
            continue

        # fin du blockquote meta
        in_meta = False
        meta_done = True
        out.append(line)

    return "".join(out)


def inject_theory(*, force: bool = True) -> None:
    del force  # toujours réécrit de façon idempotente
    for slug in HERO_SLUGS:
        path = THEORY / f"{slug}.md"
        if not path.exists():
            print(f"  skip missing {path.name}")
            continue
        text = path.read_text(encoding="utf-8")
        text = polish_module_header(text)
        text = _strip_existing_asset_block(text, slug)
        block = image_block(slug, f"../assets/{slug}.svg")
        if "\n---\n" in text:
            head, tail = text.split("\n---\n", 1)
            tail = tail.lstrip("\n")
            text = head + "\n---\n" + block + "\n" + tail
        else:
            text = text.rstrip() + block + "\n"
        text = re.sub(r"\n{3,}", "\n\n", text)
        path.write_text(text, encoding="utf-8")
        print(f"  injected {path.name}")


def inject_readme(*, force: bool = True) -> None:
    del force
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    slug = "parcours-28j"
    text = _strip_existing_asset_block(text, slug)
    meta = CAPTIONS[slug]
    block = (
        f"\n![{meta['alt']}](assets/{slug}.svg)\n\n"
        f"> **En une phrase :** {meta['phrase']}\n"
        f">\n"
        f"> **Visuel :** {meta['legend']}\n\n"
    )
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    inserted = False
    for line in lines:
        out.append(line)
        if not inserted and line.startswith("# "):
            out.append(block)
            inserted = True
    text = re.sub(r"\n{3,}", "\n\n", "".join(out))
    readme.write_text(text, encoding="utf-8")
    print("  injected README")


def write_assets_readme() -> None:
    rows = [
        f"| `{slug}.svg` | {CAPTIONS.get(slug, {}).get('alt', slug)} |"
        for slug, _ in SPECS
    ]
    content = f"""# Assets visuels — agentic-ai

**Standard qualité** (SSOT) :  
[`.claude/skills/mastering-domain-creator/references/svg-pedagogique.md`](../../../../.claude/skills/mastering-domain-creator/references/svg-pedagogique.md)

## Design system (Teal Trust)

| Rôle | Hex |
|------|-----|
| Primary | `#0F766E` |
| Accent | `#F59E0B` |
| Danger | `#DC2626` |
| Surface | `#F8FAFC` |
| Ink | `#0F172A` |

Chaque SVG : **1200×680**, ombre système, barre latérale teal, `title`+`desc`, 1 idée.

## Style KaView (légendes positives)

Sous chaque image — toujours utile, jamais un message d'échec :

```markdown
![alt descriptif](../assets/NN-slug.svg)

> **En une phrase :** takeaway pédagogique.
>
> **Visuel :** ce que le schéma montre (caption lisible hors image).
```

L'`alt` + le bloc **Visuel** couvrent accessibilité et lecture KaView hors-ligne.

## Inventaire

| Fichier | Contenu |
|---------|---------|
{chr(10).join(rows)}

## Régénérer

```bash
python domains/tech/agentic-ai/scripts/generate_assets.py
```

Structure du générateur :

```
scripts/
  generate_assets.py          # CLI
  assets_gen/
    kit.py                    # palette + primitives SVG
    captions.py               # alt / légende / takeaway
    registry.py               # SPECS
    inject.py                 # injection MD style KaView
    builds/
      s1_fondations.py        # J1–J7
      s2_multi_prod.py        # J8–J14
      s3_frontier.py          # J15–J21
      s4_scale.py             # J22–J28 + parcours
      s5_body_diagrams.py     # schémas corps lot A
      s6_enrichment.py        # enrichissement lots B/C
```
"""
    (ASSETS / "README.md").write_text(content, encoding="utf-8")
    print("  wrote assets/README.md")
