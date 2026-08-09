#!/usr/bin/env python3
"""Génère les SVG pédagogiques du domaine agentic-ai (standard Teal Trust).

Usage:
  python domains/tech/agentic-ai/scripts/generate_assets.py
  python domains/tech/agentic-ai/scripts/generate_assets.py --svg-only
  python domains/tech/agentic-ai/scripts/generate_assets.py --inject-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from assets_gen.inject import (  # noqa: E402
    ASSETS,
    inject_readme,
    inject_theory,
    write_assets_readme,
)
from assets_gen.registry import SPECS  # noqa: E402


def write_svgs() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    repo_root = ASSETS.parent.parent.parent  # mastering-believe
    for slug, builder in SPECS:
        svg = builder()
        if not svg.strip().startswith("<?xml"):
            raise SystemExit(f"SVG invalide pour {slug}")
        path = ASSETS / f"{slug}.svg"
        path.write_text(svg, encoding="utf-8")
        try:
            rel = path.relative_to(repo_root)
        except ValueError:
            rel = path
        print(f"wrote {rel}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Assets SVG agentic-ai")
    parser.add_argument(
        "--svg-only", action="store_true", help="Ne régénère que les SVG"
    )
    parser.add_argument(
        "--inject-only", action="store_true", help="N'injecte que les blocs MD"
    )
    args = parser.parse_args()

    if not args.inject_only:
        write_svgs()
    if not args.svg_only:
        inject_theory()
        inject_readme()
        write_assets_readme()
    print("done")


if __name__ == "__main__":
    main()
