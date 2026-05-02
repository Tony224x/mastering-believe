"""Post-traite l'output Quarkdown pour rendre les liens portables.

Quarkdown emit des liens entre subdocuments sans suffixe `/index.html`
(ex. `href="../01-anatomie-agent"`). Ces liens marchent sur un serveur
HTTP qui fait du directory index, mais cassent en `file://` et dans
WKWebView (KaView). Ce script reecrit les liens en `href="../01-anatomie-agent/index.html"`
explicites — le bundle devient utilisable partout sans serveur.

Usage :
    python quarkdown/post-build-fix-links.py quarkdown/output-site

Idempotent : peut etre rejoue sans casser.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HREF_RE = re.compile(r'href="([^"#?]+)"')


def fix_html_file(html_path: Path) -> int:
    """Reecrit les hrefs vers des dossiers en hrefs vers index.html."""
    text = html_path.read_text(encoding="utf-8")
    base_dir = html_path.parent
    changed = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal changed
        href = match.group(1)

        if href.startswith(("http://", "https://", "mailto:", "/", "data:")):
            return match.group(0)
        if href.endswith((".html", ".htm")) or "." in Path(href).name:
            return match.group(0)

        candidate = (base_dir / href / "index.html").resolve()
        if not candidate.is_file():
            return match.group(0)

        new_href = href.rstrip("/") + "/index.html"
        changed += 1
        return f'href="{new_href}"'

    new_text = HREF_RE.sub(replace, text)
    if changed:
        html_path.write_text(new_text, encoding="utf-8")
    return changed


def main() -> None:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <output-site-dir>", file=sys.stderr)
        sys.exit(2)

    root = Path(sys.argv[1]).resolve()
    if not root.is_dir():
        print(f"error: not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    total_files = 0
    total_links = 0
    for html_path in root.rglob("*.html"):
        n = fix_html_file(html_path)
        if n:
            total_files += 1
            total_links += n
            print(f"  fixed {n:3d} link(s) in {html_path.relative_to(root)}")

    print(f"done: {total_links} link(s) rewritten across {total_files} file(s)")


if __name__ == "__main__":
    main()
