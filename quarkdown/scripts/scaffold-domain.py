"""Scaffold un domaine Quarkdown : main.qd + .qd placeholders.

Pour un domaine donne (ex: agentic-ai), genere :
- domains/<domain>/01-theory-qd/main.qd : assembly du site avec sidebar
  nav listant tous les chapitres detectes dans 01-theory/.
- domains/<domain>/01-theory-qd/<NN-xxx>.qd : 1 placeholder par .md du
  reading path source-of-truth, avec preambule standard, sidebar nav, et
  copie brute du contenu .md.

Le site devient buildable des le jour 1. L'enrichissement (mermaid,
callouts denses, etc.) se fait ensuite chapitre par chapitre via le
skill `quarkdown-course-author`.

Idempotent : ne reecrit pas un .qd existant sauf flag --force. main.qd
est toujours regenere (c'est l'index, pas un livrable enrichi).

Usage :
    python quarkdown/scripts/scaffold-domain.py <domain>
    python quarkdown/scripts/scaffold-domain.py <domain> --force
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DOMAIN_TITLE_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
CHAPTER_TITLE_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


CHAPTER_PREFIX_RE = re.compile(r"^J\d+\s*[—\-:]\s*", re.IGNORECASE)


def _strip_subtitle(title: str) -> str:
    """Enleve un sous-titre apres "—" ou ":" (garde la partie avant).

    Si le resultat est trop court (< 2 mots), garde l'original — evite des
    labels indigents type "Build" pour un titre "Build : un agent de
    recherche complet".
    """
    for sep in ("—", " - ", " : ", ": "):
        if sep in title:
            head = title.split(sep, 1)[0].strip()
            if len(head.split()) >= 2:
                return head
    return title.strip()


def extract_domain_title(domain_dir: Path) -> str:
    """Extrait le titre court du domaine depuis README.md.

    Prend le H1 et retire le sous-titre apres "—" ou ":". Fallback : nom
    du dossier capitalize.
    """
    readme = domain_dir / "README.md"
    if readme.is_file():
        text = readme.read_text(encoding="utf-8")
        match = DOMAIN_TITLE_RE.search(text)
        if match:
            return _strip_subtitle(match.group(1))
    return domain_dir.name.replace("-", " ").title()


def extract_chapter_title(md_path: Path) -> str:
    """Extrait le titre d'un chapitre depuis le 1er H1.

    Retire les prefixes type "J1 — " (jour planning) et les sous-titres
    apres ":" ou "—" pour des labels de sidebar lisibles. Fallback : nom
    de fichier nettoye.
    """
    text = md_path.read_text(encoding="utf-8")
    match = CHAPTER_TITLE_RE.search(text)
    if match:
        title = match.group(1).strip()
        title = CHAPTER_PREFIX_RE.sub("", title)
        return _strip_subtitle(title)
    stem = md_path.stem
    cleaned = re.sub(r"^\d+-", "", stem).replace("-", " ")
    return cleaned.capitalize()


def list_chapter_mds(theory_dir: Path) -> list[Path]:
    """Liste les chapitres .md tries par prefixe numerique."""
    mds = sorted(theory_dir.glob("[0-9][0-9]-*.md"))
    return mds


def chapter_label(index: int, title: str) -> str:
    """Format de l'entree sidebar : "1. Anatomie d'un Agent IA"."""
    return f"{index}. {title}"


def render_sidebar(chapters: list[tuple[Path, str]]) -> str:
    """Rend le bloc .pagemargin/.navigation pour la sidebar."""
    lines = [".pagemargin {lefttop}", "    .navigation role:{pagelist}", "        - [Accueil](main.qd)"]
    for i, (md_path, title) in enumerate(chapters, start=1):
        qd_name = md_path.with_suffix(".qd").name
        lines.append(f"        - [{chapter_label(i, title)}]({qd_name})")
    return "\n".join(lines)


def render_main_qd(domain_title: str, chapters: list[tuple[Path, str]]) -> str:
    """Rend le main.qd : preambule + sidebar + corps avec plan."""
    sidebar = render_sidebar(chapters)
    plan_lines = []
    for i, (md_path, title) in enumerate(chapters, start=1):
        qd_name = md_path.with_suffix(".qd").name
        plan_lines.append(f"{i}. [{title}]({qd_name})")
    plan_block = "\n".join(plan_lines)

    return f""".doctype {{docs}}
.docname {{Mastering Believe — {domain_title}}}
.docauthor {{mastering-believe}}
.doclang {{fr}}
.theme {{paperwhite}} layout:{{hyperlegible}}

{sidebar}

# Mastering Believe — {domain_title}

Cours communautaire d'apprentissage accelere — theorie, code applique,
exercices progressifs.

## Plan du module

{plan_block}

## Approche pedagogique

- **Pareto-first** : 20% qui donne 80% du resultat en premier
- **Concret avant abstrait** : exemple, puis principe
- **Spaced repetition** : flash-cards en fin de chapitre
- **Capstone reel** : projet ship-able a la fin du module
"""


SIDEBAR_BLOCK_RE = re.compile(
    r"\.pagemargin\s*\{lefttop\}\n"
    r"(?:    \.navigation\s+role:\{pagelist\}\n)"
    r"(?:        -\s+\[[^\]]*\]\([^)]*\)\n)+",
    re.MULTILINE,
)


def update_sidebar_in_existing(qd_path: Path, new_sidebar: str) -> bool:
    """Remplace le bloc sidebar nav d'un .qd sans toucher au reste.

    Retourne True si le fichier a ete modifie. Sans-op si la sidebar y
    est deja identique. Echoue silencieusement si aucun bloc n'est
    detecte (cas d'un .qd structuralement different).
    """
    text = qd_path.read_text(encoding="utf-8")
    # Le bloc sidebar genere se termine par une ligne vide naturelle.
    new_block = new_sidebar.rstrip() + "\n"
    new_text, n = SIDEBAR_BLOCK_RE.subn(new_block, text, count=1)
    if n == 0:
        return False
    if new_text == text:
        return False
    qd_path.write_text(new_text, encoding="utf-8")
    return True


def render_chapter_qd(
    md_path: Path,
    chapter_title: str,
    chapters: list[tuple[Path, str]],
) -> str:
    """Rend un .qd placeholder : preambule + sidebar + copie brute du .md."""
    sidebar = render_sidebar(chapters)
    md_content = md_path.read_text(encoding="utf-8")
    # Strip le H1 du .md pour eviter le doublon avec docname
    md_content_stripped = CHAPTER_TITLE_RE.sub("", md_content, count=1).lstrip()

    return f""".doctype {{docs}}
.docname {{{chapter_title}}}
.docauthor {{mastering-believe}}
.doclang {{fr}}
.theme {{paperwhite}} layout:{{hyperlegible}}

{sidebar}

# {chapter_title}

{md_content_stripped}
"""


def scaffold(domain: str, force: bool) -> None:
    domain_dir = REPO_ROOT / "domains" / domain
    if not domain_dir.is_dir():
        sys.exit(f"error: domain '{domain}' not found at {domain_dir}")

    theory_dir = domain_dir / "01-theory"
    if not theory_dir.is_dir():
        sys.exit(f"error: no 01-theory/ in {domain_dir}")

    qd_dir = domain_dir / "01-theory-qd"
    qd_dir.mkdir(exist_ok=True)

    mds = list_chapter_mds(theory_dir)
    if not mds:
        sys.exit(f"error: no chapter .md files (NN-xxx.md) in {theory_dir}")

    chapters: list[tuple[Path, str]] = [
        (md, extract_chapter_title(md)) for md in mds
    ]

    domain_title = extract_domain_title(domain_dir)

    # main.qd : toujours regenere (c'est l'index)
    main_qd = qd_dir / "main.qd"
    main_qd.write_text(render_main_qd(domain_title, chapters), encoding="utf-8")
    print(f"  wrote {main_qd.relative_to(REPO_ROOT)}")

    # Bloc sidebar partage par tous les .qd du domaine
    sidebar_block = render_sidebar(chapters)

    # Chaque chapitre : skip contenu si existe (sauf --force), mais
    # toujours synchroniser la sidebar pour que main.qd et chapitres
    # restent coherents quand on ajoute/retire des chapitres.
    written = 0
    skipped = 0
    sidebar_updated = 0
    for md_path, title in chapters:
        qd_path = qd_dir / md_path.with_suffix(".qd").name
        if qd_path.exists() and not force:
            if update_sidebar_in_existing(qd_path, sidebar_block):
                sidebar_updated += 1
                print(f"  synced sidebar in {qd_path.relative_to(REPO_ROOT)}")
            skipped += 1
            continue
        qd_path.write_text(render_chapter_qd(md_path, title, chapters), encoding="utf-8")
        print(f"  wrote {qd_path.relative_to(REPO_ROOT)}")
        written += 1

    print(
        f"\ndone: {written} placeholder(s) written, {skipped} skipped "
        f"({sidebar_updated} sidebar sync'd, use --force to overwrite content)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("domain", help="Domain name (e.g. agentic-ai)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing .qd files (preserves nothing — destructive)",
    )
    args = parser.parse_args()
    scaffold(args.domain, args.force)


if __name__ == "__main__":
    main()
