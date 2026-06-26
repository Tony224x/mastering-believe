#!/usr/bin/env python3
"""Verifie que les liens relatifs des fichiers Markdown pointent vers une cible existante.

Garde-fou anti-regression : le reorg en 3 tracks (domains/<domaine> ->
domains/<track>/<domaine>) avait decale la profondeur de dizaines de liens
relatifs sans les reecrire. Ce script aurait fait echouer la CI sur la PR
fautive.

Ce qu'il verifie :
  - tous les `.md` SUIVIS PAR GIT (donc pas les zones gitignore : tasks/, etc.) ;
  - uniquement les liens RELATIFS (`./` ou `../`, ou chemin nu sans schema) ;
  - cible resolue depuis le dossier du fichier, ancre `#...` et `?query` retirees ;
  - existence d'un fichier OU d'un dossier a la cible.

Ce qu'il ignore (a dessein, pour zero faux positif) :
  - liens externes : http(s)://, mailto:, tel:, ftp://, protocol-relative // ;
  - ancres pures (`#section`) ;
  - tout ce qui est dans un bloc de code cloture (``` / ~~~) ou un span de code
    inline (`...`) — sinon les exemples illustratifs casseraient la CI.

stdlib only. Compatible avec build_catalog.py (meme racine repo, meme style).

Usage :
  python shared/tools/check_links.py            # echoue (exit 1) s'il existe un lien casse
  python shared/tools/check_links.py --quiet     # n'affiche que le resume
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

# Racine repo = deux niveaux au-dessus de ce fichier (shared/tools/check_links.py)
REPO = Path(__file__).resolve().parents[2]

# ](target) couvre les liens inline ET les images ![alt](target).
LINK_RE = re.compile(r"\]\(\s*<?([^)>\s]+)>?\s*\)")
# Span de code inline : `code`, ``co`de``, ... (run de backticks symetrique).
INLINE_CODE_RE = re.compile(r"(`+)(.+?)\1")
# Ouverture/fermeture d'un bloc de code cloture (au moins 3 backticks ou tildes).
FENCE_RE = re.compile(r"^(\s*)(`{3,}|~{3,})")

EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "tel:", "ftp://", "//")

# Dossiers de scaffolding : liens-gabarits volontairement non resolus (ex: ](URL)).
EXCLUDE_DIRS = ("shared/templates/",)


def _scrub(content: str) -> list[str]:
    """Renvoie les lignes avec le contenu des blocs/spans de code neutralise.

    Le nombre de lignes est preserve (report file:line fiable). Le contenu code
    est remplace par des espaces pour qu'aucun lien n'y soit detecte.
    """
    out: list[str] = []
    fence: str | None = None  # caractere de cloture courant (``` ou ~~~), None si hors bloc
    for line in content.splitlines():
        m = FENCE_RE.match(line)
        if fence is None:
            if m:
                fence = m.group(2)[0]  # '`' ou '~'
                out.append("")  # la ligne d'ouverture ne contient pas de lien a verifier
                continue
            # hors bloc : neutraliser uniquement les spans de code inline
            out.append(INLINE_CODE_RE.sub(lambda mm: " " * len(mm.group(0)), line))
        else:
            # dans un bloc : une ligne de cloture du meme type ferme le bloc
            if m and m.group(2)[0] == fence:
                fence = None
            out.append("")
    return out


def _is_relative(target: str) -> bool:
    if not target or target.startswith("#"):
        return False
    if target.startswith(EXTERNAL_PREFIXES):
        return False
    return True


def check_file(relpath: str) -> list[tuple[int, str]]:
    """Retourne [(num_ligne, cible_cassee), ...] pour un .md donne (chemin relatif repo)."""
    abspath = REPO / relpath
    content = abspath.read_text(encoding="utf-8")
    fdir = os.path.dirname(relpath)
    broken: list[tuple[int, str]] = []
    for lineno, line in enumerate(_scrub(content), start=1):
        for m in LINK_RE.finditer(line):
            raw = m.group(1).strip()
            if not _is_relative(raw):
                continue
            target = raw.split("#")[0].split("?")[0]
            if not target:  # etait une ancre pure type ?x ou vide
                continue
            resolved = os.path.normpath(os.path.join(fdir, target))
            if not (REPO / resolved).exists():
                broken.append((lineno, raw))
    return broken


def tracked_md() -> list[str]:
    res = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "*.md"],
        capture_output=True, text=True, check=True,
    )
    return sorted(
        p for p in res.stdout.splitlines()
        if p and not p.startswith(EXCLUDE_DIRS)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verifie les liens relatifs des .md suivis par git.")
    parser.add_argument("--quiet", action="store_true", help="n'affiche que le resume")
    args = parser.parse_args(argv)

    files = tracked_md()
    total_broken = 0
    for relpath in files:
        for lineno, raw in check_file(relpath):
            total_broken += 1
            if not args.quiet:
                print(f"{relpath}:{lineno}: lien relatif casse -> {raw}", file=sys.stderr)

    if total_broken:
        print(
            f"\nECHEC : {total_broken} lien(s) relatif(s) casse(s) dans {len(files)} fichiers .md.",
            file=sys.stderr,
        )
        return 1
    print(f"OK — {len(files)} fichiers .md, aucun lien relatif casse.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
