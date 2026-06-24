#!/usr/bin/env python3
"""Genere le catalogue des domaines de mastering-believe a partir des meta.toml.

Source de verite :
  - meta.toml par domaine -> donnees SEMANTIQUES non-derivables (titre, focus, niveau,
    duree, stack, pilier, garde-fou, prerequis, tags, statut).
  - le filesystem -> faits STRUCTURELS derivables (nb de modules de theorie, presence de
    code/exercices/projets). Calcules en live ici, jamais stockes dans meta.toml.

Sorties :
  - domains/CATALOG.md            (inventaire riche, groupe par track)
  - README.md (racine)            (tables resumees, injectees entre marqueurs)

Usage :
  python shared/tools/build_catalog.py            # ecrit CATALOG.md + injecte dans README
  python shared/tools/build_catalog.py --check     # CI : echoue si stale / meta manquant / incoherent

stdlib only (tomllib requiert Python 3.11+).
"""
from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

# Racine repo = deux niveaux au-dessus de ce fichier (shared/tools/build_catalog.py)
REPO = Path(__file__).resolve().parents[2]
DOMAINS = REPO / "domains"
CATALOG = DOMAINS / "CATALOG.md"
README = REPO / "README.md"

# Ordre d'affichage des tracks + libelles
TRACKS = [
    ("tech", "Track Tech — maitrise d'ingenierie"),
    ("vie", "Track Vie — l'ecole de la vie"),
    ("exploratoire", "Exploratoire — ajouts sous cadrage adverse"),
]

STATUS_LABEL = {
    "stable": "stable",
    "wip": "WIP",
    "exploratoire": "exploratoire",
    "draft": "draft",
}

REQUIRED = ["slug", "title", "track", "status", "level", "duration", "focus"]
VALID_STATUS = set(STATUS_LABEL)
VALID_LEVEL = {"debutant", "intermediaire", "avance"}

CATALOG_MARKER_START = "<!-- CATALOG:START -->"
CATALOG_MARKER_END = "<!-- CATALOG:END -->"


class DomainError(Exception):
    pass


def _count_theory_modules(domain_dir: Path) -> int:
    theory = domain_dir / "01-theory"
    if not theory.is_dir():
        return 0
    return sum(1 for p in theory.glob("*.md") if p.is_file())


def _has_content(domain_dir: Path, sub: str, pattern: str = "*") -> bool:
    d = domain_dir / sub
    if not d.is_dir():
        return False
    return any(p.is_file() for p in d.rglob(pattern))


def discover() -> list[dict]:
    """Scanne domains/<track>/<slug>/meta.toml et fusionne meta + faits structurels."""
    domains: list[dict] = []
    errors: list[str] = []
    valid_tracks = {t for t, _ in TRACKS}

    for track in sorted(valid_tracks):
        track_dir = DOMAINS / track
        if not track_dir.is_dir():
            continue
        for domain_dir in sorted(p for p in track_dir.iterdir() if p.is_dir()):
            meta_path = domain_dir / "meta.toml"
            rel = domain_dir.relative_to(REPO).as_posix()
            if not meta_path.is_file():
                errors.append(f"{rel}/ : meta.toml manquant")
                continue
            try:
                with meta_path.open("rb") as fh:
                    meta = tomllib.load(fh)
            except (tomllib.TOMLDecodeError, OSError) as exc:
                errors.append(f"{rel}/meta.toml : illisible ({exc})")
                continue

            for field in REQUIRED:
                if not meta.get(field):
                    errors.append(f"{rel}/meta.toml : champ requis manquant ou vide : {field}")
            if meta.get("slug") != domain_dir.name:
                errors.append(
                    f"{rel}/meta.toml : slug={meta.get('slug')!r} != dossier {domain_dir.name!r}"
                )
            if meta.get("track") != track:
                errors.append(
                    f"{rel}/meta.toml : track={meta.get('track')!r} != dossier parent {track!r}"
                )
            if meta.get("status") not in VALID_STATUS:
                errors.append(f"{rel}/meta.toml : status invalide : {meta.get('status')!r}")
            if meta.get("level") not in VALID_LEVEL:
                errors.append(f"{rel}/meta.toml : level invalide : {meta.get('level')!r}")

            meta.setdefault("stack", [])
            meta.setdefault("pillar", "")
            meta.setdefault("guardrail", "")
            meta.setdefault("prerequisites", [])
            meta.setdefault("tags", [])
            # Faits structurels derives (live, jamais stockes)
            meta["_modules"] = _count_theory_modules(domain_dir)
            meta["_has_code"] = _has_content(domain_dir, "02-code", "*.py")
            meta["_has_exercises"] = (domain_dir / "03-exercises").is_dir()
            meta["_has_projects"] = _has_content(domain_dir, "04-projects")
            meta["_has_projets_guides"] = _has_content(domain_dir, "05-projets-guides")
            meta["_path"] = domain_dir.relative_to(REPO).as_posix()
            domains.append(meta)

    if errors:
        raise DomainError("\n".join(errors))
    return domains


def _md_escape(text: str) -> str:
    return str(text).replace("|", "\\|")


def _link(d: dict, base: str = "") -> str:
    # _path est relatif a la racine repo (ex: domains/tech/agentic-ai).
    # base="domains" pour CATALOG.md (qui vit dans domains/) -> lien relatif au dossier ;
    # base="" pour le README racine -> lien relatif a la racine.
    path = d["_path"]
    if base:
        prefix = base + "/"
        if path.startswith(prefix):
            path = path[len(prefix):]
    return f"[{_md_escape(d['title'])}](./{path}/)"


def _slug_title_map(domains: list[dict]) -> dict[str, str]:
    return {d["slug"]: d["title"] for d in domains}


def render_catalog(domains: list[dict]) -> str:
    titles = _slug_title_map(domains)
    lines: list[str] = []
    lines.append("# Catalogue des domaines")
    lines.append("")
    lines.append(
        "> Fichier **genere** par `shared/tools/build_catalog.py` — ne pas editer a la main. "
        "Les metadonnees vivent dans `domains/<track>/<domaine>/meta.toml`."
    )
    lines.append("")
    total = len(domains)
    by_track = {t: [d for d in domains if d["track"] == t] for t, _ in TRACKS}
    counts = " · ".join(f"{t} : {len(by_track[t])}" for t, _ in TRACKS)
    lines.append(f"**{total} domaines** — {counts}.")
    lines.append("")

    for track, label in TRACKS:
        ds = by_track.get(track, [])
        if not ds:
            continue
        lines.append(f"## {label}")
        lines.append("")
        is_vie = track == "vie"
        head = ["Domaine"]
        if is_vie:
            head.append("Pilier")
        head += ["Niveau", "Duree", "Modules", "Code", "Stack / Focus", "Statut"]
        lines.append("| " + " | ".join(head) + " |")
        lines.append("|" + "|".join(["---"] * len(head)) + "|")
        for d in sorted(ds, key=lambda x: x["slug"]):
            stack = ", ".join(d["stack"]) if d["stack"] else "—"
            sf = f"{stack} · {_md_escape(d['focus'])}" if d["stack"] else _md_escape(d["focus"])
            row = [_link(d, base="domains")]
            if is_vie:
                row.append(_md_escape(d.get("pillar") or "—"))
            row += [
                d["level"],
                _md_escape(d["duration"]),
                str(d["_modules"]),
                "oui" if d["_has_code"] else "non",
                sf,
                STATUS_LABEL.get(d["status"], d["status"]),
            ]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        # Detail prerequis / garde-fous sous chaque track si presents
        for d in sorted(ds, key=lambda x: x["slug"]):
            extras = []
            if d["prerequisites"]:
                pres = ", ".join(titles.get(s, s) for s in d["prerequisites"])
                extras.append(f"prerequis : {pres}")
            if d.get("guardrail"):
                extras.append(f"garde-fou : {d['guardrail']}")
            if extras:
                lines.append(f"- **{_md_escape(d['title'])}** — " + " · ".join(extras))
        if any(d["prerequisites"] or d.get("guardrail") for d in ds):
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_readme_block(domains: list[dict]) -> str:
    """Bloc resume injecte dans le README racine (entre marqueurs)."""
    by_track = {t: [d for d in domains if d["track"] == t] for t, _ in TRACKS}
    lines: list[str] = []
    for track, label in TRACKS:
        ds = by_track.get(track, [])
        if not ds:
            continue
        is_vie = track == "vie"
        lines.append(f"**{label}** :")
        lines.append("")
        head = ["Domaine"]
        head.append("Pilier" if is_vie else "Stack")
        head += ["Focus", "Duree"]
        lines.append("| " + " | ".join(head) + " |")
        lines.append("|" + "|".join(["---"] * len(head)) + "|")
        for d in sorted(ds, key=lambda x: x["slug"]):
            col2 = (d.get("pillar") or "—") if is_vie else (", ".join(d["stack"]) if d["stack"] else "—")
            row = [_link(d), _md_escape(col2), _md_escape(d["focus"]), _md_escape(d["duration"])]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    lines.append("> Inventaire complet (modules, prerequis, garde-fous, statuts) : [`domains/CATALOG.md`](./domains/CATALOG.md).")
    return "\n".join(lines).strip() + "\n"


def inject_readme(block: str, *, write: bool) -> bool | str:
    """Injecte le bloc entre les marqueurs du README. Retourne True si a jour,
    le nouveau contenu si write=False et stale, ou leve si marqueurs absents."""
    text = README.read_text(encoding="utf-8")
    if CATALOG_MARKER_START not in text or CATALOG_MARKER_END not in text:
        raise DomainError(
            f"README.md : marqueurs {CATALOG_MARKER_START} / {CATALOG_MARKER_END} absents"
        )
    pre, rest = text.split(CATALOG_MARKER_START, 1)
    _, post = rest.split(CATALOG_MARKER_END, 1)
    new = f"{pre}{CATALOG_MARKER_START}\n{block}{CATALOG_MARKER_END}{post}"
    if new == text:
        return True
    if write:
        README.write_text(new, encoding="utf-8")
        return True
    return new


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Genere le catalogue des domaines.")
    parser.add_argument("--check", action="store_true", help="CI : echoue si stale/incoherent, n'ecrit rien")
    args = parser.parse_args(argv)

    try:
        domains = discover()
    except DomainError as exc:
        print("ERREUR de coherence des domaines :\n" + str(exc), file=sys.stderr)
        return 1

    catalog_md = render_catalog(domains)
    readme_block = render_readme_block(domains)

    if args.check:
        stale = []
        current_catalog = CATALOG.read_text(encoding="utf-8") if CATALOG.is_file() else ""
        if current_catalog != catalog_md:
            stale.append("domains/CATALOG.md")
        try:
            if inject_readme(readme_block, write=False) is not True:
                stale.append("README.md (bloc CATALOG)")
        except DomainError as exc:
            print(f"ERREUR : {exc}", file=sys.stderr)
            return 1
        if stale:
            print(
                "STALE — regenerer avec `python shared/tools/build_catalog.py` :\n  - "
                + "\n  - ".join(stale),
                file=sys.stderr,
            )
            return 1
        print(f"OK — catalogue a jour ({len(domains)} domaines).")
        return 0

    CATALOG.write_text(catalog_md, encoding="utf-8")
    inject_readme(readme_block, write=True)
    print(f"Catalogue ecrit : {CATALOG.relative_to(REPO).as_posix()} ({len(domains)} domaines).")
    print("README.md : bloc CATALOG injecte.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
