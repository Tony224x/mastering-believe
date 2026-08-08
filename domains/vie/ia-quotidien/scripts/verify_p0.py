#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Structural verification of P0 public packaging for ia-quotidien.

Run from repo root:
  python domains/vie/ia-quotidien/scripts/verify_p0.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

DOMAIN = Path(__file__).resolve().parents[1]
DAYS = [
    "01-ia-sans-panique",
    "02-prompts-qui-marchent",
    "03-hallucinations-verification",
    "04-partenaire-reflexion",
    "05-ecrire-avec-ia",
    "06-excel-bases-ia",
    "07-formules-tableaux",
    "08-nettoyer-analyser",
    "09-projet-tresorerie",
    "10-structure-pitch",
    "11-slides-visuels",
    "12-notes-orateur",
    "13-capstone-brouillon",
    "14-capstone-deck-pitch",
]


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    raise SystemExit(1)


def main() -> None:
    # 1) no internal review artifacts / pycache
    for name in ("CODEX-REVIEW.md", "REVIEW-pass1.md", "REVIEW-pass2.md"):
        if (DOMAIN / name).exists():
            fail(f"public artifact still present: {name}")
    pycaches = list(DOMAIN.rglob("__pycache__"))
    if pycaches:
        fail(f"__pycache__ still present: {pycaches}")

    # 2) no hec slug on learner path (EVAL historical allowed)
    hec_hits = []
    # needle built so this file does not match its own source
    bad_slug = "capstone-deck-" + "hec"
    for p in DOMAIN.rglob("*"):
        if not p.is_file():
            continue
        if p.name in {"EVAL-dual-agents.md", "verify_p0.py"}:
            continue
        if "scripts" in p.parts and p.name.startswith("verify_"):
            continue
        if p.suffix not in {".md", ".py", ".toml", ".svg"}:
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        if bad_slug in text.lower() or ("14-" + bad_slug) in text.lower():
            hec_hits.append(str(p.relative_to(DOMAIN)))
        if p.name.endswith("hec.md") or p.name.endswith("hec.py"):
            hec_hits.append(str(p.relative_to(DOMAIN)))
    if hec_hits:
        fail(f"hec brand/slug still present: {hec_hits}")
    pitch = DOMAIN / "01-theory" / "14-capstone-deck-pitch.md"
    if not pitch.exists():
        fail("missing renamed theory 14-capstone-deck-pitch.md")

    # 3) README 3-bullet ce soir
    readme = (DOMAIN / "README.md").read_text(encoding="utf-8")
    if "Ce soir, fais seulement" not in readme and "Ce soir, 3 gestes" not in readme:
        fail("README missing « Ce soir » entry")
    # count numbered bullets under that section
    m = re.search(r"Ce soir[^\n]*\n\n(?:[^\n]+\n\n)?((?:\d+\..+\n)+)", readme)
    if not m:
        # fallback: lines "1." "2." "3." after heading
        block = readme.split("Ce soir", 1)[1].split("## ", 1)[0]
        nums = re.findall(r"(?m)^\d+\.", block)
        if len(nums) < 3:
            fail(f"README ce soir needs 3 bullets, found {len(nums)}")
    else:
        nums = re.findall(r"(?m)^\d+\.", m.group(1))
        if len(nums) != 3:
            fail(f"README ce soir expected 3 bullets, got {len(nums)}")

    # 4) capstone min/bonus multi-soir wording
    j14 = (DOMAIN / "01-theory" / "14-capstone-deck-pitch.md").read_text(encoding="utf-8")
    if not re.search(r"minimum|Minimum", j14):
        fail("J14 theory missing minimum wording")
    if not re.search(r"bonus|Bonus", j14):
        fail("J14 theory missing bonus wording")
    if not re.search(r"soir|multi-soir|plusieurs soirs|1 à 2 soirs|1–2 soirs", j14, re.I):
        fail("J14 theory missing multi-soir wording")

    # 5) medium/hard mission or bonus
    for folder in ("02-medium", "03-hard"):
        for day in DAYS:
            p = DOMAIN / "03-exercises" / folder / f"{day}.md"
            if not p.exists():
                fail(f"missing exercise {folder}/{day}.md")
            t = p.read_text(encoding="utf-8")
            ok = bool(re.search(r"(?m)^## (But|À faire|A faire|Mission)", t)) or (
                "bonus" in t.lower()
            )
            if not ok:
                fail(f"{folder}/{day}.md not mission-style and not bonus-labeled")

    # 6) 14 markdown solutions day-themed
    for day in DAYS:
        p = DOMAIN / "03-exercises" / "solutions" / f"{day}.md"
        if not p.exists():
            fail(f"missing solution md {day}.md")
        t = p.read_text(encoding="utf-8").strip()
        if len(t) < 200:
            fail(f"solution md too thin: {day}.md")
        if not t.lower().startswith("# solution"):
            fail(f"solution md not titled as solution: {day}.md")

    # 7) FR polish spot checks
    if "Key takeaway" in readme:
        fail("Key takeaway still in README")
    for needle in ("maîtriser", "réflexion", "critères", "À retenir"):
        blob = readme + (DOMAIN / "01-theory" / "01-ia-sans-panique.md").read_text(encoding="utf-8")
        blob += (DOMAIN / "01-theory" / "04-partenaire-reflexion.md").read_text(encoding="utf-8")
        if needle not in blob:
            # allow if peer form present
            peers = {
                "maîtriser": ["Maîtriser", "maîtrise"],
                "réflexion": ["Réflexion", "réfléchir"],
                "critères": ["Critères", "critère"],
                "À retenir": ["à retenir", "A retenir"],
            }
            if not any(x in blob for x in peers.get(needle, [])):
                fail(f"expected accented form missing: {needle}")

    # confance typo
    for p in DOMAIN.rglob("*.md"):
        if p.name == "EVAL-dual-agents.md":
            continue
        if "confance" in p.read_text(encoding="utf-8", errors="replace"):
            fail(f"typo confance in {p}")

    print("OK verify_p0 — all structural P0 checks passed")
    return None


if __name__ == "__main__":
    main()
