#!/usr/bin/env python3
"""Capstone checklist validator for HEC-style deck structure (stdlib).
Does not create a .pptx binary; validates a JSON-like outline the learner can export.
requires: stdlib only
"""
from __future__ import annotations

def validate_outline(slides: list[dict]) -> list[str]:
    errors: list[str] = []
    n = len(slides)
    if n < 8 or n > 12:
        errors.append(f"slide_count={n} not in 8..12")
    for i, s in enumerate(slides, 1):
        title = (s.get("title") or "").strip()
        bullets = s.get("bullets") or []
        if not title:
            errors.append(f"slide {i}: empty title")
        if len(bullets) > 3:
            errors.append(f"slide {i}: more than 3 bullets")
        for b in bullets:
            if len(str(b).split()) > 15:
                errors.append(f"slide {i}: bullet too long (>{15} words)")
    return errors

if __name__ == "__main__":
    sample = [
        {"title": f"Slide {i} conclusion claire", "bullets": ["Point A", "Point B"]}
        for i in range(1, 11)
    ]
    err = validate_outline(sample)
    assert err == [], err
    bad = sample[:5]
    assert validate_outline(bad), "should fail short deck"
    print("OK capstone outline validator")
    print("Primary learner tool: ChatGPT + PowerPoint (not this script)")
