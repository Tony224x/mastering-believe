"""Primitives SVG Teal Trust (palette + formes)."""
from __future__ import annotations

# Palette Teal Trust (svg-pedagogique.md)
P = {
    "primary": "#0F766E",
    "primary_deep": "#115E59",
    "accent": "#F59E0B",
    "danger": "#DC2626",
    "surface": "#F8FAFC",
    "card": "#FFFFFF",
    "ink": "#0F172A",
    "muted": "#64748B",
    "line": "#E2E8F0",
    "teal_soft": "#CCFBF1",
    "amber_soft": "#FEF3C7",
    "red_soft": "#FEE2E2",
    "slate": "#94A3B8",
}

FONT = "Inter, ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif"


def esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def header_defs() -> str:
    return f"""  <defs>
    <filter id="s" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="4" stdDeviation="8" flood-color="#0F172A" flood-opacity="0.08"/>
    </filter>
    <linearGradient id="gHead" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="{P["primary"]}"/>
      <stop offset="100%" stop-color="#14B8A6"/>
    </linearGradient>
    <marker id="arr" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
      <path d="M0,0 L9,3 L0,6 Z" fill="{P["slate"]}"/>
    </marker>
    <marker id="arrT" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
      <path d="M0,0 L9,3 L0,6 Z" fill="{P["primary"]}"/>
    </marker>
    <marker id="arrA" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
      <path d="M0,0 L9,3 L0,6 Z" fill="{P["accent"]}"/>
    </marker>
  </defs>
  <rect width="1200" height="680" fill="{P["surface"]}"/>
  <rect x="0" y="0" width="8" height="680" fill="url(#gHead)"/>"""


def title_block(title: str, subtitle: str, footer: str) -> str:
    return f"""  <text x="56" y="48" font-family="{FONT}" font-size="26" font-weight="700" fill="{P["ink"]}">{esc(title)}</text>
  <text x="56" y="80" font-family="{FONT}" font-size="15" fill="{P["muted"]}">{esc(subtitle)}</text>
  <text x="56" y="660" font-family="{FONT}" font-size="12" fill="{P["muted"]}">{esc(footer)}</text>"""


def card(x: int, y: int, w: int, h: int, fill: str | None = None) -> str:
    fill = fill or P["card"]
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="{fill}" filter="url(#s)"/>'


def card_header(x: int, y: int, w: int, label: str, bg: str | None = None) -> str:
    bg = bg or P["primary"]
    return f"""  <rect x="{x}" y="{y}" width="{w}" height="48" rx="18" fill="{bg}"/>
  <rect x="{x}" y="{y + 24}" width="{w}" height="24" fill="{bg}"/>
  <text x="{x + w // 2}" y="{y + 32}" text-anchor="middle" font-family="{FONT}" font-size="16" font-weight="700" fill="#fff">{esc(label)}</text>"""


def wrap_svg(title: str, desc: str, body: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="680" viewBox="0 0 1200 680"
     role="img" aria-labelledby="t d">
  <title id="t">{esc(title)}</title>
  <desc id="d">{esc(desc)}</desc>
{header_defs()}
{body}
</svg>
"""


def node_box(
    x: int,
    y: int,
    w: int,
    h: int,
    label: str,
    sub: str = "",
    bg: str | None = None,
    ink: str | None = None,
) -> str:
    bg = bg or P["card"]
    ink = ink or P["ink"]
    lines = [f"  {card(x, y, w, h, bg)}"]
    if sub:
        lines.append(
            f'  <text x="{x + w // 2}" y="{y + h // 2 - 4}" text-anchor="middle" font-family="{FONT}" font-size="16" font-weight="700" fill="{ink}">{esc(label)}</text>'
        )
        lines.append(
            f'  <text x="{x + w // 2}" y="{y + h // 2 + 18}" text-anchor="middle" font-family="{FONT}" font-size="13" fill="{P["muted"]}">{esc(sub)}</text>'
        )
    else:
        lines.append(
            f'  <text x="{x + w // 2}" y="{y + h // 2 + 6}" text-anchor="middle" font-family="{FONT}" font-size="16" font-weight="700" fill="{ink}">{esc(label)}</text>'
        )
    return "\n".join(lines)


def arrow_h(
    x1: int, y: int, x2: int, color: str | None = None, marker: str = "arrT"
) -> str:
    color = color or P["primary"]
    return f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{color}" stroke-width="2.5" marker-end="url(#{marker})"/>'


def arrow_v(
    x: int, y1: int, y2: int, color: str | None = None, marker: str = "arrT"
) -> str:
    color = color or P["primary"]
    return f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="{color}" stroke-width="2.5" marker-end="url(#{marker})"/>'


def pill(x: int, y: int, w: int, h: int, label: str, bg: str, ink: str = "#fff") -> str:
    return f"""  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{h // 2}" fill="{bg}"/>
  <text x="{x + w // 2}" y="{y + h // 2 + 5}" text-anchor="middle" font-family="{FONT}" font-size="14" font-weight="600" fill="{ink}">{esc(label)}</text>"""


# ---------------------------------------------------------------------------
# Specs: filename, module footer, title, subtitle, desc, builder
# ---------------------------------------------------------------------------


