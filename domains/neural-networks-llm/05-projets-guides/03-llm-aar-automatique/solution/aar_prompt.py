"""
Templates de prompt pour la generation AAR automatique.

Principes :
- System prompt : role, contraintes factuelles, format de sortie
- User prompt : contexte events + question precise
- Few-shot : 2 exemples de paragraphes bien formes pour orienter le style
- Low temperature : 0.2 max pour du factuel

Le prompt est pense pour etre reutilisable avec n'importe quel LLM instructable
(Claude, Llama 3, Mistral). L'absence de features specifiques a un provider
est deliberate : air-gap = portabilite.
"""
from __future__ import annotations

SYSTEM_PROMPT = """Tu es un assistant d'analyse tactique pour un formateur militaire.

Ta mission : rediger un paragraphe d'AAR (After-Action Review) pour un moment cle d'exercice SWORD.

Regles strictes :
1. FACTUALITE : base-toi exclusivement sur les events fournis. Si une information manque, ecris "non documente". N'invente jamais un fait.
2. CITATIONS : chaque affirmation doit etre suivie de la citation de l'event source au format [ev:<id>]. Si une affirmation regroupe plusieurs events, cite-les tous : [ev:12, ev:13].
3. STYLE : phrases courtes (< 20 mots), vocabulaire militaire standard OTAN, ton neutre. Pas de jugement de valeur ("brillant", "decevant") sauf dans la section "Recommandation".
4. FORMAT : 3 sections : **Contexte** (ce qui precede), **Deroulement** (ce qui se passe), **Recommandation** (1-2 phrases orientees formation).
5. LONGUEUR : 150 a 250 mots par moment cle.
6. Si les events ne permettent pas de reconstruire un narratif coherent, ecris "Moment non reconstructible a partir des events fournis" et stop.
"""

FEW_SHOT_EXAMPLE = """EXEMPLE :

Events :
[ev:1001] t=0h42m10s DETECT Alpha-3 observes OPFOR vehicle at grid 204-718
[ev:1002] t=0h42m12s ORDER Bravo-HQ to Alpha-3: engage if possible
[ev:1003] t=0h42m30s FIRE Alpha-3 anti-tank launch
[ev:1004] t=0h42m32s IMPACT OPFOR vehicle neutralized
[ev:1005] t=0h42m45s ORDER Bravo-HQ to Alpha-3: reposition north 200m

AAR paragraphe :

**Contexte**
Alpha-3 est en posture d'observation en 204-716 depuis 0h38 [ev:1001]. Aucun contact prealable.

**Deroulement**
A 0h42m10s, Alpha-3 detecte un vehicule OPFOR a 200m en 204-718 [ev:1001]. Bravo-HQ ordonne l'engagement si possible a 0h42m12s [ev:1002]. L'unite engage a l'anti-char a 0h42m30s [ev:1003]. Impact confirme et cible neutralisee a 0h42m32s [ev:1004]. Bravo-HQ ordonne le repositionnement nord de 200m a 0h42m45s [ev:1005].

**Recommandation**
Execution conforme aux ROE, delai decision-engagement 20s, acceptable. Travailler le repositionnement immediat post-tir en zone hostile pour reduire la fenetre de riposte.

FIN DE L'EXEMPLE
"""


def build_user_prompt(
    moment_description: str,
    event_lines: list[str],
    focus_unit: str,
) -> str:
    events_str = "\n".join(event_lines)
    return f"""{FEW_SHOT_EXAMPLE}

MAINTENANT, genere l'AAR pour le moment suivant :

Moment cle : {moment_description}
Unite pivot : {focus_unit}

Events du contexte :
{events_str}

Redige le paragraphe d'AAR en suivant strictement le format de l'exemple."""


def format_event_line(event: dict) -> str:
    """Formate un event dict pour insertion dans un prompt."""
    t = event["t_sim"]
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    payload = event.get("payload", {})
    payload_str = " ".join(f"{k}={v}" for k, v in payload.items()) if payload else ""
    return f"[ev:{event['id']}] t={h}h{m:02d}m{s:02d}s {event['kind']} {event['unit_id']} {payload_str}".strip()
