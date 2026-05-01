"""
Templates de prompt pour la generation EOD Review automatique LogiSim.

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

SYSTEM_PROMPT = """Tu es un assistant d'analyse EOD pour un superviseur OCC d'entrepot.

Ta mission : rediger un paragraphe d'EOD (End-of-Day Review) pour un moment cle de shift FleetSim.

Regles strictes :
1. FACTUALITE : base-toi exclusivement sur les events fournis. Si une information manque, ecris "non documente". N'invente jamais un fait.
2. CITATIONS : chaque affirmation doit etre suivie de la citation de l'event source au format [ev:<id>]. Si une affirmation regroupe plusieurs events, cite-les tous : [ev:12, ev:13].
3. STYLE : phrases courtes (< 20 mots), vocabulaire operationnel logistique, ton neutre. Pas de jugement de valeur sauf dans la section "Recommandation".
4. FORMAT : 3 sections : **Contexte** (ce qui precede), **Deroulement** (ce qui se passe), **Recommandation** (1-2 phrases orientees ajustement SOP).
5. LONGUEUR : 150 a 250 mots par moment cle.
6. Si les events ne permettent pas de reconstruire un narratif coherent, ecris "Moment non reconstructible a partir des events fournis" et stop.
"""

FEW_SHOT_EXAMPLE = """EXEMPLE :

Events :
[ev:1001] t=0h42m10s DETECT AGV-Alpha-3 observes parcel PCL-44 at zone B-04-NE confidence=0.92
[ev:1002] t=0h42m12s ORDER OCC to AGV-Alpha-3: pickup if slot free
[ev:1003] t=0h42m30s PICKUP AGV-Alpha-3 parcel_id=PCL-44 from_slot=B04
[ev:1004] t=0h42m32s DROPOFF AGV-Alpha-3 to_slot=L4-2 ok=True
[ev:1005] t=0h42m45s ORDER OCC to AGV-Alpha-3: reposition zone B-08

EOD paragraphe :

**Contexte**
AGV-Alpha-3 est en patrouille zone B-04 depuis 0h38 [ev:1001]. Aucun pickup actif.

**Deroulement**
A 0h42m10s, AGV-Alpha-3 detecte le colis PCL-44 en zone B-04-NE avec une confiance elevee (0.92) [ev:1001]. L'OCC autorise le pickup conditionnel a 0h42m12s [ev:1002]. L'unite execute le pickup a 0h42m30s [ev:1003], puis depose en slot L4-2 sans erreur a 0h42m32s [ev:1004]. L'OCC redirige ensuite l'AGV vers la zone B-08 a 0h42m45s [ev:1005].

**Recommandation**
Cycle de pickup/dropoff conforme au SOP, delai detection-pickup 20s, acceptable. Investiguer la possibilite de pre-positionner Alpha-3 plus pres de B-04 en debut de shift pour reduire ce delai.

FIN DE L'EXEMPLE
"""


def build_user_prompt(
    moment_description: str,
    event_lines: list[str],
    focus_unit: str,
) -> str:
    events_str = "\n".join(event_lines)
    return f"""{FEW_SHOT_EXAMPLE}

MAINTENANT, genere l'EOD pour le moment suivant :

Moment cle : {moment_description}
Unite pivot : {focus_unit}

Events du contexte :
{events_str}

Redige le paragraphe d'EOD en suivant strictement le format de l'exemple."""


def format_event_line(event: dict) -> str:
    """Formate un event dict pour insertion dans un prompt."""
    t = event["t_sim"]
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    payload = event.get("payload", {})
    payload_str = " ".join(f"{k}={v}" for k, v in payload.items()) if payload else ""
    return f"[ev:{event['id']}] t={h}h{m:02d}m{s:02d}s {event['kind']} {event['unit_id']} {payload_str}".strip()
