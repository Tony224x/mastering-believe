"""
Jour 22 — Structured outputs & constrained generation
======================================================
Pure Python. On implemente un vrai constrained decoder (char-level FSA)
pour un sous-ensemble de JSON schema :
  { "category": "A"|"B"|"C", "score": <integer> }

Comparaison :
  1. "Prompt only" (le modele peut derailler)
  2. "JSON mode" (valide JSON, retry si invalide, mais pas de garantie schema)
  3. Constrained decoding (100% conforme)

Run : python 02-code/22-structured-outputs-constrained-generation.py
"""

from __future__ import annotations
import sys, io, json, random

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

random.seed(22)

ALPHABET = list('{}[]":,. _-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')


# ============================================================================
# 1) Fake LLM token-par-token (ici les "tokens" = chars pour simplifier)
# ============================================================================


class FakeLLM:
    def __init__(self, script: str, error_rate: float = 0.1):
        self.script = script
        self.idx = 0
        self.error_rate = error_rate

    def next_logits(self) -> dict[str, float]:
        """Haut logit pour le char voulu, avec proba d'erreur sinon."""
        logits = {c: random.gauss(-4, 1) for c in ALPHABET}
        if self.idx < len(self.script):
            target = self.script[self.idx]
            logits[target] = 6.0
            if random.random() < self.error_rate:
                # Le modele se trompe : on abaisse le target et on remonte
                # un autre char plausible a sa place.
                logits[target] -= 15
                wrong = random.choice([c for c in ALPHABET if c != target])
                logits[wrong] += 7

        # Le modele peut aussi mettre un EOS prematuré (fin)
        return logits

    @staticmethod
    def sample(logits: dict[str, float]) -> str:
        items = list(logits.items())
        max_l = max(v for _, v in items)
        exp = [(c, pow(2.718, v - max_l)) for c, v in items]
        s = sum(e for _, e in exp)
        r = random.random() * s
        for c, e in exp:
            r -= e
            if r <= 0:
                return c
        return items[-1][0]

    def consume(self, token: str):
        if self.idx < len(self.script) and self.script[self.idx] == token:
            self.idx += 1


# ============================================================================
# 2) Mode 1 : free sampling (prompt-only)
# ============================================================================


def generate_free(script: str, error_rate: float = 0.1, max_len: int = 100) -> str:
    llm = FakeLLM(script, error_rate)
    out = []
    for _ in range(max_len):
        logits = llm.next_logits()
        tok = llm.sample(logits)
        out.append(tok)
        llm.consume(tok)
        # Stop naif sur le } final (ce qu'un vrai LLM ferait via EOS token)
        if "".join(out).count('{') > 0 and "".join(out).count('}') >= "".join(out).count('{'):
            break
    return "".join(out)


# ============================================================================
# 3) Mode 2 : JSON mode simulé (retry x3)
# ============================================================================


def generate_json_mode(script: str, error_rate: float = 0.1, max_retries: int = 3) -> str:
    best = ""
    for _ in range(max_retries):
        out = generate_free(script, error_rate)
        best = out
        try:
            json.loads(out)
            return out
        except Exception:
            continue
    return best


# ============================================================================
# 4) Mode 3 : Constrained decoding avec une FSA simple
# ============================================================================
# La FSA est modelisee comme une liste de "slots" : chaque slot a un set de
# chars autorises, plus optionnellement une transition vers un autre slot
# apres un match. C'est une version simplifiee de ce que fait xgrammar.


def build_json_fsa():
    """FSA pour { "category": "A"|"B"|"C", "score": <digits> }.
    On genere une suite d'etapes, chacune decrit les chars permis a l'etape.
    """
    steps = []

    def lit(s):
        """Ajoute une litterale char par char."""
        for ch in s:
            steps.append(("lit", {ch}))

    lit('{"category":"')
    # Choix dans un enum : un seul char parmi A/B/C
    steps.append(("enum", {"A", "B", "C"}))
    lit('","score":')
    # Un ou plusieurs chiffres : premier chiffre obligatoire
    steps.append(("digit_first", set("0123456789")))
    # Chiffres suivants OU fermeture directe
    steps.append(("digit_or_close", set("0123456789}")))
    # Si on a eu un chiffre de plus, on reste ici tant qu'on a des chiffres
    # sinon on ferme. Pour simplifier on repete.
    for _ in range(3):
        steps.append(("digit_or_close", set("0123456789}")))
    return steps


def generate_constrained(script: str, error_rate: float = 0.1) -> tuple[str, int]:
    fsa = build_json_fsa()
    llm = FakeLLM(script, error_rate)
    out = []
    i = 0
    masked_events = 0

    while i < len(fsa):
        kind, allowed = fsa[i]

        logits = llm.next_logits()
        # Masquer tout ce qui n'est pas dans 'allowed'
        masked_logits = {c: (v if c in allowed else -1e9) for c, v in logits.items()}
        # Si le LLM voulait produire un char interdit, on le compte comme masque
        top = max(logits.items(), key=lambda x: x[1])[0]
        if top not in allowed:
            masked_events += 1

        tok = llm.sample(masked_logits)
        out.append(tok)
        llm.consume(tok)

        # Transitions
        if kind == "digit_or_close":
            if tok == '}':
                break  # on ferme, on a fini le JSON
            else:
                # on reste dans le meme etat "digit_or_close" jusqu'a }
                i += 1
                continue
        i += 1

    # Si on n'a pas ferme, ajouter un } (le contraint decoder peut forcer l'EOS)
    if not out or out[-1] != '}':
        out.append('}')
    return "".join(out), masked_events


# ============================================================================
# 5) Evaluation
# ============================================================================

SCRIPT = '{"category":"B","score":42}'
SCHEMA = {"category": ("A", "B", "C"), "score": "int"}


def check(out: str) -> tuple[bool, bool]:
    """Retourne (valid_json, valid_schema)."""
    try:
        data = json.loads(out)
    except Exception:
        return False, False
    schema_ok = (
        isinstance(data, dict)
        and data.get("category") in SCHEMA["category"]
        and isinstance(data.get("score"), int)
    )
    return True, schema_ok


print("=" * 70)
print("Comparaison : prompt-only vs JSON mode vs CONSTRAINED (200 runs)")
print("=" * 70)
print(f"  Target output : {SCRIPT}")
print(f"  Schema        : category in (A,B,C), score int\n")

TRIALS = 200
ERROR = 0.08

for label, fn in [
    ("Prompt-only", lambda: generate_free(SCRIPT, error_rate=ERROR)),
    ("JSON mode (retry x3)", lambda: generate_json_mode(SCRIPT, error_rate=ERROR)),
    ("Constrained decoding", lambda: generate_constrained(SCRIPT, error_rate=ERROR)[0]),
]:
    vj = vs = 0
    for _ in range(TRIALS):
        out = fn()
        j, s = check(out)
        vj += int(j); vs += int(s)
    print(f"  {label:<30}  json_valid={vj / TRIALS:.2%}  schema_valid={vs / TRIALS:.2%}")

print("\n  Exemple d'output constrained :")
out, masked = generate_constrained(SCRIPT, error_rate=ERROR)
print(f"    {out}")
print(f"    evenements de masquage (tokens interdits rejetes) : {masked}")

print("""
Lecons :
  - Prompt-only : le modele derive, erreurs de structure amplifient le %
    d'outputs invalides. En prod ce taux est plus faible mais non-nul.
  - JSON mode avec retry : filtre une partie des invalides, mais le retry
    coute en cost/latence et ne garantit pas le schema.
  - Constrained decoding : 100% valide (structure ET schema), aucun retry.
    Ici on voit qu'il y a ete N evenements de masquage = autant de fois ou
    le modele aurait derive sans la contrainte.

Stack reelle 2026 :
  - OpenAI     : response_format: {"type": "json_schema", "strict": true}
  - Anthropic  : tool calling natif avec input_schema
  - Gemini     : response_schema + responseMimeType "application/json"
  - Self-host  : vLLM / SGLang + xgrammar ou llguidance
  - Local/edge : llama.cpp avec GBNF grammar ou json_schema
""")
