# Jour 22 — Structured outputs & constrained generation

> **Temps estime** : 4h | **Prerequis** : J7 (mini-transformer, sampling), J11 (inference), J16 (agents)

---

## 1. Pourquoi la generation structuree a explose en 2024-2026

Tout AI engineer s'est pris ce bug en 2023 :

```python
output = llm("Return JSON with fields name, age: ...")
data = json.loads(output)  # → JSONDecodeError: Expecting property name...
```

Le modele a mis une virgule en trop, oublie un guillemet, sorti un markdown ```json...```, ajoute du commentaire. Chaque ingenieur a ecrit son parser defensif avec 15 regex. **C'etait la tax la plus insupportable des produits LLM**.

En 2024-2025, trois techniques ont rendu ca un probleme resolu :
1. **Grammar-constrained decoding** (outlines, xgrammar, llama.cpp)
2. **JSON mode / schema-constrained API** (OpenAI, Anthropic, Gemini)
3. **Tool calling natif** avec validation automatique

En 2026, tu **ne dois plus jamais** ecrire un parser JSON defensif pour un output LLM. Si tu le fais, tu utilises mal les outils.

---

## 2. Les trois niveaux de garantie structurelle

| Methode | Garantie | Latence | Support |
|---|---|---|---|
| Prompt "reply in JSON" | ≈ 95% | Aucun overhead | Tous les modeles |
| JSON mode API | 99% | +5% | OpenAI, Anthropic, Gemini, DeepSeek |
| Tool calling API | 99.9% | +0-10% | OpenAI, Anthropic, Gemini, vLLM/SGLang |
| Constrained decoding | **100%** | -5 a +15% | vLLM, SGLang, outlines, llama.cpp |

100% veut dire : **il est mathematiquement impossible** que le modele produise un output invalide. Pas "presque toujours bon" — **toujours** bon.

---

## 3. Comment fonctionne le constrained decoding

### L'idee fondamentale

A chaque step de sampling, le LLM produit des logits pour les ~100k tokens du vocab. **Constrained decoding** masque les tokens qui violeraient la grammaire, puis normalise (softmax sur les tokens autorises).

```python
logits = model(context)             # [vocab_size]
mask = grammar.allowed_tokens(current_state)   # [vocab_size], 0/1
logits = logits + (mask - 1) * 1e9  # -inf sur les tokens interdits
next_token = sample(softmax(logits))
grammar.advance(next_token)         # update state machine
```

### Les deux implementations principales

**Token-level FSA (Finite State Automaton)** — outlines, xgrammar, llguidance.
- On precompile la grammaire en un graphe d'etats
- A chaque token, on sait exactement quels tokens du vocab sont valides
- Overhead : precompilation O(grammaire × vocab), runtime O(1) par token
- Supporte : JSON schema, regex, context-free grammars

**Character-level + retokenization** — plus flexible mais plus lent.
- On check char-par-char contre la grammaire
- Plus facile pour des grammaires dynamiques

En prod 2026 : **xgrammar** (Berkeley, 2024) et **llguidance** (Microsoft, 2024) sont les SOTA. Performance quasi-identique au sampling non-contraint.

### Tokenizer alignment — le cas degenere

Un token dans un LLM moderne peut etre `"Paris,"`, `" { ""`, etc. — pas aligne sur les boundaries de la grammaire. Le moteur doit gerer :
- Un token qui represente plusieurs chars de la grammaire
- Un token qui "commence" un symbole sans le finir
- Backtracking si la grammaire interdit un tokenisation possible

C'est non-trivial a implementer, mais les libs matures (xgrammar) le gerent transparent.

---

## 4. JSON Schema — le format universel

Tous les providers 2026 acceptent une version de JSON schema pour decrire la structure attendue.

### Exemple Anthropic 2026

```python
response = client.messages.create(
    model="claude-4-7",
    messages=[{"role": "user", "content": "Parse this invoice: ..."}],
    tools=[{
        "name": "invoice_extraction",
        "input_schema": {
            "type": "object",
            "properties": {
                "amount": {"type": "number"},
                "currency": {"type": "string", "enum": ["USD", "EUR", "GBP"]},
                "vendor": {"type": "string"},
                "items": {
                    "type": "array",
                    "items": {"type": "object", "properties": {
                        "description": {"type": "string"},
                        "quantity": {"type": "integer"},
                    }}
                }
            },
            "required": ["amount", "currency", "vendor"]
        }
    }],
    tool_choice={"type": "tool", "name": "invoice_extraction"}
)
```

Le modele est **force** de produire un output conforme. Tu parses avec `json.loads` sans try/except, tu lis les champs directement.

### Limitations

- Profondeur max souvent limitee (5-10 niveaux selon provider)
- Regex patterns supportes mais parfois bugges sur les cas edge
- Les grands enums (>100 values) peuvent causer timeout
- Recursion autoreferentielle : souvent limite ou interdite

---

## 5. Quand utiliser quel niveau

### Tool calling natif API

Le defaut pour 90% des cas. Validation serveur-side par le provider, retry automatique si pas conforme, latence negligeable. Utiliser des que tu as une structure > 1 champ.

### JSON mode

Plus simple que tool calling, mais moins strict (le modele peut produire du JSON valide mais pas de la bonne structure). Utile pour des outputs ad-hoc ou la structure change souvent.

### Constrained decoding local (vLLM / SGLang / outlines)

Necessite de self-host. Avantages :
- **100% garantie** (le modele ne peut pas produire d'invalide)
- **Latence potentiellement meilleure** : le vocabulaire reduit = moins de compute sampling
- Flexibilite : grammaires custom (CFG), regex complexes
- Control total sur le tokenizer alignment

Pieces : SGLang expose `response_format: {type: "json_schema", schema: {...}}` et `xgrammar` en backend. vLLM 0.6+ supporte le meme.

### Prompt-seul

Jamais en 2026, sauf pour des modeles qui ne supportent pas les autres modes (rare). Meme Gemma 3, Phi-4, Qwen 3 supportent JSON schema via leur grammaire.

---

## 6. Les pieges specifiques au structured output

### Piege 1 — Degradation de qualite

Forcer un JSON schema tres rigide peut **degrader la qualite** du contenu produit. Le modele "focus" sur la forme au detriment du fond. Surtout visible sur les reasoning models.

Mitigation :
- Laisser un champ "reasoning" texte libre EN PREMIER (avant les champs structures)
- Utiliser "chain-of-thought then structured" : le modele pense en texte puis rempli le JSON
- Ne pas sur-contraindre : laisser des champs `string` ouverts plutot que enum restrictifs quand c'est pas critique

### Piege 2 — Perte d'informations

Un schema strict peut exclure des infos utiles. Si tu definis `category: enum[A, B, C]` et que l'input ne correspond a aucune → le modele force une categorie approximative.

Mitigation : ajouter `"other"` ou `"none"` dans les enums + un champ `explanation` texte libre.

### Piege 3 — Schema drift dans le temps

Ton schema evolue. Les anciens logs ne sont plus parseables. Les A/B tests sont fausses car les outputs ne sont plus comparables.

Mitigation : versionner le schema, maintenir une table de migration pour l'analyse historique.

### Piege 4 — Hallucination dans les champs libres

Constrained decoding garantit la **structure**, pas le **contenu**. Le champ `"vendor": "ACME"` peut etre hallucine. Toujours eval comme une tache d'extraction classique (precision/recall).

### Piege 5 — Recursion et profondeur

Les schemas recursifs (arbres, graphes) sont mal supportes. Workaround : applatir en listes avec des id + parent_id.

---

## 7. Pattern "reasoning-then-structured"

Le meilleur pattern 2026 pour combiner qualite de raisonnement et structure :

```python
schema = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string",
                      "description": "Think step by step first"},
        "category": {"type": "string", "enum": [...]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["reasoning", "category", "confidence"]
}
```

Le champ `reasoning` en premier permet au modele de penser avant de remplir les champs contraints. Qualite + structure.

Pour les reasoning models (o3, Claude thinking), ce champ n'est pas necessaire car ils ont leur propre `<thinking>` interne.

---

## 8. Alternative : structured outputs via function calling chains

Pour des structures complexes, au lieu d'un schema geant, enchainer des tool calls :

```
Step 1 : extract_customer(text) → {name, email, phone}
Step 2 : extract_invoice(text) → {amount, currency}
Step 3 : extract_line_items(text) → [{desc, qty}]
```

Avantages :
- Chaque schema reste simple → meilleure qualite
- Erreurs isolees (un fail n'annule pas tout)
- Retry ciblable

Inconvenients : plus de latence (plusieurs appels), plus de cost.

Arbitrage : si le schema global a > 10 champs ou plusieurs arrays, envisager la chaine.

---

## 9. Domain-Specific Languages (DSL) comme output

Une extension : faire produire au LLM un DSL, pas juste du JSON. Exemples :
- SQL avec grammaire SQL valide garantie
- SPARQL pour knowledge graphs
- Un mini-langage DSL custom pour ton produit

Outils 2026 : `guidance` (Microsoft, 2024-2026) permet de definir une grammaire BNF et contraindre la generation. `outlines` supporte regex complexes.

Exemple : forcer un modele a generer du SQL valide qui utilise SEULEMENT les tables autorisees :

```python
grammar = f"""
query ::= "SELECT " fields " FROM " table " WHERE " condition
fields ::= "*" | ident ("," ident)*
table  ::= "{allowed_tables}"
...
"""
```

Utile pour : NL-to-SQL securise (pas de DROP possible), DSL metier, prompts d'agents structures.

---

## 10. Combine avec eval et safety

Structured output + schema = **verifiable at compile time**. Exploit pour :

- **Eval auto** : verifier que chaque field est dans le bon range, que les listes sont non-vides, etc.
- **Safety guardrails** : schema interdit les champs PII, interdit certains enums.
- **Contract testing** : chaque prompt a un schema -> test automatise qui verifie qu'une suite d'inputs genere toujours du valid.
- **Versioning** : changer un schema casse le test -> force le review.

C'est la meme philosophie que le typage static en prog : detecter les erreurs **au plus tot**.

---

## Key takeaways (flashcards)

**Q1** — Pourquoi constrained decoding garantit-il 100% la structure ?
> Il masque les logits des tokens qui violeraient la grammaire a chaque step. Mathematiquement impossible de sampler un token invalide. Contraste : JSON mode API a ~99% (retry sur le serveur).

**Q2** — Quelle est la difference entre JSON mode et tool calling ?
> JSON mode : juste "doit etre du JSON valide". Tool calling : doit correspondre a un schema specifique. Tool calling est plus strict et integre la validation au provider.

**Q3** — Quel pattern combine raisonnement et structure ?
> "Reasoning-then-structured" : un champ `reasoning` texte libre en premier, puis les champs structures. Le modele pense avant de remplir. Pas necessaire pour les reasoning models natifs (thinking interne).

**Q4** — Quand preferer une chaine de tool calls a un schema unique ?
> Quand le schema global devient > 10 champs ou contient plusieurs arrays imbriques. Isole les erreurs, simplifie chaque appel, permet retry ciblable.

**Q5** — Qu'est-ce que le tokenizer alignment dans constrained decoding ?
> Un token LLM (ex `" {\""`) couvre plusieurs chars de la grammaire. Le moteur doit gerer ce mismatch : un token peut etre partiel dans la grammaire. Les libs matures (xgrammar, llguidance) s'en occupent.

**Q6** — Quels sont les 3 pieges principaux des structured outputs ?
> (1) Degradation qualite (modele focus sur forme au detriment fond), (2) perte d'info (enum force un choix mauvais), (3) hallucinations dans les champs libres (structure ≠ contenu correct).
