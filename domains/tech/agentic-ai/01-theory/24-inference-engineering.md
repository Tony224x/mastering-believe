# J24 — Inference engineering : fiabilite, routage et caching pour agents

> **Temps estime** : 3h | **Prerequis** : J1-J23
> **Objectif** : maitriser les trois leviers d'optimisation de l'inference LLM en production — structured outputs pour des tool calls robustes, model routing pour reduire les couts de 50-70 %, prompt caching pour compresser la latence et le budget token.

---

## 1. Structured outputs / constrained decoding

### 1.1 Le probleme : des tool calls qui cassent en prod

Un agent appelle un tool via JSON. Si le LLM hallucine une cle, omet un champ requis ou ecrit `"true"` a la place de `true`, l'appel echoue. En production, ce taux d'erreur de format s'accumule et fragilise toute la chaine.

```
# Exemple de sortie non conforme :
{"action": "search", "query": null, extra_field: "surprise"}
#                              ^                 ^
#          null sur un champ required       cle non quoted
```

### 1.2 Constrained decoding — forcer la grammaire a la source

L'idee : au lieu de laisser le LLM generer librement puis valider apres coup, on **masque les tokens incompatibles** a chaque etape de generation.

**Methode token masking** :
1. On definit une grammaire (JSON schema, EBNF, regex)
2. A chaque position, on calcule l'ensemble des tokens qui prolongent un prefixe syntaxiquement valide
3. On met a `-inf` la logit de tous les autres tokens avant le softmax
4. Le LLM ne peut **physiquement pas** generer un token invalide

```
Position : {"action": "
Tokens valides : "search", "fetch", "send", "list", "stop"
Tokens masques : TOUT LE RESTE (chiffres, autre JSON, texte libre...)
```

**Librairies en production** :
- **Outlines** (dottxt-ai) : masquage exact sur n'importe quel schema JSON, EBNF, regex ; fonctionne avec Transformers, vLLM, llama.cpp
- **Guidance** (Microsoft) : programmes mixtes texte/code, contraintes inter-fields, caching d'etat
- **Instructor** (Jason Liu) : wrapper Pydantic autour des API ; valide + re-prompt si echec (pas du vrai masquage mais simple et efficace)
- **Function calling natif** (OpenAI, Anthropic) : JSON schema fourni dans la requete, le modele est fine-tune + la plateforme applique des contraintes cote serveur

### 1.3 Validation + re-prompt (fallback)

Quand le masquage cote serveur n'est pas disponible (API distante), on valide la sortie et on re-prompt en boucle :

```python
for attempt in range(max_retries):
    raw = llm.call(prompt)
    ok, parsed, error = validate_json(raw, schema)
    if ok:
        return parsed
    prompt = f"{prompt}\n\nERREUR : {error}\nCorrige et renvoie UNIQUEMENT le JSON."
raise MaxRetriesExceeded()
```

Limite : chaque retry coute des tokens. On vise 0-1 retry en moyenne avec un bon schema.

### 1.4 JSON schema — bonnes pratiques pour les tool calls

```json
{
  "name": "search_fleet",
  "description": "Recherche des vehicules par critere",
  "parameters": {
    "type": "object",
    "properties": {
      "query":  {"type": "string", "description": "mots-cles"},
      "limit":  {"type": "integer", "minimum": 1, "maximum": 50}
    },
    "required": ["query"],
    "additionalProperties": false
  }
}
```

- `additionalProperties: false` evite les champs fantomes
- `enum` sur les champs a valeur fixe : `"status": {"enum": ["active","idle","error"]}`
- Descriptions courtes mais precises — elles vont dans le contexte

> **Analogie** : le constrained decoding, c'est comme un formulaire papier avec des cases a cocher. On ne peut pas ecrire "bleu" dans une case qui n'accepte que des chiffres.

---

## 2. Model routing — payer le bon prix pour chaque requete

### 2.1 Le probleme du modele unique

Appeler `claude-opus-4` ou `gpt-4.1` sur **toutes** les requetes est un gaspillage massif. 60-80 % des requetes d'un agent sont simples (reformatage, classification, extraction courte) et peuvent etre traitees par un modele moins cher.

Exemple de tarifs representatifs (ordre de grandeur, 2024-2025) :

| Modele     | Type   | Cout input ($/M tok) | Cout output ($/M tok) |
|------------|--------|---------------------|-----------------------|
| GPT-4.1    | Strong | ~2.00               | ~8.00                 |
| GPT-4.1-mini | Weak | ~0.40               | ~1.60                 |
| claude-haiku-3.5 | Weak | ~0.80          | ~4.00                 |
| claude-opus-4 | Strong | ~15.00           | ~75.00                |

Un routing efficace peut reduire la facture de **50-70 %** sans degradation visible de la qualite.

### 2.2 RouteLLM — routing entraine sur des preferences humaines

**Ong et al. (LMSYS/Berkeley, 2024)** ont publie RouteLLM, un routeur entraine sur des paires de preferences ChatBot Arena :

1. **Dataset** : pour chaque question, on sait quel modele a ete prefere par des humains
2. **Routeur** : classifier leger (MF, BERT-like) — prend la question en input, predit si le weak model suffit
3. **Seuil** : un hyperparametre `threshold` controle le trade-off qualite/cout
   - `threshold` haut : on envoie plus sur le weak model (moins cher, risque de degradation)
   - `threshold` bas : on envoie plus sur le strong model (plus sur mais plus cher)
4. **Resultat** : >2x reduction de cout sur MMLU, MT-Bench, GSM8K avec degradation <5 %

```
Requete -> RouteurLeger(complexity_score) -> score > threshold ?
                                                 |
                              oui (requete simple)   non (requete complexe)
                                  |                        |
                            WeakModel (cheap)       StrongModel (expensive)
```

### 2.3 Heuristiques simples sans routeur entraine

En attendant un routeur ML, des heuristiques marche correctement :

- **Longueur** : >300 tokens → strong model
- **Mots-cles de complexite** : "analyse", "compare", "explique en detail", "code" → strong
- **Type de task** : extraction JSON simple → weak ; raisonnement multi-etapes → strong
- **Historique** : si les N derniers appels ont eu besoin d'un retry → escalade vers strong

```python
def route(query: str) -> str:
    complexity = len(query.split()) / 100 + sum(
        1 for kw in COMPLEX_KEYWORDS if kw in query.lower()
    )
    return "strong" if complexity > THRESHOLD else "weak"
```

### 2.4 Table cout/latence/qualite

| Strategie        | Cout relatif | Latence | Qualite |
|------------------|-------------|---------|---------|
| All-strong       | 100 %       | haute   | max     |
| All-weak         | ~10-20 %    | basse   | degradee |
| Routing heuristique | ~40-60 % | mixte   | legere degradation |
| RouteLLM entraine | ~30-50 %   | mixte   | ~same as all-strong |

> **Analogie** : le routing, c'est comme un cabinet medical avec un infirmier et un specialiste. Les questions simples vont a l'infirmier (rapide, pas cher) ; seuls les cas complexes remontent au specialiste.

---

## 3. Prompt caching — amortir le contexte lourd

### 3.1 Le probleme : le prefixe repete

Les agents ont souvent un contexte fixe long : instructions systeme, schemas de tools, knowledge base, historique de session. Ce prefixe est envoye **a chaque appel** et facture en tokens input.

Pour un agent avec 10 000 tokens de contexte systeme et 100 appels/heure :
- Sans cache : 10 000 × 100 × prix_input = facture elevee
- Avec cache : 10 000 tokens KV calcules **une fois**, reutilises 99 fois

### 3.2 Anthropic — cache_control

Chez Anthropic, on marque explicitement les blocs a cacher avec `cache_control` :

```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Tres long contexte systeme... (10 000 tokens)",
                "cache_control": {"type": "ephemeral"}
            },
            {
                "type": "text",
                "text": "Question courte de l'utilisateur"
            }
        ]
    }
]
```

**Regles** :
- Cache valable **5 minutes** (ephemeral) — se rafraichit automatiquement si reutilise dans ce delai
- Minimum **1024 tokens** de prefixe pour activer le cache
- Cout write (premier appel) : **+25 %** sur les tokens mis en cache
- Cout read (appels suivants) : **-90 %** sur les tokens en cache
- Latence read : **-80 %** sur le time-to-first-token
- Maximum **4 points de cache** par requete

### 3.3 OpenAI — caching automatique

OpenAI cache automatiquement les prefixes d'au moins **1024 tokens** — pas de configuration requise :

```python
# Meme appel que d'habitude, le caching est transparent
response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {"role": "system", "content": long_system_prompt},  # >= 1024 tokens
        {"role": "user", "content": question}
    ]
)
# response.usage.prompt_tokens_details.cached_tokens -> tokens servis depuis le cache
```

**Regles** :
- Cache valable quelques minutes, exact-prefix matching (le prefixe doit etre identique bit-a-bit)
- Discount : **-50 %** sur les tokens caches (vs -90 % chez Anthropic)
- Fonctionne aussi pour les embeddings et les batches

### 3.4 Caching semantique (niveau applicatif)

Au-dela du cache de prefixe, on peut cacher **les reponses** a des requetes semantiquement similaires :

```
Requete 1 : "Quels camions sont disponibles a Lyon ?"
Requete 2 : "Montre-moi les camions dispos sur Lyon"
-> Meme intention -> meme reponse depuis le cache applicatif
```

Implementation : embedding de la requete → recherche dans un cache de vecteurs (cosine similarity > seuil) → si hit, retourne la reponse cachee sans appeler le LLM.

**Limites** :
- Risque de servir une reponse obsolete (TTL obligatoire)
- Faux positifs si deux requetes similaires ont des reponses differentes selon le contexte
- Overhead d'embedding a chaque requete

### 3.5 Recap des strategies de caching

| Strategie         | Economie tokens | Economie latence | Complexite impl. |
|-------------------|----------------|------------------|------------------|
| Anthropic cache_control | -90 % input cache | -80 % TTFT | Faible (marquer les blocs) |
| OpenAI auto-cache | -50 % input cache | significative | Nulle (transparent) |
| Cache semantique  | -100 % si hit   | -100 % si hit    | Elevee (embedding + store) |

---

## 4. Combiner les trois leviers

En production, on combine :

```
Requete utilisateur
       |
  [ModelRouter] -> weak ou strong ?
       |
  [PromptCache] -> prefixe deja calcule ? -> KV cache hit
       |
  [LLM call] -> sortie brute
       |
  [ConstrainedDecoder / Validator] -> JSON valide ?
       |
  [Reponse] ou [re-prompt si invalide]
```

**Exemple de gains cumules sur 1000 requetes/heure** :
- Routing 60 % weak : -50 % cout modele
- Cache prefixe 10k tokens : -70 % cout tokens prefixe
- Cache semantique 30 % hit rate : -30 % appels totaux
- Net : cout reduit de ~70 % vs baseline all-strong sans cache

> **Rappel J12** : le cost tracking basique (comptage de tokens, budget par session) a ete couvert en J12. J24 approfondit le routing intelligent (RouteLLM) et ajoute le constrained decoding, deux mecanismes absents de J12.

---

## Flash-cards

**Q1 :** Qu'est-ce que le constrained decoding et comment differe-t-il d'une validation post-generation ?
> **R :** Le constrained decoding masque les tokens incompatibles avec la grammaire **pendant** la generation (logits → -inf). La validation post-generation laisse le LLM generer librement puis rejette ou re-prompt si invalide. Le constrained decoding garantit une sortie valide sans retry.

**Q2 :** RouteLLM est entraine sur quoi et quel gain typique annonce-t-il ?
> **R :** Entraine sur des paires de preferences humaines (ChatBot Arena). Il annonce >2x reduction de cout sur des benchmarks standards avec <5 % de degradation de qualite.

**Q3 :** Quelle est la difference de discount entre Anthropic cache_control et le caching auto OpenAI ?
> **R :** Anthropic : -90 % sur les tokens en cache (mais write +25 %) ; OpenAI : -50 % sur les tokens caches. Anthropic est plus avantageux pour les longs prefixes reutilises frequemment.

**Q4 :** Pourquoi `additionalProperties: false` est-il important dans un JSON schema pour les tool calls ?
> **R :** Il empeche le LLM d'ajouter des champs inventes qui ne font pas partie du schema, evitant des erreurs de deserialization ou des comportements inattendus cote tool.

**Q5 :** Quand le caching semantique peut-il etre dangereux ?
> **R :** Quand deux requetes semantiquement proches ont des reponses differentes selon le contexte (ex : "vehicules dispos" depend de l'heure et des reservations en cours). Sans TTL strict, on sert des reponses obsoletes.

---

## Points cles a retenir

- **Structured outputs** : le constrained decoding garantit un JSON valide sans retry en masquant les tokens invalides ; Outlines, Guidance, Instructor et le function calling natif sont les principales solutions
- **Model routing** : envoyer 60-80 % des requetes simples vers un modele cheap reduit le cout de 50-70 % ; RouteLLM (Ong et al. 2024) est la reference academique ; des heuristiques simples (longueur, mots-cles) donnent 80 % du benefice
- **Prompt caching** : Anthropic (-90 % sur tokens caches) et OpenAI (-50 %, automatique) amortissent le cout des longs prefixes fixes ; minimum 1024 tokens ; le cache semantique applicatif va plus loin mais est plus complexe
- **Combinaison** : routing + caching + constrained decoding sont orthogonaux et se cumulent — un agent bien optimise peut diviser son budget par 3 a 5 vs baseline

---

## Pour aller plus loin

- Ong et al. (LMSYS/Berkeley), **"RouteLLM: Learning to Route LLMs with Preference Data"** (2024) — https://arxiv.org/abs/2406.18665
- Anthropic, **"Prompt caching"** — https://platform.claude.com/docs/en/build-with-claude/prompt-caching
- OpenAI, **"Prompt Caching"** — https://developers.openai.com/api/docs/guides/prompt-caching
- Dottxt-ai, **Outlines** (constrained decoding library) — https://github.com/dottxt-ai/outlines
- Microsoft, **Guidance** — https://github.com/guidance-ai/guidance
- Jason Liu, **Instructor** — https://github.com/jxnl/instructor
