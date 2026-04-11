# J3 — Memory & State : Short-term, Long-term, Working Memory, Checkpointing

> **Temps estime** : 3h | **Prerequis** : J1 (Anatomie d'un agent), J2 (Tool Use)
> **Objectif** : comprendre et implementer les differents types de memoire d'un agent IA, maitriser le state management, et savoir quand/comment persister l'etat.

---

## 1. Pourquoi la memoire est ce qui separe un agent utile d'un jouet

Sans memoire, un agent est un **poisson rouge**. A chaque etape, il redecouvre le monde. Il ne sait pas :
- Ce qu'il a deja fait (il re-cherche les memes infos)
- Ce qu'il a appris en cours de route (il refait les memes erreurs)
- Ce que l'utilisateur lui a dit il y a 5 minutes (il repose les memes questions)

```
Agent sans memoire :
  User: "Mon budget max est 500€"
  Agent: *cherche des produits*
  Agent: "Voulez-vous ce laptop a 1200€ ?"   ← a oublie le budget

Agent avec memoire :
  User: "Mon budget max est 500€"
  Agent: *stocke: budget_max = 500*
  Agent: *filtre les resultats > 500€*
  Agent: "Voici 3 options sous 500€"          ← se souvient
```

> **Analogie cerveau humain** : imagine un chirurgien sans memoire. A chaque geste, il oublie ou il en est dans l'operation. Un agent sans memoire, c'est pareil — il a les competences (le LLM), les outils (les instruments), mais aucune conscience de la continuite.

### Ce que la memoire permet concretement

| Capacite | Sans memoire | Avec memoire |
|----------|-------------|--------------|
| Taches multi-etapes | Perd le fil apres 2-3 etapes | Maintient le contexte sur 20+ etapes |
| Personnalisation | Traite chaque user comme un inconnu | Se souvient des preferences |
| Apprentissage | Refait les memes erreurs | "La derniere fois que j'ai essaye X, ca a echoue" |
| Continuite | Chaque session repart de zero | Reprend la ou on en etait |
| Efficacite | Re-calcule tout a chaque fois | Cache les resultats intermediaires |

---

## 2. Les 3 types de memoire — analogie cerveau humain

Le cerveau humain a plusieurs systemes de memoire. Les agents IA aussi. Comprendre cette analogie aide a choisir le bon type pour chaque besoin.

### 2.1 Short-term Memory / Context Window — le buffer de travail

**Analogie** : la memoire de travail humaine. Tu retiens un numero de telephone pendant 10 secondes — puis il s'efface. Capacite limitee (~7 elements).

Pour un agent, la short-term memory, c'est le **context window** du LLM. C'est la liste des messages envoyes dans le prompt : system prompt, historique de conversation, observations des outils.

```
Context window (ex: 128k tokens)
┌──────────────────────────────────────┐
│ System prompt          (~500 tokens) │
│ User message 1         (~100 tokens) │
│ Assistant response 1   (~200 tokens) │
│ Tool result 1          (~300 tokens) │
│ ...                                  │
│ User message N         (~100 tokens) │
│ Assistant response N   (~200 tokens) │
│ ← espace restant pour la reponse →  │
└──────────────────────────────────────┘
```

**Limites** :
- **Taille fixe** : 128k tokens (Claude), 128k (GPT-5.4). Ca parait enorme, mais un agent qui fait 30 etapes avec des tool results verbeux peut remplir le context en quelques minutes
- **Cout proportionnel** : chaque token dans le contexte est facture en input. Un contexte de 100k tokens = 100k * prix/Mtok. Ca chiffre vite
- **"Lost in the middle"** : les LLM sont moins bons pour retrouver des infos au milieu d'un long contexte. Les infos en debut et en fin sont mieux retenues

**Strategies de gestion** :

| Strategie | Principe | Trade-off |
|-----------|----------|-----------|
| **Buffer complet** | Garder tout | Simple mais explose en tokens |
| **Sliding window** | Garder les N derniers messages | Perd l'historique ancien |
| **Summarization** | Resumer les vieux messages | Perd les details |
| **Token-aware** | Couper quand on depasse un budget | Plus fin que sliding window |
| **Hybride** | Summary de l'ancien + buffer du recent | Meilleur compromis |

### 2.2 Long-term Memory / Persistent — la memoire a long terme

**Analogie** : la memoire declarative humaine. Ce que tu retiens durablement : faits, experiences, connaissances. Tu peux te rappeler ton premier jour de travail des annees apres.

Pour un agent, c'est une **base de donnees externe** qui persiste entre les sessions. Le LLM n'y a pas directement acces — il faut un mecanisme de retrieval (recherche).

**Implementations courantes** :

| Technologie | Usage | Quand l'utiliser |
|-------------|-------|-----------------|
| **Vector store** (Chroma, Pinecone, Qdrant) | Recherche semantique — "trouve les infos similaires a X" | Quand le contenu est textuel et la recherche est floue |
| **Key-value store** (Redis, fichier JSON) | Acces rapide par cle — "quel est le budget de l'utilisateur ?" | Quand les donnees sont structurees et l'acces est direct |
| **SQL database** | Donnees relationnelles complexes | Quand tu as besoin de jointures, filtres, aggregations |
| **Knowledge graph** (Neo4j) | Relations entre entites | Quand les relations comptent autant que les donnees |

**Quand ecrire en memoire long terme ?**

C'est la question la plus dure. Stocker trop = bruit. Stocker trop peu = amnesie.

```python
# Heuristiques pour decider quoi stocker :

# 1. Preference utilisateur explicite → TOUJOURS stocker
"Mon budget max est 500€"  →  store("user_prefs.budget_max", 500)

# 2. Fait appris via un outil → stocker SI reutilisable
"Le prix actuel du BTC est 63,200$"  →  NE PAS stocker (ephemere)
"L'API Stripe accepte les webhooks sur /webhook"  →  STOCKER (durable)

# 3. Resultat d'une tache complexe → stocker le RESUME, pas le detail
"J'ai analyse 500 contrats et trouve 12 clauses problematiques"  →  STOCKER
[Detail de chaque clause]  →  garder en working memory, pas en long-term
```

**Retrieval strategies** :

| Strategie | Comment ca marche | Avantage | Limite |
|-----------|-------------------|----------|--------|
| **Similarity search** | Embed la query, cherche les vecteurs proches | Trouve des infos "semantiquement" proches | Peut retourner du bruit |
| **Keyword filter + similarity** | D'abord filtrer par metadata, puis similarity | Plus precis | Necessite des metadata bien definies |
| **Recency-weighted** | Ponderer par date — les infos recentes comptent plus | Pertinent pour les donnees temporelles | Peut ignorer des faits anciens importants |
| **Relevance + recency** | Combiner score de similarite et score de recence | Le plus complet | Plus complexe a tuner |

### 2.3 Working Memory / Scratchpad — les notes de brouillon

**Analogie** : la feuille de brouillon pendant un examen. Tu notes des calculs intermediaires, des hypotheses, des resultats partiels. Ce n'est ni de la memoire a court terme (trop structure) ni du long terme (tu jettes le brouillon apres).

Pour un agent, c'est un **espace de stockage structure** accessible pendant une tache, qui contient :
- Des variables intermediaires (resultats de sous-taches)
- Des notes de planification ("il reste les etapes 3, 4, 5")
- Des hypotheses en cours ("je pense que le bug est dans la fonction X")
- Des resultats partiels ("j'ai trouve 3 produits sur 5")

```python
# Working memory = dictionnaire structure pendant l'execution
working_memory = {
    "task": "Analyser les ventes Q1 2026",
    "current_step": 3,
    "findings": [
        "Ventes France : +12%",
        "Ventes Afrique Ouest : +34%",
    ],
    "remaining": ["Calculer le total", "Generer le rapport"],
    "hypothesis": "La croissance Afrique est liee au lancement de kalira-immo",
}
```

**Pourquoi c'est different du context window** : la working memory est **structuree et compacte**. Le context window contient tout le dialogue brut (verbeux). La working memory extrait et organise les infos cles.

> **Opinion** : la working memory est le **secret des agents performants**. Un agent qui maintient un scratchpad propre est 10x plus efficace qu'un agent qui noie les infos dans le context window. C'est la difference entre un developpeur qui prend des notes et un qui essaie de tout retenir de tete.

---

## 3. State management patterns

Le state de l'agent (memoires + variables + historique), c'est ce qui le rend coherent dans le temps. Comment le gerer proprement ?

### 3.1 State as dict — simple mais fragile

```python
# Le plus simple : un dictionnaire
state = {
    "messages": [],
    "working_memory": {},
    "iteration": 0,
    "total_tokens": 0,
}

# Probleme : aucune validation, aucune structure
state["mesages"] = []  # Typo silencieuse — pas d'erreur
state["iteration"] = "cinq"  # Mauvais type — pas d'erreur
```

**Quand l'utiliser** : prototypes rapides, scripts jetables, exploration.
**Quand NE PAS l'utiliser** : production, agents multi-etapes, tout ce qui doit etre fiable.

### 3.2 State as dataclass / Pydantic model — production-grade

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class AgentState:
    """Typed, validated state. Typos and bad types are caught at definition."""
    messages: list[dict] = field(default_factory=list)
    working_memory: dict[str, Any] = field(default_factory=dict)
    iteration: int = 0
    total_tokens: int = 0
    max_iterations: int = 10

    def is_done(self) -> bool:
        return self.iteration >= self.max_iterations
```

Avec Pydantic (plus strict, validation au runtime) :

```python
from pydantic import BaseModel, Field

class AgentState(BaseModel):
    messages: list[dict] = []
    working_memory: dict[str, Any] = {}
    iteration: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    max_iterations: int = Field(default=10, ge=1)

# Erreur detectee immediatement :
state = AgentState(iteration=-1)  # ValidationError: iteration must be >= 0
```

**Quand l'utiliser** : toujours en production. Le surcout de definition est negligeable comparé au temps de debug des typos et mauvais types.

### 3.3 Immutable state + reducers — pattern fonctionnel

Inspire de Redux/Elm. L'etat n'est jamais modifie directement — on cree un nouvel etat a chaque etape.

```python
from dataclasses import dataclass, replace

@dataclass(frozen=True)  # frozen = immutable
class AgentState:
    messages: tuple = ()  # tuple car frozen exige des types immutables
    iteration: int = 0
    done: bool = False

def reducer(state: AgentState, action: dict) -> AgentState:
    """Cree un nouvel etat a partir de l'ancien + une action."""
    match action["type"]:
        case "add_message":
            return replace(state, messages=state.messages + (action["message"],))
        case "increment":
            return replace(state, iteration=state.iteration + 1)
        case "finish":
            return replace(state, done=True)
        case _:
            return state  # Action inconnue, etat inchange
```

**Avantage majeur** : **time-travel debugging**. Tu peux rejouer toutes les actions depuis l'etat initial pour reproduire un bug. Chaque etat intermediaire est conserve.

**Quand l'utiliser** : quand tu as besoin de replay, debugging avance, ou checkpointing. C'est le pattern utilise par LangGraph sous le capot.

> **Opinion** : pour les agents en production, le pattern reducer est le plus robuste. C'est plus de code au depart, mais tu gagnes enormement en debugging et en fiabilite. LangGraph l'utilise pour une bonne raison.

---

## 4. Conversation memory strategies — en detail

### 4.1 Buffer Memory — garder tout

Le plus simple : on garde tous les messages dans le context window.

```
Message 1 → Message 2 → ... → Message N → [LLM genere Message N+1]
```

**Avantages** :
- Zero perte d'information — le LLM a TOUT le contexte
- Implementation triviale — c'est juste une liste

**Inconvenients** :
- **Explose en tokens** : chaque message ajoute des tokens. 50 messages de 200 tokens = 10k tokens d'input
- **Degrade la qualite** : le LLM est moins bon avec un tres long contexte ("lost in the middle")
- **Cout croissant** : chaque appel LLM coute plus cher que le precedent

**Quand ca passe** : conversations courtes (< 20 messages), taches rapides (< 10 etapes).
**Quand ca casse** : conversations longues, agents multi-etapes, outil results volumineux.

### 4.2 Summary Memory — resumer les anciens messages

On remplace les vieux messages par un resume.

```
[Message 1...Message 20] → [Resume: "L'utilisateur cherche un laptop < 500€, a deja regarde 3 modeles"]
[Message 21...Message 30] → gardes tels quels (recents)
```

**Comment resumer** : on demande au LLM de resumer les messages anciens. C'est un appel LLM supplementaire, mais il reduit drastiquement la taille du contexte.

```python
def summarize_messages(messages: list[dict], llm_call) -> str:
    """Ask the LLM to create a concise summary of old messages."""
    prompt = f"""Summarize this conversation concisely. Keep:
    - Key facts and decisions
    - User preferences
    - Results of tool calls
    - Current state of the task

    Conversation:
    {format_messages(messages)}

    Summary:"""
    return llm_call(prompt)
```

**Trade-offs** :
- (+) Reduit le contexte de 90%+
- (+) Garde les infos essentielles
- (-) Perd les details fins (le LLM decide quoi garder)
- (-) Cout de l'appel de summarization (mais compense par la reduction du contexte futur)
- (-) Risque de "telephone arabe" si on resume un resume d'un resume

### 4.3 Sliding Window — garder les N derniers messages

On ne garde que les N derniers messages.

```
Window size = 10
[Message 1...Message 5] → supprimes
[Message 6...Message 15] → dans le context window
```

**Avantages** :
- Simple, deterministe, pas d'appel LLM supplementaire
- Cout previsible (toujours ~N messages)

**Inconvenients** :
- Perte d'information **brutale** — tout ce qui sort de la fenetre est perdu
- Si l'utilisateur a donne une info cruciale au message 3 et que la fenetre est a 10, apres le message 13, l'info disparait

**Quand l'utiliser** : taches repetitives ou les messages anciens ne sont plus pertinents. Chat assistants simples.

### 4.4 Token-aware Memory — couper au budget

Plutot que de couper par nombre de messages, on coupe par **nombre de tokens**.

```python
def trim_to_token_budget(messages: list[dict], budget: int) -> list[dict]:
    """Keep as many recent messages as fit in the token budget."""
    result = []
    total = 0
    # Iterate from newest to oldest
    for msg in reversed(messages):
        msg_tokens = count_tokens(msg["content"])
        if total + msg_tokens > budget:
            break
        result.append(msg)
        total += msg_tokens
    return list(reversed(result))  # Restore chronological order
```

**Avantage** : plus fin que le sliding window — un message court prend peu de place, un message long beaucoup. On maximise l'info dans le budget.

**Inconvenient** : un seul long tool result peut ejecter 10 messages courts. Besoin d'une bonne estimation du token count.

### 4.5 Hybrid Memory — le meilleur compromis

Combine summary + buffer : les anciens messages sont resumes, les recents sont gardes en entier.

```
┌────────────────────────────────────────────┐
│  Summary of messages 1-30                   │  ← resume compact
│  "L'utilisateur cherche un laptop < 500€,   │
│   a regarde 3 modeles, prefere ASUS"        │
├────────────────────────────────────────────┤
│  Message 31: User asks about warranty       │  ← buffer recent (complet)
│  Message 32: Agent checks warranty tool     │
│  Message 33: Tool result: 2 year warranty   │
│  Message 34: Agent responds with details    │
└────────────────────────────────────────────┘
```

**Pourquoi c'est le meilleur compromis** :
- Le resume preserve les **decisions et faits cles** du passe
- Le buffer recent garde les **details exacts** necessaires pour la tache en cours
- Le token budget reste maitrise

> **Opinion** : c'est la strategie utilisee par la quasi-totalite des agents en production. Si tu ne retiens qu'une strategie, retiens celle-ci. Le ratio "summary des 80% anciens + buffer des 20% recents" est un bon point de depart.

---

## 5. Checkpointing — sauvegarder et reprendre

### 5.1 Pourquoi le checkpointing

Un agent qui tourne 30 minutes et crashe a l'etape 25... tu veux pouvoir **reprendre a l'etape 25**, pas tout recommencer.

**Cas d'usage** :
- **Resume apres crash** : reprendre l'execution la ou elle s'est arretee
- **Time-travel debugging** : revenir a l'etape 15 pour comprendre pourquoi l'etape 16 a mal tourne
- **Branching** : "et si a l'etape 10, l'agent avait fait un choix different ?" — charger le checkpoint 10, modifier, re-executer
- **Audit** : retracer exactement ce que l'agent a fait, etape par etape
- **Long-running tasks** : sauvegarder regulierement pour survivre aux interruptions

### 5.2 Quoi sauvegarder

Un checkpoint doit contenir **tout ce qui est necessaire pour reprendre l'execution** :

```python
checkpoint = {
    "version": "1.0",
    "timestamp": "2026-04-11T14:30:00Z",
    "step": 15,
    "state": {
        "messages": [...],             # Historique complet
        "working_memory": {...},       # Scratchpad
        "tools_available": [...],      # Outils actifs
        "iteration": 15,
        "total_tokens_used": 34500,
    },
    "metadata": {
        "task": "Analyse des ventes Q1",
        "user_id": "user_123",
        "model": "claude-opus-4-6",
        "duration_so_far_seconds": 180,
    }
}
```

### 5.3 Format de serialisation

| Format | Avantage | Inconvenient | Quand l'utiliser |
|--------|----------|-------------|-----------------|
| **JSON** | Lisible, standard, debuggable | Pas de types complexes (datetime, bytes) | Defaut — toujours commencer par ca |
| **Pickle** | Supporte tout objet Python | Non-portable, problemes de securite | Jamais en production |
| **MessagePack** | Compact + rapide | Moins lisible | Quand la taille compte |
| **SQLite** | Requetable, transactionnel | Plus complexe | Quand tu as beaucoup de checkpoints |

> **Regle** : JSON pour 90% des cas. C'est lisible, debuggable, et universel. Si tu as besoin de plus, SQLite.

### 5.4 Frequence de checkpointing

| Strategie | Quand sauvegarder | Avantage | Cout |
|-----------|-------------------|----------|------|
| **A chaque etape** | Apres chaque action | Perte zero | IO a chaque etape |
| **Toutes les N etapes** | Tous les 5-10 steps | Bon compromis | Perte max de N etapes |
| **Sur evenement** | Apres un outil important, une decision cle | Intelligent | Plus complexe a implementer |
| **Timer** | Toutes les 60 secondes | Simple | Pas aligne avec la logique |

**Recommandation** : a chaque etape pour les agents courts (< 20 etapes), toutes les 5 etapes pour les agents longs. Le cout IO d'un fichier JSON est negligeable comparé au cout LLM.

### 5.5 Time-travel debugging

Le pouvoir du checkpointing : tu peux **revenir a n'importe quel point** et inspecter ou rejouer.

```
Step 1 ──→ Step 2 ──→ Step 3 ──→ Step 4 ──→ Step 5 (bug!)
  ↓           ↓           ↓           ↓
 CP-1       CP-2       CP-3       CP-4

"Pourquoi l'etape 5 a echoue ?"
→ Charger CP-4
→ Inspecter le state : ah, la working_memory n'avait pas le bon format
→ Charger CP-3 : ok, c'est l'etape 3 qui a mal ecrit en working memory
→ Bug trouve.
```

C'est exactement ce que fait LangGraph avec son systeme de checkpointing. Et c'est pour ca que le pattern immutable state + reducers est puissant : chaque etape produit un nouvel etat, et tu peux reconstruire n'importe quel point de l'execution.

---

## 6. Vector stores pour la memoire long-terme

### 6.1 Embeddings — transformer du texte en vecteurs

Un embedding, c'est une representation numerique d'un texte. Des textes semantiquement proches ont des vecteurs proches.

```
"Le chat dort sur le canape" → [0.12, -0.34, 0.56, 0.78, ...]  (1536 dimensions)
"Le felin se repose sur le sofa" → [0.11, -0.33, 0.55, 0.79, ...]  ← tres proche !
"Le prix du Bitcoin monte" → [0.89, 0.12, -0.67, 0.03, ...]  ← tres different
```

**Modeles d'embedding courants** :
- `text-embedding-3-small` (OpenAI) — 1536 dims, rapide, pas cher
- `text-embedding-3-large` (OpenAI) — 3072 dims, meilleur mais plus cher
- `voyage-3` (Voyage AI) — excellent pour le code
- Modeles locaux : `all-MiniLM-L6-v2` (384 dims, gratuit, rapide)

### 6.2 Similarity search — retrouver des souvenirs pertinents

```python
import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. 1.0 = identical, 0.0 = orthogonal."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def search_memory(query_embedding: np.ndarray,
                  stored: list[dict],    # [{"text": ..., "embedding": ...}]
                  top_k: int = 3) -> list[dict]:
    """Find the top_k most similar stored memories."""
    scored = []
    for item in stored:
        score = cosine_similarity(query_embedding, item["embedding"])
        scored.append({**item, "score": score})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]
```

### 6.3 Quand stocker quoi dans le vector store

| Type d'info | Stocker ? | Pourquoi |
|-------------|-----------|----------|
| Preferences utilisateur | Oui (key-value + vector) | Reutilisees entre sessions |
| Faits appris (stables) | Oui | "L'API X necessite le header Y" — valable longtemps |
| Resultats de recherche | Oui, avec TTL | Utiles pour eviter des re-recherches |
| Messages de conversation | Resume, pas verbatim | Le verbatim est trop volumineux |
| Erreurs et corrections | Oui | Eviter de refaire les memes erreurs |
| Donnees ephemeres | Non | Prix, meteo — perime en minutes |

### 6.4 Metadata filtering — le turbo de la recherche vectorielle

Ne pas juste chercher par similarite — filtrer d'abord par metadata.

```python
# Sans metadata filtering :
# "Qu'est-ce que l'utilisateur prefere comme couleur ?" → cherche dans TOUS les souvenirs (10k docs)

# Avec metadata filtering :
# Filtre: type="user_preference", user_id="user_123"
# Puis: similarity search dans les 50 docs restants → 100x plus rapide, 10x plus pertinent
```

Metadata utiles :
- `type` : "preference", "fact", "error", "summary"
- `source` : "user_input", "tool_result", "agent_reasoning"
- `created_at` : timestamp pour le recency weighting
- `user_id` : isolation entre utilisateurs
- `session_id` : regrouper par session
- `confidence` : score de fiabilite (0-1)

---

## 7. Comment tout s'assemble — architecture memoire d'un agent production

```
┌─────────────────────────────────────────────────────────┐
│                      AGENT LOOP                          │
│                                                          │
│   ┌────────────────┐    ┌──────────────────────┐        │
│   │  Context Window │    │   Working Memory      │        │
│   │  (short-term)   │    │   (scratchpad)         │        │
│   │                 │    │                        │        │
│   │  - System prompt│    │  - task: "..."          │        │
│   │  - Summary old  │    │  - step: 5              │        │
│   │  - Recent msgs  │    │  - findings: [...]      │        │
│   │  - Tool results │    │  - hypothesis: "..."    │        │
│   └────────┬───────┘    └──────────┬─────────────┘        │
│            │                       │                      │
│            └───────┐   ┌──────────┘                      │
│                    ▼   ▼                                  │
│              ┌──────────────┐                            │
│              │     LLM      │                            │
│              └──────┬───────┘                            │
│                     │                                    │
│            ┌────────┴────────┐                           │
│            ▼                 ▼                           │
│   ┌──────────────┐  ┌──────────────┐                    │
│   │ Tool Execution│  │ Checkpoint   │                    │
│   └──────────────┘  │ (save state) │                    │
│                     └──────────────┘                    │
└──────────────────────────┬──────────────────────────────┘
                           │
                    ┌──────┴──────┐
                    ▼             ▼
            ┌────────────┐  ┌────────────┐
            │ Vector Store│  │ Key-Value  │
            │ (long-term) │  │ (prefs,    │
            │             │  │  facts)    │
            └────────────┘  └────────────┘
```

**Flux typique** :
1. L'agent recoit une tache
2. Il **consulte la memoire long-terme** : "est-ce que j'ai deja des infos pertinentes ?"
3. Il charge le contexte : summary + messages recents + infos du vector store
4. Il raisonne (LLM), met a jour sa **working memory**
5. Il execute des outils, observe les resultats
6. Il **checkpointe** son etat
7. Il decide s'il faut **stocker** quelque chose en memoire long-terme
8. Il boucle jusqu'a completion

---

## 8. Flash Cards — Test de comprehension

**Q1 : Quels sont les 3 types de memoire d'un agent et a quoi sert chacun ?**
> R : (1) **Short-term / Context window** : le buffer de travail, contient les messages actuels, limite par la taille du contexte. (2) **Long-term / Persistent** : base de donnees externe (vector store, KV store), persiste entre sessions. (3) **Working memory / Scratchpad** : espace structure pour les variables intermediaires, hypotheses, resultats partiels pendant une tache.

**Q2 : Pourquoi la strategie "buffer complet" (garder tous les messages) ne scale pas ?**
> R : Parce que le nombre de tokens croit lineairement a chaque message. Chaque appel LLM coute plus cher (tokens input factures), la qualite degrade avec un long contexte ("lost in the middle"), et on finit par depasser la taille max du context window.

**Q3 : Quelle est la strategie de conversation memory recommandee pour la production, et pourquoi ?**
> R : La strategie **hybride** : summary des anciens messages + buffer complet des messages recents. Ca preserve les faits cles du passe (via le resume) tout en gardant les details exacts des echanges recents. Le token budget reste maitrise.

**Q4 : Que doit contenir un checkpoint pour permettre de reprendre l'execution d'un agent ?**
> R : L'etat complet : historique des messages, working memory (scratchpad), outils disponibles, compteur d'iteration, tokens utilises, et metadata (tache, user, modele, duree). Tout ce qui est necessaire pour re-creer l'etat exact de l'agent au moment du checkpoint.

**Q5 : Pourquoi le metadata filtering est-il crucial pour un vector store utilise comme memoire long-terme ?**
> R : Sans filtering, la similarity search parcourt tous les documents (potentiellement des milliers). Avec filtering (type, user_id, session_id, date...), on reduit l'espace de recherche avant la recherche vectorielle. C'est plus rapide et plus pertinent — on evite de retourner des souvenirs d'un autre utilisateur ou d'un type different.

---

## Points cles a retenir

- Sans memoire, un agent est un poisson rouge — il oublie tout a chaque etape
- 3 types de memoire : short-term (context window), long-term (vector store / BDD), working memory (scratchpad)
- La strategie hybride (summary ancien + buffer recent) est le standard en production
- State management : utiliser des dataclass/Pydantic, jamais des dicts nus en production
- Le pattern immutable state + reducers permet le time-travel debugging (utilise par LangGraph)
- Checkpointing = filet de securite : reprendre apres crash, debugger, brancher des scenarios alternatifs
- Vector stores + metadata filtering = memoire long-terme performante et pertinente
- La working memory (scratchpad) est le secret des agents performants — extraire et structurer les infos cles plutot que de tout noyer dans le context window
