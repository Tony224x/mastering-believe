# J16 — Memoire long-horizon : episodique, semantique, procedurale et consolidation

> **Temps estime** : 3h | **Prerequis** : J1-J15
> **Objectif** : comprendre les trois types de memoire d'un agent autonome, maitriser l'architecture hierarchique MemGPT/Letta, implémenter le scoring de pertinence (recence + importance + similarite) et la consolidation par reflection de style Generative Agents.

---

## 1. Pourquoi la memoire "courte" ne suffit pas

Un agent sans memoire persistante est comme un humain amnésique : chaque conversation repart de zero.

> **Analogie** : imagine un medecin qui oublie tout entre chaque consultation. Il ne peut pas suivre l'evolution d'un patient, n'apprend pas de ses erreurs, repose les memes questions. Un agent long-horizon a besoin de se souvenir de **ce qu'il a vecu** (episodique), **ce qu'il sait** (semantique) et **comment il fait les choses** (procedural).

En J3 tu as vu les bases : short-term (contexte LLM), long-term (vector store), checkpointing LangGraph.
En J15 tu as vu le context engineering : comment gerer la fenetre de contexte et offloader sur fichiers.

**Ici, on monte d'un cran** : architectures completes de memoire structuree persistante qui survivent des semaines, categories claires, strategies de consolidation et scoring de pertinence.

---

## 2. Taxonomie des trois types de memoire

### 2.1 Memoire episodique

**Definition** : souvenir d'evenements vecus, lies a un contexte temporel precis.

```
Episode {
    id: "ep_001"
    timestamp: "2024-01-15T14:23:00"
    content: "L'utilisateur a demande un rapport sur les ventes Q3"
    outcome: "Rapport genere, approuve par Alice"
    importance: 0.8
}
```

Caracteristiques :
- Lie a **quand** et **ou** l'evenement s'est passe
- Degrade naturellement avec le temps (oubli physiologique)
- Peut etre reactivee par la **reminiscence** (cue → retrieval)

Exemples d'usage agent :
- "La derniere fois que j'ai genere un rapport pour cet utilisateur, il preferait le format CSV"
- "Le tool `fetch_weather` a echoue 3 fois cette semaine avec timeout"

### 2.2 Memoire semantique

**Definition** : faits, concepts, connaissances generales, independants d'un episode specifique.

```
Fact {
    id: "fact_007"
    content: "L'utilisateur prefere les rapports en CSV, pas en PDF"
    source: "consolide depuis ep_001, ep_003, ep_012"
    confidence: 0.92
    last_updated: "2024-01-20"
}
```

Caracteristiques :
- **Decontextualise** : la connaissance existe independamment de quand tu l'as apprise
- Peut etre extraite par consolidation de plusieurs episodes
- Plus stable dans le temps (ne decay pas aussi vite)

Exemples d'usage agent :
- Preferences utilisateur persistantes
- Caracteristiques d'outils, contraintes metier
- Faits appris sur le domaine de l'utilisateur

### 2.3 Memoire procedurale

**Definition** : savoir-faire, patterns d'action, heuristiques operationnelles.

```
Skill {
    id: "skill_003"
    name: "generer_rapport_csv"
    steps: [
        "Requete SQL sur la table ventes",
        "Formater avec pandas",
        "Exporter via tool export_csv",
        "Envoyer par email a l'utilisateur"
    ]
    success_rate: 0.94
    last_used: "2024-01-20"
}
```

Caracteristiques :
- **Executable** : c'est un plan ou une procedure
- S'ameliore avec la pratique (reinforcement implicite)
- Peut etre composee (skill qui appelle d'autres skills)

> **Analogie** : la difference entre "je sais que faire du velo c'est pedaler" (semantique) et "je sais faire du velo" (procedural). La memoire procedurale est encodee dans les schemas d'action, pas dans les propositions.

---

## 3. Architecture MemGPT / Letta : la memoire comme OS

Le papier **MemGPT (2023)** de Packer et al. propose une analogie avec les systemes d'exploitation :

| OS concept | MemGPT concept |
|---|---|
| RAM (rapide, limitee) | Main context (fenetre de contexte LLM) |
| Disque dur (lent, illimite) | External context (base persistante) |
| Paging in/out | Memory manager : charge/decharge des blocs |
| Page fault | Context overflow → trigger paging |

### 3.1 Main context vs External context

```
┌─────────────────────────────────────────────┐
│  MAIN CONTEXT (fenetre LLM ~8k tokens)      │
│  ┌───────────────┐  ┌──────────────────┐    │
│  │ System prompt │  │ Conversation     │    │
│  │ + memory      │  │ history (recent) │    │
│  │ summary       │  │                  │    │
│  └───────────────┘  └──────────────────┘    │
└─────────────────────────────────────────────┘
              ↕  paging in/out
┌─────────────────────────────────────────────┐
│  EXTERNAL CONTEXT (illimite)                 │
│  ┌─────────────┐  ┌──────────┐  ┌────────┐ │
│  │ Episodic DB │  │ Semantic │  │Skills  │ │
│  │ (episodes)  │  │   DB     │  │  DB    │ │
│  └─────────────┘  └──────────┘  └────────┘ │
└─────────────────────────────────────────────┘
```

### 3.2 Paging : chargement a la demande

Quand l'agent a besoin d'un souvenir :
1. **Query** : "Qu'est-ce que l'utilisateur prefere pour les rapports ?"
2. **Search** : scoring des entrees dans l'external context
3. **Page in** : charger les top-K dans le main context
4. **Eviction** : si le main context deborde, page out les entrees les moins pertinentes

Le trigger de paging peut etre :
- **Automatique** : avant chaque generation LLM, retrieval des N entrees les plus pertinentes
- **Explicite** : le LLM appelle un tool `memory_search(query)` quand il en a besoin

---

## 4. Scoring de pertinence : recence + importance + similarite

Inspire de Generative Agents (Park et al., 2023), le score de pertinence combine trois dimensions :

```
score = w1 * recency + w2 * importance + w3 * similarity
```

### 4.1 Recence

```python
import math

def recency_score(entry_timestamp: float, now: float, decay: float = 0.995) -> float:
    """Score exponentiellement decroissant avec le temps."""
    hours_elapsed = (now - entry_timestamp) / 3600
    return decay ** hours_elapsed
```

- `decay=0.995` : apres 24h, score ~0.89 ; apres 7j, score ~0.41
- Parametre reglable selon le domaine (info tres ephemere vs stable)

### 4.2 Importance

Score attribue au moment de la creation (ou mis a jour par reflection) :
- 0.0 → information banale ("l'utilisateur a dit bonjour")
- 0.5 → information moderement utile ("l'utilisateur a mentionne une deadline Q3")
- 1.0 → information critique ("l'utilisateur a change son email de contact")

En pratique : on peut demander au LLM d'evaluer l'importance sur 0-10 lors de la creation.

### 4.3 Similarite

Similarite cosinus entre l'embedding de la query et celui de l'entree memoire.

```python
def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x**2 for x in a) ** 0.5
    norm_b = sum(x**2 for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
```

En production : on utilise des embeddings reels (text-embedding-3-small, etc.).
En dev/test : on peut utiliser un embedding deterministique (bag-of-words, hash).

### 4.4 Scoring global

```python
def relevance_score(
    entry, query_embedding, now: float,
    w1: float = 0.3, w2: float = 0.3, w3: float = 0.4
) -> float:
    r = recency_score(entry.timestamp, now)
    i = entry.importance
    s = cosine_similarity(entry.embedding, query_embedding)
    return w1 * r + w2 * i + w3 * s
```

Poids typiques : similarite prime (0.4), recence et importance a poids egaux (0.3 chacun).

---

## 5. Consolidation et reflection

### 5.1 Memory stream (Generative Agents)

Le principe : tout evenement est stocke dans un **stream chronologique** d'observations.

```
Stream (ordre chronologique) :
  [obs_1] Alice a demande un rapport CSV
  [obs_2] Rapport genere avec succes
  [obs_3] Alice a dit "merci, parfait"
  [obs_4] Bob a demande un rapport PDF
  [obs_5] Rapport PDF echoue (format non supporte)
  [obs_6] Bob a dit "c'est nul, je voulais CSV aussi"
```

### 5.2 Reflection : consolidation des observations en faits

Periodiquement (ou quand le stream depasse N entrees), un processus de **reflection** :

1. **Detecte les patterns** dans les N derniers episodes
2. **Extrait des faits semantiques** (knowledge distillation)
3. **Stocke** dans la memoire semantique avec `source = [obs_1, obs_3, obs_6]`

```
Reflection sur obs_1 a obs_6 :
→ Fait : "Les utilisateurs preferent generalement les exports CSV aux PDF"
  source: [obs_1, obs_4, obs_6], confidence: 0.75
→ Fait : "Le tool export_pdf est defectueux"
  source: [obs_5], confidence: 0.60
```

Mock de reflection (sans LLM) : extraire les entites communes, agregger les outcomes.
Avec LLM : prompt "Quels sont les faits generaux que tu peux extraire de ces episodes ?"

### 5.3 Oubli et decay

Un systeme de memoire realiste implemente l'oubli :

```python
def should_forget(entry, now: float, threshold: float = 0.05) -> bool:
    """Oublie les entrees dont le score de recence descend sous le seuil."""
    r = recency_score(entry.timestamp, now)
    # L'importance protege de l'oubli
    effective_score = max(r, entry.importance * 0.5)
    return effective_score < threshold
```

Strategies :
- **Hard decay** : suppression si score < seuil
- **Soft decay** : compression (resume) plutot que suppression
- **Importance shield** : les entrees importantes ne sont jamais oubliees

---

## 6. Vue d'ensemble : flux de memoire d'un agent

```
Evenement (tool result / user message)
    │
    ▼
EpisodicMemory.add(event)          ← stocke le "vecu"
    │
    │  (si stream > seuil)
    ▼
consolidate()                       ← LLM/mock extrait les faits
    │
    ▼
SemanticMemory.add(facts)           ← met a jour les connaissances
    │
ProceduralMemory.update_skill()     ← met a jour les heuristiques si action reussie/echouee
    │
    ▼
HierarchicalMemory.retrieve(query)  ← scoring recence+importance+similarite
    │
    ▼
Page in les top-K dans le contexte LLM
```

---

## 7. Comparaison MemGPT vs Generative Agents

| Dimension | MemGPT / Letta | Generative Agents |
|---|---|---|
| Focus | Gestion de la fenetre de contexte (paging) | Comportement social realiste sur longue duree |
| Memoire | Main context + archival (key-value + vector) | Memory stream + reflection + planning |
| Retrieval | Tool calls explicites du LLM | Automatique avant chaque action |
| Consolidation | Summarization du contexte | Reflection periodique |
| Application | Assistants personnels persistants | Simulation multi-agent |
| Open-source | Letta (successeur MemGPT) | Code GitHub Park et al. |

---

## 8. OKF — formaliser la memoire semantique en format portable

Jusqu'ici, on a vu *comment* consolider des episodes en faits semantiques (§5 reflection) et *comment* MemGPT separe main context et **external context** (§3). Reste une question pratique : **sous quel format** persister cette memoire semantique externalisee pour qu'elle survive a un changement de framework, de vector store, ou de fournisseur de modele ?

Le 12 juin 2026, Google Cloud a publie l'**Open Knowledge Format (OKF)** : une **specification ouverte, vendor-neutral**, qui formalise le pattern **"LLM-Wiki" d'Andrej Karpathy** en un format **portable et interoperable**. But : representer les **metadonnees, le contexte et la connaissance curee** dont les systemes IA ont besoin — lisible par les humains *et* parsable par les agents.

> **Analogie** : MemGPT te dit *comment paginer* la memoire entre RAM et disque. OKF te dit *en quel format ecrire sur le disque* — un format que n'importe quel autre agent, demain, saura relire sans ton SDK.

### 8.1 Le format : des fichiers markdown + frontmatter YAML

OKF (version **v0.1**, point de depart amene a evoluer) tient sur trois choix simples :
- **"just markdown"** : lisible, rendu et indexable partout
- **"just files"** : un repertoire — qui vit dans un tarball, un repo git, un filesystem
- **"just YAML frontmatter"** pour le petit jeu de champs requetables

OKF est **minimalement opinione** : il n'exige **qu'une seule chose** de chaque concept — un champ **`type`**. Tout le reste (quels types existent, quels autres champs, quelles sections dans le corps) est laisse au producteur. Les champs frontmatter standard : `type` (seul obligatoire), `title`, `description`, `resource`, `tags`, `timestamp`.

```
memory/
├── index.md            # optionnel : divulgation progressive (table des matieres)
├── log.md              # optionnel : historique chronologique des changements
├── facts/
│   ├── user-prefers-csv.md
│   └── export-pdf-tool-broken.md
├── episodes/
│   └── ep-001-rapport-q3.md
└── skills/
    └── generer-rapport-csv.md
```

```yaml
---
type: fact
title: "Preference de format de rapport"
description: "L'utilisateur prefere les exports CSV aux PDF"
resource: "facts/user-prefers-csv.md"
tags: [user-preference, reporting]
timestamp: 2026-01-20
---
Consolide depuis [[ep-001-rapport-q3]] et [[ep-012]]. Confidence 0.92.
```

Les concepts se **lient entre eux par des liens markdown normaux** : le repertoire cesse d'etre une simple hierarchie parent/enfant du filesystem et devient un **graphe** de relations — exactement la structure dont a besoin une memoire semantique riche.

### 8.2 `index.md` et `log.md` : le parallele avec le chapitre

Deux fichiers **optionnels** aux noms reserves portent directement les idees de ce chapitre :
- **`index.md`** — sert la **divulgation progressive** (progressive disclosure) : quand l'agent navigue la hierarchie, il lit d'abord la table des matieres, puis ne charge en contexte que les pages utiles. C'est le retrieval scope de §3.2 (paging in a la demande), cote stockage.
- **`log.md`** — un **historique chronologique** des changements. C'est le pendant durable du **memory stream** de Generative Agents (§5.1) : la suite ordonnee des observations, mais persistee dans un fichier que la **reflection** (§5.2) vient distiller en pages `facts/`.

La spec v0.1 complete (criteres de conformite, regles de cross-linking, petit nombre de noms reserves) tient sur une page.

### 8.3 L'agent fait le bookkeeping, l'humain curate

Pourquoi un format markdown maintenu par un agent, et pas un wiki humain de plus ? Karpathy le resume :

> « Les LLM ne s'ennuient pas, n'oublient pas de mettre a jour une cross-reference, et peuvent toucher 15 fichiers en une seule passe. »

Le travail de tenue de registre (bookkeeping) qui pousse les humains a abandonner leurs wikis personnels est exactement ce que les LLM font bien. Applique a notre chapitre : la **consolidation** (§5) et la mise a jour des cross-references entre `episodes/`, `facts/` et `skills/` deviennent une tache d'agent, l'humain se contentant de curer.

### 8.4 Format, pas plateforme

OKF est un **format, pas une plateforme** : non lie a un cloud, une base de donnees, un fournisseur de modele ou un framework d'agent ; il ne requiert jamais de compte ni de SDK proprietaire. D'ou son interet pour ce chapitre : la memoire semantique externalisee (l'`external context` de MemGPT, §3.1) devient **portable** entre systemes et vendeurs, et survit a leurs changements. Il resout la **fragmentation de la connaissance** — schema d'une table, sens metier d'une metrique, runbook d'incident, avis de depreciation — eparpillee entre catalogues (chacun son API/SDK), wikis, commentaires de code et « la tete de quelques ingenieurs seniors ». Il manquait un **format**, pas un service de plus.

En resume : OKF complete le J15 (context offloading) en donnant un standard d'ecriture au savoir deporte, et donne a la memoire semantique de ce chapitre un format d'echange durable.

---

## Flash-cards

**Q1 : Quelle est la difference entre memoire episodique et semantique ?**
> R : Episodique = souvenir d'un evenement lie a un contexte temporel ("le 15 jan, l'user a demande X"). Semantique = fait decontextualise ("l'user prefere CSV"). La consolidation transforme plusieurs episodes en faits semantiques.

**Q2 : Comment MemGPT gere-t-il le depassement de la fenetre de contexte ?**
> R : Par un mecanisme de paging inspire des OS : le main context (RAM = fenetre LLM) est limite ; les donnees moins pertinentes sont pagees out vers l'external context (disque = DB persistante) et rechargees a la demande.

**Q3 : Quelles sont les trois composantes du scoring de pertinence de Generative Agents ?**
> R : Recence (score decroissant avec le temps), Importance (score attribue a la creation), Similarite (cosinus entre embedding query et embedding memoire). Score = w1*recency + w2*importance + w3*similarity.

**Q4 : Qu'est-ce que la reflection dans l'architecture Generative Agents ?**
> R : Un processus periodique qui analyse les N derniers episodes du memory stream et en extrait des faits semantiques de plus haut niveau (knowledge distillation). C'est la consolidation episodique → semantique.

**Q5 : Pourquoi l'importance protege-t-elle de l'oubli ?**
> R : Les entrees importantes (score proche de 1.0) ont un `effective_score = max(recency, importance * 0.5)` qui reste elevee meme quand la recence decroit. Sans ce mecanisme, des faits critiques seraient oublies apres quelques jours.

**Q6 : Qu'est-ce que l'Open Knowledge Format (OKF) et quel pattern formalise-t-il ?**
> R : OKF (Google Cloud, juin 2026) est une specification ouverte, vendor-neutral, qui formalise le pattern "LLM-Wiki" d'Andrej Karpathy en un format portable et interoperable : un repertoire de fichiers markdown + frontmatter YAML, lisible par les humains et parsable par les agents, pour persister la memoire semantique externalisee.

**Q7 : Quels sont les champs de frontmatter OKF, et lequel est le seul obligatoire ?**
> R : Les champs standard sont `type`, `title`, `description`, `resource`, `tags`, `timestamp`. Le seul obligatoire est `type` : OKF est minimalement opinione et laisse tout le reste (types disponibles, autres champs, sections du corps) au producteur. `index.md` (divulgation progressive) et `log.md` (historique chronologique) sont des fichiers optionnels a noms reserves.

---

## Points cles a retenir

- **Trois types** : episodique (vecu), semantique (savoir), procedural (savoir-faire) — chacun a son cycle de vie et son mode de retrieval
- **MemGPT** : la memoire fonctionne comme un OS, avec paging in/out entre main context (fenetre LLM) et external context (DB persistante)
- **Scoring** : `score = w1*recency + w2*importance + w3*similarity` — les trois dimensions ensemble permettent un retrieval pertinent meme sur de tres longs horizons
- **Consolidation** : la reflection periodique distille les episodes en faits semantiques, reduisant la masse de donnees a gerer
- **Oubli selectif** : decay exponentiel + bouclier d'importance — oublier intelligemment est aussi important que se souvenir
- **OKF / LLM-Wiki** : un repertoire de markdown + frontmatter YAML (seul champ obligatoire : `type`) donne a la memoire semantique externalisee un format portable et vendor-neutral — `log.md` joue le role du memory stream, `index.md` celui de la divulgation progressive

---

## Pour aller plus loin

- Packer, Wooders, Lin, Patil, Gonzalez et al., **"MemGPT: Towards LLMs as Operating Systems"** (2023) — https://arxiv.org/abs/2310.08560
- Park, O'Brien, Cai, Morris, Liang, Bernstein, **"Generative Agents: Interactive Simulacra of Human Behavior"** (2023) — https://arxiv.org/abs/2304.03442
- Shinn et al., **"Reflexion: Language Agents with Verbal Reinforcement Learning"** (2023) — https://arxiv.org/abs/2303.11366
- Letta (successeur open-source de MemGPT) — https://github.com/letta-ai/letta
- Sam McVeety & Amir Hormati (Google Cloud Data Analytics), **"How the Open Knowledge Format can improve data sharing"** (12 juin 2026) — https://cloud.google.com/blog/products/data-analytics/how-the-open-knowledge-format-can-improve-data-sharing (repo : https://github.com/GoogleCloudPlatform/knowledge-catalog/tree/main/okf)
- Andrej Karpathy, **"LLM-Wiki"** (gist fondateur du pattern formalise par OKF) — https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f
