# J15 — Context engineering : compaction, offloading & token budgeting

> **Temps estime** : 3h | **Prerequis** : J1-J14
> **Objectif** : maitriser la curation du context window a grande echelle — detecter le "context rot", compacter l'historique, deporter les donnees sur un systeme externe, isoler les sous-agents, et allouer un budget de tokens fin par sous-agent et par profondeur d'arbre ReAct.

---

## 1. Le probleme fondamental : la fenetre de contexte n'est pas infinie

Un modele a une limite physique — son **context window** (ex: 200 k tokens pour Claude 3.5 Sonnet). Au-dela, il tronque ou refuse. Mais les vrais problemes commencent bien avant la limite :

**Exemple concret** : un agent de refactoring de code tourne pendant 40 etapes. Il accumule :
- le system prompt (3 k tokens)
- 40 paires (user + assistant) dont beaucoup sont redondantes (80 k tokens)
- 60 appels de tools avec leurs resultats (120 k tokens)

Total : 203 k tokens. L'agent depasse la fenetre, echoue, et a gaspille 3$ de compute.

**Regles issues de l'ingenierie de terrain** (Anthropic, Cognition 2025) :
1. Plus le contexte grandit, plus le LLM "oublie" les instructions du debut (context rot)
2. Chaque token supplementaire en entree a un cout marginal non-nul
3. Les resultats de tools longs (HTML scrappe, JSON complet) sont les pires coupables

---

## 2. Context rot : quand le contexte se degrade

> **Analogie** : imagine un chef cuisinier qui doit se rappeler d'une recette en relisant tout le brouillon de session en meme temps. Au bout de 200 lignes, il a "lu" les instructions du debut mais ne les "tient" plus clairement en tete — il se concentre sur ce qui est recent.

Le phenomene "context rot" designe la degradation des performances du LLM quand le contexte devient tres long :

- Les instructions du **system prompt** sont "diluees" par le volume de conversation
- Le LLM commence a **repeter** des erreurs deja corrigees en debut de session
- La **coherence** entre les etapes diminue (le modele "perd le fil")
- La **latence** augmente car le LLM doit processer plus de tokens a chaque tour

**Symptomes observables** :
- L'agent redemande des informations qu'il a deja recues
- Il ignore des contraintes etablies dans les premiers messages
- Il regenere du code qu'il a deja genere (loops de duplication)

**Seuils empiriques** (calibres sur Claude 3.5 Sonnet) :

| Volume contexte | Comportement |
|----------------|-------------|
| < 20 k tokens | Nominal — aucun probleme |
| 20–80 k tokens | Leger glissement — surveiller les instructions critiques |
| 80–150 k tokens | Context rot notable — compaction recommandee |
| > 150 k tokens | Comportement erratique — compaction obligatoire |

---

## 3. Compaction : resumer l'historique actif

La **compaction** consiste a remplacer une portion de l'historique par un resume, reduisant le nombre de tokens sans perdre les faits critiques.

### 3.1 Strategies de compaction

**Strategy 1 — Sliding window** : on garde les N derniers messages et on compacte tout ce qui precede.

```
[system] [summary of msgs 1-30] [msg 31] [msg 32] ... [msg 40]
```

**Strategy 2 — Hierarchical summary** : un sous-agent cree un resume par "blocs" tematiques (ex: "exploration", "implementation", "tests").

**Strategy 3 — Selective keep** : certains messages sont marques "pinned" (instructions critiques, decisions cles) et ne sont jamais compactes.

### 3.2 Ce que le resume doit contenir

Un bon resume de compaction preserves :
- **Decisions prises** ("on a choisi Redis plutot que Postgres pour le cache")
- **Etat courant** ("le fichier auth.py est valide, le fichier user.py a 2 bugs non-resolus")
- **Contraintes actives** ("ne jamais utiliser eval(), tout doit etre type-safe")
- **Faits discovers** ("l'API tierces retourne HTTP 429 apres 100 req/min")

Ce qu'on peut supprimer :
- Les messages intermediaires ("OK, je vais reflechir…")
- Les resultats de tools redondants (memes donnees vues plusieurs fois)
- Les erreurs corrigees (garder la correction, pas l'erreur)

### 3.3 Quand declencher la compaction

Deux modes :
1. **Reactive** : on mesure le compte de tokens avant chaque tour ReAct et on compacte si `tokens > seuil`
2. **Proactive** : l'agent estime combien de tours il reste et compacte préventivement si `tokens_actuels + tokens_estimes_restants > limite`

> Note : voir J3 pour les bases des strategies memoire (short/long/working) — ici on se concentre sur la mecanique des tokens.

---

## 4. Context offloading : externaliser sur un systeme de fichiers virtuel

Quand le contexte est tres grand, la compaction ne suffit pas — il faut **externaliser** les donnees hors du context window.

> **Analogie** : un developpeur travaillant sur un vrai projet ne garde pas tout le code en memoire vive — il ouvre et ferme des fichiers selon ses besoins. Un agent deep peut faire pareil.

### 4.1 Virtual filesystem (VFS)

Un **VFS** est un dictionnaire persistant (ou un vrai systeme de fichiers) que l'agent peut lire/ecrire via des tools :

```
vfs_read(path: str) -> str
vfs_write(path: str, content: str) -> None
vfs_list() -> list[str]
vfs_delete(path: str) -> None
```

L'agent ne garde dans son contexte que la **table des matieres** du VFS (liste des fichiers + tailles), pas le contenu. Il charge un fichier uniquement quand il en a besoin.

### 4.2 Scratchpad et todo list

Deux patterns de memoire externe tres efficaces pour les deep agents (voir LangChain Deep Agents docs) :

**Scratchpad** : un fichier texte libre ou l'agent note ses hypotheses, plans et resultats intermediaires. Remplace avantageusement la memoire implicite dans l'historique de conversation.

**Todo list** : une liste structuree de taches avec statuts (`pending` / `in_progress` / `done` / `blocked`). Permet a l'agent de reprendre apres une interruption sans avoir a relire tout le contexte.

```
# todo.md
- [done] Analyser la structure du projet
- [done] Identifier les fichiers a modifier
- [in_progress] Corriger le bug dans auth.py
- [pending] Ecrire les tests
- [pending] Mettre a jour la documentation
```

### 4.3 Isolation de contexte par sous-agent

> Note : voir J9 pour les topologies multi-agents (supervisor/swarm) — ici on se concentre sur l'aspect contexte/tokens.

Quand un superviseur delegue a un sous-agent, **chaque sous-agent a son propre context window vierge**. C'est un avantage majeur :
- Le sous-agent ne voit pas les 80 k tokens d'historique du superviseur
- Il recoit uniquement le prompt de la tache, specifique et compact
- Son contexte reste propre tout au long de sa sous-tache

**Patron "context isolation via subagent"** :
```
Superviseur (80k tokens d'historique)
    |
    |-- delegue la tache X avec un prompt de 500 tokens
    |
Sous-agent X (context frais, 500 tokens de depart)
    |-- execute X, renvoie le resultat compacte (200 tokens)
    |
Superviseur integre 200 tokens (pas 10k de logs intermediaires)
```

Le superviseur ne recoit que le **resultat final**, pas l'historique interne du sous-agent. Gain net : 10 x a 100 x de reduction selon la complexite de la sous-tache.

---

## 5. Token & cost budgeting fin

> Note : voir J12 pour le cost tracking basique (comptage global par session) — ici on fait du budgeting **par sous-agent et par profondeur d'arbre ReAct**.

### 5.1 Anatomie du cout d'un tour ReAct

Un tour ReAct = Think + Act + Observe :

```
Cout total = cout_think + cout_tool_call + cout_observe
           = (N_input * prix_input) + latence_tool + (N_output * prix_output)
```

Le cout **marginal** du Nieme tour est plus eleve que le 1er car le contexte a grandi :
```
N_input_au_tour_N = N_input_initial + somme(outputs des N-1 tours precedents)
```

**Consequence** : les derniers tours d'un agent long coutent beaucoup plus cher que les premiers.

### 5.2 Budget par sous-agent

**Principe** : le superviseur alloue un budget en tokens (ou en dollars) a chaque sous-agent avant de le lancer. Le sous-agent doit s'arreter et rendre la main avant de depasser son budget.

```
allocation = {
    "agent_search":    budget_tokens=5_000,   # tache simple
    "agent_code_gen":  budget_tokens=20_000,  # tache complexe
    "agent_review":    budget_tokens=8_000,   # verification
}
```

**Mecanisme de veto** : si un sous-agent estime que sa tache requiert plus que son budget, il doit le signaler au superviseur plutot que de continuer et exploser le budget global.

### 5.3 Profondeur d'arbre et cout exponentiel

Dans un systeme multi-agents hierarchique (superviseur → sous-agents → sous-sous-agents), le cout peut etre exponentiel si chaque niveau spawne K agents avec un budget non-controle :

```
Profondeur 0 : superviseur         1 agent    x budget_0
Profondeur 1 : K sous-agents       K agents   x budget_1
Profondeur 2 : K^2 sous-agents     K^2 agents x budget_2
```

**Regles pratiques** :
- Limiter la profondeur a 2-3 niveaux maximum
- Reduire le budget alloue par facteur 2-5 a chaque niveau
- Placer un **budget global** en plus des budgets individuels (circuit breaker)

### 5.4 Arbitrage profondeur vs budget

La question strategique : vaut-il mieux un seul agent avec un grand contexte, ou plusieurs agents avec des contextes isoles ?

| Scenario | Approche recommandee | Justification |
|---------|---------------------|--------------|
| Tache lineaire courte | Agent unique | Overhead de spawning non-justifie |
| Tache avec phases distinctes | Multi-agents | Isolation de contexte evite le rot |
| Tache avec parallelisme | Multi-agents | Gain de latence justifie le cout |
| Budget serre | Agent unique compacte | Pas de duplication de tokens systeme |

---

## 6. Patterns concrets pour les deep agents

Les deep agents (agents qui tournent des heures, pas des secondes) ont besoin de patterns specifiques.

### 6.1 Checkpoint de contexte

Sauvegarder periodiquement l'etat complet de l'agent (contexte + VFS + todo list) pour pouvoir reprendre apres un crash ou une interruption (voir J3 pour le checkpointing general — ici la mecanique est la meme mais on inclut le VFS).

### 6.2 Context budget check avant chaque tour

```python
# Avant chaque tour ReAct
if context_manager.token_count() > threshold:
    summary = context_manager.compact()   # resumer les vieux messages
    context_manager.replace_history(summary)
```

### 6.3 Tool result trimming

Les resultats de tools sont souvent verbeux (HTML, JSON complet). Trimer automatiquement :
- HTML → extraire le texte utile (BeautifulSoup ou regex)
- JSON long → extraire les champs pertinents selon le schema attendu
- Logs → garder les N derniers + les lignes ERROR/WARN

```python
def trim_tool_result(result: str, max_tokens: int = 500) -> str:
    # approx: 1 token ≈ 4 chars
    max_chars = max_tokens * 4
    if len(result) > max_chars:
        return result[:max_chars] + f"\n[... {len(result) - max_chars} chars trimmed ...]"
    return result
```

---

## 7. Flash Cards — Test de comprehension

**Q1 : Qu'est-ce que le "context rot" et a partir de quel seuil devient-il critique pour Claude Sonnet ?**
> R : Le "context rot" est la degradation des performances du LLM quand le contexte devient tres long — les instructions du debut sont "diluees", le modele repete des erreurs corrigees et perd la coherence entre les etapes. Pour Claude Sonnet, le phenomene devient notable au-dela de 80 k tokens et critique au-dela de 150 k tokens.

**Q2 : Quelle est la difference entre compaction et offloading, et quand utiliser l'un vs l'autre ?**
> R : La **compaction** remplace une portion de l'historique par un resume (reduction de volume, contenu reste dans le contexte). L'**offloading** externalise des donnees sur un VFS ou des fichiers, retirant le contenu du contexte (seule la reference reste). Utiliser la compaction d'abord (moins d'overhead) ; si le contexte reste trop large, passer a l'offloading.

**Q3 : Pourquoi le cout marginal du Nieme tour ReAct est-il plus eleve que celui du 1er ?**
> R : Parce que le contexte grandit a chaque tour. Le LLM doit processer en input N_initial tokens + tous les outputs des N-1 tours precedents. Le prix d'input etant proportionnel au nombre de tokens, chaque tour supplementaire est plus couteux que le precedent.

**Q4 : Comment l'isolation de contexte par sous-agent reduit-elle les couts par rapport a un agent unique ?**
> R : Chaque sous-agent demarre avec un context window vierge et ne recoit que le prompt de sa sous-tache (~500 tokens). Il renvoie un resultat compacte (~200 tokens) au superviseur, pas l'integralite de son historique interne. Un agent unique aurait accumule tous ces tokens intermediaires dans un seul contexte. Gain typique : 10 x a 100 x selon la complexite.

**Q5 : Quelles sont les 3 informations indispensables qu'un resume de compaction doit preserver ?**
> R : (1) Les **decisions prises** (choix d'architecture, options retenues), (2) l'**etat courant** des artefacts (quels fichiers sont corrects, lesquels ont des bugs), (3) les **contraintes actives** (regles de code, limites de l'API, invariants a respecter). Tout le reste (messages intermediaires, erreurs corrigees, resultats redondants) peut etre supprime.

---

## Points cles a retenir

- Le **context rot** degrade les performances bien avant la limite physique du context window — surveiller les seuils 80 k / 150 k tokens
- La **compaction** (resumer l'historique) est la premiere ligne de defense ; l'**offloading** (VFS, scratchpad, todo list) est la seconde
- Les **sous-agents** ont un context window isole : ils recoivent un prompt compact et renvoient un resultat compact — c'est le principal levier de reduction de cout a grande echelle
- Le **cout marginal** d'un tour ReAct augmente lineairement avec la profondeur — les derniers tours coutent beaucoup plus que les premiers
- Allouer un **budget par sous-agent** et placer un **budget global** comme circuit breaker
- En architecture multi-niveaux, limiter la **profondeur a 2-3 niveaux** et reduire le budget par facteur 2-5 a chaque niveau
- Le **tool result trimming** est souvent le gain le plus rapide : les resultats HTML/JSON non-filres peuvent peser 10-50 k tokens

---

## Pour aller plus loin

- Anthropic, **"Effective context engineering for AI agents"** (2025) — https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
- Anthropic Claude Cookbook, **"Context engineering: memory, compaction, and tool clearing"** (2025) — https://platform.claude.com/cookbook/tool-use-context-engineering-context-engineering-tools
- Walden Yan (Cognition), **"Don't Build Multi-Agents"** (2025) — https://cognition.ai/blog/dont-build-multi-agents
- LangChain, **"Deep Agents — overview"** (2025) — https://docs.langchain.com/oss/python/deepagents/overview
