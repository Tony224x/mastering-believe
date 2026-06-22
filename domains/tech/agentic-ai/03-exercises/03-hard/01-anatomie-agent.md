# Exercices Hard — Anatomie d'un Agent (J1)

---

## Exercice 1 : Agent auto-correctif avec boucle de reflexion

### Objectif
Implementer un agent qui detecte ses propres erreurs et se corrige — le pattern Reflexion (Shinn et al., 2023).

### Consigne
Construis un agent ReAct augmente d'une boucle de reflexion :

1. **Inner loop (ReAct classique)** : l'agent execute des actions pour resoudre la tache
2. **Evaluator** : apres chaque tentative, un second appel LLM evalue la reponse :
   - "Is this answer correct and complete? If not, what went wrong?"
   - Retourne un verdict : `PASS` ou `RETRY` + explication
3. **Reflexion** : si `RETRY`, l'agent recoit le feedback et recommence avec l'historique de ce qui n'a pas marche
4. **Outer loop** : max 3 tentatives. Si 3 echecs, retourne la meilleure tentative

Architecture :
```
User Question
    │
    ▼
┌──────────────────────┐
│   ReAct Agent (try)  │◄───── Reflexion feedback
│   Thought → Act →    │
│   Observe → Answer   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Evaluator (judge)  │
│   PASS → return      │
│   RETRY → reflect    │
└──────────────────────┘
```

Outils disponibles pour l'agent :
- `calculator` : calcul mathematique
- `search` : recherche web (mock)
- `python_exec` : execute du code Python et retourne le resultat (sandboxed avec `exec()` dans un namespace isole)

Teste avec un probleme deliberement piege :
- "What is the sum of the first 100 prime numbers?"
  (L'agent doit probablement se corriger si sa premiere tentative est fausse)

**Mode simule** : hardcode un scenario ou la premiere tentative echoue (mauvais calcul) et la deuxieme reussit apres reflexion.

### Criteres de reussite
- [ ] L'inner loop ReAct fonctionne normalement (Thought → Action → Observation)
- [ ] L'evaluateur juge la reponse et retourne PASS ou RETRY avec une explication
- [ ] En cas de RETRY, l'agent recoit le feedback precedent dans son contexte
- [ ] L'agent s'ameliore entre les tentatives (la tentative 2 est meilleure que la 1)
- [ ] Max 3 tentatives, avec early exit sur PASS
- [ ] La trace complete (toutes tentatives + reflexions) est affichee
- [ ] `python_exec` est sandbox (namespace isole, pas d'import os/subprocess)

---

## Exercice 2 : Framework d'agent generique et configurable

### Objectif
Extraire les patterns communs et construire un mini-framework d'agent reutilisable — comprendre ce que font LangGraph/CrewAI sous le capot.

### Consigne
Cree un module `agent_framework.py` avec les classes suivantes :

```python
class Tool:
    """Encapsulates a tool: name, description, schema, implementation."""

class Memory:
    """Working memory with add/get/summarize."""

class AgentConfig:
    """Configuration: model, temperature, max_iterations, tools, system_prompt."""

class AgentTrace:
    """Records every step: thought, action, observation, duration, tokens."""

class ReActAgent:
    """The main agent class. Configurable, observable, extensible."""

    def __init__(self, config: AgentConfig): ...
    def run(self, question: str) -> AgentResult: ...
    def add_tool(self, tool: Tool): ...
    def add_hook(self, event: str, callback: Callable): ...
```

Fonctionnalites requises :

1. **Configuration par objet** : pas de params scattered — tout dans `AgentConfig`
2. **Hooks** : callbacks sur les evenements (before_llm_call, after_tool_call, on_error, on_finish)
3. **Trace automatique** : chaque execution produit un `AgentTrace` avec toutes les etapes
4. **Retry avec backoff** : si l'appel LLM echoue (rate limit, timeout), retry 3x avec exponential backoff
5. **Multiple stopping conditions** : max_iterations, max_tokens, timeout, custom predicate
6. **Serialisable** : `AgentTrace.to_json()` pour persister les traces

Teste en reconstruisant 3 agents differents avec le meme framework :
- Agent calculatrice (1 outil)
- Agent recherche (2 outils : search + calculator)
- Agent multi-step avec working memory (4 outils)

### Criteres de reussite
- [ ] Les 3 agents fonctionnent avec le meme framework, differencies uniquement par la config
- [ ] Les hooks sont appeles aux bons moments (verifiable via des prints dans les callbacks)
- [ ] `AgentTrace` contient toutes les infos : steps, durations, token counts, tools used
- [ ] `AgentTrace.to_json()` produit du JSON valide et parseable
- [ ] Le retry avec backoff fonctionne (testable en simulant un echec API)
- [ ] Le code est type (type hints partout) et documente
- [ ] Le framework fait < 300 lignes (complexite maitrisee)
