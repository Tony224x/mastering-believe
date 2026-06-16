# REFERENCES — `agentic-ai` (14 jours)

> Sources de tier-1 par module : papiers fondateurs (arXiv + année), docs officielles, articles de référence.
> Source de vérité pour les modules J1..J14. Toutes les URLs et métadonnées (titre exact, auteurs, année, arXiv id) ont été vérifiées via WebSearch (juin 2026).
> Actualité prise en compte : MCP révision stable **2025-11-25**, **LangGraph v1.0** (oct. 2025, docs migrées vers `docs.langchain.com`), OWASP Top 10 for LLM Applications **2025**.

---

## Module 01 — Anatomie d'un agent (boucle ReAct, agent vs chatbot vs pipeline)

- **Yao, Zhao, Yu, Du, Shafran, Narasimhan, Cao (2022/ICLR 2023). *ReAct: Synergizing Reasoning and Acting in Language Models*. arXiv:2210.03629** — https://arxiv.org/abs/2210.03629 — papier fondateur du paradigme reason+act interleaved : la boucle exacte (Thought → Action → Observation) implémentée from scratch en J1.
- **Anthropic (déc. 2024). *Building Effective Agents*.** — https://www.anthropic.com/engineering/building-effective-agents — distinction canonique workflow vs agent, et le conseil "commence simple, n'utilise un framework que si nécessaire" — colonne vertébrale du discours "quand un agent vs un pipeline".
- **Park, O'Brien, Cai, Morris, Liang, Bernstein (2023/UIST 2023). *Generative Agents: Interactive Simulacra of Human Behavior*. arXiv:2304.03442** — https://arxiv.org/abs/2304.03442 — architecture agent complète (perceive→plan→reflect→act) à grande échelle ; illustre la boucle d'agent au-delà du simple ReAct.

---

## Module 02 — Tool Use & Function Calling

- **Schick, Dwivedi-Yu, Dessì, Raileanu, Lomeli, Hambro, Zettlemoyer, Cancedda, Scialom (2023/NeurIPS 2023). *Toolformer: Language Models Can Teach Themselves to Use Tools*. arXiv:2302.04761** — https://arxiv.org/abs/2302.04761 — référence fondatrice sur l'apprentissage de l'usage d'outils (quand appeler, quels arguments) ; cadre le "pourquoi" du function calling.
- **Anthropic. *Tool use with Claude* (Claude API docs).** — https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview — doc officielle du tool use Claude : schéma JSON des tools, cycle tool_use → tool_result, parallel tool calls. Source de vérité pour le code J2.
- **OpenAI. *Function calling* (API guide).** — https://platform.openai.com/docs/guides/function-calling — doc officielle côté OpenAI : `tools`/`tool_choice`, structured outputs, strict mode. Complément multi-provider du module.

---

## Module 03 — Memory & State (short-term, long-term, working memory, checkpointing)

- **Packer, Wooders, Lin, Fang, Patil, Stoica, Gonzalez (2023). *MemGPT: Towards LLMs as Operating Systems*. arXiv:2310.08560** — https://arxiv.org/abs/2310.08560 — virtual context management inspiré des OS (memory tiers, paging hors/dans le context window) — modèle mental canonique du long-term memory d'agent.
- **Lewis, Perez, Piktus, Petroni, Karpukhin, Goyal, Küttler, Lewis, Yih, Rocktäschel, Riedel, Kiela (2020/NeurIPS 2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. arXiv:2005.11401** — https://arxiv.org/abs/2005.11401 — papier d'origine du RAG = mémoire externe non-paramétrique ; base du long-term memory via vector store.
- **LangChain. *LangGraph — Persistence / Memory* (docs v1).** — https://docs.langchain.com/oss/python/langgraph/persistence — doc officielle du checkpointing, des threads et de la persistance d'état (short-term via checkpointer, long-term via store). Source de vérité pour la partie state management de J3.

---

## Module 04 — Planning & Reasoning (CoT, ToT, Self-Consistency, Plan-and-Execute, Reflexion)

- **Wei, Wang, Schuurmans, Bosma, Ichter, Xia, Chi, Le, Zhou (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*. arXiv:2201.11903** — https://arxiv.org/abs/2201.11903 — le papier qui démontre le gain du raisonnement explicite (GSM8K 18%→57%) ; chiffres directement cités dans J4.
- **Wang, Wei, Schuurmans, Le, Chi, Narang, Chowdhery, Zhou (2022/ICLR 2023). *Self-Consistency Improves Chain of Thought Reasoning in Language Models*. arXiv:2203.11171** — https://arxiv.org/abs/2203.11171 — échantillonner plusieurs chaînes puis voter ; technique simple à fort levier sur le raisonnement.
- **Yao, Yu, Zhao, Shafran, Griffiths, Cao, Narasimhan (2023/NeurIPS 2023). *Tree of Thoughts: Deliberate Problem Solving with Large Language Models*. arXiv:2305.10601** — https://arxiv.org/abs/2305.10601 — généralise CoT en arbre avec exploration/backtracking et self-evaluation ; le "ToT" du module.
- **Wang, Xu, Lan, Hu, Lan, Lee, Lim (2023/ACL 2023). *Plan-and-Solve Prompting*. arXiv:2305.04091** — https://arxiv.org/abs/2305.04091 — pattern plan-then-execute en zero-shot ; fondement du "plan-and-execute".
- **Shinn, Cassano, Berman, Gopinath, Narasimhan, Yao (2023/NeurIPS 2023). *Reflexion: Language Agents with Verbal Reinforcement Learning*. arXiv:2303.11366** — https://arxiv.org/abs/2303.11366 — self-critique verbale stockée en mémoire épisodique ; le pattern Reflexion du module.

---

## Module 05 — LangGraph fondamentaux (StateGraph, nodes, edges, conditional routing, HITL)

- **LangChain. *LangGraph overview* (docs v1, Python).** — https://docs.langchain.com/oss/python/langgraph/overview — point d'entrée officiel : modèle State + Nodes + Edges, exécution type Pregel (super-steps). Source de vérité du modèle mental de J5.
- **LangChain (oct. 2025). *LangGraph 1.0 is now generally available*.** — https://changelog.langchain.com/announcements/langgraph-1-0-is-now-generally-available — annonce v1.0 (stabilité jusqu'à 2.0, backward-compat) ; contexte de la note de version du module (dépréciation de `create_react_agent` → `create_agent`).
- **LangChain. *LangGraph — Low-level concepts (StateGraph, reducers, conditional edges)*.** — https://langchain-ai.github.io/langgraph/concepts/low_level/ — glossaire de référence des primitives bas niveau utilisées tout au long du module.

---

## Module 06 — LangGraph avancé (subgraphs, parallel/Send, streaming, persistence, time-travel)

- **LangChain. *LangGraph — Subgraphs* (docs).** — https://docs.langchain.com/oss/python/langgraph/subgraphs — composition de graphs réutilisables ; section 2 du module.
- **LangChain. *LangGraph — Streaming* (docs).** — https://docs.langchain.com/oss/python/langgraph/streaming — modes `values` / `updates` / `messages` ; couvre la partie streaming du module.
- **LangChain. *LangGraph — Persistence & time-travel* (docs).** — https://docs.langchain.com/oss/python/langgraph/persistence — checkpointers, reprise après crash, time-travel et branching depuis un état passé ; cœur de la 2e moitié de J6.
- **Anthropic (juin 2025). *How we built our multi-agent research system*.** — https://www.anthropic.com/engineering/multi-agent-research-system — retour d'expérience sur l'orchestration parallèle de sous-agents (fan-out) ; complément production aux patterns Send/parallel.

---

## Module 07 — Build : agent de recherche + analyse complet (capstone S1)

- **Anthropic (juin 2025). *How we built our multi-agent research system*.** — https://www.anthropic.com/engineering/multi-agent-research-system — étude de cas directe d'un agent de recherche (planner → sous-agents → synthèse) ; gabarit du capstone semaine 1.
- **DeepLearning.AI (Harrison Chase & Rotem Weiss). *AI Agents in LangGraph* (short course).** — https://www.deeplearning.ai/courses/ai-agents-in-langgraph — construit un agent de recherche from scratch puis en LangGraph avec agentic search + HITL ; parcours quasi identique à J7.
- **Yao et al. (2022). *ReAct*. arXiv:2210.03629** — https://arxiv.org/abs/2210.03629 — boucle de base réutilisée dans l'exécuteur du capstone (rappel inter-module).

---

## Module 08 — RAG agentique (query decomposition, routing, retrieval grading, multi-hop, adaptive RAG)

- **Asai, Wu, Wang, Sil, Hajishirzi (2023/ICLR 2024). *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*. arXiv:2310.11511** — https://arxiv.org/abs/2310.11511 — retrieval on-demand + reflection tokens qui notent pertinence/support ; fondement du "retrieval grading".
- **Yan, Gu, Zhu, Ling (2024). *Corrective Retrieval Augmented Generation (CRAG)*. arXiv:2401.15884** — https://arxiv.org/abs/2401.15884 — évaluateur de retrieval + fallback web search quand les docs sont mauvais ; le pattern corrective/retry du module.
- **Jeong, Baek, Cho, Hwang, Park (2024/NAACL 2024). *Adaptive-RAG: Learning to Adapt Retrieval-Augmented LLMs through Question Complexity*. arXiv:2403.14403** — https://arxiv.org/abs/2403.14403 — routeur qui choisit no-retrieval / single-hop / multi-hop selon la complexité ; cœur de l'"adaptive RAG".
- **Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. arXiv:2005.11401** — https://arxiv.org/abs/2005.11401 — le RAG "vanilla" de référence dont le module montre les limites.

---

## Module 09 — Multi-agent patterns (supervisor, hierarchical, swarm, debate)

- **Du, Li, Torralba, Tenenbaum, Mordatch (2023). *Improving Factuality and Reasoning in Language Models through Multiagent Debate*. arXiv:2305.14325** — https://arxiv.org/abs/2305.14325 — papier de référence du pattern "debate" (plusieurs agents proposent/critiquent sur N rounds).
- **Anthropic (juin 2025). *How we built our multi-agent research system*.** — https://www.anthropic.com/engineering/multi-agent-research-system — supervisor + sous-agents parallèles en production, avec les pièges réels (sur-spawn, coûts) ; arbitrage complexité/qualité du module.
- **LangChain. *How and when to build multi-agent systems*.** — https://www.langchain.com/blog/how-and-when-to-build-multi-agent-systems — cadre de décision supervisor vs swarm vs single-agent ; complément "quand passer au multi-agent".
- **LangChain. *LangGraph — Multi-agent* (docs, incl. swarm & supervisor).** — https://docs.langchain.com/oss/python/langgraph/multi-agent — implémentations de référence des handoffs (swarm) et du supervisor utilisées en code.

---

## Module 10 — MCP (Model Context Protocol : servers, resources, tools, prompts)

- **Model Context Protocol. *Specification 2025-11-25* (révision stable).** — https://modelcontextprotocol.io/specification/2025-11-25 — spec officielle à jour : primitives tools/resources/prompts, transports, autorisation (OAuth, OIDC discovery), Tasks. Source de vérité du module.
- **Anthropic. *Introducing the Model Context Protocol* (annonce, nov. 2024).** — https://www.anthropic.com/news/model-context-protocol — annonce d'origine et motivation ("le USB-C des intégrations LLM") ; cadre le "pourquoi" du module.
- **Model Context Protocol. *Documentation & SDKs* (modelcontextprotocol.io).** — https://modelcontextprotocol.io — guides serveur/client et SDKs officiels (Python, TypeScript, Java, C#, Rust, Kotlin) ; base pour construire le serveur MCP du module.
- **Model Context Protocol Blog (nov. 2025). *One Year of MCP: November 2025 Spec Release*.** — https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/ — récap des nouveautés 2025-11-25 (icons, scope consent incrémental, tool calling in sampling, Tasks) pour rester à jour mi-2026.

---

## Module 11 — Evaluation & Testing (agent evals, trajectory, LLM-as-judge, benchmarks, regression)

- **Zheng, Chiang, Sheng, Zhuang, Wu, Zhuang, Lin, Li, Li, Xing, Zhang, Gonzalez, Stoica (2023/NeurIPS 2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. arXiv:2306.05685** — https://arxiv.org/abs/2306.05685 — référence du LLM-as-judge : accord ~80% avec l'humain et biais connus (position, verbosité, self-enhancement) ; section LLM-as-judge du module.
- **Yao, Shinn, Razavi, Narasimhan (2024). *τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains*. arXiv:2406.12045** — https://arxiv.org/abs/2406.12045 — benchmark agent réaliste (état de DB final vs goal + métrique pass^k pour la fiabilité) ; modèle d'éval de trajectoire et de reproductibilité.
- **Jimenez, Yang, Wettig, Yao, Pei, Press, Narasimhan (2023/ICLR 2024). *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?*. arXiv:2310.06770** — https://arxiv.org/abs/2310.06770 — benchmark agentique exécutable de référence (2 294 issues GitHub réelles) ; exemple d'éval end-to-end basée sur l'exécution.
- **LangChain. *LangSmith — Evaluation* (docs).** — https://docs.langchain.com/langsmith/evaluation — outillage pratique : datasets, evaluators (LLM-as-judge, heuristiques), regression testing ; pont vers l'implémentation du harness d'éval.

---

## Module 12 — Production & Observabilité (tracing, cost/latency, error recovery, guardrails)

- **LangChain. *LangSmith — Observability / Tracing* (docs).** — https://docs.langchain.com/langsmith/observability — concepts traces/spans, debugging et monitoring d'agents ; référence pour la partie tracing de J12.
- **Langfuse. *Documentation* (open-source LLM observability).** — https://langfuse.com/docs — alternative open-source/self-hostable : traces, cost tracking, latency, scores ; cité dans le module comme option vendor-neutral.
- **OpenTelemetry. *Generative AI / LLM semantic conventions*.** — https://opentelemetry.io/docs/specs/semconv/gen-ai/ — standard d'instrumentation des spans GenAI (tokens, coûts, modèle) ; socle interopérable du tracing.
- **Anthropic (déc. 2024). *Building Effective Agents*.** — https://www.anthropic.com/engineering/building-effective-agents — principes de guardrails, budgets et garde-fous repris dans la partie production du module.

---

## Module 13 — Sécurité & Robustesse (prompt injection, tool abuse, sandboxing, human oversight)

- **Greshake, Abdelnabi, Mishra, Endres, Holz, Fritz (2023/AISec 2023). *Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection*. arXiv:2302.12173** — https://arxiv.org/abs/2302.12173 — papier fondateur de l'injection indirecte (payload caché dans le contenu lu via un tool) ; l'attaque centrale du module.
- **OWASP. *Top 10 for LLM Applications 2025*.** — https://genai.owasp.org/llm-top-10/ — taxonomie de référence (LLM01 Prompt Injection en tête, + system prompt leakage, vector/embedding weaknesses) ; structure la section "taxonomie des attaques".
- **Anthropic. *Mitigating jailbreaks & prompt injections* (Claude docs).** — https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/mitigate-jailbreaks — défenses pratiques (input/output filtering, prompts robustes) côté Claude ; complément défensif officiel.
- **Greshake et al. — *indirect prompt injection* (rappel) :** voir arXiv:2302.12173 ci-dessus pour les démonstrations concrètes (Bing Chat, exfiltration de données).

---

## Module 14 — Capstone : assistant de recherche autonome production-ready

- **Anthropic (juin 2025). *How we built our multi-agent research system*.** — https://www.anthropic.com/engineering/multi-agent-research-system — blueprint complet (supervisor + sous-agents, prompts, coûts, éval) qui correspond presque ligne à ligne au système AcmeResearcher du capstone.
- **Anthropic (déc. 2024). *Building Effective Agents*.** — https://www.anthropic.com/engineering/building-effective-agents — checklist design (workflows vs agents, guardrails, HITL) pour cadrer une archi production-credible.
- **Yao et al. (2024). *τ-bench*. arXiv:2406.12045** — https://arxiv.org/abs/2406.12045 — modèle pour l'eval-harness du capstone (cas de test + fiabilité multi-trials).
- **LangChain. *LangGraph — Multi-agent* (docs).** — https://docs.langchain.com/oss/python/langgraph/multi-agent — référence d'implémentation du supervisor/workers assemblé dans le capstone.

---

## Ressources transversales

### Docs officielles (stack du domaine)

- **LangGraph (v1)** — https://docs.langchain.com/oss/python/langgraph/overview — orchestration d'agents stateful ; doc de référence pour J5-J9 et le capstone.
- **LangGraph low-level concepts** — https://langchain-ai.github.io/langgraph/concepts/low_level/ — glossaire State/Node/Edge/reducer.
- **Model Context Protocol (spec 2025-11-25 + docs)** — https://modelcontextprotocol.io — protocole + SDKs officiels (Python/TS/Java/C#/Rust/Kotlin).
- **Anthropic Claude — Tool use** — https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview — function calling natif Claude.
- **OpenAI — Function calling** — https://platform.openai.com/docs/guides/function-calling — function calling natif OpenAI (multi-provider).
- **LangSmith** — https://docs.langchain.com/langsmith — tracing + evaluation (J11-J12).
- **Langfuse** — https://langfuse.com/docs — observabilité LLM open-source (J12).
- **OWASP Top 10 for LLM Applications (2025)** — https://genai.owasp.org/llm-top-10/ — taxonomie sécurité (J13).

### Articles de référence (Anthropic Engineering)

- **Building Effective Agents** (déc. 2024) — https://www.anthropic.com/engineering/building-effective-agents — design patterns workflows vs agents ; transversal J1-J14.
- **How we built our multi-agent research system** (juin 2025) — https://www.anthropic.com/engineering/multi-agent-research-system — orchestration multi-agent en production ; transversal J6-J9, J14.

### Cours

- **DeepLearning.AI — AI Agents in LangGraph** (Harrison Chase & Rotem Weiss) — https://www.deeplearning.ai/courses/ai-agents-in-langgraph — agent from scratch puis LangGraph + agentic search + HITL ; aligné J1-J7.
- **LangChain Academy — Introduction to LangGraph** — https://academy.langchain.com/courses/intro-to-langgraph — cours officiel gratuit ; renforce J5-J6.

---

## Notes

- **arXiv non-fetchable directement** : arxiv.org bloque le fetch automatisé (HTTP 403) ; les métadonnées (titre, auteurs, année, id) ont été vérifiées via recherche web croisée (alphaXiv / HuggingFace papers / Semantic Scholar / pages projet). Les URLs `https://arxiv.org/abs/<id>` restent valides en navigateur.
- **Versions / actualité (juin 2026)** : LangGraph **v1.0** (oct. 2025) — `create_react_agent` déprécié au profit de `create_agent` (cf. note de version dans J5) ; les docs LangChain/LangGraph ont migré vers `docs.langchain.com/oss/...`. MCP : révision stable **2025-11-25** retenue comme source de vérité.
- **Sources non vérifiées / à confirmer** : aucune référence inventée. Deux URLs de docs officielles sont des liens canoniques par section qui peuvent se réorganiser au fil des refontes (la doc LangChain a été redesignée avec la v1) — fonctionnels en juin 2026 mais à re-vérifier si 404 : `docs.langchain.com/langsmith/observability`, `docs.langchain.com/oss/python/langgraph/multi-agent`. L'URL Anthropic tool-use a migré de `docs.anthropic.com` vers `platform.claude.com` ; les deux redirigent.
</content>
</invoke>


---

## REFERENCES — agentic-ai (extension J15–J28)

Sources tier-1 par module pour l'extension « 2 semaines avancées » du domaine.
Les modules J1–J14 (fondations) précèdent ; cette liste couvre les modules
frontier J15–J28. Source-of-truth pour les passes de vérification (Phase 5/6).

> Note : plusieurs docs officielles (`modelcontextprotocol.io`,
> `docs.langchain.com`, `docs.temporal.io`, `genai.owasp.org`, pages produit
> OpenAI) renvoient un 403 anti-bot au fetch automatisé mais sont des URL
> canoniques stables, confirmées par recherche.

## Semaine 3 — Frontier patterns & orchestration

### J15 — Context engineering & compaction
- **Effective context engineering for AI agents** — Anthropic (Applied AI), 2025. https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents — référence canonique (curation du contexte, compaction, note-taking, sub-agent isolation, « context rot »).
- **Context engineering: memory, compaction, and tool clearing** — Anthropic (Claude Cookbook), 2025. https://platform.claude.com/cookbook/tool-use-context-engineering-context-engineering-tools — implémentation concrète (memory tool + context editing beta).
- **Don't Build Multi-Agents** — Walden Yan (Cognition), 2025. https://cognition.ai/blog/dont-build-multi-agents — contre-argument single-agent vs sub-agents (débat structurant).
- **LangChain Deep Agents — overview** — LangChain (docs), 2025. https://docs.langchain.com/oss/python/deepagents/overview — planning/`write_todos`, sub-agents isolés, virtual filesystem (offloading).

### J16 — Mémoire avancée & agents long-horizon
- **MemGPT: Towards LLMs as Operating Systems** — Packer, Wooders, Lin, Patil, Gonzalez et al. (UC Berkeley), 2023. https://arxiv.org/abs/2310.08560 — mémoire hiérarchique virtuelle (main/external context), socle de Letta.
- **Generative Agents: Interactive Simulacra of Human Behavior** — Park, O'Brien, Cai, Morris, Liang, Bernstein (Stanford), 2023. https://arxiv.org/abs/2304.03442 — memory stream + retrieval + reflection (consolidation), épisodique/sémantique en pratique.
- **Reflexion: Language Agents with Verbal Reinforcement Learning** — Shinn, Cassano, Berman, Gopinath, Narasimhan, Yao, 2023 (NeurIPS). https://arxiv.org/abs/2303.11366 — buffer de mémoire épisodique de réflexions verbales (pont mémoire/self-improvement).

### J17 — Self-improving agents
- **Reflexion** — Shinn et al., 2023. https://arxiv.org/abs/2303.11366 — self-improvement par feedback verbal, sans fine-tuning.
- **Self-Refine: Iterative Refinement with Self-Feedback** — Madaan, Tandon et al., 2023 (NeurIPS). https://arxiv.org/abs/2303.17651 — boucle generator/critic/refiner avec un seul LLM.
- **Let's Verify Step by Step** — Lightman, Kosaraju, Burda, Edwards, Baker, Leike, Schulman, Sutskever, Cobbe (OpenAI), 2023. https://arxiv.org/abs/2305.20050 — verifier / process reward models (PRM).
- **Scaling LLM Test-Time Compute Optimally...** — Snell, Lee, Xu, Kumar (UC Berkeley / Google DeepMind), 2024. https://arxiv.org/abs/2408.03314 — test-time scaling (recherche vs verifiers + révision adaptative).

### J18 — Frameworks d'orchestration comparés
- **LangGraph — Multi-agent** — LangChain (docs), 2025. https://docs.langchain.com/oss/python/langchain/multi-agent — supervisor / swarm / handoffs, modèle graphe d'états.
- **AutoGen v0.4: Reimagining the foundation of agentic AI** — Microsoft Research, jan. 2025. https://www.microsoft.com/en-us/research/blog/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/ — architecture event-driven/actor (Core + AgentChat).
- **OpenAI Agents SDK — Handoffs** — OpenAI (docs), 2025. https://openai.github.io/openai-agents-python/handoffs/ — handoff léger tool-centric.
- **openai/swarm** — OpenAI, 2024 (archivé/éducatif, remplacé par l'Agents SDK). https://github.com/openai/swarm — contraste stateless vs stateful.
- **CrewAI — documentation** — CrewAI Inc. https://docs.crewai.com/ — modèle role-based crews/tasks/process (sequential vs hierarchical).

### J19 — Protocoles inter-agents
- **A2A (Agent2Agent) — spécification** — Linux Foundation (Google), v1.0 2025. https://a2a-protocol.org/ (repo https://github.com/a2aproject/A2A) — JSON-RPC/HTTP/SSE, Agent Cards, lifecycle des tasks.
- **Model Context Protocol — Specification (2025-11-25)** — Anthropic / MCP. https://modelcontextprotocol.io/specification/2025-11-25 — tools/resources/prompts/transports (rappel + complémentarité avec A2A).
- **Introducing the Model Context Protocol** — Anthropic, nov. 2024. https://www.anthropic.com/news/model-context-protocol — problème M×N, motivation.
- **Agent Communication Protocol (ACP)** — IBM / BeeAI (LF AI & Data), 2025. https://github.com/i-am-bee/acp — async-first JSON-RPC ; nuance d'actualité : convergence ACP → A2A.

### J20 — Durable & event-driven agents
- **Durable Agent with Tools — OpenAI Agents SDK (Temporal)** — Temporal (AI cookbook), 2025. https://docs.temporal.io/ai-cookbook/openai-agents-sdk-python — LLM/tool calls en activities, contrôle déterministe dans le workflow.
- **Durable Execution meets AI** — Temporal (blog), 2025. https://temporal.io/blog/durable-execution-meets-ai-why-temporal-is-the-perfect-foundation-for-ai — event history append-only, replay, reprise sur crash, idempotence.
- **LangGraph — Persistence/checkpointing** — LangChain. https://docs.langchain.com/oss/python/langgraph/persistence — checkpointers, time-travel, fault tolerance.
- **AutoGen 0.4 — saving/restoring & resuming** — Microsoft. https://microsoft.github.io/autogen/stable/ — reprise des tâches en archi event-driven.

## Semaine 4 — Computer/coding agents, production & capstone

### J21 — Computer use & GUI/browser agents
- **Computer use tool (doc)** — Anthropic, 2024-2025. https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool — actions (`screenshot`/`left_click`/`type`/`scroll`), boucle action→screenshot, classifiers anti-injection.
- **Introducing computer use (Claude 3.5 Sonnet)** — Anthropic, oct. 2024. https://www.anthropic.com/news/3-5-models-and-computer-use — philosophie « compétences générales d'ordinateur ».
- **Computer-Using Agent (CUA) / Operator** — OpenAI, jan. 2025. https://openai.com/index/computer-using-agent/ — comparatif Anthropic vs OpenAI.
- **Set-of-Mark Prompting...** — Yang et al. (Microsoft, HKUST, UW-Madison), 2023. https://arxiv.org/abs/2310.11441 — grounding visuel (segmentation + marques).
- **browser-use** — 2024-. https://github.com/browser-use/browser-use — agent navigateur open-source (DOM + LLM).
- **WebArena** — Zhou et al., 2023. https://arxiv.org/abs/2307.13854 — difficulté quantifiée des agents web (GPT-4 14,4% vs humain 78,2%).

### J22 — Architecture des coding agents
- **SWE-agent: Agent-Computer Interfaces...** — Yang, Jimenez, Wettig, Lieret, Yao, Narasimhan, Press (Princeton), 2024 (NeurIPS). https://arxiv.org/abs/2405.15793 — concept d'ACI (interfaces dédiées au LM).
- **SWE-agent (repo)** — 2024-. https://github.com/SWE-agent/SWE-agent — boucle navigation/édition/test sur issue GitHub.
- **SWE-bench: Can LMs Resolve Real-World GitHub Issues?** — Jimenez et al. (Princeton), 2023 (ICLR'24). https://arxiv.org/abs/2310.06770 (https://www.swebench.com/verified.html) — 2 294 issues réelles.
- **Aider (repo + doc)** — Gauthier / Aider-AI, 2023-. https://github.com/Aider-AI/aider (https://aider.chat) — repo-map, edit blocks, auto-commit git.

### J23 — Sandboxing & exécution sûre + voice/realtime
- **sandbox-runtime** — anthropic-experimental, 2025 (Apache-2.0). https://github.com/anthropic-experimental/sandbox-runtime — sandboxing OS-level (sandbox-exec/bubblewrap) + proxy réseau.
- **gVisor — Application Kernel for Containers** — Google, 2018-. https://gvisor.dev/ (https://github.com/google/gvisor) — noyau userspace, interception syscalls, runtime `runsc`.
- **Operator System Card** — OpenAI, jan. 2025. https://cdn.openai.com/operator_system_card.pdf — sandboxing + safety d'un computer-use agent.
- **Voice agents / Realtime guide** — OpenAI, 2024-2025. https://developers.openai.com/api/docs/guides/voice-agents — Realtime API speech-to-speech vs pipeline STT→LLM→TTS.
- **openai-realtime-agents (repo)** — OpenAI, 2024-. https://github.com/openai/openai-realtime-agents — handoffs/tools/guardrails sur Realtime.

### J24 — Model routing & coût/latence
- **RouteLLM: Learning to Route LLMs with Preference Data** — Ong et al. (LMSYS / UC Berkeley), 2024. https://arxiv.org/abs/2406.18665 — routeur strong/weak, >2× réduction de coût.
- **Prompt caching (Anthropic)** — doc, 2025. https://platform.claude.com/docs/en/build-with-claude/prompt-caching — `cache_control`, TTL.
- **Prompt Caching (OpenAI)** — doc. https://developers.openai.com/api/docs/guides/prompt-caching — caching automatique préfixe ≥1024 tokens, -80% latence/-90% coût input.

### J25 — Serving stateful & sessions à l'échelle
- **Persistence (LangGraph)** — LangChain. https://docs.langchain.com/oss/python/langgraph/persistence — checkpointers vs store, threads, fault tolerance.
- **Checkpoints API reference (langgraph)** — LangChain. https://reference.langchain.com/python/langgraph/checkpoints — `PostgresSaver`/`AsyncPostgresSaver`, `BaseCheckpointSaver`.
- **langgraph-redis** — Redis, 2025. https://github.com/redis-developer/langgraph-redis — `RedisSaver`, TTL, store cross-thread.

### J26 — Benchmarks agents & guardrails production
- **τ-bench: Tool-Agent-User Interaction...** — Yao et al. (Sierra), 2024. https://arxiv.org/abs/2406.12045 — pass^k (fiabilité multi-essais).
- **GAIA: a benchmark for General AI Assistants** — Mialon, Fourrier et al. (Meta AI, HF, AutoGPT), 2023. https://arxiv.org/abs/2311.12983 — humains 92% vs GPT-4 15%.
- **WebArena** — Zhou et al. (CMU), 2023. https://arxiv.org/abs/2307.13854 — 812 tâches web long-horizon.
- **SWE-bench** — Jimenez, Yang et al., 2023 (ICLR'24). https://arxiv.org/abs/2310.06770 — édition multi-fichiers + exécution.
- **AgentBench: Evaluating LLMs as Agents** — Liu et al. (Tsinghua, OSU, UC Berkeley), 2023 (ICLR'24). https://arxiv.org/abs/2308.03688 — 8 environnements.
- **OWASP Top 10 for LLM Applications 2025 (v2.0)** — OWASP GenAI Security Project, nov. 2024. https://genai.owasp.org/resource/owasp-top-10-for-llm-applications-2025/ — LLM01 Prompt Injection … LLM06 Excessive Agency, LLM10 Unbounded Consumption.

### J27–J28 — Capstone (assemblage)
Réutilise les sources ci-dessus : deep agent + durable execution (J16/J20) + coding/computer tools (J22/J23) + routing (J24) + persistence (J25) + harness d'éval style τ-bench/SWE-bench (J26).
