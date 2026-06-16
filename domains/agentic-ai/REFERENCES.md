# REFERENCES — agentic-ai (extension J15–J28)

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
