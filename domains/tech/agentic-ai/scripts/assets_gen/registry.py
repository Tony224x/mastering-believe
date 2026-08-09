"""Registre des schémas à générer."""

from __future__ import annotations

from collections.abc import Callable

from .builds import s1_fondations as s1
from .builds import s2_multi_prod as s2
from .builds import s3_frontier as s3
from .builds import s4_scale as s4
from .builds import s5_body_diagrams as s5
from .builds import s6_enrichment as s6
from .kit import wrap_svg


def _mem03() -> str:
    """J3 — mémoire (body via build_03_fixed)."""
    return wrap_svg(
        "Trois mémoires d'un agent",
        "Couches short-term, working memory, long-term, plus checkpoint pour reprise et debug.",
        s1.build_03_fixed(),
    )


SPECS: list[tuple[str, Callable[[], str]]] = [
    ("01-anatomie-agent", s1.build_01),
    ("02-tool-use-function-calling", s1.build_02),
    ("03-memory-state", _mem03),
    ("04-planning-reasoning", s1.build_04),
    ("05-langgraph-fondamentaux", s1.build_05),
    ("06-langgraph-avance", s1.build_06),
    ("07-agent-complet", s1.build_07),
    ("08-rag-agentique", s2.build_08),
    ("09-multi-agent-patterns", s2.build_09),
    ("10-mcp", s2.build_10),
    ("11-evaluation-testing", s2.build_11),
    ("12-production-observabilite", s2.build_12),
    ("13-securite-robustesse", s2.build_13),
    ("14-capstone", s2.build_14),
    ("15-context-engineering-compaction", s3.build_15),
    ("16-memoire-long-horizon", s3.build_16),
    ("17-verifiers-self-improvement", s3.build_17),
    ("18-orchestration-comparee-failure-modes", s3.build_18),
    ("19-protocoles-inter-agents", s3.build_19),
    ("20-durable-event-driven-agents", s3.build_20),
    ("21-coding-agents-architecture", s3.build_21),
    ("22-computer-use-gui-agents", s4.build_22),
    ("23-sandboxing-execution-sure", s4.build_23),
    ("24-inference-engineering", s4.build_24),
    ("25-serving-stateful-sessions", s4.build_25),
    ("26-benchmarking-pratique", s4.build_26),
    ("27-capstone-architecture", s4.build_27),
    ("28-capstone-build-eval", s4.build_28),
    ("parcours-28j", s4.build_parcours),
    # Schémas corps (remplacent ASCII / mermaid denses)
    ("08-rag-pipeline-complet", s5.build_08_pipeline),
    ("14-capstone-pipeline", s5.build_14_pipeline),
    ("03-memory-production", s5.build_03_production),
    ("13-defense-profondeur", s5.build_13_defense),
    ("07-research-agent-archi", s5.build_07_research_archi),
    ("11-eval-pipeline", s5.build_11_eval_pipeline),
    # Lot B/C — enrichissement
    ("09-hierarchical", s6.build_09_hierarchical),
    ("09-debate", s6.build_09_debate),
    ("09-swarm-handoff", s6.build_09_swarm),
    ("09-decision-tree", s6.build_09_decision_tree),
    ("06-agent-prod-complet", s6.build_06_agent_prod),
    ("06-parallel-vs-seq", s6.build_06_parallel),
    ("16-main-vs-external", s6.build_16_main_external),
    ("16-memory-flow", s6.build_16_memory_flow),
    ("10-mcp-lifecycle", s6.build_10_lifecycle),
    ("04-plan-execute", s6.build_04_plan_execute),
    ("04-tree-of-thought", s6.build_04_tot),
    ("22-perceive-mark-act", s6.build_22_pma),
    ("22-set-of-marks", s6.build_22_som),
    ("25-stateful-problem", s6.build_25_stateful),
    ("03-hybrid-memory", s6.build_03_hybrid),
    ("08-hybrid-rrf", s6.build_08_hybrid_rrf),
    ("14-deployment", s6.build_14_deployment),
    ("19-a2a-task-lifecycle", s6.build_19_a2a_lifecycle),
    ("20-durable-combined", s6.build_20_combined),
    ("24-three-levers", s6.build_24_levers),
    ("23-isolation-layers", s6.build_23_layers),
    ("15-context-isolation", s6.build_15_isolation),
    ("17-self-refine-loop", s6.build_17_self_refine),
]
