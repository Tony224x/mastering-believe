# REFERENCES — Gouvernance de l'IA

> Sources tier-1 du domaine `gouvernance-ia` (Track Tech). Chaque source a ete **verifiee** (existence, editeur, annee, URL primaire) par un subagent de recherche le 2026-06-21. C'est la **source de verite** pour la creation des modules (Phase 4) et les passes de verification (Phase 5/6). Citer ces sources, pas des "docs officielles" generiques.
>
> **Angle** : hybride — cadres durables (EU AI Act, NIST AI RMF, ISO/IEC 42001) + colonne vertebrale agentique (identite, owner, permissions, audit trail). Ancrage **EU-first + international**.

---

## ⚠️ Faits a manier avec precaution (corrections verifiees)

Le poste LinkedIn d'origine (Clare Kitching) contient des imprecisions. **Ne PAS les reproduire** :

| Claim du poste | Realite verifiee | Regle |
|----------------|------------------|-------|
| « 58 % des dirigeants s'attendent a ce que l'IA remodele la gouvernance » | 58 % **parmi les adoptants extensifs de l'agentique** (vs 37 % chez les non-adoptants), source BCG AI Radar 2026 | Toujours preciser « parmi les adoptants extensifs » |
| « Cyber Signals 2026 » | Le rapport s'appelle **Cyber Pulse (Issue 1)**, Microsoft Security, 10 fev. 2026 | Nom exact = Cyber Pulse |
| « 29 % du personnel a utilise des modeles non approuves » (attribue a la telemetrie) | Vient d'une **enquete Microsoft/Hypothesis Group (juil. 2025, 1 700+ pros securite)**, PAS de la telemetrie Fortune 500 | Ne pas fusionner les deux sources |
| « 80 % du Fortune 500 utilisent des agents IA actifs » | CONFIRME (telemetrie first-party Copilot Studio/Agent Builder, fenetre nov. 2025) | OK a citer |

Autres points de vigilance :
- **AI Index 2026** : bien publie (edition 2026 confirmee). Stats Responsible AI verifiees : entreprises sans politique RAI 24 %→11 %, roles de gouvernance IA +17 % en 2025, 362 incidents IA recenses en 2025 (vs 233 en 2024).
- **McKinsey "State of AI"** : pas d'edition 2026 a ce jour ; la plus recente est **nov. 2025**.
- **OpenTelemetry GenAI semconv** : statut **Development/Experimental** au S1 2026 (PAS stable). Le code qui s'en inspire doit le signaler.

---

## Axe 1 — Adoption & ecart adoption/garde-fous (the gap) → J1, J3

1. **The 2026 AI Index Report — Ch. 3 Responsible AI** — Stanford HAI, 2026.
   https://hai.stanford.edu/ai-index/2026-ai-index-report/responsible-ai
   *Pourquoi* : barometre academique de reference sur l'ecart capacites/garde-fous (gouvernance, incidents, transparence).
2. **Cyber Pulse, Issue 1 — "80% of the Fortune 500 use active AI agents"** — Microsoft Security, 10 fev. 2026.
   https://www.microsoft.com/en-us/security/blog/2026/02/10/80-of-fortune-500-use-active-ai-agents-observability-governance-and-security-shape-the-new-frontier/
   *Pourquoi* : la telemetrie qui chiffre la proliferation d'agents et le shadow AI. **Source directe du poste.**
3. **AI agents are scaling faster than their guardrails (State of AI in the Enterprise 2026)** — Deloitte Insights, 24 avr. 2026.
   https://www.deloitte.com/us/en/insights/topics/emerging-technologies/ai-agents-scaling-faster.html
   *Pourquoi* : nomme et quantifie l'ecart « adoption > gouvernance » (seulement 21 % ont une gouvernance agentique mature ; 74 % prevoient un deploiement d'agents d'ici 2027, n=3 235 dirigeants).
4. **The State of AI: How organizations are rewiring to capture value** — McKinsey (QuantumBlack), nov. 2025.
   https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai
   *Pourquoi* : 88 % des organisations utilisent regulierement l'IA, mais 23 % seulement scalent un systeme agentique et 39 % rapportent un impact EBIT — l'ecart valeur.
5. **AI Radar 2026 — "As AI Investments Surge, CEOs Take the Lead"** — BCG, janv. 2026.
   https://www.bcg.com/publications/2026/as-ai-investments-surge-ceos-take-the-lead
   *Pourquoi* : chiffre l'attente d'une refonte de la gouvernance et des droits de decision (58 % parmi les adoptants extensifs). Complement : MIT SMR × BCG, *The Emerging Agentic Enterprise*, nov. 2025.

## Axe 2 — Cadres reglementaires & normatifs → J5, J6 (NIST aussi J4)

1. **Reglement (UE) 2024/1689 — AI Act (texte officiel)** — UE / EUR-Lex, adopte 13 juin 2024, en vigueur **1 aout 2024**.
   https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng
   *Faits verifies* : 4 tiers de risque (inacceptable / haut / limite / minimal). Calendrier (Art. 113) : interdictions + litteratie IA → **2 fev. 2025** ; obligations GPAI → **2 aout 2025** ; application generale + haut risque Annexe III → **2 aout 2026** ; haut risque produits Annexe I → **2 aout 2027**.
2. **AI Act — page officielle Commission & AI Act Explorer (timeline)** — Commission europeenne / Future of Life Institute.
   https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai · https://artificialintelligenceact.eu/implementation-timeline/
   *Pourquoi* : navigation article-par-article + calendrier consolide sans halluciner les dates.
3. **NIST AI Risk Management Framework (AI RMF 1.0), NIST AI 100-1** — NIST, **26 janv. 2023**.
   https://nvlpubs.nist.gov/nistpubs/ai/nist.ai.100-1.pdf
   *Faits verifies* : 4 fonctions cœur **Govern, Map, Measure, Manage** (Govern transversale). Le pivot du module risque.
4. **AI RMF: Generative AI Profile, NIST AI 600-1** — NIST, **26 juil. 2024**.
   https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf
   *Pourquoi* : profil GenAI du RMF, 12 categories de risques + 200+ actions suggerees.
5. **ISO/IEC 42001:2023 — AI Management System (AIMS)** — ISO/IEC, 2023.
   https://www.iso.org/standard/42001
   *Pourquoi* : 1re norme certifiable de management de l'IA (l'« ISO 27001 de l'IA »).
6. **OECD AI Principles (Recommendation on AI)** — OCDE, 2019, **mis a jour mai 2024**.
   https://oecd.ai/en/ai-principles
   *Pourquoi* : 1er standard intergouvernemental (5 principes + 5 recommandations), socle commun G20/UE.
7. **Model AI Governance Framework for Generative AI** — IMDA & AI Verify Foundation (Singapour), 30 mai 2024.
   https://aiverifyfoundation.sg/wp-content/uploads/2024/05/Model-AI-Governance-Framework-for-Generative-AI-May-2024-1-1.pdf
   *Pourquoi* : cadre asiatique pro-innovation (9 dimensions), contrepoint a l'approche UE.
8. **Model AI Governance Framework for Agentic AI** — IMDA (Singapour), **22 janv. 2026**.
   https://www.imda.gov.sg/-/media/imda/files/about/emerging-tech-and-research/artificial-intelligence/mgf-for-agentic-ai.pdf
   *Pourquoi* : **premier cadre officiel au monde dedie specifiquement a l'IA agentique** — « les humains restent ultimement responsables ». Pile dans l'axe du domaine.

## Axe 2bis — Data governance & RGPD pour l'IA agentique → J6

1. **Reglement (UE) 2016/679 — RGPD / GDPR (texte officiel)** — UE / EUR-Lex, 2016, applicable depuis 25 mai 2018.
   https://eur-lex.europa.eu/eli/reg/2016/679/oj
   *Pourquoi* : socle de la protection des donnees en UE — finalite, minimisation, base legale, droits des personnes, DPIA (Art. 35). S'applique des qu'un agent traite des donnees personnelles.
2. **EDPB — Opinion 28/2024 on data protection aspects of AI models** — Comite europeen de la protection des donnees (EDPB), adoptee 17 dec. 2024.
   https://www.edpb.europa.eu/our-work-tools/our-documents/opinion-board-art-64/opinion-282024-certain-data-protection-aspects_en
   *Pourquoi* : interprete l'articulation modeles IA ↔ RGPD (anonymat des modeles, interet legitime, consequences d'un traitement illicite). *(URL a reconfirmer en Phase 6.)*
3. **CNIL — IA & RGPD (recommandations / fiches pratiques)** — CNIL, 2024-2025.
   https://www.cnil.fr/en/artificial-intelligence
   *Pourquoi* : doctrine operationnelle francaise (base legale d'un entrainement, reutilisation de donnees, information des personnes) — directement applicable en mission FR/UE.

## Axe 3 — Gestion du risque, taxonomies & documentation → J4, J12, J13

1. **The AI Risk Repository** — Slattery, Saeri, Grundy et al., 2024 (arXiv:2408.12622 ; version *Patterns* 2026).
   https://airisk.mit.edu/ · https://arxiv.org/abs/2408.12622
   *Pourquoi* : base vivante de 1 700+ risques IA extraits de 65 frameworks, 2 taxonomies (causale + par domaine).
2. **Model Cards for Model Reporting** — Mitchell, Wu, Zaldivar, Barnes, Vasserman, Hutchinson, Spitzer, Raji, Gebru, 2019, *FAT\* '19* (ACM).
   https://arxiv.org/abs/1810.03993 · DOI 10.1145/3287560.3287596
   *Pourquoi* : paper seminal de la documentation standardisee des modeles.
3. **GPT-4 System Card** — OpenAI, 15 mars 2023.
   https://cdn.openai.com/papers/gpt-4-system-card.pdf
   *Pourquoi* : exemple primaire canonique de « system card » d'un modele frontier (alternatives : GPT-4o System Card, system cards Anthropic/Claude).
4. **Safety Cases: How to Justify the Safety of Advanced AI Systems** — Clymer, Gabrieli, Krueger, Larsen, 2024 (arXiv:2403.10462).
   https://arxiv.org/abs/2403.10462
   *Pourquoi* : cadre fondateur des « safety cases » IA (inability / control / trustworthiness / deference).
5. **NIST AI RMF Playbook** — NIST (ITL), 2023.
   https://www.nist.gov/itl/ai-risk-management-framework/nist-ai-rmf-playbook
   *Pourquoi* : actions concretes pour operationnaliser Govern/Map/Measure/Manage.
6. **Adversarial Machine Learning: A Taxonomy and Terminology (NIST AI 100-2 E2025)** — Vassilev, Oprea, Fordyce, Anderson (UK AISI), Hamin (US AISI), mars 2025.
   https://csrc.nist.gov/pubs/ai/100/2/e2025/final
   *Pourquoi* : taxonomie officielle des attaques (prompt injection, fuite d'info) ; 1re edition a nommer les agents autonomes comme surface de menace.

## Axe 4 — Gouvernance d'entreprise & boardroom → J9, J10

1. **Agentic AI at Scale: Redefining Management for a Superhuman Workforce** — MIT Sloan Management Review × BCG, 16 sept. 2025.
   https://sloanreview.mit.edu/article/agentic-ai-at-scale-redefining-management-for-a-superhuman-workforce/
   *Pourquoi* : la supervision concue pour des humains ne suffit plus ; redevabilite explicite pour des agents autonomes. **Source du poste.**
2. **To Thrive in the AI Era, Companies Need Agent Managers** — Harvard Business Review (Srinivasan & Wei), fev. 2026.
   https://hbr.org/2026/02/to-thrive-in-the-ai-era-companies-need-agent-managers
   *Pourquoi* : emergence du role de « gestionnaire d'agents » pour orchestrer des fleets. **Source du poste.**
3. **The board's role in managing emerging AI risks** — McKinsey (avec la NACD), 2025.
   https://www.mckinsey.com/capabilities/mckinsey-technology/overview/cybersecurity/the-boards-role-in-managing-emerging-ai-risks
   *Pourquoi* : qui detient l'oversight (board vs comites vs management) ; <25 % ont une politique IA approuvee par le conseil. **Source du poste.**
4. **Six Steps to Enhance Governance and Increase Agentic AI's Value** — EY (Canada, Zhu & Cobey), janv. 2026.
   https://www.ey.com/en_ca/insights/assurance/technology-risk/six-steps-to-enhance-agentic-ai-governance
   *Pourquoi* : 6 pratiques calibrees sur le risque (bornes du domaine d'operation de l'agent, controle des appels d'API). **Source du poste.**
5. **Governance, Trust, and the Data Foundation** — Bain & Company, 2025.
   https://www.bain.com/insights/governance-trust-and-the-data-foundation/
   *Pourquoi* : la gouvernance comme prerequis (pas couche a posteriori) ; s'etend des outputs aux **actions** des agents. **Source du poste.**
6. **The IIA's Three Lines Model (update of the Three Lines of Defense)** — Institute of Internal Auditors, juil. 2020.
   https://www.theiia.org/globalassets/documents/resources/the-iias-three-lines-model-an-update-of-the-three-lines-of-defense-july-2020/three-lines-model-updated-english.pdf
   *Pourquoi* : cadre canonique de redevabilite distribuee (governing body / management / internal audit) applique a l'IA.

## Axe 5 — Identite, IAM, securite & tracabilite des agents → J2, J7, J8

1. **Agentic AI Identity and Access Management: A New Approach** — Cloud Security Alliance (CSA), 2025.
   https://cloudsecurityalliance.org/artifacts/agentic-ai-identity-and-access-management-a-new-approach
   *Pourquoi* : IAM purpose-built pour agents (autonomie, ephemerite, delegation) via DID + verifiable credentials + Zero Trust. La reference « identite des agents / Non-Human Identity ».
2. **OWASP Top 10 for LLM Applications 2025 (v2.0)** — OWASP Gen AI Security Project, publie 18 nov. 2024.
   https://genai.owasp.org/resource/owasp-top-10-for-llm-applications-2025/
   *Pourquoi* : taxonomie des risques LLM (LLM01–LLM10:2025, dont excessive agency, system prompt leakage).
3. **OWASP Top 10 for Agentic Applications for 2026** — OWASP Gen AI Security Project, publie 9 dec. 2025.
   https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/
   *Pourquoi* : taxonomie ASI01–ASI10 specifique aux agents (ASI01 Agent Goal Hijack, ASI02 Tool Misuse, **ASI03 Identity and Privilege Abuse**, … ASI10 Rogue Agents). Pendant agentique exact des 4 piliers.
4. **NIST SP 800-207 — Zero Trust Architecture** — NIST (Rose et al.), aout 2020.
   https://csrc.nist.gov/pubs/sp/800/207/final
   *Pourquoi* : standard d'autorite pour le moindre privilege et l'acces per-request base identite/contexte.
5. **OpenTelemetry Semantic Conventions for Generative AI** — OpenTelemetry GenAI SIG (CNCF), statut **Development/Experimental** (S1 2026).
   https://opentelemetry.io/docs/specs/semconv/gen-ai/
   *Pourquoi* : convention vendor-neutre d'instrumentation (spans LLM, tool calls, token usage) — base technique de la « piste d'audit ». **Statut non stable a signaler.**

## Axe 6 — Policy-as-code, garde-fous, evals & tooling → J12, J13, J14

1. **Open Policy Agent (OPA) & Rego** — projet OPA (CNCF *graduated*), doc 2026.
   https://www.openpolicyagent.org/docs
   *Pourquoi* : la reference « politiques executables » — modele mental reimplementable en mini en Python stdlib.
2. **Model Context Protocol — Specification (rev. 2025-11-25, section Security & Trust)** — Anthropic / projet MCP open-source.
   https://modelcontextprotocol.io/specification/2025-11-25
   *Pourquoi* : surface de gouvernance n°1 des agents (consentement, *Tool Safety*, permissions tools/resources). JSON-RPC reproductible en stdlib.
3. **NVIDIA NeMo Guardrails** — NVIDIA, open-source (Apache 2.0).
   https://docs.nvidia.com/nemo/guardrails/latest/index.html
   *Pourquoi* : garde-fous runtime (input/output rails, moderation, jailbreak/injection). Archetype « intercepter entrees/sorties ».
4. **Guardrails AI — Validators & Hub** — Guardrails AI, open-source (Apache 2.0), v0.9.2 (mars 2026).
   https://www.guardrailsai.com/docs
   *Pourquoi* : 2e framework runtime, plus « Pythonic » (validators composables) — facile a mimer en stdlib.
5. **Inspect AI** — UK AI Security Institute (AISI) + Meridian Labs, open-source.
   https://inspect.aisi.org.uk/
   *Pourquoi* : framework d'evaluation/assurance (datasets, scorers, agentic tasks) — socle pour des evals de conformite.
6. **(Emergent) Agent registry / inventaire** — Microsoft Entra Agent ID (*Preview* dec. 2025) ; Google A2A Agent Card (avr. 2025).
   *Pourquoi* : « agent inventory = controle live, pas un tableur » (registry + Agent Card JSON). **Pratique emergente, non figee** — le reproductible (registre JSON + Agent Card) est plus durable que les produits cites.

---

## Mapping source → modules (recapitulatif — plan fige a 15 jours)

| Module | Sources principales |
|--------|---------------------|
| J1 — Pourquoi gouverner l'IA agentique (+ enjeu reglementaire) | Axe 1 (Cyber Pulse, Deloitte, AI Index, BCG, McKinsey) |
| J2 — Les 4 piliers d'un agent gouvernable | Axe 5 (CSA IAM, OWASP Agentic 2026) |
| J3 — Inventaire & registry d'agents | Axe 1 + Axe 6 (Entra Agent ID, A2A) |
| J4 — Taxonomie des risques & NIST AI RMF | Axe 3 (AI Risk Repository, RMF Playbook) + Axe 2 (NIST AI 100-1, 600-1) |
| J5 — EU AI Act & gouvernance des tiers/GPAI | Axe 2 (Reglement 2024/1689, AI Act Explorer) |
| J6 — Data governance & RGPD pour l'IA | Axe 2bis (RGPD, EDPB Opinion 28/2024, CNIL) |
| J7 — Normes & AIMS (ISO 42001, OECD, IMDA) | Axe 2 (ISO 42001, OECD, IMDA GenAI + Agentic) |
| J8 — Identite & IAM des agents (deep technique) | Axe 5 (CSA IAM, NIST SP 800-207, OWASP Agentic) |
| J9 — Audit, observabilite & tracabilite runtime | Axe 5 (OTel GenAI semconv) + Axe 3 |
| J10 — Autonomie, garde-fous & operations (incident, lifecycle) | Axe 4 (EY, Bain) + Axe 2 (IMDA Agentic) |
| J11 — Design organisationnel & boardroom | Axe 4 (MIT SMR/BCG, HBR, McKinsey board, IIA Three Lines) |
| J12 — Documentation & assurance (preuve statique) | Axe 3 (Model Cards, System Cards, Safety Cases) |
| J13 — Evaluation, red-teaming & mesure (ex-ante) | Axe 3 (NIST AI 100-2) + Axe 6 (Inspect AI) + Axe 5 (OWASP LLM) |
| J14 — Policy-as-code & enforcement (runtime) | Axe 6 (OPA/Rego, MCP security, NeMo/Guardrails AI) |
| J15 — Capstone : Agent Governance Toolkit | Axe 6 (MCP, OPA, registry) + synthese transverse |
