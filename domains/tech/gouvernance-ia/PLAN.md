# Plan fige — domaine `gouvernance-ia` (Track Tech, 15 jours)

> **Contrat fige.** Chaque subagent Phase 4 lit CE fichier pour le contexte global, puis construit UNIQUEMENT son jour. Les autres jours n'existent pas encore — ne pas tenter de les lire.

## Contexte global (a lire par tous les subagents)

- **Domaine** : gouvernance de l'IA, angle **hybride** — cadres durables (EU AI Act, NIST AI RMF, ISO/IEC 42001, RGPD) + **colonne vertebrale agentique** (les 4 piliers : identite, owner nomme, permissions, audit trail).
- **Public cible** : profil **T-shaped** — capable de (a) cadrer la gouvernance d'une flotte d'agents pour un board/DSI ET (b) en coder l'ossature technique. Contexte **France / UE** (ancrage EU-first + international).
- **Langue** : prose en **francais**, code (identifiers + commentaires) en **anglais**.
- **Stack** : **Python 3.11+ stdlib UNIQUEMENT**. Pas de dependance externe, pas d'API LLM, pas de cle. Le code illustre les mecanismes de gouvernance par des mini-implementations runnables (registry, audit log, policy engine, risk scorer). Si un concept renvoie a un outil reel (OPA, MCP, OTel), le **re-implementer en miniature en stdlib** et citer l'outil reel en commentaire.
- **Posture** : factuelle et non-dogmatique. La gouvernance est un domaine ou les cadres evoluent — **dater les faits**, distinguer ce qui est **obligatoire** (loi) de ce qui est **recommande** (norme/cadre). Ne PAS halluciner de dates : utiliser exactement les dates de REFERENCES.md. Voir l'encadre ⚠️ de REFERENCES.md (corrections : "58% adoptants extensifs", "Cyber Pulse", "29% enquete distincte").
- **Format par module** (rappel) : theorie 30-60 min, exemple concret AVANT principe abstrait, sections H2 numerotees, "Key takeaway" par section, ≥1 citation `[Auteur/Org, Annee]` vers une source autorisee, bloc final `## Spaced repetition` (3-5 Q&A). Code = fichier `.py` plat, docstring d'entete, `if __name__ == "__main__":`, commentaires sur le WHY. Exercices = meme slug en easy/medium/hard avec `## Objectif` / `## Consigne` / `## Criteres de reussite`. Solutions = 1 fichier `# === EASY/MEDIUM/HARD ===` + smoke test.
- **Continuite du fil-rouge code** : autant que possible, les jours reutilisent un mini-modele commun d'agent (un `dict`/`dataclass` d'agent avec `id`, `owner`, `permissions/scopes`, `risk_tier`). Le capstone J15 integre tout. Reste neanmoins **autonome** : chaque fichier tourne seul (ne pas importer un autre jour).

---

## J1 — Pourquoi gouverner l'IA agentique
- **Concepts cles** : l'ecart adoption/garde-fous (agent sprawl, shadow AI) ; la question fondatrice « combien d'agents tournent chez nous, et qui les possede ? » ; pourquoi la gouvernance **agentique** differe de la gouvernance IA classique (les agents *agissent* : appellent des outils, executent des transactions) ; apercu de l'enjeu reglementaire (l'EU AI Act existe et contraint — motive tout le parcours) ; apercu des 4 piliers (identite/owner/permissions/audit).
- **Acquis a la fin du jour** : savoir cadrer et **chiffrer** le probleme avec des donnees verifiees ; expliquer a un dirigeant pourquoi l'absence d'inventaire d'agents est un risque ; situer les grands cadres (EU AI Act, NIST, ISO).
- **Sources autorisees** : Microsoft Cyber Pulse (2026) ; Deloitte State of AI in the Enterprise 2026 ; Stanford HAI AI Index 2026 (ch. Responsible AI). *(Respecter les corrections de l'encadre ⚠️ : 80% Fortune 500 = OK ; 29% shadow AI = enquete distincte Hypothesis Group ; ne pas inventer le "58%".)*
- **Stack** : python stdlib — *agent sprawl census* : compter des agents, detecter ceux sans owner (« shadow/orphelins »), produire un chiffre de couverture de gouvernance.
- **Slug** : `01-pourquoi-gouverner`

## J2 — Les 4 piliers d'un agent gouvernable
- **Concepts cles** : les 4 piliers — **identite** (qui est l'agent, de maniere unique), **owner nomme** (un humain responsable), **permissions** (ce qu'il a le droit de faire — moindre privilege en apercu), **audit trail** (trace verifiable de ce qu'il a fait) ; la notion d'**Agent Card** (declaration de capacites/auth) ; pourquoi les 4 sont indissociables (un agent sans owner = ingerable ; sans audit = non-prouvable) ; le « smell test » d'un agent *ungoverned*.
- **Acquis a la fin du jour** : definir pour un agent donne ses 4 attributs de gouvernance ; reconnaitre et qualifier un agent non gouverne.
- **Sources autorisees** : CSA — Agentic AI Identity and Access Management (2025) ; OWASP Top 10 for Agentic Applications 2026 (ASI03 = Identity & Privilege Abuse, etc.).
- **Stack** : python stdlib — `@dataclass GovernedAgent` + validateur de completude de gouvernance (les 4 piliers presents et non vides ?).
- **Slug** : `02-quatre-piliers`

## J3 — Inventaire & registry d'agents
- **Concepts cles** : inventaire d'agents vs simple tableur ; le **registry comme controle live** (source de verite, pas un export fige) ; decouverte (telemetrie/scan) ; enrolement et cycle de vie d'inscription ; Agent Card (Google A2A) et Microsoft Entra Agent ID (**pratiques emergentes**, non figees) ; requetes de gouvernance (« qui possede quoi », « agents orphelins », « agents par tier de risque »).
- **Acquis a la fin du jour** : construire un mini-registry d'agents avec owner/permissions ; repondre par requete a « combien d'agents, qui les possede ».
- **Sources autorisees** : Microsoft Cyber Pulse (la question) ; Google A2A / Microsoft Entra Agent ID (emergent — presenter comme tel) ; CSA Agentic IAM.
- **Stack** : python stdlib — registry JSON (CRUD + requetes : by_owner, orphans, by_risk). Persistance fichier JSON.
- **Slug** : `03-inventaire-registry`

## J4 — Taxonomie des risques & NIST AI RMF
- **Concepts cles** : NIST AI RMF — 4 fonctions **Govern / Map / Measure / Manage** (Govern transversale) ; AI Risk Repository (taxonomie causale + par domaine) ; risques **propres aux agents** (excessive agency, tool misuse, actions irreversibles) ; risque = vraisemblance × impact ; profil GenAI (NIST 600-1) en complement.
- **Acquis a la fin du jour** : classer un risque dans la taxonomie ; appliquer les 4 fonctions du RMF a un systeme agentique ; produire un score de risque defendable.
- **Sources autorisees** : NIST AI RMF 1.0 (AI 100-1, 2023) ; MIT AI Risk Repository (Slattery et al., 2024) ; NIST GenAI Profile (AI 600-1, 2024).
- **Stack** : python stdlib — risk register + scorer (vraisemblance × impact, tagging par fonction RMF, tri par criticite).
- **Slug** : `04-risques-nist-rmf`

## J5 — EU AI Act & gouvernance des tiers/GPAI
- **Concepts cles** : les **4 tiers de risque** (inacceptable / haut / limite / minimal) ; haut risque Annexe III ; obligations **GPAI** ; **calendrier** d'application (2 fev. 2025 interdictions ; 2 aout 2025 GPAI ; 2 aout 2026 haut risque Annexe III ; 2 aout 2027 Annexe I) ; ou se situent les **systemes agentiques** ; **gouvernance des tiers** : due diligence fournisseur, modeles/agents achetes, obligations downstream GPAI.
- **Acquis a la fin du jour** : classer un systeme par tier ; lister ses obligations ; situer la deadline applicable ; cadrer une due diligence fournisseur minimale.
- **Sources autorisees** : Reglement (UE) 2024/1689 (texte officiel) ; AI Act Explorer / page Commission (timeline). *(Dates exactes : voir REFERENCES.md, ne pas approximer.)*
- **Stack** : python stdlib — classifieur de tier a base de regles (questionnaire → tier → obligations → deadline).
- **Slug** : `05-eu-ai-act`

## J6 — Data governance & RGPD pour l'IA agentique
- **Concepts cles** : **finalite** & **minimisation** ; **base legale** (consentement, interet legitime, contrat) ; donnees d'entrainement vs d'inference ; **memoire d'agent** & donnees personnelles ; **DPIA/AIPD** (Art. 35 RGPD) ; transferts ; droits des personnes (acces, effacement) ; **articulation AI Act ↔ RGPD** (deux regimes cumulatifs) ; position EDPB sur les modeles IA.
- **Acquis a la fin du jour** : determiner si un agent traite des donnees personnelles et sous quelle base legale ; declencher (ou non) une DPIA ; articuler obligations RGPD et AI Act.
- **Sources autorisees** : RGPD — Reglement (UE) 2016/679 ; EDPB Opinion 28/2024 (modeles IA) ; CNIL — IA & RGPD.
- **Stack** : python stdlib — mini-DPIA / data-flow assessor pour un agent (detecte donnees perso, exige base legale & retention, signale si DPIA requise).
- **Slug** : `06-data-governance-rgpd`

## J7 — Normes & systemes de management de l'IA (AIMS)
- **Concepts cles** : **ISO/IEC 42001** (AI Management System, boucle PDCA, certifiable — l'« ISO 27001 de l'IA ») ; **OECD AI Principles** (5 valeurs + 5 recommandations) ; **IMDA** (Model AI Governance Framework GenAI + **Agentic 2026**) ; **crosswalk** des cadres (un meme controle satisfait plusieurs referentiels) ; obligatoire (loi) vs volontaire (norme).
- **Acquis a la fin du jour** : comprendre ce qu'est un AIMS et la logique PDCA ; mapper une exigence a travers EU AI Act / NIST / ISO ; situer l'apport specifique du cadre agentique IMDA.
- **Sources autorisees** : ISO/IEC 42001:2023 ; OECD AI Principles (maj 2024) ; IMDA Model AI Governance Framework for Agentic AI (jan. 2026).
- **Stack** : python stdlib — control crosswalk mapper (controle interne → {article AI Act, fonction NIST RMF, clause ISO 42001}, calcule la couverture).
- **Slug** : `07-normes-aims`

## J8 — Identite & IAM des agents (deep technique)
- **Concepts cles** : **non-human / machine identity** ; **ephemerite** (identites a duree de vie courte) ; **delegation** (agir *on-behalf-of* un humain, chaine de delegation) ; **scopes OAuth** & moindre privilege operationnel ; **Zero Trust** (NIST SP 800-207 : verifier chaque requete) ; gestion des secrets/credentials ; deprovisionnement. *(Distinct de J2 : J2 = anatomie/4 piliers cote board ; J8 = mecanique d'acces cote ingenieur.)*
- **Acquis a la fin du jour** : concevoir un modele d'identite + permissions scoped pour un agent ; appliquer le moindre privilege ; modeliser une delegation.
- **Sources autorisees** : CSA Agentic AI IAM (2025) ; NIST SP 800-207 (Zero Trust) ; OWASP Top 10 Agentic 2026 (Identity & Privilege Abuse).
- **Stack** : python stdlib — scope-based access control engine (decision allow/deny per-request, verification de scope, chaine de delegation, expiration).
- **Slug** : `08-identite-iam`

## J9 — Audit, observabilite & tracabilite runtime
- **Concepts cles** : audit trail **append-only** & **tamper-evident** (chaine de hash) ; **OTel GenAI semantic conventions** (spans LLM, tool calls, token usage — **statut experimental** a signaler) ; reconstruction d'incident (qui / quoi / quand / sur quelle autorisation) ; retention & integrite ; « preuve **machine** » (par opposition a la doc statique de J12).
- **Acquis a la fin du jour** : construire un audit log verifiable (detecte toute alteration) ; reconstruire la trace complete d'une action d'agent.
- **Sources autorisees** : OpenTelemetry GenAI semconv (experimental, S1 2026) ; CSA Agentic IAM ; (NIST AI 100-2 pour le lien incident/attaque).
- **Stack** : python stdlib (`hashlib`) — audit log chaine par hash + verifier d'integrite + reconstruction d'incident.
- **Slug** : `09-audit-tracabilite`

## J10 — Autonomie, garde-fous & operations
- **Concepts cles** : **niveaux d'autonomie** (human-in-the-loop / on-the-loop / out-of-the-loop) ; garde-fous (guardrails) ; **kill-switch** ; **budgets** (cout / nb d'actions / tokens) & escalade ; **reponse a incident** (detect → contain → eradicate → recover) ; **cycle de vie & decommission** d'un agent. *(Recupere la charge "incident/lifecycle" pour ne pas surcharger J14.)*
- **Acquis a la fin du jour** : calibrer un niveau d'autonomie sur le risque ; implementer un budget + un kill-switch ; derouler une reponse a incident et un decommission propre.
- **Sources autorisees** : EY — Six Steps to Enhance Agentic AI Governance (2026) ; Bain — Governance, Trust & Data Foundation (2025) ; IMDA Agentic (2026).
- **Stack** : python stdlib — autonomy gate (enforce budget, escalade HITL au-dela d'un seuil de risque, kill-switch, machine a etats d'incident).
- **Slug** : `10-autonomie-gardefous`

## J11 — Design organisationnel & boardroom
- **Concepts cles** : **agent managers** (nouveau role, HBR) ; **RACI** applique aux agents ; **Three Lines Model** (IIA — governing body / management / audit interne) ; **role du board** (oversight, qui decide quoi — McKinsey/NACD) ; comites IA ; accountability distribuee. *(Distinct de J3 : J3 = registry technique ; J11 = qui-est-responsable cote organisation.)*
- **Acquis a la fin du jour** : dessiner un RACI + un mapping three-lines pour une flotte d'agents ; definir le mandat d'oversight du board.
- **Sources autorisees** : HBR — Companies Need Agent Managers (2026) ; McKinsey — The board's role in managing emerging AI risks (2025) ; IIA — Three Lines Model (2020).
- **Stack** : python stdlib — RACI / ownership resolver sur le registry (qui est accountable par agent, mapping three-lines, detection de trous de responsabilite).
- **Slug** : `11-design-organisationnel`

## J12 — Documentation & assurance (preuve statique)
- **Concepts cles** : **model cards** (Mitchell et al., 2019) ; **system cards** (ex. GPT-4) ; **agent cards** ; **safety cases** (Clymer et al., 2024 : arguments inability / control / trustworthiness / deference) ; argument d'assurance structure ; transparence ; « preuve **humaine**/conformite » (par opposition a la trace runtime de J9).
- **Acquis a la fin du jour** : rediger une agent/system card complete ; structurer un safety case minimal et defendable.
- **Sources autorisees** : Model Cards for Model Reporting (Mitchell et al., 2019) ; GPT-4 System Card (OpenAI, 2023) ; Safety Cases (Clymer et al., 2024).
- **Stack** : python stdlib — generateur d'agent/system card (template structure → markdown) + squelette de safety case (claims → evidence → gaps).
- **Slug** : `12-documentation-assurance`

## J13 — Evaluation, red-teaming & mesure (ex-ante)
- **Concepts cles** : **evals de gouvernance** (tester AVANT de deployer) ; **red-teaming** & taxonomie d'attaques (NIST AI 100-2 : prompt injection, exfiltration) ; **Inspect AI** (datasets / scorers / agentic tasks — concept) ; metriques de conformite (taux de blocage, couverture) ; fonction **Measure** du RMF ; OWASP LLM (excessive agency). *(Distinct de J14 : J13 = mesurer ex-ante ; J14 = enforcer en runtime.)*
- **Acquis a la fin du jour** : concevoir une eval de conformite simple ; lancer un scenario de red-team contre un garde-fou ; produire un scorecard.
- **Sources autorisees** : NIST AI 100-2 E2025 (adversarial ML) ; Inspect AI (UK AISI) ; OWASP Top 10 LLM 2025.
- **Stack** : python stdlib — mini eval harness (suite de prompts adverses + scorers) testant un garde-fou simple, produit un compliance scorecard.
- **Slug** : `13-evaluation-redteaming`

## J14 — Policy-as-code & enforcement (runtime)
- **Concepts cles** : **policy-as-code** (modele mental OPA/Rego : politiques declaratives versionnees) ; **PDP/PEP** (policy decision/enforcement point) ; regles executables (autonomie / budget / scope / donnees) ; **MCP** comme surface de permission (Tool Safety, consentement, permissions tools/resources) ; test & drift des politiques. *(Distinct de J13 : ici on **bloque** au moment de l'action.)*
- **Acquis a la fin du jour** : ecrire des politiques de gouvernance executables ; les faire respecter a un point d'enforcement ; comprendre MCP comme surface de controle.
- **Sources autorisees** : Open Policy Agent / Rego (CNCF) ; MCP Specification (section Security & Trust, rev. 2025-11-25) ; NVIDIA NeMo Guardrails OU Guardrails AI.
- **Stack** : python stdlib — mini policy engine (regles declaratives evaluees contre une action d'agent → allow / deny / oblige) + gate de permission facon MCP.
- **Slug** : `14-policy-as-code`

## J15 — Capstone : Agent Governance Toolkit
- **Concepts cles** : **integration end-to-end** des 14 modules — registry (identite + owner + permissions) → policy engine (autonomie/budget/scope) → audit trail tamper-evident → risk scorer → **mapping conformite** (EU AI Act / NIST RMF / ISO 42001) → **generateur de rapport board-ready** (markdown + JSON). Le capstone est un **outil reutilisable** (utilisable en mission), pas un exercice jetable.
- **Acquis a la fin du jour** : disposer d'un toolkit runnable qui inventorie une flotte d'agents, applique des politiques, journalise de maniere verifiable, score le risque, mappe la conformite et emet un rapport de gouvernance pour un comite.
- **Sources autorisees** : synthese transverse — MCP, OPA/Rego, NIST AI RMF, EU AI Act, IMDA Agentic. (Capstone : citer le module-source de chaque brique.)
- **Stack** : python stdlib — **toolkit complet (~2-3× un jour normal)**, oriente CLI/fonctions : `ingest → enforce → log → score → map → report`. Doit tourner end-to-end sur un jeu d'agents d'exemple et produire un rapport.
- **Slug** : `15-capstone-governance-toolkit`
