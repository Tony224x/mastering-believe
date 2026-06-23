# Gouvernance de l'IA — Gouverner une flotte d'agents en entreprise

## Scope

Ce domaine enseigne **comment gouverner l'IA quand elle devient agentique** : non pas l'ethique IA abstraite, mais le probleme operationnel concret qu'aucun dirigeant ne sait encore trancher — *combien d'agents tournent dans mon organisation, et qui les possede ?*

Angle **hybride** : on ancre la pratique dans les **cadres durables** (EU AI Act, NIST AI RMF, ISO/IEC 42001, RGPD) tout en suivant une **colonne vertebrale agentique** — les 4 piliers d'un agent gouvernable : **identite**, **proprietaire nomme**, **permissions**, **piste d'audit**. C'est le denominateur commun de tous les grands rapports 2025-2026 (Stanford HAI, McKinsey, Deloitte, BCG, EY, IMDA, Bain).

Le domaine est **Track Tech** : a code modere, tout est **runnable en Python stdlib** (aucune dependance, aucune cle API). On re-implemente en miniature les mecanismes reels (registry, contrôle d'acces scoped, audit log tamper-evident, policy engine, risk scorer) pour les *comprendre de l'interieur*, pas seulement en parler.

**Ce qu'on EXCLUT** : la construction d'agents (voir le domaine `agentic-ai`) ; la politique publique / geopolitique de l'IA ; le detail juridique exhaustif (on vise la maitrise operationnelle, pas le barreau).

**Public** : profil **T-shaped** — a la sortie, on sait **cadrer** la gouvernance d'agents pour un board/DSI **et** en **coder** l'ossature. Ancrage **EU-first + international** (contexte France/UE).

## Prerequisites

- Python intermediaire (dataclasses, fichiers, `hashlib`, JSON). Le domaine `algorithmie-python` suffit largement.
- Aucune connaissance juridique requise — les cadres sont expliques a partir de zero.
- Utile mais non requis : avoir parcouru `agentic-ai` (on gouverne ce qu'il apprend a construire).

## Planning (15 modules, ~45-60 min chacun)

| Jour | Module | Focus |
|------|--------|-------|
| J1  | Pourquoi gouverner l'IA agentique | L'ecart adoption/garde-fous, agent sprawl, shadow AI, l'enjeu reglementaire |
| J2  | Les 4 piliers d'un agent gouvernable | Identite · owner · permissions · audit trail · Agent Card |
| J3  | Inventaire & registry d'agents | Registry comme controle live, decouverte, requetes de gouvernance |
| J4  | Taxonomie des risques & NIST AI RMF | Govern/Map/Measure/Manage, AI Risk Repository, scoring |
| J5  | EU AI Act & gouvernance des tiers | 4 tiers de risque, GPAI, calendrier 2025-2027, due diligence fournisseur |
| J6  | Data governance & RGPD | Finalite, base legale, DPIA, articulation AI Act ↔ RGPD |
| J7  | Normes & AIMS | ISO/IEC 42001, OECD, IMDA agentique, crosswalk des cadres |
| J8  | Identite & IAM des agents | Non-human identity, scopes OAuth, Zero Trust, delegation |
| J9  | Audit, observabilite & tracabilite | Audit log tamper-evident (hash chain), OTel GenAI, reconstruction d'incident |
| J10 | Autonomie, garde-fous & operations | Niveaux d'autonomie, kill-switch, budgets, reponse a incident, decommission |
| J11 | Design organisationnel & boardroom | Agent managers, RACI, three lines model, role du board |
| J12 | Documentation & assurance | Model/system/agent cards, safety cases (preuve statique) |
| J13 | Evaluation, red-teaming & mesure | Evals de conformite ex-ante, attaques, scorecard |
| J14 | Policy-as-code & enforcement | Politiques executables (OPA/Rego, MCP), PDP/PEP runtime |
| J15 | **Capstone — Agent Governance Toolkit** | Integration end-to-end + rapport board-ready |

## Structure du contenu

- `01-theory/` — 15 modules theoriques (source-of-truth, francais)
- `02-code/` — scripts Python autonomes stdlib (registry, audit log, policy engine, risk scorer…)
- `03-exercises/` — exercices progressifs easy/medium/hard + solutions
- `04-projects/` — mini-projets libres lies au domaine
- `05-projets-guides/` — 3 projets appliques au contexte logistique FleetSim (voir `shared/logistics-context.md`)
- `REFERENCES.md` — 41 sources tier-1 verifiees par module
- `PLAN.md` — brief fige de chaque jour

## Capstone (J15) — Agent Governance Toolkit

Un **outil reutilisable** (pas un exercice jetable), runnable en stdlib pur, qui enchaine :

1. **Ingest** — charge une flotte d'agents dans un registry (identite + owner + permissions)
2. **Enforce** — applique des politiques de gouvernance executables (autonomie, budget, scope)
3. **Log** — journalise chaque action dans un audit trail tamper-evident (hash chain)
4. **Score** — evalue le risque de chaque agent (vraisemblance × impact, fonctions NIST RMF)
5. **Map** — mappe la conformite (EU AI Act tier, NIST RMF, ISO 42001)
6. **Report** — emet un **rapport de gouvernance board-ready** (markdown + JSON)

Directement transposable en mission : c'est l'ossature d'un audit de gouvernance d'agents.

## Criteres de reussite

- [ ] Repondre, donnees a l'appui, a « combien d'agents tournent ici et qui les possede »
- [ ] Definir pour un agent ses 4 piliers (identite, owner, permissions, audit) et reperer un agent non gouverne
- [ ] Classer un systeme IA dans les 4 tiers de l'EU AI Act et lister ses obligations + la deadline applicable
- [ ] Determiner si un agent declenche une DPIA et sous quelle base legale RGPD il opere
- [ ] Appliquer les 4 fonctions du NIST AI RMF (Govern/Map/Measure/Manage) a un systeme agentique
- [ ] Concevoir un modele d'identite + permissions scoped (moindre privilege) pour un agent
- [ ] Construire un audit log tamper-evident et reconstruire la trace d'une action
- [ ] Calibrer un niveau d'autonomie + budget + kill-switch sur le risque
- [ ] Dessiner un RACI / three-lines pour une flotte d'agents et le mandat board
- [ ] Ecrire une agent card + un safety case, et des politiques de gouvernance executables
- [ ] Livrer le capstone : un toolkit qui inventorie, enforce, journalise, score, mappe et rapporte

## Ressources externes (top 5 — liste exhaustive dans `REFERENCES.md`)

1. **EU AI Act** — Reglement (UE) 2024/1689 (texte officiel EUR-Lex)
2. **NIST AI Risk Management Framework 1.0** (AI 100-1) + GenAI Profile (AI 600-1)
3. **ISO/IEC 42001:2023** — AI Management System
4. **IMDA — Model AI Governance Framework for Agentic AI** (2026, 1er cadre dedie a l'agentique)
5. **OWASP Top 10 for Agentic Applications 2026** + CSA Agentic AI IAM
