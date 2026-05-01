# 05 — Projets guides (Neural Networks & LLMs)

> Voir `shared/logistics-context.md` pour le contexte metier de LogiSim.

Trois projets ML / LLM appliques aux traces FleetSim. Le fil rouge : **transformer des traces de shift en intelligence exploitable** pour les operateurs OCC et pour enrichir les comportements autonomes des flottes.

## Projets

| # | Projet | Concepts couverts | Difficulte |
|---|---|---|---|
| 01 | **Detection de collisions inter-flotte** | classification desequilibree, F1/AUC-PR, threshold tuning | medium |
| 02 | **Imitation learning (behavioral cloning)** | dataset sequentiel, LSTM/Transformer minimal, teacher forcing | hard |
| 03 | **LLM pour EOD Review automatique** | prompt engineering, RAG sur traces, streaming, eval | medium |

## Methodologie

Chaque projet contient :
1. Un **dataset synthetique** (generateur Python) qui simule des traces FleetSim avec des patterns plausibles
2. Un notebook / script de training ou un prompt template
3. Une **baseline** a battre (ex: majority class, heuristique simple)
4. Des **metriques** et un script d'eval

Principe : pas de GPU requis pour les v0 (MLP simple, petits transformers, API LLM). Si tu as un GPU, tu peux scaler.

## Contraintes metier rappelees

- **Determinisme** — pour l'auditabilite et la reconstitution d'incident, les inferences doivent etre reproductibles. Fixer les seeds, desactiver non-determinisme CUDA pendant l'eval.
- **Explicabilite** — les clients audites (SOC 2 / ISO 9001) veulent comprendre pourquoi un modele a predit X. Privilegier modeles simples quand possible, ajouter SHAP / attention maps sinon.
- **On-premise / quasi air-gap** — pour les LLM, tout doit pouvoir tourner offline chez le client. Penser Llama/Mistral local, ou distillation d'un gros modele vers un petit.
