# 05 — Projets guides (Neural Networks & LLMs)

> Voir `shared/masa-context.md` pour le contexte metier de MASA Group.

Trois projets ML / LLM appliques aux traces SWORD. Le fil rouge : **transformer des traces de simulation en intelligence exploitable** pour les formateurs et pour enrichir les comportements autonomes des unites.

## Projets

| # | Projet | Concepts couverts | Difficulte |
|---|---|---|---|
| 01 | **Detection de fratricide** | classification desequilibree, F1/AUC-PR, threshold tuning | medium |
| 02 | **Imitation learning (behavioral cloning)** | dataset sequentiel, LSTM/Transformer minimal, teacher forcing | hard |
| 03 | **LLM pour AAR automatique** | prompt engineering, RAG sur traces, streaming, eval | medium |

## Methodologie

Chaque projet contient :
1. Un **dataset synthetique** (generateur Python) qui simule des traces SWORD avec des patterns plausibles
2. Un notebook / script de training ou un prompt template
3. Une **baseline** a battre (ex: majority class, heuristique simple)
4. Des **metriques** et un script d'eval

Principe : pas de GPU requis pour les v0 (MLP simple, petits transformers, API LLM). Si tu as un GPU, tu peux scaler.

## Contraintes metier rappelees

- **Determinisme** — pour la certification, les inferences doivent etre reproductibles. Fixer les seeds, desactiver non-determinisme CUDA pendant l'eval.
- **Explicabilite** — les armees clientes veulent comprendre pourquoi un modele a predit X. Privilegier modeles simples quand possible, ajouter SHAP / attention maps sinon.
- **Air-gap** — pour les LLM, tout doit pouvoir tourner offline. Penser Llama/Mistral local, ou distillation d'un gros modele vers un petit.
