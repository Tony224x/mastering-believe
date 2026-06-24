# Catalogue des domaines

> Fichier **genere** par `shared/tools/build_catalog.py` — ne pas editer a la main. Les metadonnees vivent dans `domains/<track>/<domaine>/meta.toml`.

**11 domaines** — tech : 6 · vie : 5 · exploratoire : 0.

## Track Tech — maitrise d'ingenierie

| Domaine | Niveau | Duree | Modules | Code | Stack / Focus | Statut |
|---|---|---|---|---|---|---|
| [Systemes IA Agentiques](./domains/tech/agentic-ai/) | avance | 4 semaines (J1-J28) | 28 | oui | Python, LangGraph, MCP, Claude/OpenAI APIs · Concevoir des agents autonomes du single-agent au multi-agent en production, jusqu'aux patterns frontier 2025-2026. | stable |
| [Algorithmie & Data Structures — Live Coding Python](./domains/tech/algorithmie-python/) | intermediaire | ~45h sur 2 semaines (14 modules) | 14 | oui | Python, stdlib · Structures de donnees et algorithmes pour le live coding : patterns LeetCode, complexite, entretiens tech FAANG. | stable |
| [Gouvernance de l'IA](./domains/tech/gouvernance-ia/) | intermediaire | 15 modules (~45-60 min chacun) | 15 | oui | Python, stdlib · Gouverner une flotte d'agents : EU AI Act, NIST RMF, ISO 42001, RGPD + 4 piliers (identite, owner, permissions, audit). | stable |
| [Réseaux de Neurones & LLMs](./domains/tech/neural-networks-llm/) | avance | 3 semaines (core J1-J14 ≈ 75-80h + bloc frontière J15-J22 ≈ 38h optionnel) | 22 | oui | Python, numpy, PyTorch · Mécanismes internes des LLMs from scratch, du neurone au Transformer puis frontière NN 2026. | stable |
| [Robotics & AI](./domains/tech/robotics-ai/) | avance | 28 jours | 28 | oui | Python, PyTorch, MuJoCo, Gymnasium, NumPy, LeRobot · Robotique moderne : fondations classiques (SE(3), FK/IK, contrôle), RL/IL, diffusion policies, VLA frontier, capstone Diffusion Policy from scratch. | WIP |
| [System Design — Architecture Backend & IA](./domains/tech/system-design/) | avance | 2 semaines (14 modules / ~43h complet, 12-15h express) | 14 | oui | Python, Kafka, RabbitMQ, Redis, gRPC, GraphQL, TorchServe, Triton, vLLM, Langfuse · Architectures scalables backend et IA (RAG, agents) pour entretiens senior/staff et ML en production. | stable |

- **Gouvernance de l'IA** — prerequis : Algorithmie & Data Structures — Live Coding Python
- **Robotics & AI** — prerequis : Réseaux de Neurones & LLMs

## Track Vie — l'ecole de la vie

| Domaine | Pilier | Niveau | Duree | Modules | Code | Stack / Focus | Statut |
|---|---|---|---|---|---|---|---|
| [Apprendre a apprendre](./domains/vie/apprendre-a-apprendre/) | Esprit | debutant | 14 modules x ~45 min (~10 h 30) | 14 | oui | Python · Meta-competence fondatrice : retrieval practice, spaced repetition, deep work, metacognition, apprendre avec l'IA. | stable |
| [Communication, persuasion & influence](./domains/vie/communication-persuasion/) | Relations | debutant | 14 modules (~45 min chacun), sur 2 semaines | 14 | non | Communiquer clairement et persuader honnetement : ecoute, clarte, recit, negociation, feedback, prise de parole, influence ethique. | stable |
| [Finance personnelle & investissement](./domains/vie/finance-personnelle/) | Argent | debutant | 14 modules (~45 min chacun), ~11 h | 14 | oui | Python · Fondamentaux Pareto-first de la finance perso : interets composes, budget, dette, fonds indiciels, psychologie, independance financiere. | stable |
| [Pensee critique, rationalite & decision](./domains/vie/rationalite-decision/) | Jugement | debutant | 14 modules (~45 min chacun) | 14 | oui | Python · Methode de raisonnement neutre : probas/Bayes, biais, decision sous incertitude, calibration (Brier), verification (SIFT). | stable |
| [Sante, nutrition & longevite](./domains/vie/sante-longevite/) | Corps | debutant | 7 modules (~45 min chacun), ~8-10h sur 2-3 semaines | 7 | non | Fondations evidence-based: sommeil, activite physique, nutrition, sante metabolique, stress & lien social, plan sante personnel. | stable |

- **Communication, persuasion & influence** — garde-fou : Persuasion ethique (charte CTR : Consentement, Transparence, Reciprocite) ; pas de manipulation ni dark patterns.
- **Finance personnelle & investissement** — garde-fou : Contenu purement educatif, pas un conseil financier personnalise ; actif vs passif par la donnee (SPIVA) ; risque de perte en capital.
- **Pensee critique, rationalite & decision** — garde-fou : Methode > conclusions ; exemples 100% neutres, aucun sujet politique/religieux/clivant.
- **Sante, nutrition & longevite** — garde-fou : Strictement educatif, pas un avis medical: toute decision sante a valider avec un professionnel de sante qualifie.
