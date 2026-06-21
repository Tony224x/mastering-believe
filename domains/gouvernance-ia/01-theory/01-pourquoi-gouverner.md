# Pourquoi gouverner l'IA agentique

## Pourquoi ce module

Avant d'apprendre *comment* gouverner des agents IA, il faut comprendre *pourquoi* c'est devenu urgent. Ce module chiffre l'écart entre l'adoption (massive, rapide) et les garde-fous (rares, immatures), et pose la question fondatrice qui structure tout le parcours : **« Combien d'agents tournent chez nous, et qui les possède ? »**

---

## 1. Un cas concret : la flotte d'agents que personne ne connaît

Imaginez une entreprise de taille moyenne, « Atlas Logistique ». En 18 mois, plusieurs équipes ont créé des assistants IA :

- L'équipe support a déployé un agent qui répond aux clients **et émet des avoirs** jusqu'à 200 €.
- Le marketing a branché un agent qui poste seul sur les réseaux sociaux.
- Un ingénieur a bricolé, un week-end, un agent qui interroge la base de production pour générer des rapports.
- La finance teste un agent qui rapproche les factures et **déclenche des paiements** en dessous d'un seuil.

Question simple posée par le nouveau RSSI : *« Combien d'agents avons-nous, et qui est responsable de chacun ? »*

Personne ne sait répondre. Il n'existe ni liste, ni propriétaire désigné pour l'agent de l'ingénieur (parti depuis), ni trace de ce que l'agent finance a réellement payé. Trois de ces agents **agissent sur le monde réel** (avoirs, posts publics, paiements) — pas seulement « répondent ». Cet écart entre *ce qui tourne* et *ce qui est maîtrisé* est exactement le problème que la gouvernance de l'IA agentique cherche à fermer.

Le principe abstrait derrière l'histoire : **on ne peut pas gouverner ce qu'on n'a pas inventorié.** Un agent sans inventaire est un agent sans propriétaire ; un agent sans propriétaire est un risque sans responsable.

> **Key takeaway** — La gouvernance commence par un fait, pas par une politique : savoir *combien* d'agents existent et *qui* les possède. Tant que cette question reste sans réponse, tout cadre réglementaire reste théorique.

---

## 2. L'écart adoption / garde-fous, en chiffres vérifiés

Le cas « Atlas » n'est pas une fiction isolée : c'est le portrait statistique de l'entreprise de 2026.

**L'adoption est quasi généralisée.** D'après la télémétrie first-party de Microsoft (Copilot Studio / Agent Builder, fenêtre novembre 2025), **80 % des entreprises du Fortune 500 utilisent des agents IA actifs** [Microsoft Security — Cyber Pulse, Issue 1, 2026]. Ce ne sont pas des pilotes : ce sont des agents *en production*.

**Les garde-fous, eux, sont à la traîne.** Selon Deloitte, **seules 21 % des organisations disposent d'une gouvernance agentique mature**, alors que **74 % prévoient un déploiement d'agents d'ici 2027** (enquête auprès de n = 3 235 dirigeants) [Deloitte Insights — State of AI in the Enterprise 2026]. Le titre même du rapport résume le diagnostic : *« AI agents are scaling faster than their guardrails »* — les agents passent à l'échelle plus vite que leurs garde-fous.

**Et le « shadow AI » brouille le décompte.** Une enquête Microsoft / Hypothesis Group (juillet 2025, 1 700+ professionnels de la sécurité) révèle que **29 % du personnel a utilisé des modèles d'IA non approuvés** par leur organisation. Attention : ce chiffre vient d'une *enquête déclarative*, distincte de la télémétrie Fortune 500 ci-dessus — ne pas confondre les deux sources.

**Côté académique, le baromètre confirme la tendance.** Le *AI Index 2026* de Stanford HAI documente à la fois des progrès (part d'entreprises *sans* politique de Responsible AI passée de 24 % à 11 %, postes de gouvernance IA en hausse de +17 % sur 2025) et l'aggravation du risque : **362 incidents IA recensés en 2025, contre 233 en 2024** [Stanford HAI — AI Index 2026, ch. Responsible AI].

Un dernier signal sur la perception dirigeante : d'après le BCG AI Radar 2026, **58 % des dirigeants — parmi les adoptants extensifs de l'agentique — s'attendent à ce que l'IA remodèle leur gouvernance** (contre 37 % chez les non-adoptants) [BCG — AI Radar 2026]. La précision « parmi les adoptants extensifs » est essentielle : ce n'est *pas* 58 % de tous les dirigeants.

> **Key takeaway** — Adoption massive (80 % du Fortune 500), gouvernance immature (21 %), shadow AI répandu (29 % en déclaratif), incidents en hausse (+55 % en un an). L'écart n'est pas une opinion : il est mesuré, et il se creuse.

---

## 3. Pourquoi gouverner un *agent* diffère de gouverner un *modèle*

La gouvernance « classique » de l'IA s'est construite autour de modèles qui **produisent une sortie** : un classifieur de crédit, un détecteur de fraude, un chatbot informatif. On gouverne alors *le contenu* : biais, exactitude, transparence, explicabilité.

Un agent change la nature du problème : il ne se contente pas de répondre, **il agit**. Il appelle des outils (API, bases, e-mail), enchaîne des étapes de manière autonome, et **exécute des transactions** dans le monde réel. La différence est de degré *et* de nature :

| Axe | IA « classique » (modèle) | IA **agentique** |
|---|---|---|
| Sortie | un texte, un score, une prédiction | **une action** (paiement, e-mail, suppression) |
| Réversibilité | souvent réversible (on ignore la sortie) | souvent **irréversible** (le virement est parti) |
| Autonomie | un appel = une réponse | **chaîne d'appels** sans humain dans la boucle |
| Surface de risque | qualité de la prédiction | **excessive agency**, mésusage d'outil, identité |
| Question clé | « la sortie est-elle correcte ? » | « **qui** a autorisé **quelle** action, **quand** ? » |

C'est pour cela que la gouvernance agentique introduit ses propres concepts (identité de l'agent, propriétaire nommé, permissions, piste d'audit) que la gouvernance modèle ne couvrait pas. Les taxonomies de sécurité l'ont acté : l'OWASP a publié en décembre 2025 un *Top 10 for Agentic Applications* distinct du *Top 10 for LLM Applications*, où le tout premier risque est précisément l'abus d'identité et de privilèges [OWASP Gen AI Security Project, 2026].

> **Key takeaway** — Un modèle *dit*, un agent *fait*. Gouverner un agent, ce n'est plus seulement contrôler une sortie : c'est tracer **qui** a autorisé **quelle action**, et pouvoir la **prouver** ou l'**arrêter**.

---

## 4. L'enjeu réglementaire : ce n'est plus optionnel

Même si l'on ne voyait la gouvernance que comme une « bonne pratique », un fait la rend incontournable : **la loi contraint déjà**, avec un calendrier daté.

L'**EU AI Act** (Règlement (UE) 2024/1689) est entré en vigueur le **1ᵉʳ août 2024** et s'applique par paliers (article 113) :

| Date | Ce qui s'applique |
|---|---|
| **2 février 2025** | Interdictions (pratiques inacceptables) + obligation de littératie IA |
| **2 août 2025** | Obligations pour les modèles d'usage général (**GPAI**) |
| **2 août 2026** | Application générale + systèmes **à haut risque (Annexe III)** |
| **2 août 2027** | Haut risque pour produits réglementés (**Annexe I**) |

Le texte classe les systèmes en **4 tiers de risque** : inacceptable, haut, limité, minimal [Règlement (UE) 2024/1689, EUR-Lex]. Beaucoup de systèmes agentiques d'entreprise tomberont dans le tier « haut risque » dès lors qu'ils touchent l'emploi, le crédit, ou des infrastructures — avec une échéance ferme au 2 août 2026.

L'AI Act ne vit pas seul. Trois familles de cadres se complètent et reviendront tout au long du parcours :
- **EU AI Act** — *obligatoire* (loi UE), approche par les risques.
- **NIST AI RMF** (NIST AI 100-1, 2023) — *recommandé* (volontaire), 4 fonctions Govern / Map / Measure / Manage.
- **ISO/IEC 42001** (2023) — *certifiable*, le système de management de l'IA (« l'ISO 27001 de l'IA »).

Distinction à garder en tête dès le premier jour : **obligatoire ≠ recommandé.** L'AI Act *oblige* ; le NIST RMF et l'ISO 42001 *outillent* la conformité sans avoir force de loi. Confondre les deux fait perdre en crédibilité devant un board comme devant un juriste.

> **Key takeaway** — La gouvernance de l'IA n'est plus un choix : l'EU AI Act impose des obligations datées (palier clé : haut risque au **2 août 2026**). Savoir distinguer le *contraignant* (loi) du *recommandé* (norme) est la première compétence du gouverneur d'IA.

---

## 5. Les 4 piliers d'un agent gouvernable (aperçu)

Tout ce parcours se ramène à une ossature simple. Un agent est *gouvernable* s'il possède ces quatre attributs — et *ingérable* dès qu'il en manque un :

1. **Identité** — l'agent est identifiable de manière unique (pas « le script de Paul »).
2. **Propriétaire nommé (owner)** — un humain responsable, joignable, redevable.
3. **Permissions** — ce que l'agent a le droit de faire, idéalement au **moindre privilège**.
4. **Piste d'audit (audit trail)** — une trace vérifiable de ce qu'il a *réellement* fait.

Reprenons « Atlas » : l'agent de l'ingénieur parti a une *identité* floue, **pas d'owner**, des permissions trop larges (accès production), et aucune *piste d'audit*. Il échoue sur 3 des 4 piliers — c'est la définition même d'un agent *ungoverned*. Le « smell test » tient en une phrase : *si un incident survient à 3 h du matin, sait-on qui appeler et ce que l'agent a fait ?* Si la réponse est non, l'agent n'est pas gouverné.

Ces quatre piliers sont la colonne vertébrale du domaine : le module de demain les détaille un par un, et le code de ce module en pose déjà la première brique (identité + owner) en comptant les agents et en repérant ceux sans propriétaire.

> **Key takeaway** — Identité, owner, permissions, audit : quatre piliers indissociables. Un agent qui en manque un seul est, par définition, un agent qu'on ne peut ni responsabiliser, ni prouver, ni arrêter proprement.

---

## Spaced repetition

1. **Q.** Quel est le chiffre exact de l'adoption d'agents dans le Fortune 500, et de quelle source provient-il ?
   **R.** **80 %** du Fortune 500 utilisent des agents IA actifs, d'après la télémétrie first-party de Microsoft (*Cyber Pulse, Issue 1*, 10 fév. 2026) — pas une enquête, une mesure produit (Copilot Studio / Agent Builder).

2. **Q.** Le chiffre « 58 % des dirigeants attendent une refonte de la gouvernance » est-il vrai tel quel ?
   **R.** Non — il faut préciser : **58 % parmi les adoptants extensifs de l'agentique** (vs 37 % chez les non-adoptants), source BCG AI Radar 2026. Affirmé sans cette précision, c'est faux.

3. **Q.** Pourquoi gouverner un agent diffère-t-il fondamentalement de gouverner un modèle ?
   **R.** Un modèle produit une *sortie* (réversible, qu'on peut ignorer) ; un agent produit une *action* (souvent irréversible : paiement, e-mail, suppression). La question passe de « la sortie est-elle correcte ? » à « qui a autorisé quelle action, quand ? ».

4. **Q.** Citez l'échéance de l'EU AI Act la plus structurante pour les systèmes d'entreprise à haut risque (Annexe III).
   **R.** Le **2 août 2026** (application générale + haut risque Annexe III). Rappel : interdictions au 2 fév. 2025, GPAI au 2 août 2025, Annexe I au 2 août 2027.

5. **Q.** Quels sont les 4 piliers d'un agent gouvernable, et qu'arrive-t-il s'il en manque un ?
   **R.** **Identité, owner nommé, permissions, audit trail.** S'il en manque un, l'agent est *ungoverned* : on ne peut pas le responsabiliser (sans owner), ni le prouver (sans audit), ni le borner (sans permissions), ni même le désigner (sans identité).
