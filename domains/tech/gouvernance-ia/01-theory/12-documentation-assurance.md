# Documentation & assurance — la preuve statique de la gouvernance

## Pourquoi ce module

Un auditeur, un régulateur ou un comité de direction ne peut pas « lire » un agent en train de tourner. Il faut lui **présenter un dossier** : qui est l'agent, ce qu'il sait faire, ce qu'on a vérifié, et **pourquoi on affirme qu'il est sûr**. Ce module construit cette preuve écrite, durable et défendable — par opposition à la trace runtime (J9).

---

## 1. L'exemple qui rend tout concret : la fiche d'un agent « rembourse-client »

Imaginez un agent déployé chez un e-commerçant. Il lit les tickets SAV, décide d'accorder ou non un remboursement, et **déclenche le virement**. Six mois plus tard, la CNIL pose une question simple : *« Sur quelles données décide-t-il, et qui répond de ses erreurs ? »*

Si la seule réponse est « il faut demander à l'équipe data, ils se souviendront peut-être », la gouvernance a échoué. La bonne réponse tient dans une fiche d'une page — une **agent card** :

```
Nom            : refund-agent v2.3
Owner          : Camille Roux (Head of CX)
Finalité       : trancher les demandes de remboursement < 200 €
Entrées        : ticket SAV (texte), historique commande (DB)
Données perso  : oui — email, n° commande (base légale : exécution du contrat)
Actions        : peut créditer ≤ 200 € ; au-delà → escalade humaine
Limites connues: langue FR uniquement ; biais possible sur tickets agressifs
Évaluations    : taux de bon classement 94 % (jeu de 500 tickets, 2026-05)
Statut         : production, revue trimestrielle
```

Cette fiche n'est pas de la paperasse : c'est une **interface contractuelle**. Elle dit à un tiers, sans lire le code, ce que l'agent fait, sur quoi, sous quelle responsabilité, et ce qu'on **ne garantit pas**.

Le **principe abstrait** derrière l'exemple : la gouvernance produit deux familles de preuves. La **preuve dynamique** (logs, traces — J9) montre *ce qui s'est passé*. La **preuve statique** — ce module — montre *ce que le système est, et pourquoi on le croit acceptable*, **avant** et **indépendamment** de toute exécution. Les deux sont nécessaires ; aucune ne remplace l'autre.

> **Key takeaway** — La documentation d'assurance transforme un agent opaque en un objet inspectable par un humain : une fiche d'une page répond à « qui, quoi, sur quelles données, sous quelle responsabilité, avec quelles limites » sans lire une ligne de code.

---

## 2. Model cards : standardiser ce qu'on déclare sur un modèle

L'idée fondatrice vient de **[Mitchell et al., 2019]** (*Model Cards for Model Reporting*, FAT\* '19). Le constat : un modèle publié sans contexte est dangereux — on ignore sur quelle population il a été évalué, donc on ignore où il échoue. La réponse : une **fiche normalisée** accompagnant tout modèle.

Les sections canoniques d'une model card :

1. **Model details** — qui l'a fait, version, date, type, licence.
2. **Intended use** — usages prévus **et** usages hors-périmètre (le « *out-of-scope* » est aussi important que le « in-scope »).
3. **Factors** — sous-populations / conditions pertinentes (langue, démographie, environnement).
4. **Metrics** — quelles mesures, quels seuils, pourquoi celles-là.
5. **Evaluation data** & **Training data** — d'où viennent les données, comment elles ont été choisies.
6. **Ethical considerations** — risques, populations sensibles.
7. **Caveats & recommendations** — limites connues, conditions d'usage.

Le point pédagogique central de Mitchell : **la performance doit être ventilée** (*disaggregated*). Un modèle « 90 % de précision » peut être à 98 % sur un groupe et 60 % sur un autre. Une model card honnête montre la **désagrégation**, pas seulement la moyenne — c'est ce qui transforme la fiche en outil d'audit plutôt qu'en plaquette marketing.

> **Key takeaway** — Une model card [Mitchell et al., 2019] standardise la déclaration d'un modèle : usages **prévus et exclus**, facteurs, et surtout des métriques **désagrégées** par sous-population. La moyenne ment ; la ventilation révèle où le modèle échoue.

---

## 3. System cards & agent cards : du modèle au système qui agit

Une model card décrit **un modèle**. Mais un système réel empile un modèle, des prompts, des filtres, des outils, une mémoire. La **system card** documente l'**ensemble déployé** et ses comportements émergents.

L'exemple canonique est la **[GPT-4 System Card, OpenAI, 2023]** (publiée le 15 mars 2023). Au-delà des métriques, elle décrit le **processus** : red-teaming par des experts externes, comportements à risque observés (hallucinations, contenus dangereux, « *emergent behaviors* »), et les **mitigations** appliquées avant déploiement. La leçon : à l'échelle système, on ne documente plus seulement *ce que le modèle sait*, mais *ce qu'on a cherché à casser et ce qu'on a corrigé*.

L'**agent card** est l'adaptation agentique — la fiche du module 1. Par rapport à une system card, elle ajoute la dimension **action** : ce que l'agent a le **droit de faire** (scopes/outils), son **owner nommé**, ses **garde-fous** (seuils d'escalade, budget), et l'**autonomie** accordée. C'est la jonction directe avec les 4 piliers (identité / owner / permissions / audit) : l'agent card est la **forme documentaire** des piliers.

| Fiche | Décrit | Champs spécifiques |
|-------|--------|--------------------|
| Model card | un modèle | facteurs, métriques désagrégées, données d'entraînement |
| System card | un système déployé | red-teaming, comportements émergents, mitigations |
| Agent card | un agent qui agit | scopes/outils, owner, autonomie, seuils d'escalade, budget |

> **Key takeaway** — Model card = le modèle ; system card = le système déployé et ce qu'on a tenté de casser [OpenAI, 2023] ; **agent card** = l'agent qui *agit*, avec scopes, owner et garde-fous. L'agent card est la forme documentaire des 4 piliers.

---

## 4. Safety cases : argumenter la sûreté, pas seulement la décrire

Une fiche **décrit**. Un **safety case** **démontre** : c'est un **argument structuré, étayé par des preuves, qui justifie qu'un système est suffisamment sûr pour un usage donné dans un contexte donné**. Le concept est emprunté à l'aéronautique et au nucléaire ; **[Clymer et al., 2024]** (*Safety Cases: How to Justify the Safety of Advanced AI Systems*) l'adapte à l'IA avancée.

Leur taxonomie distingue **quatre types d'arguments** de sûreté, du plus fort au plus fragile :

1. **Inability** (incapacité) — « le système ne *peut pas* causer ce dommage » (il n'a pas la capacité). L'argument le plus solide : si l'agent n'a pas accès à l'API de virement, il ne peut pas détourner d'argent.
2. **Control** (contrôle) — « même s'il en était capable, nos garde-fous l'**empêchent** » (sandbox, limites, supervision, kill-switch). C'est l'argument du moindre privilège et des budgets (J8, J10).
3. **Trustworthiness** (fiabilité comportementale) — « on a des raisons de croire qu'il ne le *ferait pas*, même s'il le pouvait » (évaluations, alignement). Plus fragile : repose sur une inférence sur le comportement.
4. **Deference** (déférence à un jugement) — « une autorité crédible (expert, comité, autre système vérifié) atteste qu'il est sûr ». Le plus indirect : on délègue le jugement.

La hiérarchie est intentionnelle : **on préfère l'incapacité au contrôle, le contrôle à la fiabilité, la fiabilité à la déférence.** Pourquoi ? Parce que « il *ne peut pas* » se vérifie ; « il *ne le ferait pas* » se parie. Un bon safety case construit son argument principal sur le type le plus fort **disponible** pour le risque visé.

> **Key takeaway** — Un safety case [Clymer et al., 2024] *argumente* la sûreté par preuves, avec 4 types ordonnés par robustesse : **inability > control > trustworthiness > deference**. Privilégier « il ne *peut* pas » (vérifiable) à « il ne le *ferait* pas » (parié).

---

## 5. Structurer un argument d'assurance : claims → evidence → gaps

Comment écrire concrètement un safety case sans tomber dans la prose creuse ? On le décompose en un **arbre d'assurance** à trois niveaux, lisible comme un argument GSA (*Goal Structuring* simplifié) :

- **Claim (allégation)** — l'affirmation à défendre. Ex. : *« refund-agent ne peut pas créditer plus de 200 € sans validation humaine. »*
- **Evidence (preuve)** — ce qui soutient le claim, idéalement **vérifiable** : un test, un log, une config, une revue de code, un résultat d'éval. Ex. : *« le policy engine bloque tout montant > 200 € — 200 tests adverses passés (cf. J13). »*
- **Gap (lacune)** — ce qui **manque** pour que le claim tienne pleinement, déclaré **honnêtement**. Ex. : *« non testé sous charge concurrente ; pas d'audit indépendant du policy engine. »*

La règle d'or, directement issue de la culture safety case : **un claim sans evidence est une opinion ; un safety case sans gaps déclarés est suspect.** Un dossier qui prétend « zéro risque » n'est pas rassurant — il est non-crédible. Déclarer ses lacunes est ce qui rend l'argument *honnête* et donc *audité-able*.

On obtient alors une mesure simple de maturité : **le ratio claims couverts par au moins une evidence**, et la **liste priorisée des gaps**. C'est exactement ce que le code du jour produit : un squelette `claim → [evidence] → [gaps]` qui calcule la couverture et refuse de faire passer un claim non étayé pour acquis.

> **Key takeaway** — Décomposer un safety case en **claim → evidence → gap**. Un claim sans evidence est une opinion ; un dossier sans gap déclaré est suspect. La maturité se mesure : % de claims étayés + gaps priorisés.

---

## 6. Pourquoi la preuve statique est obligatoire (et pas une option de confort)

La documentation d'assurance n'est pas seulement une bonne pratique : elle est **exigée par les cadres** que couvre ce domaine.

- **EU AI Act** — les systèmes à haut risque doivent fournir une **documentation technique** (Annexe IV) et des **instructions d'usage** : on retrouve précisément les champs d'une model/system card (capacités, limites, données, surveillance humaine). La fiche **est** une obligation de conformité.
- **NIST AI RMF** — la fonction **Govern** demande de documenter rôles, responsabilités et limites ; **Map** demande de documenter le contexte et les usages prévus. La preuve statique alimente directement ces fonctions.
- **ISO/IEC 42001** — un système de management de l'IA repose sur des **enregistrements** auditables ; cards et safety cases en sont la matière.

Le fil rouge : ce module produit le **dossier de conformité humain** (preuve statique) qui se présente à un auditeur, là où J9 produit la **preuve machine** (logs vérifiables). Un audit sérieux croise les deux : la fiche dit « l'agent ne crédite jamais > 200 € sans humain » (statique), le log prouve qu'**en effet** il ne l'a jamais fait (dynamique). La gouvernance défendable, c'est **la fiche + la trace + l'argument qui les relie**.

> **Key takeaway** — La preuve statique (cards, safety cases) n'est pas optionnelle : elle est exigée par l'EU AI Act (doc technique Annexe IV), le NIST AI RMF (Govern/Map) et l'ISO 42001 (enregistrements). Elle se croise avec la preuve machine (J9) pour un audit défendable.

---

## Spaced repetition

**Q1.** Quelle est la différence entre une *model card* et une *system card* ?
**R1.** Une model card décrit **un modèle** (facteurs, métriques désagrégées, données d'entraînement [Mitchell et al., 2019]). Une system card décrit le **système déployé complet** — modèle + prompts + outils + mémoire — et documente le red-teaming, les comportements émergents et les mitigations appliquées avant déploiement [OpenAI, 2023].

**Q2.** Citez les 4 types d'arguments d'un safety case, du plus robuste au plus fragile, et expliquez pourquoi cet ordre.
**R2.** **Inability** (il ne *peut* pas) > **Control** (les garde-fous l'empêchent) > **Trustworthiness** (il ne le *ferait* pas) > **Deference** (une autorité l'atteste) [Clymer et al., 2024]. L'ordre va du **vérifiable** (une incapacité se teste) vers le **parié/délégué** (un comportement futur s'infère).

**Q3.** Pourquoi un safety case qui ne déclare **aucun** gap est-il suspect ?
**R3.** Parce qu'un dossier prétendant « zéro risque » n'est pas crédible : tout système réel a des limites. Déclarer ses gaps honnêtement est ce qui rend l'argument auditable. Un claim sans evidence est une opinion ; un dossier sans gap déclaré masque ses lacunes.

**Q4.** En quoi une *agent card* diffère-t-elle d'une *system card* classique ?
**R4.** Elle ajoute la dimension **action** : scopes/outils autorisés, owner nommé, niveau d'autonomie, seuils d'escalade et budget. C'est la forme documentaire des **4 piliers** (identité, owner, permissions, audit).

**Q5.** Comment la preuve statique (ce module) et la preuve dynamique (J9) se complètent-elles dans un audit ?
**R5.** La preuve statique **affirme** ce que le système est et pourquoi il est sûr (« il ne crédite jamais > 200 € sans humain ») ; la preuve machine (logs tamper-evident) **prouve** qu'il s'est effectivement comporté ainsi. Un audit défendable croise la fiche, la trace, et l'argument (safety case) qui les relie.
