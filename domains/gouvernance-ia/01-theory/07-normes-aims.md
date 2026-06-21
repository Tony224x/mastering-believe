# Normes & systemes de management de l'IA (AIMS)

## Pourquoi ce module

Une fois qu'on a vu l'EU AI Act (obligatoire) et le RGPD, il reste une question : **comment piloter durablement** la gouvernance dans le temps, et comment prouver qu'on le fait. C'est le role des **normes** (ISO/IEC 42001), des **principes** (OECD) et des **cadres** (IMDA) — volontaires mais structurants.

---

## 1. Le probleme concret : trois auditeurs, une seule organisation

Imaginez une banque francaise qui deploie une flotte d'agents IA (scoring credit, support client, conformite). En six mois, elle recoit trois sollicitations :

- un **auditeur ISO** veut verifier qu'elle a un *systeme de management* qui s'ameliore (certification ISO/IEC 42001) ;
- une **autorite** (au titre de l'EU AI Act) veut la preuve que le systeme de scoring credit — **haut risque, Annexe III** — respecte ses obligations ;
- un **client grand compte** demande comment elle s'aligne sur les **principes OCDE** (transparence, redevabilite) avant de signer.

La tentation : monter trois projets de conformite separes, trois jeux de documents, trois equipes. Le piege : un cout multiplie par trois pour des exigences qui se **recouvrent largement** (gestion du risque, documentation, supervision humaine, journalisation).

Le principe a extraire : **un meme controle interne satisfait souvent plusieurs referentiels a la fois**. La bonne strategie n'est pas d'empiler les conformites, mais de batir **un** systeme de management et de **mapper** ses controles vers chaque referentiel (un *crosswalk*). C'est exactement ce que formalise ISO/IEC 42001.

> **Key takeaway** — Les referentiels de gouvernance IA se recouvrent. On ne construit pas N systemes ; on construit **un** systeme de management et on le mappe vers les N referentiels.

---

## 2. ISO/IEC 42001 : l'AIMS et la boucle PDCA

**ISO/IEC 42001:2023** est la **premiere norme internationale certifiable** d'un *AI Management System* (AIMS) — souvent surnommee « l'ISO 27001 de l'IA » [ISO/IEC, 2023]. Un AIMS n'est pas un document : c'est un **systeme vivant** de politiques, roles, processus et controles que l'organisation exploite et **ameliore en continu**.

Le moteur d'un AIMS est la boucle **PDCA** (Plan-Do-Check-Act), heritee de la famille des systemes de management ISO :

- **Plan** — definir la politique IA, le perimetre, les objectifs, l'analyse de risque (qui rejoint *Map* du NIST RMF) ;
- **Do** — operer les controles (registry d'agents, permissions, journalisation) ;
- **Check** — mesurer, auditer, detecter les ecarts (audits internes, indicateurs) ;
- **Act** — corriger, ajuster la politique, relancer un tour de boucle (amelioration continue).

Le point cle de PDCA pour l'IA agentique : la gouvernance n'est **jamais figee**. Un agent ajoute, une permission elargie, un incident — chacun declenche un nouveau tour de boucle. Une certification ISO 42001 atteste justement que **la boucle tourne**, pas qu'un etat parfait a ete atteint a un instant T.

Concretement, un AIMS impose (entre autres) : une politique IA documentee, des **roles et responsabilites** nommes (rejoint le pilier *owner*), une **evaluation d'impact** des systemes d'IA, la **gestion des ressources et donnees**, et un **suivi operationnel** avec journalisation (rejoint le pilier *audit trail*).

> **Key takeaway** — ISO/IEC 42001 = un **systeme de management** (AIMS) certifiable, motorise par **PDCA** : la valeur prouvee n'est pas « tout est parfait » mais « la boucle d'amelioration tourne ».

---

## 3. OECD AI Principles : le socle commun intergouvernemental

Avant les normes techniques, il y a les **principes**. Les **OECD AI Principles** (Recommendation on AI) sont le **premier standard intergouvernemental** sur l'IA, adopte en 2019 et **mis a jour en mai 2024** [OECD, 2024]. Ils servent de socle commun a de nombreux pays (et ont inspire des textes ulterieurs).

La structure est en deux familles :

**5 valeurs (principes) pour une IA digne de confiance :**
1. croissance inclusive, developpement durable et bien-etre ;
2. respect de l'etat de droit, des droits humains, valeurs democratiques, **equite et vie privee** ;
3. **transparence et explicabilite** ;
4. **robustesse, surete et securite** ;
5. **redevabilite** (accountability).

**5 recommandations aux pouvoirs publics :**
1. investir dans la R&D en IA ;
2. favoriser un ecosysteme numerique pour l'IA ;
3. mettre en place un cadre d'action favorable ;
4. renforcer les capacites humaines et preparer la transition du marche du travail ;
5. cooperer a l'international pour une IA digne de confiance.

Pour le praticien, les OECD Principles ne sont **pas executables** tels quels — ils ne disent pas *comment* journaliser ou *comment* scoper une permission. Mais ils donnent le **vocabulaire partage** (transparence, redevabilite, robustesse) qu'on retrouve, decline, dans le NIST RMF, l'EU AI Act et ISO 42001. C'est la **boussole**, pas la carte routiere.

> **Key takeaway** — Les **OECD AI Principles** (2019, maj **2024**) = 5 valeurs + 5 recommandations. Socle de vocabulaire commun (transparence, redevabilite, robustesse), non executable mais structurant pour tous les autres cadres.

---

## 4. IMDA : le premier cadre dedie a l'IA agentique

Les cadres precedents visent l'IA en general. Or les agents **agissent** (appellent des outils, executent des transactions). Singapour, via l'**IMDA** (Infocomm Media Development Authority) et l'AI Verify Foundation, a produit deux cadres complementaires :

- le **Model AI Governance Framework for Generative AI** (30 mai 2024) — approche pro-innovation, 9 dimensions [IMDA, 2024] ;
- le **Model AI Governance Framework for Agentic AI** (**22 janvier 2026**) — **premier cadre officiel au monde dedie specifiquement a l'IA agentique** [IMDA, 2026].

Le message central du cadre agentique IMDA : **« les humains restent ultimement responsables »**. Meme quand un agent decide et agit seul, la redevabilite ne se delegue pas a la machine. Cela se traduit par des exigences concretes : borner le **domaine d'operation** de l'agent, exiger un **owner humain**, tracer les actions, prevoir des points de supervision et d'escalade.

L'apport specifique d'IMDA face a EU AI Act / NIST / ISO : ces derniers sont surtout penses pour des **systemes** d'IA (qui produisent des sorties) ; IMDA Agentic adresse directement la dimension **action autonome** — exactement la colonne vertebrale agentique (identite, owner, permissions, audit) de ce parcours. C'est un **complement**, pas un substitut : on l'utilise pour combler ce que les cadres generalistes ne couvrent pas finement.

> **Key takeaway** — IMDA **Agentic AI** (22 janv. **2026**) = **premier cadre mondial dedie aux agents**. Sa these : **les humains restent ultimement responsables**. Il complete EU AI Act / NIST / ISO sur la dimension *action autonome*.

---

## 5. Le crosswalk : un controle, plusieurs referentiels

C'est ici que tout se rejoint. Reprenons la banque du §1. Elle implemente un controle interne : **« chaque agent journalise ses actions dans un audit trail inviolable »** (le pilier *audit*). Ce **seul** controle satisfait *simultanement* plusieurs referentiels :

| Controle interne | EU AI Act | NIST AI RMF | ISO/IEC 42001 |
|---|---|---|---|
| Audit trail des actions d'agent | Art. 12 (tenue de registres / logging, haut risque) | fonction **Measure** / **Manage** (tracabilite) | clause de **suivi operationnel** & journalisation |
| Owner humain nomme par agent | Art. 14 (supervision humaine) | fonction **Govern** (roles & responsabilites) | **roles & responsabilites** de l'AIMS |
| Evaluation de risque par agent | Art. 9 (systeme de gestion des risques) | fonction **Map** | **AI risk assessment** de l'AIMS |

Un **crosswalk** est cette table de correspondance : chaque controle → l'ensemble des exigences qu'il couvre. Sa valeur est double :

1. **Efficacite** — on implemente le controle **une fois**, on prouve la conformite **N fois**.
2. **Couverture** — on voit immediatement les **trous** : une exigence d'un referentiel qu'aucun controle interne ne couvre encore. C'est exactement ce que calcule le code du jour (`02-code/07-normes-aims.py`).

Attention a la distinction fondamentale que le crosswalk rend visible :

- **Obligatoire (loi)** — EU AI Act, RGPD : on **doit** se conformer, sous peine de sanction.
- **Volontaire (norme / cadre)** — ISO 42001, NIST RMF, OECD, IMDA : on **choisit** de s'aligner (pour la confiance, l'audit, les appels d'offres). Mais une norme volontaire devient souvent le **moyen de preuve** d'une obligation legale : se conformer a ISO 42001 aide a demontrer la conformite a l'EU AI Act.

> **Key takeaway** — Un **crosswalk** mappe chaque controle interne vers {article AI Act, fonction NIST, clause ISO}. Il maximise la **couverture** (un controle prouve N conformites) et revele les **trous**. Toujours distinguer **obligatoire** (loi) de **volontaire** (norme qui sert de moyen de preuve).

---

## Spaced repetition

1. **Q :** Que signifie PDCA dans un AIMS, et quelle est l'idee centrale de cette boucle pour l'IA agentique ?
   **R :** Plan-Do-Check-Act. Idee centrale : la gouvernance n'est jamais figee — chaque agent ajoute, permission elargie ou incident relance un tour de boucle. Une certification ISO 42001 atteste que **la boucle tourne**, pas qu'un etat parfait est atteint.

2. **Q :** Pourquoi surnomme-t-on ISO/IEC 42001 « l'ISO 27001 de l'IA » ?
   **R :** C'est la premiere norme internationale **certifiable** d'un systeme de management dedie a l'IA (AIMS), construite sur la meme logique de systeme de management + amelioration continue que l'ISO 27001 (securite de l'information).

3. **Q :** Quel est l'apport specifique du cadre IMDA Agentic (2026) face a EU AI Act / NIST / ISO ?
   **R :** Il est le **premier cadre mondial dedie a l'IA agentique** et adresse la dimension **action autonome** (les agents agissent, pas seulement produisent des sorties). Sa these : **les humains restent ultimement responsables**. Complement, pas substitut.

4. **Q :** Qu'est-ce qu'un crosswalk de controles, et quelle est sa double valeur ?
   **R :** Une table mappant chaque controle interne vers les exigences de plusieurs referentiels (AI Act, NIST, ISO). Double valeur : (1) efficacite — implementer une fois, prouver N fois ; (2) couverture — reperer les exigences non couvertes (les trous).

5. **Q :** Quelle est la difference entre referentiel obligatoire et volontaire, et comment s'articulent-ils ?
   **R :** Obligatoire = loi (EU AI Act, RGPD), sanction a la cle. Volontaire = norme/cadre (ISO 42001, NIST, OECD, IMDA), choix d'alignement. Mais une norme volontaire sert souvent de **moyen de preuve** d'une obligation legale (ISO 42001 aide a demontrer la conformite a l'EU AI Act).
