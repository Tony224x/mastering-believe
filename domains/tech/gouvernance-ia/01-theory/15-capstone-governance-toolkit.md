# J15 — Capstone : Agent Governance Toolkit

> **Temps estime** : 50-60 min | **Prerequis** : J1 a J14 (l'ensemble du parcours)
> **Objectif** : assembler les 14 briques du domaine en **un seul outil reutilisable** qui inventorie une flotte d'agents, applique des politiques en runtime, journalise de maniere verifiable, score le risque, mappe la conformite et **emet un rapport de gouvernance pret pour un comite**.

## Pourquoi ce module

Quatorze jours durant, on a forge des pieces isolees : registry, scorer, policy engine, audit chaine, crosswalk. Une piece seule ne gouverne rien — c'est leur **chainage end-to-end** (`ingest → enforce → log → score → map → report`) qui produit la preuve qu'un board attend. Aujourd'hui on les soude.

---

## 1. Le livrable concret : un rapport qu'un comite peut signer

Avant toute architecture, regardons la **sortie**. Lundi matin, le comite IA d'une entreprise demande : *"Ou en est-on sur la gouvernance de nos agents ?"* Le toolkit produit, en une commande, ceci :

```text
==================================================================
AGENT GOVERNANCE REPORT — Acme Corp — 2026-06-21
==================================================================
Fleet: 4 agents | Governance coverage: 75% (3/4 fully governed)
Orphan agents (no named owner): 1  -> agent-scraper-07
Audit trail: 9 entries, integrity = VERIFIED (head 3f9a1c..)

Risk posture (NIST AI RMF):
  HIGH  agent-finance-01   crit=20  TREAT     (irreversible + autonomous)
  MED   agent-support-02   crit=8   MONITOR
  ...
Policy enforcement (runtime):
  attempts=6  blocked=3  -> 3 risky actions stopped before execution

Compliance crosswalk:
  EU AI Act     2/3 (67%)   ! gap Art. 14 (human oversight) — MANDATORY
  NIST AI RMF   4/4 (100%)
  ISO/IEC 42001 2/3 (67%)
VERDICT: 1 mandatory legal gap -> remediation required before scale-up.
==================================================================
```

Ce rapport n'est pas decoratif : chaque chiffre est **derive d'un mecanisme** (pas saisi a la main), **horodate**, et **rejouable**. C'est la difference entre *"je crois qu'on est gouverne"* et *"voici la preuve, machine et humaine, datee de ce matin"*. Tout le reste du module explique comment chaque ligne est calculee.

> **Key takeaway** : un capstone de gouvernance se juge a son **livrable** — un rapport board-ready ou chaque chiffre est *derive* d'un mecanisme verifiable, horodate et rejouable, jamais saisi a la main.

---

## 2. L'architecture en pipeline : six etapes, un fil de donnees

Le toolkit est un **pipeline** : la sortie de chaque etape nourrit la suivante. Le fil-rouge commun de tout le domaine — un agent decrit par `id`, `owner`, `permissions/scopes`, `risk_tier` — circule du debut a la fin.

```text
  fleet.json
      │  (1) INGEST   -> registry: charge, valide les 4 piliers, detecte les orphelins
      ▼
  Registry ──(2) ENFORCE  -> policy engine: chaque action passe par PDP/PEP (J14)
      │                        allow / deny / oblige, avec consentement MCP
      ▼
  decisions ──(3) LOG     -> audit trail tamper-evident (J9): chaque decision chainee par hash
      │
      ▼
  Registry ──(4) SCORE    -> risk scorer (J4): likelihood × impact + modulateurs agentiques
      │
      ▼
  controls ──(5) MAP      -> crosswalk (J7): controles -> {EU AI Act, NIST RMF, ISO 42001}
      │
      ▼
  tout ────(6) REPORT    -> generateur board-ready: markdown + JSON, verdict + gaps
```

Pourquoi un pipeline et pas un gros bloc ? Trois raisons d'ingenierie :

1. **Testabilite** : chaque etage a une entree/sortie claire, donc se teste isolement.
2. **Tracabilite** : on peut rejouer une seule etape pour comprendre un chiffre du rapport.
3. **Composition** : on peut substituer un etage (un autre scorer, un autre referentiel) sans casser les autres.

Le **toolkit reste autonome** : il ne dependra d'aucun module externe. Le capstone re-implemente en stdlib les briques necessaires — exactement comme un livrable de mission qu'on remet a un client sans lui imposer 14 fichiers.

> **Key takeaway** : le toolkit est un **pipeline a 6 etages** (`ingest → enforce → log → score → map → report`) traversant un meme modele d'agent. Pipeline = testable etage par etage, tracable, et composable.

---

## 3. Etape INGEST : du fichier brut au registry gouverne

L'entree est un `fleet.json` — la realite d'une mission : un export imparfait, avec des trous. INGEST le charge dans un **registry** (J3) qui n'est pas un tableur fige mais un **controle live** : il **valide les 4 piliers** (identite, owner, permissions, audit — J2) et signale ce qui manque.

La regle centrale : un agent **sans owner nomme est un orphelin** (shadow agent). On ne le supprime pas du registry — au contraire, on le rend **visible** et on le compte. La metrique cle qui en sort est la **couverture de gouvernance** : `agents pleinement gouvernes / total`. La question fondatrice de tout le domaine — *"combien d'agents tournent chez nous, et qui les possede ?"* [Microsoft Security, 2026] — recoit ici une reponse chiffree.

> **Key takeaway** : INGEST transforme un export brut en registry-controle qui **valide les 4 piliers** et **expose les orphelins** plutot que de les ignorer. Sortie cle : un taux de couverture de gouvernance.

---

## 4. Etapes ENFORCE + LOG : decider, bloquer, prouver

ENFORCE rejoue le couple **PDP/PEP** de J14 sur un flux d'actions tentees par la flotte. Le PDP (cerveau) evalue des regles declaratives — scope (J8), budget (J10), autonomie/tier (J4), donnees/RGPD (J6) — et renvoie `allow` / `deny` / `oblige`, avec precedence de surete **deny > oblige > allow**. Le PEP (muscle) **applique** : il laisse passer, bloque, ou suspend. Un gate facon MCP exige le **consentement explicite** pour les outils sensibles [MCP Specification, 2025-11-25].

LOG est indissociable d'ENFORCE : **chaque** decision (allow comme deny) est ecrite dans un **audit trail tamper-evident** (J9). Chaque entree est chainee par hash (`hash = SHA256(prev_hash + canonical(payload))`), si bien qu'une edition silencieuse d'une decision passee est detectee a sa position exacte. C'est la **preuve machine** : non pas *"on bloque, promis"* mais *"voici la trace verifiable de chaque action et de la decision qui l'a precedee"*.

La metrique board qui en sort : `actions risquees bloquees / actions tentees`. Un chiffre defendable, adosse a une trace inviolable.

> **Key takeaway** : ENFORCE **decide et bloque** (PDP/PEP + consentement MCP) ; LOG **prouve** (chaine de hash tamper-evident). Le couple repond a *"bloque-t-on vraiment, et peut-on le prouver ?"* — pas seulement le premier.

---

## 5. Etapes SCORE + MAP : risque defendable et conformite mappee

SCORE applique le scorer de J4 a chaque agent : `criticality = likelihood × impact` sur des echelles ancrees 1..5, **modulee** par le contexte agentique — une action irreversible monte l'impact, l'autonomie totale monte la vraisemblance. Le score est *explicable* (on stocke les ancres et les modulateurs), donc **defendable** devant un auditeur, conformement a la logique Measure → Manage du **NIST AI RMF** [NIST, 2023]. Sortie : un classement TREAT / MONITOR / ACCEPT.

MAP applique le **crosswalk** de J7 : un meme controle interne (ex. *"audit trail inviolable"*) satisfait **plusieurs** referentiels a la fois — un article de l'EU AI Act, une fonction du NIST RMF, une clause ISO/IEC 42001. On ne construit pas N systemes de conformite ; on construit **un** systeme de management et on le mappe vers les N referentiels. Le rapport calcule la **couverture par referentiel** et, surtout, isole les **trous OBLIGATOIRES** (loi) des trous volontaires (norme) — la distinction qui declenche, ou non, une alerte de non-conformite legale.

> **Key takeaway** : SCORE produit un risque **explicable** (ancres + modulateurs agentiques) donc defendable ; MAP relie **un** controle a N referentiels et separe les trous **obligatoires** (loi) des volontaires (norme). Cette separation est le cœur du verdict.

---

## 6. Etape REPORT : du calcul a la decision du board

REPORT agrege tout en **deux formats complementaires** :

- **Markdown** pour l'humain : un comite le lit, le commente, le signe.
- **JSON** pour la machine : un SI le rejoue, le diffe d'un mois sur l'autre, l'archive comme preuve datee.

La regle d'or du rapport board-ready : **finir sur un verdict actionnable**, pas sur un tableau de chiffres. Le verdict agrege les signaux de gouvernance (orphelins, integrite de l'audit, gaps obligatoires) en une phrase de decision : *"1 trou legal obligatoire → remediation requise avant passage a l'echelle"*. Un dirigeant n'a pas besoin de lire les 9 entrees d'audit ; il a besoin de savoir **s'il peut signer**, et **quoi corriger sinon**.

Cette posture — *les humains restent ultimement responsables, l'outil produit la preuve qui eclaire leur decision* — est exactement celle du premier cadre officiel dedie a l'IA agentique [IMDA, 2026]. Le toolkit n'automatise pas la responsabilite ; il la **rend exercable** en mettant la preuve sous les yeux de qui decide.

> **Key takeaway** : un rapport board-ready emet **deux formats** (markdown pour signer, JSON pour rejouer/archiver) et **se termine sur un verdict actionnable**, pas un tableau brut. L'outil eclaire la decision ; la responsabilite reste humaine.

---

## 7. Antiseche — les 6 briques en un coup d'oeil

Avant le code, ancrez la vue d'ensemble. Chaque etage du pipeline reutilise un module precis du parcours et produit **une** metrique que le board comprend :

| Etape | Ce qu'elle fait | Vient du jour | Metrique board |
|---|---|---|---|
| **1. INGEST** | charge `fleet.json`, valide les 4 piliers, expose les orphelins | J2 / J3 | couverture de gouvernance (%) |
| **2. ENFORCE** | PDP/PEP : `allow / deny / oblige` + consentement MCP | J14 (+ J8 scope, J10 budget) | actions risquees bloquees / tentees |
| **3. LOG** | audit chaine par hash, tamper-evident | J9 | integrite = VERIFIED |
| **4. SCORE** | `likelihood x impact` + modulateurs agentiques | J4 | classement TREAT / MONITOR / ACCEPT |
| **5. MAP** | crosswalk : 1 controle -> N referentiels | J7 | couverture par referentiel + **gaps obligatoires** |
| **6. REPORT** | agrege en markdown + JSON, finit sur un verdict | (synthese) | verdict : signable / a corriger |

Le fil de donnees ne change jamais : un meme agent (`id, owner, scopes, risk_tier`) traverse les six etapes. Si vous perdez le fil dans le code, revenez a cette table : chaque fonction du toolkit correspond a une ligne.

---

## Spaced repetition

1. **Q :** Pourquoi structurer le toolkit en pipeline `ingest → enforce → log → score → map → report` plutot qu'en un seul bloc monolithique ?
   **R :** Pour la **testabilite** (chaque etage a une entree/sortie claire, testable isolement), la **tracabilite** (rejouer une etape pour expliquer un chiffre du rapport) et la **composition** (substituer un etage — autre scorer, autre referentiel — sans casser les autres).

2. **Q :** Dans l'etape INGEST, que fait le toolkit d'un agent **sans owner** ? Pourquoi ce choix ?
   **R :** Il ne le supprime pas : il le **rend visible** comme orphelin (shadow agent) et le **compte** dans la couverture de gouvernance. Cacher un orphelin reproduirait le risque qu'on veut mesurer (« combien d'agents, qui les possede »).

3. **Q :** Pourquoi LOG est-il indissociable d'ENFORCE, et qu'apporte le chainage par hash ?
   **R :** Decider/bloquer (ENFORCE) sans trace verifiable ne *prouve* rien. LOG ecrit **chaque** decision dans un audit chaine par hash : une edition silencieuse d'une decision passee est detectee a sa position exacte → preuve **machine** tamper-evident.

4. **Q :** Un controle interne *"audit trail inviolable"* est mappe par le crosswalk. Que signifie qu'il « couvre » a la fois EU AI Act Art. 12, NIST RMF Measure et ISO 42001 §8.2 ?
   **R :** Qu'**un seul** controle satisfait **plusieurs** referentiels simultanement. On ne construit pas N systemes de conformite ; on construit un systeme de management mappe vers N referentiels — et on isole les trous **obligatoires** (loi) des volontaires (norme).

5. **Q :** Pourquoi un rapport board-ready doit-il finir sur un **verdict** et emettre du **JSON** en plus du markdown ?
   **R :** Le verdict transforme des chiffres en **decision** (peut-on signer ? quoi corriger ?) — un dirigeant decide, il ne lit pas 9 lignes d'audit. Le JSON rend le rapport **rejouable, diffable mois sur mois et archivable** comme preuve datee, la ou le markdown sert a etre lu et signe.
