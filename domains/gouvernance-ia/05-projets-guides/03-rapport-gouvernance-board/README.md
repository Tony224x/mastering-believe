# Projet 03 — Rapport de gouvernance board-ready + crosswalk de conformite

> **Difficulte** : medium | **Contexte metier** : [`shared/logistics-context.md`](../../../../shared/logistics-context.md) | **Solution** : [`solution/governance_report.py`](solution/governance_report.py)
> **Theorie mobilisee** : J4 (scoring NIST AI RMF), J5 (EU AI Act, obligatoire vs volontaire), J7 (normes & crosswalk AIMS), J12 (documentation & assurance).

## Contexte metier

Un client LogiSim — operateur logistique **audite ISO 9001 / SOC 2** — exploite une flotte d'agents FleetSim : un agent qui regle automatiquement les transporteurs, un assistant OCC, un classifieur d'events de shift, un coordinateur de flotte externe (livreurs tiers). Il veut **passer a l'echelle** (plus d'agents, plus d'autonomie) et doit d'abord presenter a son **comite de direction** un etat de la gouvernance.

Le piege classique : arriver avec un slide ou les niveaux de risque et les pourcentages de conformite ont ete **saisis a la main**. Dans un environnement audite, c'est disqualifiant — un auditeur SOC 2 demandera *« d'ou vient ce 75 % ? »* et *« pourquoi cet agent est-il classe haut risque ? »*. La reponse « on l'a estime » ne passe pas : il faut que **chaque chiffre soit derive d'un mecanisme** et rejouable.

Voici le livrable que le comite veut pouvoir signer (sortie reelle du script, tronquee) :

```
======================================================================
RAPPORT DE GOUVERNANCE DES AGENTS — LogiSim Client — Plateforme Nord — 2026-06-21
======================================================================
Flotte FleetSim : 4 agents
Agents orphelins (sans owner nomme) : external-fleet-coordinator-04

1. POSTURE DE RISQUE (NIST AI RMF — criticite = vraisemblance x impact)
----------------------------------------------------------------------
  [TREAT  ] fleet-finance-billing-01 crit=25  (L5 x I5, base L4/I4)
            modulateurs : action irreversible -> impact +1; autonomie totale -> vraisemblance +1
  [TREAT  ] external-fleet-coordinator-04 crit=12  (L4 x I3, base L3/I3)
  [MONITOR] occ-support-assistant-02 crit= 9  (L3 x I3, base L3/I3)
  [ACCEPT ] shift-event-classifier-03 crit= 4  (L2 x I2, base L2/I2)

2. CONFORMITE — COUVERTURE PAR REFERENTIEL (crosswalk)
----------------------------------------------------------------------
  EU AI Act        2/3 ( 67%)   [OBLIGATOIRE (loi)]
  ISO/IEC 42001    2/3 ( 67%)   [volontaire (norme)]
  NIST AI RMF      3/4 ( 75%)   [volontaire (norme)]

3. TROUS DE CONFORMITE (separes par nature)
----------------------------------------------------------------------
  TROUS OBLIGATOIRES (loi — bloquants) :
    ! EU AI Act Art. 14 — Supervision humaine
  Trous volontaires (norme — dette a planifier) :
    - NIST AI RMF Govern — Roles & responsabilites
    - ISO/IEC 42001 5.3 — Roles & responsabilites de l'AIMS

4. VERDICT
----------------------------------------------------------------------
  1 trou legal OBLIGATOIRE (EU AI Act Art. 14) -> remediation requise AVANT passage a l'echelle.
```

Tout l'enjeu du projet est de **construire l'outil qui produit ce rapport** — ou le `25`, le `67%` et surtout le verdict final sont **calcules**, pas tapes.

## Objectif technique

Ecrire un script Python **stdlib uniquement** qui, a partir d'une description de flotte FleetSim et de l'etat des controles internes du client, produit un rapport de gouvernance board-ready. Trois briques :

1. **Scorer le risque** de chaque agent (NIST AI RMF) avec un score *explicable*.
2. **Crosswalk de conformite** : controles internes → {EU AI Act, NIST AI RMF, ISO/IEC 42001}, avec separation des trous **obligatoires** (loi) et **volontaires** (norme).
3. **Rapport** en **deux formats** (markdown signable + JSON archivable) finissant sur un **verdict actionnable**.

Le fil conducteur pedagogique : la distinction **obligatoire (loi) / volontaire (norme)** est le **coeur du verdict**. Un trou legal bloque le passage a l'echelle ; un trou volontaire est une dette a planifier. Tout repose sur cette frontiere.

## Consigne

### 1. Scoring de risque (NIST AI RMF)

Pour chaque agent, calculer `criticite = vraisemblance x impact` sur des **echelles ancrees 1..5** (chaque niveau a une definition, pas un ressenti — cf. theorie J4 §5.1), puis appliquer les **modulateurs agentiques** :

- **action irreversible** (virement, suppression) → impact **+1** (rien a annuler) ;
- **autonomie totale** (out-of-the-loop, aucun humain dans la boucle) → vraisemblance **+1** (aucun filet pour intercepter).

Le score doit etre **explicable** : on conserve les ancres (L/I bruts), les modulateurs appliques et le score final. Sortie : classement **TREAT** (criticite ≥ 12) / **MONITOR** (6..11) / **ACCEPT** (< 6).

### 2. Crosswalk de conformite

Definir une table de **controles internes** (ex. « audit trail inviolable », « owner nomme par agent », « kill-switch teste », « revue humaine des actions a haut risque ») et, pour chacun, l'ensemble des exigences qu'il couvre dans **trois referentiels** :

| Referentiel | Nature | Exemples d'exigences |
|---|---|---|
| **EU AI Act** | **OBLIGATOIRE (loi)** | Art. 9 (gestion des risques), Art. 12 (logging), Art. 14 (supervision humaine) |
| **NIST AI RMF** | volontaire (cadre) | Govern, Map, Measure, Manage |
| **ISO/IEC 42001** | volontaire (norme) | 6.1 (eval. risques), 8.2 (journalisation), 5.3 (roles) |

Calculer la **couverture par referentiel** (%) et, surtout, **isoler les trous obligatoires des trous volontaires**. Un controle **non implemente ne couvre rien** (sinon le crosswalk devient du theatre).

### 3. Rapport board-ready (2 formats + verdict)

- **Markdown** — lisible et signable par un humain (le comite).
- **JSON** — rejouable / archivable / **diffable** d'un trimestre sur l'autre (la preuve qu'exige un auditeur SOC 2 / ISO 9001).
- Le rapport DOIT **finir sur un verdict actionnable**, pas sur un tableau brut. Ordre de priorite : trou **legal** > agent **orphelin** > trou **volontaire** > flotte gouvernee.

### Contraintes

- **Python 3.11+, stdlib uniquement** (`dataclasses`, `json`, `typing`). Aucune dependance externe, aucune cle API.
- Sortie **deterministe** : date figee en constante, `json.dumps(..., sort_keys=True)`. Deux executions = sortie identique octet pour octet.
- Demo dans `__main__` : flotte de **≥ 4 agents varies** (un finance/irreversible HIGH, un support MED, un classifieur LOW, un external-fleet), impression du markdown + JSON + verdict final.

## Etapes guidees

1. **Modeliser l'agent** — une dataclass `FleetAgent` portant `risk_tier`, `autonomous`, `handles_irreversible`, `owner` (et un flag `external_fleet`). Ces champs sont les entrees du scoring.
2. **Ecrire le scorer** — `score_agent()` : ancres par tier → modulateurs (cap a 5) → criticite → decision TREAT/MONITOR/ACCEPT. Conserver les ancres et la liste des modulateurs pour l'explicabilite.
3. **Definir les exigences** — une liste de `Requirement(framework, ref, label, mandatory)`. Le booleen `mandatory` (True pour EU AI Act seulement) est ce qui sera trie a la fin.
4. **Definir les controles** — une liste de `Control(control_id, label, implemented, covers)`. `covers` = les cles `framework::ref` couvertes. Un controle absent (`implemented=False`) ne compte pas.
5. **Calculer le crosswalk** — couverture par referentiel + liste des trous, separes `mandatory_gaps` / `voluntary_gaps`.
6. **Relier owner ↔ controle** — un agent **orphelin** (sans owner) doit faire tomber le controle « owner nomme ». C'est ainsi qu'un fait sur la flotte se propage mecaniquement jusqu'au crosswalk.
7. **Rendre le rapport** — `render_markdown()` et `render_json()` a partir d'une meme structure `GovernanceReport` ; une fonction `_verdict()` partagee par les deux formats.
8. **Demo** — flotte d'exemple, etat des controles realiste (presque tout en place sauf un controle), impression complete.

## Criteres de reussite

- [ ] `python solution/governance_report.py` tourne et **exit 0**, sortie deterministe.
- [ ] Chaque agent recoit un score **explicable** (ancres + modulateurs visibles) et un classement TREAT/MONITOR/ACCEPT coherent ; l'agent finance/irreversible/autonome sort en **crit=25 (TREAT)** en tete.
- [ ] Le crosswalk affiche la **couverture par referentiel** et separe clairement **trous obligatoires** (loi) et **trous volontaires** (norme).
- [ ] Le rapport est produit en **markdown ET JSON**, et **finit sur un verdict actionnable**.
- [ ] **Probe adversariale** : si on **corrige le trou legal** (controle implemente) et qu'on **nomme l'owner manquant**, le verdict bascule de « remediation requise » a « passage a l'echelle autorise ». La logique du verdict est donc bien pilotee par les mecanismes, pas codee en dur.
- [ ] Aucune dependance externe, aucun chiffre saisi a la main.

## Solution

Voir [`solution/governance_report.py`](solution/governance_report.py). Points cles du corrige :

- **Score explicable** (`RiskScore`) — on stocke `base_likelihood/impact`, `eff_likelihood/impact` et la liste `modulators`. Le comite peut auditer *pourquoi* un agent sort en TREAT, pas seulement *qu'il* en sort.
- **`mandatory` au coeur du modele** — `Requirement.mandatory` distingue la loi (EU AI Act) de la norme. Le tri `mandatory_gaps` / `voluntary_gaps` en decoule, et le verdict prioritise le legal.
- **Controles gates sur l'evidence** — `Control.implemented` : un controle prevu mais absent ne prouve rien. Dans la demo, le **kill-switch non teste** est le seul controle portant Art. 14 → il ouvre un **trou legal** que le verdict fait remonter comme bloquant. C'est l'illustration centrale.
- **Owner manquant → trou** — un agent orphelin force `CTRL-OWNER=False`, ce qui cree des trous (ici volontaires : Govern, ISO 5.3). Un fait sur la flotte devient un chiffre dans le rapport.
- **Deux formats, un seul contenu** — markdown pour la signature humaine, JSON `sort_keys` pour le diff d'un mois sur l'autre.

## Questions de reflexion

1. **Pourquoi separer obligatoire et volontaire change-t-il le verdict ?** Un trou EU AI Act (loi) est une non-conformite sanctionnable : il **bloque**. Un trou NIST/ISO (norme) est une dette d'alignement : il se **planifie**. Si on les agregait dans un seul pourcentage (« 73 % conforme »), on noierait le trou legal — exactement l'erreur que le rapport doit empecher. (cf. theorie J5 et J7 §5.)
2. **Pourquoi un score « explicable » et pas juste un chiffre ?** Devant un comite audite, un score non justifie n'est pas defendable : un tiers doit pouvoir **rejouer** le calcul et obtenir le meme resultat. Stocker les ancres et les modulateurs transforme « 25/25 » en « L5×I5, dont +1 irreversible et +1 autonomie ».
3. **Pourquoi deux formats (markdown + JSON) ?** Le markdown sert a la **lecture et la signature** par un humain ; le JSON sert a la **tracabilite** — archiver le rapport du trimestre et le **differ** avec le suivant pour prouver la progression a un auditeur SOC 2 / ISO 9001. Meme contenu, deux usages complementaires (cf. preuve statique vs machine, J12).
4. **L'outil decide-t-il a la place du comite ?** Non. Conformement a la these IMDA Agentic 2026 — *« les humains restent ultimement responsables »* — l'outil produit la **preuve** qui eclaire la decision. Le verdict est une recommandation tracee, pas une autorisation automatique.

## Pour aller plus loin

- **Mitigation et re-score** — apres avoir applique un controle (ex. HITL sur l'agent finance), recalculer le **risque residuel** et montrer le delta avant/apres dans le rapport.
- **Diff de deux rapports** — ecrire une fonction qui prend deux JSON (T et T+1) et liste ce qui a change (trous fermes, nouveaux agents, scores qui ont bouge) — exactement ce qu'un auditeur veut voir.
- **Due diligence fournisseur** — ajouter aux agents un champ `vendor` + `eu_ai_act_tier` (cf. J5 §5) et faire remonter dans le rapport les briques achetees dont la deadline EU AI Act approche.
- **Safety case par agent** — pour chaque agent en TREAT, generer un squelette `claim → evidence → gap` (cf. J12 §5) et refuser de declarer « sur » un claim sans evidence.
- **Export PDF** — transformer le markdown en document signable (en-tete, date, espace signature) pour le dossier de comite.
