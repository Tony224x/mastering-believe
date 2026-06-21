# J10 — Autonomie, garde-fous & operations

> **Temps estime** : 45-60 min | **Prerequis** : J2 (les 4 piliers), J4 (score de risque), J9 (audit trail)
> **Objectif** : calibrer le **niveau d'autonomie** d'un agent sur son risque, implementer **budget + kill-switch**, derouler une **reponse a incident** et un **decommission** propre.

## Pourquoi ce module

Un agent gouvernable n'est pas un agent qu'on surveille en permanence — c'est un agent dont l'**autonomie est calibree sur le risque** et dont on peut **reprendre le controle** a tout instant. Ce module donne les leviers operationnels : autonomie graduee, garde-fous, kill-switch, reponse a incident, fin de vie.

---

## 1. Le probleme concret : l'agent qui rembourse 10 000 € tout seul

Un agent de support client a le droit d'emettre des remboursements. Lundi, il traite 200 tickets : remboursements de 5 a 40 €, tout va bien. Mardi, un client formule sa demande de maniere ambigue ; l'agent interprete « rembourse-moi l'annee » comme « rembourse 12 mois d'abonnement » **et le fait** — 1 800 €. Personne n'a valide. Mercredi, un prompt injection dans un e-mail client pousse l'agent a enchainer 30 remboursements vers le meme IBAN. Le total atteint 10 000 € **avant** que quiconque ne regarde le tableau de bord.

Qu'est-ce qui a manque ? Trois choses, dans l'ordre :

1. **Pas de seuil d'autonomie** : un remboursement de 5 € et un de 1 800 € passaient par le meme chemin (aucune validation humaine au-dela d'un montant).
2. **Pas de budget** : rien ne plafonnait le **cumul** d'actions par fenetre de temps.
3. **Pas de kill-switch** : une fois la derive detectee, il a fallu redeployer pour l'arreter — plusieurs minutes pendant lesquelles l'agent continuait.

Le principe abstrait derriere ces trois failles : **l'autonomie d'un agent doit etre une variable que l'on regle, pas une constante**. On la regle selon le risque de l'action, on la borne par des budgets, et on garde un interrupteur. C'est exactement ce que recommande EY : borner explicitement le **domaine d'operation** de l'agent et controler ses appels d'API [EY, 2026].

> **Key takeaway** : Les incidents agentiques naissent rarement d'une seule action enorme, mais de l'**absence de seuil, de budget et d'interrupteur**. L'autonomie se regle ; elle ne se subit pas.

---

## 2. Les niveaux d'autonomie : human-in / on / out-of-the-loop

L'erreur de debutant est binaire : « l'agent est autonome » ou « l'agent ne l'est pas ». La realite est graduee. Trois niveaux de reference :

| Niveau | L'humain... | L'agent agit... | Exemple |
|--------|-------------|-----------------|---------|
| **Human-in-the-loop (HITL)** | valide **avant** chaque action sensible | seulement apres feu vert | Remboursement > 500 € : l'agent prepare, un humain approuve. |
| **Human-on-the-loop (HOTL)** | **surveille** et peut interrompre | de lui-meme, sous supervision | Tri de tickets : l'agent route seul, un humain voit le flux et peut reprendre la main. |
| **Human-out-of-the-loop** | n'intervient pas en temps reel | totalement seul | Classement d'e-mails en « spam/non-spam » : aucun humain par decision. |

Le bon niveau **depend du risque de l'action** (vraisemblance x impact, cf. J4), pas du confort de l'equipe. Une regle robuste : **plus l'action est irreversible et a fort impact, plus l'humain doit etre en amont** (in-the-loop). Le classement spam est reversible et a faible impact -> out-of-the-loop acceptable. Le virement bancaire est irreversible et a fort impact -> in-the-loop obligatoire.

Le point cardinal partage par tous les cadres recents : meme out-of-the-loop, **l'humain reste ultimement responsable** de l'agent. Le cadre agentique de Singapour le pose comme principe fondateur : l'autonomie technique ne transfere jamais la responsabilite [IMDA, 2026].

> **Attention au piege** : un meme agent peut operer a des niveaux differents **selon l'action**. Le bon design n'attribue pas un niveau a l'agent, mais **un niveau par type d'action** (lire = out-of-the-loop, ecrire = on-the-loop, payer = in-the-loop).

> **Key takeaway** : L'autonomie est un **curseur par action**, calibre sur le risque (impact x reversibilite). Out-of-the-loop ne veut jamais dire « personne n'est responsable ».

---

## 3. Garde-fous (guardrails) : la difference avec un test

Un **garde-fou** (guardrail) est un controle qui s'execute **au moment de l'action**, en production, et qui peut la **bloquer** ou la **modifier**. C'est different d'un test (qui s'execute *avant* le deploiement, cf. J13) : le garde-fou vit dans le chemin chaud.

Trois familles de garde-fous, du plus simple au plus fin :

- **Input rails** : filtrent ce qui entre (detecter une instruction injectee, un PII, une demande hors-perimetre).
- **Output rails** : filtrent ce qui sort (bloquer une reponse toxique, un secret, une donnee non autorisee).
- **Action rails** : interceptent les **actions** (l'agent veut appeler `issue_refund(1800)` -> le garde-fou verifie le montant, le budget, l'autorisation avant de laisser passer).

Pour un agent qui *agit*, les **action rails** sont les plus critiques : c'est la qu'on attrape le remboursement de 1 800 €. Un garde-fou d'action rend une decision parmi trois :

- **ALLOW** — l'action passe telle quelle.
- **DENY** — l'action est bloquee (et journalisee).
- **ESCALATE / require approval** — l'action est mise en attente d'une validation humaine (transition vers HITL pour CETTE action).

Des frameworks reels existent (NVIDIA NeMo Guardrails, Guardrails AI) ; on en reimplemente ici une mini-version en stdlib pour comprendre le mecanisme : une fonction qui prend une action proposee et retourne `ALLOW | DENY | ESCALATE` selon des regles. Bain insiste sur ce glissement : la gouvernance ne porte plus seulement sur les **outputs** d'un modele, mais sur ses **actions** — d'ou la necessite de garde-fous au niveau de l'action [Bain, 2025].

> **Key takeaway** : Un garde-fou s'execute **en runtime, sur le chemin de l'action**, et tranche ALLOW / DENY / ESCALATE. Pour un agent qui agit, le **action rail** est le garde-fou qui compte.

---

## 4. Budgets & escalade : plafonner le cumul

Un seul garde-fou par action ne suffit pas : il faut aussi plafonner le **cumul**. Trois budgets classiques pour un agent :

- **Budget de cout** (€) : plafond de depense par fenetre (ex. 500 €/jour de remboursements cumules).
- **Budget d'actions** (compte) : nombre max d'actions sensibles par fenetre (ex. 20 remboursements/heure).
- **Budget de tokens / appels** : plafond de consommation LLM (anti-boucle infinie, anti-cout qui derape).

Le pattern d'enforcement : a chaque action, on **incremente un compteur** et on verifie le plafond **avant** d'agir. Deux comportements possibles au depassement :

- **Hard cap** : on bloque (DENY) jusqu'a la fenetre suivante.
- **Soft cap + escalade** : on n'autorise plus l'agent seul ; chaque action supplementaire passe en HITL (ESCALATE).

Le soft cap est souvent preferable : il ne casse pas le service, il **rehausse le niveau de supervision** au moment ou le risque cumule monte. C'est l'idee EY de calibrer les controles sur le risque plutot que de tout interdire en bloc [EY, 2026].

```
action proposee
   |
   v
[budget restant ?] --non--> ESCALATE (ou DENY si hard cap)
   | oui
   v
[garde-fou ALLOW ?] --non--> DENY / ESCALATE
   | oui
   v
incremente compteur -> ACT -> journalise (cf. J9)
```

> **Key takeaway** : Les budgets (cout, actions, tokens) plafonnent le **cumul** que les garde-fous par action ne voient pas. Au depassement, preferer **escalader vers HITL** (soft cap) plutot que tout bloquer.

---

## 5. Kill-switch : reprendre le controle immediatement

Un **kill-switch** est un mecanisme qui **suspend instantanement** un agent (ou une flotte) sans redeploiement. Propriete essentielle : il doit etre **hors du chemin de decision de l'agent** — l'agent ne doit pas pouvoir l'ignorer, le contourner ni se l'auto-desactiver.

Bonnes proprietes d'un kill-switch :

- **Externe a l'agent** : un flag/etat partage lu **avant** chaque action, pas une variable interne que l'agent peut ecraser.
- **Granulaire** : pouvoir tuer un seul agent, un type d'action, ou toute la flotte (« big red button »).
- **Reversible et trace** : qui a active/desactive, quand, pourquoi (journalise — cf. J9).
- **Fail-safe** : en cas de doute (etat illisible), l'agent **s'arrete** plutot que de continuer (default deny).

Implementation minimale : un `dict` d'etat `{agent_id: "active" | "paused" | "killed"}` que l'autonomy gate consulte en tout premier. `killed` = l'agent ne fait plus rien jusqu'a reactivation explicite par un humain. C'est la traduction operationnelle du principe « l'humain garde le controle ultime » [IMDA, 2026].

> **Key takeaway** : Le kill-switch doit etre **externe a l'agent, granulaire, trace et fail-safe**. S'il est une variable que l'agent peut ecraser, ce n'est pas un kill-switch.

---

## 6. Reponse a incident : detect -> contain -> eradicate -> recover

Quand un garde-fou se declenche ou qu'une derive est detectee, on ne bricole pas : on deroule un **cycle de reponse a incident**. Le modele classique (herite de la securite, NIST) en quatre phases, applique aux agents :

1. **Detect** — un signal (budget depasse, garde-fou DENY repete, anomalie d'audit) ouvre un incident. On le **date** et on l'**identifie**.
2. **Contain** — on limite la casse **immediatement** : kill-switch sur l'agent concerne, revocation de ses scopes (cf. J8). On arrete l'hemorragie avant de comprendre.
3. **Eradicate** — on traite la **cause racine** : corriger la politique trop permissive, patcher le prompt vulnerable a l'injection, retirer l'outil fautif.
4. **Recover** — on **reactive** prudemment (souvent a un niveau d'autonomie reduit, ex. HITL force pendant 48 h), on surveille, puis on **documente** (post-mortem) ce qui alimente les garde-fous futurs.

L'ordre est **non negociable** : on **contient avant d'eradiquer** (on coupe d'abord, on enquete ensuite). Une machine a etats simple modelise bien ce cycle : `OPEN -> CONTAINED -> ERADICATED -> RECOVERED -> CLOSED`, avec des transitions interdites (on ne passe pas de `OPEN` directement a `CLOSED`). EY recommande precisement ce type de processus operationnel calibre, plutot qu'une reaction ad hoc [EY, 2026].

> **Key takeaway** : Reponse a incident = **detect -> contain -> eradicate -> recover**, dans cet ordre. On **contient (kill-switch) avant d'eradiquer (cause racine)**. Le post-mortem nourrit les garde-fous suivants.

---

## 7. Cycle de vie & decommission : la fin de vie compte

Un agent gouverne nait, vit et **doit mourir proprement**. Le decommission est la phase la plus negligee — et celle qui laisse les agents **orphelins** (cf. J1/J3) trainer avec des acces actifs.

Cycle de vie minimal d'un agent : `PROPOSED -> APPROVED -> ACTIVE -> SUSPENDED -> DECOMMISSIONED`. La transition `DECOMMISSIONED` n'est pas un simple « on l'eteint » : c'est une **checklist** :

- [ ] **Revoquer tous les acces** : scopes, credentials, secrets (cf. J8) — sinon identite zombie exploitable.
- [ ] **Retirer du registry** (ou marquer `decommissioned`, pas supprimer la ligne) — pour la tracabilite.
- [ ] **Conserver l'audit trail** : on archive, on ne detruit pas (obligations de retention, cf. J9).
- [ ] **Reassigner ou clore l'owner** : aucun acces actif sans owner vivant.
- [ ] **Dater et tracer** la decommission (qui, quand, pourquoi).

Le garde-fou de fin de vie : **un acces actif sans agent actif est une faille**. Le decommission propre, c'est garantir qu'apres la mort de l'agent, il ne reste aucun scope exploitable.

> **Key takeaway** : Decommissionner = **revoquer les acces + conserver l'audit + clore l'owner**, pas juste « eteindre ». Un agent inactif avec des scopes actifs est une **identite zombie** exploitable.

---

## Spaced repetition

1. **Q :** Pour decider du niveau d'autonomie d'une action (in / on / out-of-the-loop), quel critere prime ?
   **R :** Le **risque de l'action** = impact x reversibilite. Plus c'est irreversible et a fort impact, plus l'humain doit etre **in-the-loop** (validation en amont). L'autonomie est un curseur **par action**, pas par agent.

2. **Q :** Quelle est la difference entre un garde-fou (guardrail) et un test de gouvernance (J13) ?
   **R :** Le test s'execute **ex-ante** (avant deploiement) ; le garde-fou s'execute **en runtime**, sur le chemin chaud, et peut **bloquer/modifier** l'action (ALLOW / DENY / ESCALATE).

3. **Q :** Pourquoi un kill-switch ne doit-il pas etre une variable interne a l'agent ?
   **R :** Parce que l'agent pourrait l'**ecraser ou l'ignorer**. Le kill-switch doit etre **externe**, lu avant chaque action, **fail-safe** (en cas de doute, on s'arrete) et **trace**.

4. **Q :** Dans la reponse a incident, pourquoi « contain » vient-il avant « eradicate » ?
   **R :** Pour **arreter l'hemorragie immediatement** (kill-switch, revocation de scopes) avant d'enqueter sur la cause racine. Contenir d'abord, comprendre ensuite. Ordre : detect -> contain -> eradicate -> recover.

5. **Q :** Qu'est-ce qu'une « identite zombie » et comment le decommission l'evite-t-il ?
   **R :** Un agent inactif dont les **acces/scopes restent actifs** — exploitable par un attaquant. Le decommission propre **revoque tous les acces** (scopes, secrets), conserve l'audit trail, et clot l'owner.
