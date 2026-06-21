# Les 4 piliers d'un agent gouvernable

## Pourquoi ce module

J1 a pose la question fondatrice — *combien d'agents tournent chez nous, et qui les possede ?* Ce module donne l'unite de mesure : les **4 attributs** sans lesquels un agent est ingouvernable (identite, owner, permissions, audit). C'est le vocabulaire commun de tout le parcours.

---

## 1. Un incident concret avant la theorie

Lundi 9h. Le support client d'une banque recoit un appel : un client s'est vu rembourser **4 200 €** qu'il n'a jamais reclames. Le virement est parti d'un agent IA de traitement des litiges, deploye trois mois plus tot par une equipe produit. Le RSSI ouvre l'enquete et pose quatre questions banales :

1. **Quel** agent a fait ce virement ? — Personne ne sait. Il y a « plusieurs bots de support », aucun ne porte d'identifiant unique dans les logs.
2. **Qui** en est responsable ? — L'equipe produit a ete reorganisee ; le dev qui l'a deploye est parti.
3. **Avait-il le droit** d'emettre un virement ? — Il avait la cle API d'un humain (le chef de projet), donc *tous* ses droits, y compris virer de l'argent.
4. **Que** s'est-il passe exactement, etape par etape ? — Les logs montrent des appels API bruts, sans lien avec la decision metier ni l'autorisation invoquee.

Quatre questions, quatre angles morts. Ce n'est pas un bug de code : c'est une **absence de gouvernance**. Chaque question correspond exactement a un pilier manquant.

| Question du RSSI | Pilier | Ce qui manquait |
|---|---|---|
| Quel agent ? | **Identite** | Pas d'ID unique attribue a l'agent |
| Qui est responsable ? | **Owner nomme** | Pas d'humain redevable |
| Avait-il le droit ? | **Permissions** | Heritait de *tous* les droits d'un humain |
| Que s'est-il passe ? | **Audit trail** | Logs techniques, pas de trace decision/autorisation |

> **Key takeaway** : un agent ingouvernable ne se reconnait pas a son code, mais a l'incapacite a repondre a 4 questions simples — *quel, qui, quel droit, quoi*. Chaque question pointe un pilier.

---

## 2. Pilier 1 — Identite : un agent, un identifiant unique et verifiable

**Le principe.** Avant toute chose, un agent doit etre une **entite distincte et adressable**, pas une fonction anonyme noyee dans un service. Concretement : un identifiant unique, stable, non reutilise, qui le distingue (a) des autres agents et (b) des humains.

L'erreur classique de l'incident bancaire : l'agent agissait *sous l'identite d'un humain* (la cle API du chef de projet). Resultat — impossible de distinguer dans les logs ce qu'a fait l'humain de ce qu'a fait l'agent, et l'agent herite mecaniquement de tous les droits de cet humain. La Cloud Security Alliance nomme cette categorie la **Non-Human Identity** (NHI) : les agents sont des identites a part entiere, ni humaines ni de simples comptes de service classiques [CSA, 2025].

Une identite d'agent porte au minimum :

- un **ID unique** (ex. `agent://refund-bot/7f3a` — un URN, pas « le bot de support ») ;
- un **type** (humain / agent / service) pour ne jamais confondre les trois ;
- de quoi **prouver** cette identite a l'execution (un secret, un certificat, une *verifiable credential* — re-implemente en miniature dans le code du jour).

> **Key takeaway** : pas d'identite unique = pas d'attribution. Un agent qui emprunte l'identite d'un humain est, du point de vue de la gouvernance, **invisible**.

---

## 3. Pilier 2 — Owner nomme : un humain redevable, toujours

**Le principe.** Chaque agent doit avoir un **proprietaire humain nomme** — une personne (pas « l'equipe data », pas « le service IT ») qui repond de son comportement. C'est le principe directeur du premier cadre officiel dedie a l'agentique : *« les humains restent ultimement responsables »* [IMDA, 2026].

Pourquoi un humain, pas une equipe ? Parce qu'une responsabilite diffuse est une responsabilite nulle. Quand le RSSI demande « qui ? », « l'equipe produit » ne permet ni d'escalader, ni de decider d'un kill-switch, ni d'imputer. L'owner est le point de contact unique pour :

- **valider** le perimetre et les permissions de l'agent ;
- **etre alerte** en cas d'incident ou de comportement anormal ;
- **decider** de suspendre ou decommissionner l'agent ;
- **rendre des comptes** au board ou au regulateur.

Un owner peut deleguer l'exploitation, mais pas la redevabilite. C'est une distinction d'**accountability** (qui repond) vs **responsibility** (qui execute) — on y reviendra au J11 avec le RACI. L'absence d'owner est l'archetype de l'**agent orphelin** : techniquement vivant, organisationnellement abandonne.

> **Key takeaway** : un agent sans owner humain nomme est ingerable. L'owner est *une personne*, jamais une boite ou une equipe.

---

## 4. Pilier 3 — Permissions : moindre privilege, pas l'identite d'un humain

**Le principe.** Un agent ne doit detenir que les permissions **strictement necessaires** a sa tache — le **moindre privilege**. L'incident bancaire est l'exact contraire : l'agent de *traitement de litiges* avait le droit d'*emettre des virements* parce qu'il portait la cle d'un humain qui, lui, l'avait.

Le moindre privilege se decline en **scopes** explicites et bornes :

- non pas « acces a l'API bancaire » mais `read:disputes`, `write:dispute_notes`, jamais `transfer:funds` ;
- des bornes : montant maximal, plage horaire, environnement (sandbox vs prod) ;
- idealement des permissions **a duree de vie courte** (ephemerite), revoquees apres la tache.

C'est exactement le risque que l'OWASP classe en tete de sa taxonomie agentique : **ASI01 — Identity and Privilege Abuse** (usurpation d'identite et abus de privileges), aux cotes de *Tool Misuse* et de *Rogue Agents* [OWASP, 2026]. Un agent sur-permissionne n'est pas un agent qui *va* mal tourner : c'est un agent dont *chaque* defaillance (bug, hallucination, injection) devient maximale au lieu d'etre bornee.

Le pilier permissions repond a une question testable : *« quelle est la pire chose que cet agent puisse faire, et est-ce acceptable ? »* Si la reponse est « virer 4 200 € » pour un bot de notes de litige, le scope est faux.

> **Key takeaway** : permissions = moindre privilege exprime en scopes explicites et bornes. Heriter des droits d'un humain n'est pas une permission, c'est une bombe.

---

## 5. Pilier 4 — Audit trail : une trace verifiable de ce qu'il a fait

**Le principe.** Tout ce que l'agent fait doit laisser une **trace exploitable** : pas un log technique brut, mais un enregistrement qui relie *qui* (identite) a fait *quoi* (action), *quand*, et *sous quelle autorisation* (permission invoquee). C'est ce qui permet de **reconstruire** un incident apres coup.

L'audit trail repond a la 4e question du RSSI. Dans l'incident, les logs existaient — mais c'etaient des appels HTTP, sans lien avec la decision metier ni la permission. Un audit trail de gouvernance enregistre des entrees comme :

```
2026-06-21T08:42:10Z  agent=agent://refund-bot/7f3a
  action=funds_transfer  amount=4200  scope_used=transfer:funds
  decision="dispute #88421 ruled in favor of customer"  result=executed
```

Trois proprietes le rendent *utile* (developpees au J9) :

- **append-only** : on ajoute, on ne reecrit jamais ;
- **tamper-evident** : toute alteration est detectable (chainage par hash — J9) ;
- **complet** : assez de contexte pour rejouer la decision, pas juste l'effet de bord.

Sans audit trail, rien n'est **prouvable** : ni la conformite a un regulateur, ni l'innocuite d'un agent, ni les responsabilites en cas de litige.

> **Key takeaway** : sans audit trail, l'action d'un agent est non-prouvable. Logguer un appel HTTP n'est pas auditer une decision.

---

## 6. Pourquoi les 4 sont indissociables — et l'Agent Card

**Les piliers forment une chaine, pas un menu.** Otez-en un, les autres perdent leur valeur :

- **Identite sans permissions** → un agent identifie mais qui peut tout faire. On sait *qui*, mais le *qui* est dangereux.
- **Permissions sans audit** → des droits bornes, mais aucune preuve de ce qui a ete fait avec. Non-prouvable.
- **Audit sans identite** → une trace d'actions… attribuees a personne. Inexploitable.
- **Tout sans owner** → un agent parfaitement instrumente que *personne* ne peut suspendre ni dont personne ne repond.

L'enchainement logique : l'**identite** rend l'agent attribuable → l'**owner** rend l'attribution actionnable (un humain a appeler) → les **permissions** bornent la casse possible → l'**audit** prouve ce qui s'est reellement passe. Les quatre se tiennent.

**L'Agent Card.** En pratique, on materialise ces 4 piliers (et d'autres metadonnees) dans une declaration structuree, l'**Agent Card** : une fiche d'identite lisible par machine qui declare *qui est l'agent, qui le possede, ce qu'il a le droit de faire et comment il s'authentifie*. C'est une pratique **emergente** (Google A2A, Microsoft Entra Agent ID — couverte au J3), pas encore un standard fige ; mais l'idee — *une fiche d'identite de gouvernance par agent* — est durable. Dans ce module, on en code une version minimale (`@dataclass GovernedAgent`).

**Le « smell test » d'un agent ungoverned.** Reflexe de terrain : prenez un agent au hasard et posez les 4 questions. S'il en manque une seule reponse claire, l'agent est **non gouverne** :

- [ ] A-t-il un **ID unique** (pas l'identite d'un humain) ?
- [ ] A-t-il un **owner humain nomme** (une personne) ?
- [ ] Ses **permissions** sont-elles explicites et bornees au moindre privilege ?
- [ ] Laisse-t-il un **audit trail** reliant identite / action / autorisation ?

> **Key takeaway** : les 4 piliers sont une chaine — un maillon manquant casse l'ensemble. L'Agent Card est leur materialisation ; le smell test (4 questions) est leur diagnostic rapide.

---

## Spaced repetition

**Q1.** Un agent porte la cle API d'un humain (le chef de projet) pour agir. Lesquels des 4 piliers cela viole-t-il directement, et pourquoi ?
**R1.** Au moins deux : **Identite** (l'agent n'a pas d'ID propre, il est invisible dans les logs, confondu avec l'humain) et **Permissions** (il herite de *tous* les droits de l'humain au lieu du moindre privilege). C'est l'archetype d'OWASP **ASI01 — Identity and Privilege Abuse**.

**Q2.** Pourquoi exiger un owner *humain nomme* (une personne) plutot qu'une equipe responsable ?
**R2.** Une responsabilite diffuse est nulle : on ne peut ni escalader, ni decider d'un kill-switch, ni imputer aupres d'« une equipe ». L'IMDA (2026) pose que *les humains restent ultimement responsables* — il faut un point de contact unique et redevable.

**Q3.** Quelle est la difference entre « logguer les appels API d'un agent » et « tenir un audit trail de gouvernance » ?
**R3.** Un log API enregistre des effets techniques bruts. Un audit trail de gouvernance relie *identite → action → quand → autorisation invoquee*, assez complet pour **reconstruire** la decision et la **prouver** (idealement append-only et tamper-evident — J9).

**Q4.** En quoi « identite sans permissions » et « permissions sans audit » sont-ils tous deux insuffisants ?
**R4.** Identite sans permissions = on sait *qui*, mais le qui peut tout faire (casse non bornee). Permissions sans audit = la casse est bornee, mais on ne peut rien *prouver* de ce qui a ete fait. Les piliers forment une chaine : il en faut les 4.

**Q5.** Qu'est-ce qu'une Agent Card, et quel est son statut de maturite ?
**R5.** Une declaration structuree, lisible par machine, qui materialise les attributs de gouvernance d'un agent (identite, owner, permissions, auth). C'est une pratique **emergente** (Google A2A, Microsoft Entra Agent ID), pas un standard fige — mais l'idee d'une fiche d'identite de gouvernance par agent est durable.
