# Design organisationnel & boardroom

## Pourquoi ce module

Un agent sans humain accountable est ingouvernable : quand il dérape, personne ne décide. Ce module installe la **structure organisationnelle** qui répond à « qui est responsable de quoi » — du board jusqu'à l'agent manager opérationnel.

---

## 1. Le problème concret : un agent qui dérape, et personne au bout du fil

Lundi 9h. L'agent « invoice-reconciler » de la flotte finance a re-catégorisé 1 200 écritures comptables pendant le week-end. Une partie est fausse. Le DAF appelle l'IT. L'IT répond : « ce n'est pas notre agent, il a été monté par l'équipe finance avec un outil low-code ». L'équipe finance répond : « c'est l'IT qui gère les agents ». Le board, lui, n'était même pas au courant que cet agent existait.

Personne n'est *accountable*. Le registre (J3) sait peut-être que l'agent existe ; il ne dit pas **qui répond de ses actes devant l'organisation**. C'est exactement le trou que ce module comble : la gouvernance technique (identité, permissions, audit) ne suffit pas s'il n'y a pas, en face, une **structure de redevabilité humaine**.

Le principe à extraire : **chaque agent doit avoir, en plus de son owner technique, une chaîne de responsabilité claire** — quelqu'un qui le supervise au quotidien, quelqu'un qui en répond devant la direction, et un dispositif d'assurance indépendant qui vérifie que ça tient. C'est le rôle du design organisationnel.

> **Key takeaway** — La gouvernance technique d'un agent (J2-J10) est inutile si aucune *personne* n'est désignée comme redevable de ses actions. Le design organisationnel transforme « l'agent X existe » en « X est supervisé par A, qui en répond devant B, sous le contrôle de C ».

---

## 2. L'agent manager : un rôle qui émerge

Quand une équipe passe de 3 agents à 30, superviser « à la main » ne tient plus. Un nouveau rôle apparaît : l'**agent manager** — la personne (ou l'équipe) dont le métier est d'orchestrer une flotte d'agents comme un manager orchestre une équipe humaine.

Concrètement, un agent manager :
- **Définit le mandat** de chaque agent (ce qu'il a le droit de faire, ses bornes d'opération) ;
- **Surveille la performance et les dérives** (taux d'erreur, escalades, coûts) ;
- **Arbitre les escalades** quand l'agent atteint sa limite d'autonomie (lien avec le human-in-the-loop de J10) ;
- **Décide du décommissionnement** d'un agent qui n'est plus fiable ou pertinent.

[HBR, 2026] décrit l'émergence de ce rôle : pour prospérer à l'ère de l'IA, les entreprises ont besoin de *gestionnaires d'agents* capables d'orchestrer des « équipes » d'agents, pas seulement de les déployer. En complément, [MIT SMR & BCG, 2025] insiste sur un point essentiel : **la supervision conçue pour des humains ne se transpose pas telle quelle à des agents** — un agent peut agir 24h/24, en parallèle, à une vitesse qu'aucun manager humain n'égale. La redevabilité doit donc être **explicite et instrumentée**, pas implicite comme dans une équipe humaine.

Attention : l'agent manager est un rôle **émergent**, pas (encore) un titre standardisé. Dans beaucoup d'organisations, c'est une responsabilité ajoutée à un poste existant (un tech lead, un product owner). Ce qui compte n'est pas le titre mais la **fonction** : quelqu'un possède explicitement la supervision opérationnelle de la flotte.

> **Key takeaway** — L'agent manager est le maillon opérationnel : il « manage » les agents comme une équipe (mandat, performance, escalades, décommission). Rôle émergent — l'important est que la fonction soit attribuée nommément, pas qu'un titre existe.

---

## 3. RACI appliqué aux agents

Pour éviter le « ce n'est pas mon agent » de la section 1, on outille la responsabilité avec une matrice **RACI** — un classique de la gestion de projet, adapté aux agents.

RACI distingue **quatre rôles** pour une activité donnée (ici : faire fonctionner et superviser un agent) :

| Lettre | Rôle | Pour un agent |
|--------|------|---------------|
| **R** — Responsible | Exécute le travail | L'**agent manager** qui opère et surveille l'agent au quotidien |
| **A** — Accountable | Répond du résultat (un·e seul·e) | Le **propriétaire métier** (business owner) qui en répond devant la direction |
| **C** — Consulted | Donne un avis avant décision | Sécurité, juridique/conformité, DPO |
| **I** — Informed | Tenu au courant après coup | Board, audit interne, parties prenantes |

**La règle d'or de RACI** : il y a **exactement un A** par agent (un seul accountable). Zéro A = personne ne répond (le cas de la section 1). Deux A = diffusion de responsabilité, donc personne ne répond non plus. Le **R** peut être multiple (plusieurs personnes exécutent), mais **l'accountability ne se partage pas**.

Différence avec J3 : le **registry** (J3) est la source de vérité *technique* (l'agent existe, voici son owner technique et ses scopes). Le **RACI** est la couche *organisationnelle* posée par-dessus : il dit qui, parmi des humains, occupe chacun des quatre rôles. Un bon outil de gouvernance lit le registry et **détecte les trous de RACI** : agent sans A, agent dont l'A n'existe plus dans l'organigramme, etc. (c'est ce que code le `02-code` du jour).

> **Key takeaway** — RACI = Responsible / Accountable / Consulted / Informed. Exactement **un** Accountable par agent — c'est la règle non négociable. Un agent sans A est un agent orphelin de responsabilité, même s'il a un owner technique dans le registry.

---

## 4. Le Three Lines Model : trois lignes de redevabilité

Le RACI répond « qui pour CET agent ». Le **Three Lines Model** de l'[IIA, 2020] répond à l'échelle de l'organisation : **comment structurer la redevabilité pour que la supervision ne soit pas juge et partie**.

C'est une mise à jour du vieux « Three Lines of Defense ». Trois rôles distincts :

1. **Première ligne — la gestion opérationnelle.** Ceux qui *font* et qui *possèdent* le risque au quotidien : les équipes métier qui déploient et opèrent les agents, et leurs agent managers. Ils gèrent le risque directement.
2. **Deuxième ligne — les fonctions de risque et conformité.** Elles *appuient et challengent* la première ligne : risk management, sécurité, conformité IA, DPO. Elles posent les politiques (lien avec J14, policy-as-code) et surveillent leur application. Elles **ne sont pas indépendantes** du management — elles en font partie.
3. **Troisième ligne — l'audit interne.** *Indépendante* du management, elle fournit une **assurance objective** au governing body : « les contrôles des lignes 1 et 2 fonctionnent-ils vraiment ? ». Elle rapporte au board, pas au management opérationnel.

Au-dessus des trois lignes : le **governing body** (le board / conseil), qui est *accountable in fine* devant les parties prenantes (actionnaires, régulateur, société). Les trois lignes le servent ; lui assure la supervision globale (voir section 5).

Pourquoi trois lignes et pas une ? Pour **l'indépendance de l'assurance**. Si l'équipe qui opère les agents est aussi celle qui audite leur conformité, l'audit ne vaut rien. La 3e ligne existe précisément pour que quelqu'un puisse dire au board « non, ça ne tient pas » sans conflit d'intérêt. Pour une flotte d'agents : la 1re ligne opère les agents, la 2e ligne écrit et surveille les politiques de gouvernance, la 3e ligne audite que tout le dispositif (registry, audit trail de J9, RACI) est réel et efficace.

> **Key takeaway** — Three Lines Model [IIA, 2020] : **1re ligne** opère le risque (équipes + agent managers), **2e ligne** appuie/challenge (risque, conformité, sécurité), **3e ligne** audite en toute indépendance et rapporte au board. La séparation garantit que l'assurance n'est ni juge ni partie.

---

## 5. Le rôle du board : oversight, pas micro-management

Dernière strate : le **board**. Sa question n'est pas « comment marche l'agent » mais « **l'organisation gouverne-t-elle ses agents de façon responsable, et qui décide quoi** ».

[McKinsey, 2025] (avec la NACD) cadre cinq missions d'oversight pour le conseil face aux risques IA émergents : s'assurer que (1) la responsabilité de l'IA est clairement attribuée au niveau exécutif, (2) les risques IA sont intégrés à la gestion des risques d'entreprise (ERM), (3) le board monte en compétence sur l'IA, (4) un cadre de gouvernance et des politiques existent, (5) la supervision est régulière (pas un one-shot). Le même article relève un signal d'alarme : **moins de 25 % des conseils ont une politique IA formellement approuvée par le board** — l'oversight reste largement à construire.

Le partage des décisions, en pratique :

| Niveau | Décide… | Exemple |
|--------|---------|---------|
| **Board** | Le mandat, l'appétence au risque, la politique de gouvernance | « Aucun agent out-of-the-loop sur des transactions > 50 k€ » |
| **Management / comité IA** | La déclinaison opérationnelle, l'allocation des moyens | Choisir les outils, nommer les agent managers, valider les tiers de risque |
| **Agent manager (1re ligne)** | Le déploiement et la supervision d'un agent donné | Mettre en prod, surveiller, escalader, décommissionner |

Beaucoup d'organisations instaurent un **comité IA** (AI governance committee) comme relais entre le board et le management : il instruit les dossiers, suit les indicateurs de gouvernance et fait remonter au board ce qui relève de son mandat. Le piège à éviter dans les deux sens : un board qui **micro-manage** (valide chaque agent — ingérable à 200 agents) ou un board **absent** (ne fixe ni mandat ni appétence — le cas de la section 1). Le bon niveau : fixer les bornes et **vérifier qu'elles sont tenues**, via les rapports de la 3e ligne.

> **Key takeaway** — Le board fait de l'**oversight** : il fixe mandat + appétence au risque + politique, et vérifie que c'est tenu — il ne valide pas chaque agent. [McKinsey, 2025] : <25 % des boards ont une politique IA approuvée — l'écart est béant. Un comité IA sert souvent de courroie entre board et management.

---

## 6. Du tableau au code : résoudre la responsabilité depuis le registry

Toute cette structure (agent manager, RACI, Three Lines, board) ne vaut que si on peut **la vérifier sur la flotte réelle**. D'où l'idée du `02-code` du jour : un **ownership resolver** qui lit le registry (J3) et répond, par agent :

- Qui est **Accountable** (et y a-t-il bien exactement un A) ?
- À quelle **ligne** du Three Lines chaque acteur appartient-il ?
- Où sont les **trous de responsabilité** (agent sans A, A inconnu, aucune 3e ligne en couverture) ?

Le passage du tableau de gouvernance au contrôle exécutable est ce qui distingue une gouvernance *réelle* d'un slide de board. On ne se contente pas de dessiner le RACI : on **interroge la flotte** pour trouver les agents qui n'y rentrent pas.

> **Key takeaway** — Une structure de responsabilité n'a de valeur que si elle est **interrogeable** : un resolver qui parcourt le registry et signale les trous (agent sans Accountable, acteur hors Three Lines) transforme l'organigramme en contrôle vivant.

---

## Spaced repetition

1. **Q.** Dans un RACI appliqué à un agent, combien de personnes peuvent porter le « A » (Accountable), et pourquoi ?
   **R.** Exactement **une**. Zéro = personne ne répond ; deux = diffusion de responsabilité (donc personne ne répond non plus). Le R (Responsible) peut être multiple, mais l'accountability ne se partage pas.

2. **Q.** Quelle est la fonction de la **3e ligne** du Three Lines Model, et qu'est-ce qui la rend crédible ?
   **R.** Fournir une **assurance objective** au governing body que les contrôles des lignes 1 et 2 fonctionnent. Sa crédibilité vient de son **indépendance** vis-à-vis du management : elle rapporte au board, pas aux équipes qu'elle audite — elle n'est ni juge ni partie.

3. **Q.** Pourquoi le board doit-il faire de l'« oversight » plutôt que valider chaque agent un par un ?
   **R.** Valider chaque agent ne passe pas à l'échelle (200 agents) et déplace l'accountability au mauvais niveau. Le board fixe le **mandat, l'appétence au risque et la politique**, puis **vérifie qu'ils sont tenus** via les rapports de la 3e ligne.

4. **Q.** Quelle différence entre le rôle de l'**agent manager** (J11) et l'**owner technique** du registry (J3) ?
   **R.** L'owner technique (J3) est l'attribut du registry qui dit « à qui appartient l'agent ». L'agent manager (J11) est la **fonction organisationnelle** de supervision opérationnelle : il définit le mandat, surveille les dérives, arbitre les escalades, décide du décommission — c'est le R/parfois A du RACI, pas une simple étiquette de propriété.

5. **Q.** Selon [McKinsey, 2025], quel chiffre illustre le retard des conseils sur la gouvernance IA, et que recommande l'article ?
   **R.** **Moins de 25 %** des boards ont une politique IA formellement approuvée par le conseil. L'article recommande notamment d'attribuer clairement la responsabilité IA au niveau exécutif, d'intégrer le risque IA à l'ERM, de monter le board en compétence, de doter l'organisation d'un cadre/politique, et d'assurer une supervision régulière (pas ponctuelle).
