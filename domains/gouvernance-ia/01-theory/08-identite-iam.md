# Identité & IAM des agents (deep technique)

## Pourquoi ce module

Un agent qui *agit* (appelle des outils, déclenche des transactions) doit s'authentifier comme une identité à part entière, pas se cacher derrière le compte d'un humain. Ce module passe du « qui est responsable ? » (côté board) à la mécanique d'accès côté ingénieur : identité machine, scopes, délégation, expiration, Zero Trust.

> **Lien avec J2.** J2 a posé les 4 piliers d'un agent gouvernable vus du board (identité / owner / permissions / audit). J8 plonge dans **un seul** de ces piliers — l'identité et les permissions — et le rend *exécutable* : comment une requête d'agent est autorisée ou refusée, requête par requête.

---

## 1. Le problème concret : un agent qui « emprunte » les droits d'un humain

Voici une situation banale et dangereuse. Une équipe finance déploie un agent « réconciliation factures ». Pour aller vite, l'ingénieur lui donne **le jeton de service partagé de l'équipe** — celui qui peut lire les comptes, écrire dans le grand livre, et déclencher des virements. L'agent tourne 24/7 avec ce jeton qui n'expire jamais.

Trois mois plus tard, une injection de prompt via une facture PDF piégée pousse l'agent à initier un virement vers un IBAN inconnu. Le virement passe : l'agent **avait le droit** de virer. Personne ne sait dire *quel* agent a agi (le jeton est partagé), ni *au nom de qui*, ni *pourquoi il avait ce scope*.

Décomposons ce qui a mal tourné :

| Symptôme | Faute de gouvernance d'identité |
|----------|--------------------------------|
| Jeton partagé entre équipe et agent | Pas d'**identité machine distincte** (non-human identity) |
| Jeton qui n'expire jamais | Pas d'**éphémérité** (durée de vie courte) |
| L'agent peut virer alors qu'il devait juste réconcilier | Violation du **moindre privilège** (scopes trop larges) |
| Impossible de dire « au nom de qui » | Pas de **chaîne de délégation** explicite |
| Le virement passe sans re-vérification | Pas de posture **Zero Trust** (on a fait confiance une fois pour toutes) |

Le principe abstrait derrière ces cinq fautes : **un agent est une identité non-humaine, autonome, qui doit être authentifiée, scopée au plus juste, traçable jusqu'à un principal humain, et vérifiée à chaque requête.** C'est exactement ce que recommande le cadre IAM dédié aux agents [CSA, 2025], qui décrit une approche IAM *purpose-built* pour des agents combinant autonomie, éphémérité et délégation.

> **Key takeaway.** Donner à un agent le jeton d'un humain ou d'une équipe casse les quatre garanties d'un accès gouverné (identité distincte, moindre privilège, traçabilité, vérification). Chaque agent mérite *sa* propre identité machine.

---

## 2. Identité machine (non-human identity) : l'agent est un *principal*

En IAM classique on distingue depuis longtemps les **identités humaines** (un employé, un compte nominatif) des **identités machine** : services, scripts, robots. Un agent IA est une identité machine d'un genre nouveau — il *décide* et *agit*, là où un service classique exécute un flux figé.

Concrètement, donner une identité à un agent signifie lui attribuer :

- un **identifiant unique et stable** (`agent-invoice-recon-007`), jamais réutilisé pour un autre agent ;
- un **principal** auquel le rattacher : son **owner humain** (responsable nommé) et éventuellement l'**organisation/équipe** ;
- un **type** (`agent`) distinct de `human` et de `service`, pour que l'audit puisse filtrer.

L'industrie converge vers des représentations standardisées de cette identité — par exemple les **Agent Cards** (Google A2A) ou un annuaire d'agents type **Microsoft Entra Agent ID** (en *Preview* fin 2025). Ce sont des **pratiques émergentes, non figées** : ce qui est durable, c'est le concept (un agent = un principal nommé, distinct), pas le produit. Le CSA propose même de fonder l'identité d'agent sur des **DID** (Decentralized Identifiers) + *verifiable credentials* pour la rendre vérifiable cryptographiquement [CSA, 2025].

Pourquoi c'est non-négociable : si deux agents partagent une identité, l'audit ne peut plus attribuer une action à *un* agent, et le déprovisionnement de l'un coupe l'autre. **Une identité = un agent.**

> **Key takeaway.** Un agent est un *principal* de première classe : un identifiant unique et stable, de type `agent`, rattaché à un owner humain. Pas de partage d'identité entre agents — sinon l'audit et le déprovisionnement deviennent impossibles.

---

## 3. Scopes & moindre privilège : ce que l'agent a le *droit* de faire

Une identité dit *qui*. Les **scopes** disent *quoi*. Un scope est une permission fine et nommée, sur le modèle des **scopes OAuth 2.0** : `invoices:read`, `ledger:write`, `payments:execute`.

Le **moindre privilège** (least privilege) : un agent ne reçoit *que* les scopes strictement nécessaires à sa mission, et rien de plus. L'agent de réconciliation a besoin de `invoices:read` et `ledger:write` — **pas** de `payments:execute`. Avec le bon découpage de scopes, l'injection de prompt du module 1 aurait échoué à l'autorisation : l'agent n'avait simplement pas le scope « virer ».

Deux raffinements opérationnels :

1. **Granularité.** Un scope trop large (`finance:*`) annule le bénéfice du moindre privilège. Préférer des scopes-actions explicites. C'est le « moindre privilège *opérationnel* » : on scope au verbe métier, pas au domaine entier.
2. **Contexte/ressource.** Un scope peut être borné à une ressource : `invoices:read` *sur le tenant A uniquement*. C'est la transition naturelle vers le module 5 (Zero Trust : la décision dépend du contexte de la requête).

OWASP a fait de cet écart un risque de premier plan : dans le **Top 10 for Agentic Applications 2026**, **Identity & Privilege Abuse** figure parmi les menaces majeures spécifiques aux agents (avec Tool Misuse et Rogue Agents) [OWASP, 2026]. Le risque jumeau, l'*excessive agency* (LLM08 dans le Top 10 LLM 2025), désigne précisément un agent doté de plus de capacités que nécessaire.

> **Key takeaway.** Scope au verbe métier, jamais au domaine entier. Le moindre privilège transforme une faille de sécurité (« l'agent a été détourné ») en simple refus d'autorisation (« il n'avait pas le scope »).

---

## 4. Délégation : agir *on-behalf-of* un humain

Un agent agit rarement « pour lui-même » : il agit **au nom d'un humain** (l'utilisateur qui l'a invoqué) ou **au nom d'un autre agent** qui le sous-traite. C'est la **délégation**, et elle forme une **chaîne**.

Exemple : Alice (analyste) demande à l'agent *orchestrateur* de préparer un rapport ; l'orchestrateur délègue la collecte de données à un *sous-agent data*. La chaîne de délégation est :

```
Alice (human)  →  agent-orchestrator  →  agent-data-fetch
   principal         délégué 1               délégué 2
```

Deux règles structurent une délégation saine :

1. **Atténuation des privilèges (privilege attenuation).** Un délégué ne peut pas avoir *plus* de droits que son délégant. Si Alice n'a pas `payments:execute`, l'agent qu'elle invoque ne peut pas l'obtenir « par magie ». À chaque maillon, les scopes ne peuvent que **rétrécir ou rester égaux**, jamais s'élargir.
2. **Traçabilité du principal humain.** Quel que soit le nombre de maillons, on doit pouvoir remonter au **principal humain à la racine**. Le cadre agentique de Singapour le formule comme un invariant : *« les humains restent ultimement responsables »* [IMDA, 2026]. La délégation déplace l'*exécution*, jamais la *responsabilité*.

Techniquement, cela se modélise par des jetons portant un champ `act` / `on_behalf_of` (l'OAuth *token exchange*, RFC 8693, formalise ce *delegation grant*). On stocke le délégant dans le jeton, on vérifie l'atténuation à chaque échange.

> **Key takeaway.** La délégation est une chaîne où les privilèges ne font que rétrécir et où l'on remonte toujours à un humain racine. Déléguer l'action ne délègue jamais la responsabilité.

---

## 5. Éphémérité & expiration : des identités à durée de vie courte

Le jeton « qui n'expire jamais » du module 1 est l'anti-pattern par excellence. Un *credential* longue durée est une bombe à retardement : volé, il reste valable indéfiniment.

La réponse est l'**éphémérité** : des identités/jetons à **durée de vie courte** (minutes ou heures), renouvelés à la demande, et **révocables**. Avantages :

- une fuite a une fenêtre d'exploitation bornée (le jeton expire) ;
- le **déprovisionnement** devient trivial : on arrête de renouveler, l'agent perd l'accès tout seul ;
- on peut lier la durée de vie à la *tâche* : le jeton vit le temps de la mission, puis meurt.

Le CSA insiste sur l'éphémérité comme propriété centrale d'une IAM d'agents, là où les comptes de service classiques traînaient des secrets statiques pendant des années [CSA, 2025].

Le **déprovisionnement** mérite son propre réflexe. Un agent décommissionné dont le jeton reste actif est un *orphelin* dangereux. Règle : **désactiver l'identité ⇒ tous ses jetons cessent d'être honorés**, immédiatement, sans attendre leur expiration naturelle. Sur la gestion des **secrets** (clés, credentials), le principe est le même que pour les humains : ne jamais coder un secret en dur, le faire tourner (rotation), le stocker dans un coffre — mais avec des durées encore plus courtes parce qu'un agent peut être exploité à la vitesse machine.

> **Key takeaway.** Préférer des identités éphémères et révocables aux secrets statiques. Une identité désactivée doit invalider *tous* ses jetons sur-le-champ — l'expiration courte n'est qu'un filet de sécurité, pas le mécanisme principal de révocation.

---

## 6. Zero Trust : vérifier *chaque* requête

Le fil conducteur des cinq sections précédentes est une posture : **ne jamais faire confiance implicitement, vérifier en permanence**. C'est le **Zero Trust**, standardisé par le **NIST SP 800-207** [NIST, 2020]. Son principe directeur : *« never trust, always verify »* — chaque requête d'accès est évaluée **per-request**, en fonction de l'identité, du contexte et de la politique, **quelle que soit** la position dans le réseau.

Appliqué à un agent, le point de décision (un *Policy Decision Point*, PDP, dans le vocabulaire 800-207) évalue, **à chaque action** :

1. **L'identité est-elle connue et active ?** (pas déprovisionnée)
2. **Le jeton est-il valide ?** (non expiré, non révoqué)
3. **Le scope demandé est-il accordé ?** (moindre privilège)
4. **La délégation est-elle valide ?** (chaîne remonte à un humain, atténuation respectée)
5. **Le contexte autorise-t-il ?** (ressource, heure, tenant…)

Si une seule réponse est « non » → **deny**. Aucune action n'est « pré-autorisée » sous prétexte que la précédente est passée. C'est ce qui distingue Zero Trust d'un modèle périmétrique (« une fois dans le réseau, tu es de confiance ») : pour un agent qui enchaîne des centaines d'appels d'outils, *chaque* appel repasse par le PDP.

Le code applique du module (`02-code/08-identite-iam.py`) implémente précisément ce PDP : un mini **scope-based access control engine** qui, pour chaque requête, vérifie identité + expiration + scope + chaîne de délégation et renvoie `allow`/`deny` avec un motif.

> **Key takeaway.** Zero Trust = décision d'accès *per-request*. Pour un agent, chaque appel d'outil est ré-évalué (identité active, jeton valide, scope accordé, délégation atténuée, contexte OK) ; le moindre « non » bloque. La confiance ne se gagne jamais « une fois pour toutes ».

---

## Spaced repetition

1. **Q.** Un ingénieur veut « aller vite » et donne à un agent le jeton de service partagé de l'équipe. Cite trois garanties de gouvernance qu'il vient de casser.
   **R.** Identité distincte (le jeton est partagé → l'audit ne peut plus attribuer l'action à *un* agent) ; moindre privilège (le jeton a tous les scopes de l'équipe, pas seulement ceux de la mission) ; éphémérité (le jeton de service ne meurt pas avec la tâche). On peut ajouter : traçabilité de la délégation (impossible de dire « au nom de qui »).

2. **Q.** Dans une chaîne de délégation `Humain → agent A → agent B`, quelle est la règle d'**atténuation des privilèges** et pourquoi protège-t-elle ?
   **R.** Un délégué ne peut jamais avoir plus de scopes que son délégant ; à chaque maillon les scopes ne font que rétrécir ou rester égaux. Cela empêche un agent d'acquérir « par magie » un droit que le principal humain racine n'a pas — la délégation déplace l'exécution, jamais la responsabilité.

3. **Q.** Pourquoi l'**éphémérité** d'un jeton ne dispense-t-elle PAS d'un mécanisme de révocation explicite ?
   **R.** L'expiration courte ne fait que *borner* la fenêtre d'exploitation d'une fuite ; entre l'incident et l'expiration, le jeton reste valable. Une identité désactivée (déprovisionnement, agent compromis) doit invalider *tous* ses jetons immédiatement. L'expiration est un filet, pas le levier principal.

4. **Q.** Qu'est-ce que le principe Zero Trust [NIST SP 800-207] change concrètement pour un agent qui enchaîne 200 appels d'outils dans une tâche ?
   **R.** Chacun des 200 appels est ré-évalué *per-request* par le point de décision (identité active, jeton valide, scope accordé, délégation atténuée, contexte OK). Aucun appel n'est « pré-autorisé » parce que le précédent est passé : « never trust, always verify ».

5. **Q.** L'agent de réconciliation est détourné par une injection de prompt et tente un virement. Avec un découpage de scopes correct, où et comment l'attaque échoue-t-elle — et quel risque OWASP cela illustre-t-il ?
   **R.** Elle échoue à l'**autorisation** : l'agent n'a que `invoices:read` + `ledger:write`, pas `payments:execute`, donc le PDP renvoie `deny`. C'est l'illustration directe de *Identity & Privilege Abuse* / excessive agency [OWASP, 2026] : le moindre privilège transforme un détournement réussi en simple refus d'accès.
