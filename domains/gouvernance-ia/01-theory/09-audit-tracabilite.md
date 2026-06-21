# Audit, observabilite & tracabilite runtime

## Pourquoi ce module
Quand un agent fait une action irreversible (virement, suppression, e-mail client), la seule preuve qui tient un audit ou un tribunal est une **trace runtime infalsifiable** — pas un souvenir, pas un export retouchable. Ce module construit cette preuve **machine**.

---

## 1. Le probleme concret : un agent a vire 40 000 €, qui prouve quoi ?

Lundi 9h. Un agent « finance-ops » declenche un virement fournisseur de 40 000 €. Le fournisseur conteste : « ce virement n'etait pas autorise ». Le DSI vous demande trois choses, dans l'ordre :

1. **Qui** a declenche l'action (quel agent, pour quel owner humain) ?
2. **Sur quelle autorisation** (quel scope, quelle politique l'a laisse passer) ?
3. **La trace a-t-elle ete modifiee** depuis (quelqu'un a-t-il « nettoye » les logs apres coup) ?

Si votre journal est un fichier `app.log` que n'importe quel admin peut editer, vous ne pouvez repondre **a aucune** des trois questions de maniere defendable. Un attaquant — ou un employe couvrant une erreur — efface ou reecrit la ligne genante, et votre « preuve » s'evapore.

Le reflexe naif : « on a les logs ». Mais un log mutable n'est pas une preuve, c'est un **recit**. La gouvernance runtime exige de transformer le recit en **preuve verifiable** : append-only (on ajoute, on ne reecrit jamais) et tamper-evident (toute alteration se **detecte**).

> **Key takeaway** — Un log qu'on peut editer en silence ne prouve rien. La question de gouvernance n'est pas « avons-nous des logs ? » mais « notre journal **detecte-t-il** toute alteration et permet-il de **reconstruire** l'action complete ? »

---

## 2. Append-only & tamper-evident : la chaine de hash

Comment rendre un journal « infalsifiable » sans base de donnees ni blockchain ? On chaine chaque entree a la precedente par un **hash cryptographique**, exactement comme un registre de caisse ou chaque page reference la somme de la precedente.

Principe (concret d'abord) :

```
entree[0] : { action: "transfer", amount: 40000, prev_hash: "GENESIS" }
            -> hash_0 = SHA256( "GENESIS" + payload_0 )

entree[1] : { action: "email", to: "client@x", prev_hash: hash_0 }
            -> hash_1 = SHA256( hash_0 + payload_1 )

entree[2] : { ..., prev_hash: hash_1 }
            -> hash_2 = SHA256( hash_1 + payload_2 )
```

Chaque entree embarque le hash de la precedente. Le hash d'une entree depend donc de **tout l'historique anterieur**. Consequence : si un attaquant modifie l'entree 0 (passer 40000 a 400), son `hash_0` recalcule change, donc `hash_1` ne correspond plus a ce qui est stocke, donc la verification casse **a la position 0**. L'alteration n'est pas seulement detectee : on sait **ou** elle a eu lieu.

C'est le meme mecanisme que les arbres de Merkle ou les ledgers Git : **tamper-evident**, pas tamper-proof. On ne peut pas *empecher* quelqu'un d'ecrire dans le fichier (controle d'acces a part) ; on garantit qu'une modification **ne passe pas inapercue**.

Deux proprietes a ne pas confondre :

- **Tamper-evident** (ce qu'on construit ici) : l'alteration est detectable. Faisable en pur stdlib avec `hashlib`.
- **Tamper-proof** (plus fort) : l'alteration est impossible. Exige du materiel (HSM), un WORM storage, ou un ancrage externe (timestamping signe, notarisation). On le **cite** mais on ne le simule pas.

Une nuance importante : une chaine de hash seule protege contre la modification **interne** d'une entree passee, mais un acteur qui controle tout le fichier peut **recalculer toute la chaine** apres son edit. Pour s'en premunir, on **ancre** periodiquement le dernier hash ailleurs (le **checkpoint** : envoye a un systeme tiers append-only, signe, ou horodate par une autorite). Le checkpoint fige « voici l'etat du journal a 9h05 » hors de portee de l'attaquant.

> **Key takeaway** — Chainer chaque entree par `hash = SHA256(prev_hash + payload)` rend le journal **append-only et tamper-evident** : toute alteration casse la chaine a la position exacte de l'edit. Pour resister a un attaquant qui recalcule tout, on **ancre** periodiquement le dernier hash dans un systeme tiers (checkpoint).

---

## 3. Que mettre dans une entree : le quintuple qui / quoi / quand / autorisation / resultat

Une trace qui ne dit pas *sur quelle autorisation* l'action est passee est inutile pour la gouvernance. L'entree d'audit minimale d'un agent contient **cinq** champs indissociables :

| Champ | Question | Exemple |
|-------|----------|---------|
| **who** | Quel agent, pour quel owner ? | `agent="finance-ops"`, `owner="a.dupont"` |
| **what** | Quelle action + parametres ? | `action="bank_transfer"`, `amount=40000` |
| **when** | Horodatage (UTC, monotone) | `2026-06-21T09:00:03Z` |
| **authorization** | Quel scope/politique a autorise ? | `scope="payments:execute"`, `policy="auto<50k"`, `decision="ALLOW"` |
| **outcome** | Resultat + correlation | `status="success"`, `trace_id="..."` |

Le champ **authorization** est ce qui distingue un audit trail de gouvernance d'un simple log applicatif. Il relie l'action a la decision de la chaine d'identite (J8) et de policy (J14) : « l'agent avait le scope `payments:execute`, la politique `auto<50k` a renvoye ALLOW ». C'est exactement ce qui permet la reconstruction d'incident de la section 5.

Les conventions semantiques **OpenTelemetry GenAI** (OTel GenAI semconv) standardisent ces champs au niveau industriel : spans pour les appels LLM, attributs pour les **tool calls** et le **token usage**, propagation d'un `trace_id` a travers une chaine d'agents. **Statut a signaler** : ces conventions sont en statut **Development/Experimental** au S1 2026 — pas encore stables [OpenTelemetry GenAI SIG, 2026]. On s'en inspire (les *noms de champs* convergeront vers ce standard), mais on ne traite pas l'API comme figee.

> **Key takeaway** — Une entree d'audit gouvernable = **who + what + when + authorization + outcome**. Le champ `authorization` (scope + politique + decision) est ce qui transforme un log applicatif en preuve de gouvernance. Les noms de champs s'alignent sur OTel GenAI semconv, statut **experimental** en 2026.

---

## 4. Correlation : suivre une action a travers une chaine d'agents

Un agent en appelle un autre, qui appelle un outil, qui declenche un sous-agent. Sans **identifiant de correlation**, l'audit est une pile de lignes deconnectees. La solution (empruntee au tracing distribue / OTel) : un **`trace_id`** unique par requete de bout en bout, et un **`span_id`** par etape, chaque span referencant son parent.

```
trace_id = T-7f3a (toute la requete "payer le fournisseur")
  span A  agent=orchestrator       parent=none
  span B  agent=finance-ops        parent=A     action=bank_transfer
  span C  tool=banking_api         parent=B     status=200
```

Avec ce fil, on reconstruit l'arbre causal complet d'une seule action utilisateur, meme si elle a traverse cinq agents. C'est exactement la « reconstruction de la trace complete » visee par le module. La CSA insiste sur ce point pour l'IAM agentique : l'auditabilite suppose de tracer la **chaine de delegation** et la propagation d'identite a travers les appels [Cloud Security Alliance, 2025].

> **Key takeaway** — Un `trace_id` partage + un `span_id` par etape (referencant son parent) permettent de reconstruire l'**arbre causal** d'une action a travers plusieurs agents/outils — sinon l'audit n'est qu'une pile de lignes orphelines.

---

## 5. Reconstruction d'incident : du log brut au recit defendable

Revenons au virement de 40 000 €. Avec un journal chaine, correle et horodate, la reconstruction devient mecanique :

1. **Verifier l'integrite** de toute la chaine — si elle casse, on sait que la preuve a ete alteree (et ou).
2. **Filtrer par `trace_id`** la requete incriminee → toutes les etapes, dans l'ordre.
3. **Lire le quintuple** : qui (agent + owner), quoi (montant), quand (timestamp), **sur quelle autorisation** (scope + politique + decision), resultat.
4. **Produire le recit** : « A 9h00:03 UTC, l'agent `finance-ops` possede par `a.dupont`, disposant du scope `payments:execute`, a execute un virement de 40 000 € ; la politique `auto<50k` a renvoye ALLOW ; statut success. Chaine d'audit verifiee, integre. »

Ce lien entre trace et **incident/attaque** est aussi un angle securite : la taxonomie d'attaques de NIST (prompt injection, exfiltration) suppose qu'on puisse, apres coup, **reconstruire ce que l'agent a reellement fait** pour qualifier l'attaque [NIST AI 100-2 E2025, 2025]. Un journal tamper-evident est la condition prealable d'une reponse a incident credible (qu'on operationnalisera J10).

C'est ici que se joue la distinction avec J12 (documentation & assurance) : J12 produit la **preuve humaine** (model/system/agent cards, safety case — statique, redigee ex-ante). J9 produit la **preuve machine** (trace runtime, generee par le systeme lui-meme, ex-post). Les deux sont complementaires : l'une argumente *que le systeme devrait etre sur*, l'autre prouve *ce qu'il a effectivement fait*.

> **Key takeaway** — Reconstruire un incident = verifier l'integrite, filtrer par `trace_id`, lire le quintuple, ecrire le recit. La trace runtime est la **preuve machine** (ce que l'agent a fait) — complementaire de la **preuve humaine** statique de J12 (model/system cards, safety case).

---

## 6. Retention, integrite dans la duree & limites

Une trace n'a de valeur que si elle est **disponible quand on en a besoin** — souvent des annees apres (un AIPD/DPIA, un controle CNIL, un litige). Trois decisions de gouvernance :

- **Duree de retention** : alignee sur l'obligation legale et le risque. Trop court = on detruit la preuve ; trop long = on accumule des donnees personnelles (tension avec la minimisation RGPD vue J6). On documente une duree justifiee, pas « pour toujours ».
- **Integrite dans la duree** : le checkpoint (section 2) doit etre re-verifiable apres archivage. On verifie la chaine **a chaque lecture critique**, pas seulement a l'ecriture.
- **Confidentialite** : un audit trail contient des donnees sensibles (montants, identites, parametres). Le journal lui-meme est un actif a proteger — chiffrement au repos, acces restreint en lecture, pseudonymisation des champs personnels si possible.

Limites a enoncer honnetement :

- Un journal tamper-evident **ne previent pas** l'action malveillante ; il la rend **prouvable apres coup**. L'enforcement (bloquer avant) est le sujet de J14.
- La chaine de hash detecte l'edit d'une entree, **pas** la suppression du fichier entier ni le rejeu complet par un acteur tout-puissant — d'ou l'ancrage externe.
- « Tamper-evident » ≠ « tamper-proof » : la vraie inviolabilite exige du materiel (HSM/WORM), hors stdlib.

> **Key takeaway** — La retention est un arbitrage **preuve vs minimisation** (RGPD) : duree justifiee, ni trop courte ni infinie. Le journal est lui-meme un actif sensible (a chiffrer/restreindre), et il **prouve** sans **prevenir** — la prevention, c'est l'enforcement de J14.

---

## Spaced repetition

**Q1.** Pourquoi un fichier `app.log` editable ne constitue-t-il pas une preuve de gouvernance, meme s'il contient toute l'information ?
**R1.** Parce qu'il est mutable : un acteur (attaquant ou employe couvrant une erreur) peut reecrire ou supprimer la ligne genante **sans laisse de trace**. Une preuve doit etre append-only et tamper-evident — toute alteration doit se **detecter**. Un log mutable est un recit, pas une preuve.

**Q2.** Comment une chaine de hash detecte-t-elle qu'une entree passee a ete modifiee, et indique-t-elle *ou* ?
**R2.** Chaque entree stocke `hash = SHA256(prev_hash + payload)`. Modifier une entree change son hash recalcule, donc le `prev_hash` de l'entree suivante ne correspond plus a ce qui est stocke : la verification casse exactement **a la position de l'edit**. On detecte l'alteration ET sa localisation.

**Q3.** Quels sont les cinq champs indissociables d'une entree d'audit gouvernable, et lequel distingue un audit trail d'un simple log applicatif ?
**R3.** who / what / when / **authorization** / outcome. Le champ **authorization** (scope + politique + decision ALLOW/DENY) est le differenciateur : il relie l'action a la chaine d'identite (J8) et de policy (J14), permettant la reconstruction d'incident.

**Q4.** Quel est le statut des conventions semantiques OpenTelemetry GenAI au S1 2026, et pourquoi est-ce important de le signaler ?
**R4.** Statut **Development/Experimental** — pas stable [OpenTelemetry GenAI SIG, 2026]. Important car les noms de champs/attributs peuvent encore changer : on s'en inspire pour la convergence future, mais on ne traite pas l'API comme figee dans une implementation de production.

**Q5.** Une chaine de hash interne suffit-elle face a un attaquant qui controle tout le fichier de log ? Sinon, quelle parade ?
**R5.** Non : un acteur tout-puissant peut **recalculer toute la chaine** apres son edit. La parade est l'**ancrage externe** (checkpoint) : envoyer/signer/horodater periodiquement le dernier hash dans un systeme tiers append-only, ce qui fige l'etat du journal hors de sa portee.
