# J4 — Taxonomie des risques & NIST AI RMF

> **Temps estime** : 45-60 min | **Prerequis** : J1 (pourquoi gouverner), J2 (4 piliers), J3 (registry)
> **Objectif** : savoir classer un risque dans une taxonomie defendable, appliquer les 4 fonctions du NIST AI RMF a un systeme agentique, et produire un score de risque qu'on peut justifier devant un comite.

## Pourquoi ce module

Inventorier les agents (J3) ne suffit pas : il faut savoir **lesquels sont dangereux et pourquoi**. Ce module donne le langage commun (taxonomie) et la methode officielle (NIST AI RMF) pour transformer "ca m'inquiete" en un risque nomme, score et assigne a une fonction de gouvernance.

---

## 1. Un risque concret avant la theorie

Prenons un agent reel d'une flotte d'entreprise.

```
Agent : "invoice-reconciler"
Owner : equipe Finance
Permissions : lire la boite mail comptable, ecrire dans l'ERP, declencher un virement < 5 000 EUR
Autonomie : execute sans validation humaine
```

Question : quel est son risque ? "Il pourrait mal payer" est une intuition, pas un risque gouvernable. Decomposons :

- **Quoi** : l'agent lit un email piege ("nouveau RIB fournisseur"), l'interprete comme une instruction et **execute un virement** vers un compte frauduleux. C'est de la **prompt injection** qui devient une **action irreversible** parce que l'agent a la permission de virer.
- **Vraisemblance** : moyenne — les emails comptables sont une cible classique, l'agent lit du contenu non fiable.
- **Impact** : eleve — argent sorti, irreversible, exposition reputationnelle.
- **Cause racine** : un agent qui *agit* (virement) sur une entree *non fiable* (email externe), sans humain dans la boucle.

On vient de faire, intuitivement, le travail que les cadres formalisent : **nommer** le risque, le **situer** dans une taxonomie (injection → excessive agency → action irreversible), et le **scorer** (vraisemblance × impact). Le reste du module rend ce reflexe systematique et reproductible.

> **Key takeaway** : un risque gouvernable n'est pas une inquietude vague — c'est un triplet *cause → effet → impact* nomme, classe et score. Si vous ne pouvez pas l'ecrire en une phrase, vous ne pouvez pas le gerer.

---

## 2. La taxonomie : un langage partage pour nommer les risques

Une **taxonomie de risques** est une classification stable qui permet a deux personnes de parler du meme risque avec les memes mots. Sans elle, chacun reinvente ses categories et rien ne se compare.

### 2.1 L'AI Risk Repository (MIT) — deux axes

Le **MIT AI Risk Repository** consolide 1 700+ risques extraits de 65 frameworks, organises en **deux taxonomies complementaires** [Slattery et al., 2024] :

- **Taxonomie causale** — classe le risque par *comment* il survient, selon 3 dimensions :
  - **Entite** : *humain* vs *IA* (qui cause le risque ?)
  - **Intention** : *intentionnel* vs *non intentionnel* (etait-ce voulu ?)
  - **Timing** : *pre-deploiement* vs *post-deploiement* (avant ou apres la mise en prod ?)
- **Taxonomie par domaine** — classe le risque par *quel sujet* il touche, en 7 grands domaines (ex. discrimination & toxicite ; vie privee & securite ; desinformation ; acteurs malveillants ; interaction homme-machine ; impacts socio-economiques ; securite & defaillances des systemes IA).

Reprenons l'agent `invoice-reconciler`. Le virement frauduleux se classe ainsi :

| Axe causal | Valeur |
|------------|--------|
| Entite | IA (l'agent execute) — declenche par un humain malveillant externe |
| Intention | non intentionnel **cote systeme** (l'agent ne "veut" pas frauder) |
| Timing | post-deploiement (survient en exploitation) |

Domaine : *securite & defaillances des systemes IA* + *acteurs malveillants*. On a maintenant deux coordonnees stables pour ce risque.

### 2.2 Pourquoi deux taxonomies plutot qu'une

La taxonomie **causale** sert a choisir la *contre-mesure* (un risque post-deploiement non intentionnel appelle du monitoring runtime, pas une revue de conception). La taxonomie **par domaine** sert a *l'agregation* ("combien de risques vie privee dans la flotte ?") et au mapping reglementaire. On ne choisit pas — on tague selon les deux.

> **Key takeaway** : une taxonomie n'est pas de la bureaucratie, c'est une *adresse* pour chaque risque. L'AI Risk Repository en propose deux : *causale* (entite/intention/timing) pour choisir la parade, *par domaine* (7 domaines) pour agreger et mapper. Taguez les deux.

---

## 3. Les risques propres aux agents

La plupart des taxonomies generiques (vie privee, biais, toxicite) s'appliquent a tout systeme IA. Mais un agent **agit** : il appelle des outils, execute des transactions, enchaine des etapes. Cela cree une famille de risques que les modeles "qui ne font que parler" n'ont pas.

- **Excessive agency** — l'agent a *plus de pouvoir que necessaire* pour sa tache (permissions trop larges, autonomie trop haute). C'est la cause racine de la plupart des incidents agentiques : ce n'est pas le LLM qui est dangereux, c'est ce qu'on l'a autorise a *faire*.
- **Tool misuse / Tool abuse** — l'agent utilise un outil legitime de facon nuisible (envoyer 10 000 emails, supprimer une table, appeler une API couteuse en boucle).
- **Actions irreversibles** — virement, suppression, envoi externe, publication. Une fois faites, on ne revient pas. Le critere "reversible ?" est decisif pour calibrer les garde-fous.
- **Prompt injection menant a une action** — une entree non fiable (email, page web, document) detourne l'agent. Pour un agent qui *agit*, l'injection ne produit pas qu'un mauvais texte : elle produit un mauvais *acte*.

Ces categories se retrouvent dans les taxonomies de securite : l'**excessive agency** est un risque nomme par l'OWASP cote LLM, et l'**adversarial ML** (dont la prompt injection) est formalise par le NIST [NIST, 2025, AI 100-2]. Le NIST AI 100-2 (2025) est d'ailleurs la premiere edition a nommer explicitement les agents autonomes comme surface de menace.

> **Key takeaway** : un agent qui *agit* herite de tous les risques d'un modele **plus** une famille propre : excessive agency, tool misuse, actions irreversibles, injection-vers-action. Le bon reflexe : pour chaque permission d'agent, demander "et si elle etait detournee ?".

---

## 4. Le NIST AI RMF : 4 fonctions pour structurer la reponse

Nommer et scorer un risque, c'est bien. Mais *qui fait quoi, quand* ? Le **NIST AI Risk Management Framework 1.0** [NIST, 2023, AI 100-1] structure la gestion du risque en **4 fonctions** :

| Fonction | Question qu'elle repond | Pour l'agent `invoice-reconciler` |
|----------|------------------------|-----------------------------------|
| **GOVERN** | Qui est responsable, quelles regles, quelle culture ? | Finance est owner ; politique "pas de virement auto > 1 000 EUR sans humain". |
| **MAP** | Quels risques, dans quel contexte ? | Identifier "injection → virement frauduleux", le contexte (emails externes), les parties prenantes. |
| **MEASURE** | A quel point, mesure quantifiable ? | Scorer vraisemblance × impact ; tester l'agent contre des emails pieges (eval). |
| **MANAGE** | Que fait-on, priorisation et traitement ? | Decider : mitiger (HITL au-dela de 1 000 EUR), accepter, ou transferer le risque residuel. |

Point cle : **GOVERN est transversale**. Elle n'est pas une etape "avant" les trois autres — elle les *entoure*. Sans culture/responsabilite (Govern), Map/Measure/Manage sont des exercices techniques sans force d'application.

### 4.1 Map → Measure → Manage : le cycle operationnel

Les trois fonctions "actives" forment un cycle, pas une ligne :

```
MAP      : "voici les risques et leur contexte"   (on les liste)
   |
MEASURE  : "voici a quel point ils sont graves"   (on les score / teste)
   |
MANAGE   : "voici ce qu'on fait de chacun"        (mitiger / accepter / transferer)
   |
   +--> boucle : un nouveau risque emerge -> retour MAP
```

Le RMF est volontairement **non prescriptif** : il dit *quoi faire* (les 4 fonctions), pas *comment exactement*. C'est un cadre **volontaire** (pas une loi) — a distinguer d'une obligation reglementaire comme l'EU AI Act (J5). Le **NIST AI RMF Playbook** [NIST, 2023] fournit les actions concretes pour operationnaliser chaque fonction.

### 4.2 Le profil GenAI (NIST AI 600-1) en complement

Le RMF 1.0 est generique. Pour l'IA generative, le NIST a publie un **profil GenAI** [NIST, 2024, AI 600-1] : il liste **12 categories de risques** propres a la GenAI (dont confabulation/hallucination, contenu CBRN, vie privee, homogeneisation) et **200+ actions suggerees** rattachees aux 4 fonctions. On ne remplace pas le RMF : on l'**applique** via ce profil quand le systeme est generatif.

> **Key takeaway** : le NIST AI RMF repond a "qui fait quoi" avec 4 fonctions — **GOVERN** (transversale : responsabilite et culture) entoure **MAP** (lister) → **MEASURE** (scorer/tester) → **MANAGE** (traiter). C'est un cadre *volontaire* et non prescriptif ; le profil GenAI (600-1) l'instancie pour l'IA generative.

---

## 5. Scorer un risque de facon defendable

Un score doit pouvoir etre **explique a un comite** et **reproduit** par quelqu'un d'autre. La formule de base est universelle :

```
criticite = vraisemblance x impact
```

### 5.1 Echelles explicites

Le piege est d'utiliser des mots ("eleve", "moyen") sans definition. On fixe des echelles ancrees :

| Niveau | Vraisemblance | Impact |
|--------|---------------|--------|
| 1 | rare (pas vu en 1 an) | negligeable (gene mineure) |
| 2 | peu probable | limite (rattrapable, cout faible) |
| 3 | possible (quelques fois/an) | serieux (perte notable, manuel requis) |
| 4 | probable (mensuel) | grave (financier/legal, difficile a annuler) |
| 5 | quasi-certain (continu) | critique (irreversible, securite, sanction) |

`invoice-reconciler` : vraisemblance 3 (les fraudes au RIB existent), impact 5 (virement irreversible) → **criticite 15/25**. Defendable parce que **chaque chiffre est ancre** sur une definition, pas sur un ressenti.

### 5.2 Modulateur agentique : reversibilite & autonomie

Pour un agent, deux facteurs aggravent un score "brut" :

- **Action irreversible** → l'impact monte d'un cran (on ne peut pas annuler).
- **Autonomie sans humain** → la vraisemblance qu'un risque se *materialise* monte (aucun filet humain pour intercepter).

C'est pourquoi le meme risque "logique" est plus grave sur un agent *autonome + irreversible* que sur un agent *en lecture seule + human-in-the-loop*. Le score doit refleter cela, sinon il ment.

### 5.3 De Measure a Manage : le seuil de traitement

Un score sert a **prioriser**, pas a decorer. On fixe un seuil : par ex. criticite ≥ 12 → traitement obligatoire (Manage), 6-11 → surveillance, < 6 → accepte et documente. Le seuil est une **decision de gouvernance** (Govern), pas un chiffre magique — il s'assume et se trace.

> **Key takeaway** : un score defendable = formule simple (`vraisemblance × impact`) + echelles *ancrees* (chaque niveau a une definition) + modulateurs agentiques (reversibilite, autonomie) + un seuil de traitement assume. Le but n'est pas la precision illusoire, c'est la *reproductibilite* et la *priorisation*.

---

## 6. Boucler le tout : du risque a la fonction RMF

Mettons les briques ensemble pour `invoice-reconciler` :

1. **MAP** — risque nomme : "email piege (prompt injection) → virement frauduleux (action irreversible)". Taxonomie causale : IA / non intentionnel / post-deploiement. Domaine : securite & acteurs malveillants.
2. **MEASURE** — vraisemblance 3, impact 5 (modulateur : irreversible + autonome) → criticite 15/25.
3. **MANAGE** — 15 ≥ 12 → traitement : imposer HITL au-dela de 1 000 EUR, restreindre les RIB autorises, journaliser. Risque residuel re-score apres mitigation.
4. **GOVERN** (transversale) — Finance reste owner ; la politique de seuil et le risk register sont revus en comite.

C'est exactement ce que le **risk register** code de ce module produit : une ligne par risque, taguee par fonction RMF, triee par criticite, avec un seuil de traitement. Un artefact qu'on montre a un auditeur ou un board.

> **Key takeaway** : la chaine complete est MAP (nommer + classer) → MEASURE (scorer) → MANAGE (traiter selon seuil), le tout sous GOVERN (responsabilite). Le livrable concret est un *risk register* trie par criticite et tague par fonction — pas un slide, un artefact reproductible.

---

## Spaced repetition

1. **Q** : Quelles sont les 4 fonctions du NIST AI RMF, et laquelle est transversale ?
   **R** : GOVERN, MAP, MEASURE, MANAGE. **GOVERN** est transversale — elle entoure les trois autres (responsabilite, culture, regles) plutot que d'etre une simple etape sequentielle.

2. **Q** : L'AI Risk Repository (MIT) propose deux taxonomies. Nommez-les et donnez l'usage de chacune.
   **R** : (1) **Causale** — entite (humain/IA) × intention (intentionnel/non) × timing (pre/post-deploiement), sert a choisir la contre-mesure. (2) **Par domaine** — 7 grands domaines, sert a agreger et mapper au reglementaire.

3. **Q** : Pourquoi "excessive agency" est-elle souvent la cause racine d'un incident agentique plutot que le LLM lui-meme ?
   **R** : Parce que le danger ne vient pas de ce que le modele *dit* mais de ce qu'on l'a autorise a *faire*. Des permissions trop larges + une autonomie trop haute transforment une erreur de raisonnement en action nuisible (virement, suppression). Restreindre l'agency coupe le risque a la racine.

4. **Q** : Deux agents ont le meme risque "logique" mais l'un est autonome + action irreversible, l'autre en lecture seule + human-in-the-loop. Pourquoi leurs scores doivent-ils differer ?
   **R** : Les modulateurs agentiques : l'irreversibilite augmente l'**impact** (rien a annuler), l'autonomie sans humain augmente la **vraisemblance** de materialisation (aucun filet). Un score qui les ignore sous-estime le vrai risque.

5. **Q** : Qu'est-ce qui rend un score de risque "defendable" devant un comite ?
   **R** : Des echelles **ancrees** (chaque niveau 1-5 a une definition explicite, pas un ressenti), une formule simple et reproductible (vraisemblance × impact), des modulateurs justifies, et un **seuil de traitement** assume comme decision de gouvernance — de sorte qu'un tiers obtienne le meme score.
