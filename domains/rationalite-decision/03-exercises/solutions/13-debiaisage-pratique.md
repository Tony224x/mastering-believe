# Solutions — Module 13 : Débiaisage en pratique

## Exercice 1 — Mener un pre-mortem

### 1. Énoncé d'hypothèse (exemple modèle)

> "Nous sommes dans 4 semaines. Le service de livraison express Nord-Sud a été lancé il y a une semaine et c'est un échec : les délais de 4 h ne sont pas tenus, des clients ont annulé leurs contrats. Chacun écrit toutes les raisons pour lesquelles ça s'est passé ainsi."

*Pourquoi cette formulation est correcte* : l'échec est posé comme un fait accompli ("a été lancé", "c'est un échec"), pas comme une hypothèse ("pourrait échouer"). Cela active la rétrospection prospective.

### 2. Cinq causes d'échec plausibles (exemple modèle)

1. **Technique** : Le logiciel de géolocalisation des camions a bugué dès le premier jour — on n'avait pas fait de test de charge à 50 livraisons simultanées.
2. **Humain** : Deux chauffeurs sur trois n'avaient pas été formés au protocole de priorisation des colis urgents ; les colis express partaient en dernier.
3. **Organisationnel** : Le délai de 4 h avait été fixé sans consulter le service douane interne — certains colis nécessitent 2 h de vérification supplémentaire.
4. **Externe** : Le prestataire de maintenance des véhicules frigorifiques était en sous-effectif en période estivale ; une panne le jour J n'a pas été réparée avant 6 h.
5. **Hypothèse erronée** : On avait supposé que les deux entrepôts étaient à 45 min de route ; en réalité, la route directe est fermée pour travaux depuis 2 semaines.

### 3. Cause la plus critique et action préventive

**Cause retenue** : l'absence de test de charge du logiciel (cause 1) — car une défaillance logicielle affecte 100 % des livraisons simultanément, alors que les autres causes ont des effets partiels.

**Action préventive** : organiser, deux semaines avant le lancement, un test de charge complet simulant 60 livraisons simultanées (120 % de la capacité prévue), avec un responsable technique désigné et un critère d'acceptation explicite (zéro bug bloquant sur 2 h de test).

### 4. Biais cognitif identifié

**Biais de confirmation** : l'équipe avait tenu des tests partiels concluants (20 livraisons) et en avait déduit que le système fonctionnerait à plus grande échelle, sans chercher à le réfuter sur des volumes supérieurs. Les signaux d'alerte (lenteurs observées à 30 livraisons lors d'un test informel) avaient été rationalisés comme "temporaires". Sans pre-mortem, personne n'aurait mentionné ce doute pour ne pas "freiner l'élan" de l'équipe.

---

## Exercice 2 — Appliquer une checklist anti-biais

| # | Question | Réponse | Ce qui manque / action corrective |
|---|----------|---------|----------------------------------|
| 1 | L'équipe a-t-elle un intérêt direct dans le résultat ? | **Non** | Le responsable achats gère la relation fournisseur habituel depuis 3 ans — un conflit d'intérêts implicite existe. Action : désigner un évaluateur sans lien avec ce fournisseur pour valider l'offre. |
| 2 | L'analyse part-elle d'un cas de référence comparable (Outside View) ? | **Non** | Aucun benchmark prix marché n'a été fait. Action : demander 2 devis concurrents ou consulter un catalogue de référence (ex : prix catalogue distributeur industriel). |
| 3 | Des informations défavorables ont-elles été activement recherchées ? | **Inconnu** | Aucune mention de recherche d'avis négatifs sur le fournisseur (fiabilité, SAV, délais). Action : consulter les bons de maintenance des imprimantes actuelles de ce fournisseur. |
| 4 | L'avis d'experts avec des hypothèses différentes a-t-il été sollicité ? | **Non** | Les utilisateurs finaux (techniciens ateliers) n'ont pas été consultés sur leurs besoins réels. Action : recueillir leurs retours avant toute validation. |
| 5 | Les hypothèses clés ont-elles été formulées explicitement ? | **Non** | L'hypothèse implicite est "le fournisseur actuel est le meilleur choix" — non challengée. Action : lister et voter explicitement les 3 critères de décision (prix, fiabilité, SAV). |
| 6 | Un scénario d'échec a-t-il été construit (pre-mortem) ? | **Non** | Aucun. Action : poser la question "si ces imprimantes tombent toutes en panne en 6 mois, qu'avons-nous raté ?" avant de signer. |
| 7 | Les options alternatives ont-elles été évaluées sur les mêmes critères ? | **Non** | Une seule offre présentée. Action : évaluer a minima une offre alternative sur prix, garantie et délai de remplacement. |
| 8 | La décision serait-elle la même si les chiffres changeaient de ±20 % ? | **Inconnu** | Si le fournisseur augmente ses tarifs de 20 % l'an prochain ou ne peut livrer qu'en 8 semaines, l'offre reste-t-elle valide ? Action : vérifier les conditions contractuelles de révision de prix. |

**Biais identifié** : la rapidité de la discussion (8 min) et l'unanimité immédiate suggèrent un **biais de statu quo** (préférence pour le fournisseur habituel) combiné à un début de **groupthink** (le responsable a présenté une conclusion, pas une question ouverte).

---

## Exercice 3 — Repérer le groupthink dans un scénario neutre

### 1. Trois symptômes identifiés

**Symptôme 1 — Autocensure**
> *Illustration* : "Aucun des deux ne prend la parole."
Les deux techniciens ont des informations pertinentes (usure de M7, risque fournisseur) mais retiennent leurs doutes. Ils anticipent la pression sociale ou l'inutilité perçue de s'opposer à la décision déjà orientée par le chef.

**Symptôme 2 — Illusion d'unanimité**
> *Illustration* : "Silence. Il note 'Validé à l'unanimité.'"
Le chef interprète l'absence de parole comme un accord. Or le silence est produit par l'autocensure — chacun pense être seul à douter, ce qui renforce le mutisme général.

**Symptôme 3 — Ancrage/leadership fermant le débat (pression vers la conformité)**
> *Illustration* : "Le chef d'équipe ouvre en déclarant : 'J'ai déjà regardé les chiffres, le planning de l'an dernier a bien fonctionné, on devrait reconduire.'"
Le chef exprime sa préférence dès l'ouverture, ce qui positionne toute objection comme une contradiction directe de l'autorité hiérarchique. Les membres moins enclins à affronter ce signal de clôture se taisent.

### 2. Garde-fous proposés

**Avant la réunion :**
1. **Recueil écrit individuel des signaux d'alerte** : envoyer par email, 24 h avant, la question "Y a-t-il des points de vigilance sur ce planning que tu veux porter en réunion ?" — les techniciens auraient écrit leurs doutes sans pression sociale directe.
2. **Vote anonyme sur les risques perçus** : via un formulaire simple (3 questions, réponse 1-5), les résultats sont affichés en début de séance avant toute prise de parole du chef.

**Pendant la réunion :**
3. **Leader parle en dernier** : le chef demande à chaque technicien d'exprimer ses observations sur les machines dont il est responsable *avant* de donner son point de vue — l'ordre inverse empêche l'ancrage.

### 3. Pourquoi le "courage" ne suffit pas

Le problème n'est pas un défaut de caractère des techniciens : ils ont eu peur de passer pour incompétents ou pour des perturbateurs, ce qui est une réaction humaine normale face à une pression hiérarchique implicite. L'autocensure est un **mécanisme social prévisible** dans tout groupe cohésif sous leadership orientant : même des individus courageux dans d'autres contextes peuvent se taire ici. Seule une **procédure structurelle** — qui rend le dissensus sûr et attendu — peut court-circuiter ce mécanisme de façon fiable. Compter sur le courage individuel, c'est espérer que chaque personne surmonte seule une pression collective — un pari perdu d'avance à l'échelle d'une organisation.
