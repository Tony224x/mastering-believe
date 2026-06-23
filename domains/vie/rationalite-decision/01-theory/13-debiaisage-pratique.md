# Module 13 — Débiaisage en pratique

> **Temps estimé** : 45 min | **Prérequis** : Modules 01-12

> **Objectif** : Maîtriser quatre outils concrets de débiaisage — pre-mortem, red teaming, checklists anti-biais, détection du groupthink — et savoir les appliquer à une décision d'équipe ou individuelle.

---

## 1. Le problème : connaître les biais ne suffit pas

Après douze modules, vous savez nommer l'ancrage, le biais de confirmation, la négligence du taux de base. Mais la recherche montre que la **conscience d'un biais ne le réduit pas** significativement (Larrick, 2004). Il faut des **procédures externalisées** — des dispositifs qui agissent sur le processus de décision *avant* que le biais s'installe, et non après.

Ce module présente quatre outils à ce niveau : le pre-mortem, le red teaming, la checklist anti-biais et la vigilance groupthink.

> **À retenir** : le débiaisage efficace s'applique au *processus* de décision, pas au raisonnement a posteriori.

---

## 2. Le pre-mortem (Klein, 2007)

### 2.1 Mécanisme

Gary Klein a observé que, avant une décision, les équipes ont tendance à **surestimer le succès** (optimisme, biais de planification) et à taire leurs doutes pour ne pas paraître négatifs. Le pre-mortem retourne le problème : on suppose que le projet **a déjà échoué**, on se projette dans le futur, et chacun écrit *pourquoi* c'est arrivé.

Ce dispositif de **"rétrospection prospective"** (*prospective hindsight*) exploite un mécanisme cognitif réel : se souvenir d'un événement passé active plus d'explications causales que d'en imaginer un futur. Klein rapporte une augmentation de ~30 % de l'identification des risques.

### 2.2 Protocole en 5 étapes

1. **Poser l'hypothèse** : "Nous sommes dans 6 mois. Le projet a échoué — complètement, irrémédiablement."
2. **Écriture individuelle silencieuse** (3-5 min) : chaque participant note toutes les causes plausibles d'échec.
3. **Tour de table non interruptif** : chaque cause est énoncée et consignée, sans débat immédiat.
4. **Regroupement** : les causes sont agrégées par thème (ressources, technique, coordination, hypothèses erronées…).
5. **Plan de mitigation** : pour chaque thème critique, une action préventive est décidée.

**Exemple neutre — projet logistique fictif** : une équipe s'apprête à ouvrir un nouvel entrepôt régional. Le responsable lance un pre-mortem. Les causes d'échec identifiées : délais de certification douanière sous-estimés, logiciel WMS non testé en conditions réelles, sous-effectif en période de montée en charge. Résultat : trois mesures préventives ajoutées au plan de lancement.

> **À retenir** : le pre-mortem crée un **espace psychologiquement sûr** pour les signaux faibles, parce que l'échec est déjà posé comme hypothèse — personne ne "joue les oiseaux de mauvais augure".

---

## 3. Le red teaming (Nemeth et al., 2001)

### 3.1 Mécanisme

Le **red teaming** consiste à désigner un sous-groupe chargé de **challenger activement** un plan, en jouant l'adversaire ou en cherchant toutes les failles. À distinguer de l'avocat du diable formel (*devil's advocate*) : Nemeth et ses collègues montrent que le **dissensus authentique** — une opposition sincère, non jouée — génère davantage d'idées originales que l'opposition assignée par rôle, perçue comme un jeu.

La différence est cognitive : une critique assignée déclenche une défense du plan ; une critique authentique force une révision réelle.

### 3.2 Protocole en 4 étapes

1. **Désigner le red team** : 1-2 personnes volontaires (idéalement avec un point de vue déjà sceptique) — pas de rotation forcée.
2. **Brief séparé** : le red team reçoit le plan *sans* avoir participé à son élaboration ; il prépare ses attaques indépendamment.
3. **Session de confrontation structurée** : le red team présente ses contre-arguments ; le plan team répond factuellement ; un modérateur empêche les dérives personnelles.
4. **Décision révisée** : les vulnérabilités identifiées sont intégrées ou documentées avec leur justification si elles sont écartées.

**Exemple neutre — décision d'achat d'équipement** : un comité envisage l'achat d'un parc de scanners industriels. Le red team (deux ingénieurs maintenance) souligne : coûts de formation non budgétés, fournisseur unique sans pièces détachées en stock local, incompatibilité avec le SCADA existant. Le comité révise le cahier des charges.

> **À retenir** : un red team fonctionne si ses membres ont une **opposition sincère**, pas un rôle théâtral. La diversité de perspective du red team est une ressource, pas un obstacle.

---

## 4. Checklists anti-biais (Gawande, 2009 ; Kahneman, Lovallo & Sibony, 2011)

### 4.1 Pourquoi les checklists fonctionnent

Atul Gawande documente comment de simples checklists réduisent les erreurs d'*ineptie* — erreurs non pas de méconnaissance, mais d'oubli de ce qu'on sait déjà — dans des environnements complexes (bloc opératoire, aviation). Le principe est identique pour les biais cognitifs : ils ne surgissent pas faute de savoir, mais faute d'une procédure qui force la vérification.

Kahneman, Lovallo et Sibony (2011) proposent une checklist de 12 questions à poser sur toute décision d'équipe. En voici les thèmes principaux, transposés dans un cadre opérationnel :

### 4.2 Checklist anti-biais (version condensée, 8 questions)

| # | Question | Biais ciblé |
|---|----------|-------------|
| 1 | L'équipe a-t-elle un intérêt direct dans le résultat de cette décision ? | Conflit d'intérêts / biais de désirabilité |
| 2 | L'analyse part-elle d'un cas de référence comparable (Outside View) ? | Biais de planification / ancrage |
| 3 | Des informations défavorables ont-elles été activement recherchées ? | Biais de confirmation |
| 4 | L'avis d'experts avec des hypothèses différentes a-t-il été sollicité ? | Excès de confiance / pensée de groupe |
| 5 | Les hypothèses clés ont-elles été formulées explicitement et challengées ? | Point aveugle cognitif |
| 6 | Un scénario d'échec a-t-il été construit (pre-mortem) ? | Optimisme / excès de confiance |
| 7 | Les options alternatives ont-elles été évaluées sur les mêmes critères ? | Cadrage / disponibilité |
| 8 | La décision serait-elle la même si les chiffres changeaient de ±20 % ? | Fragilité aux hypothèses |

**Exemple neutre — comité d'évaluation de risque météo fictif** : un comité doit valider un protocole d'alerte pour événements de grêle. En appliquant la checklist, il réalise que (Q2) aucun cas de référence d'autres régions n'a été consulté, et que (Q4) le service agronomique — dont les cultures seraient affectées — n'a pas été interrogé. Deux lacunes corrigées avant validation.

> **À retenir** : une checklist ne remplace pas le jugement — elle garantit que le jugement s'exerce sur des informations complètes et équilibrées.

---

## 5. Groupthink — biais de groupe (Janis, 1982)

### 5.1 Mécanisme

Le **groupthink** (Janis, 1982) désigne la tendance d'un groupe soudé à privilégier la cohésion au détriment de l'évaluation critique. Il émerge typiquement quand :

- Le groupe est très cohésif (forte identité partagée)
- Il est sous pression de temps ou de stress
- Il manque de procédures formelles d'évaluation critique
- Le leader exprime clairement ses préférences en amont

**Symptômes principaux identifiés par Janis** :

| Symptôme | Description |
|----------|-------------|
| **Illusion d'invulnérabilité** | Optimisme excessif ; prise de risque injustifiée ("ça ne peut pas rater") |
| **Rationalisation collective** | Les signaux d'alerte sont minimisés ou réinterprétés pour maintenir la décision |
| **Pression vers la conformité** | Les membres qui doutent se taisent pour ne pas "trahir" l'équipe |
| **Autocensure** | Chaque membre évite d'exprimer ses doutes, croyant être le seul à en avoir |
| **Illusion d'unanimité** | Le silence est interprété comme un accord — ce qui renforce l'autocensure |
| **Mindguards** | Certains membres filtrent les informations qui pourraient troubler la décision |

### 5.2 Exemple neutre

**Scénario fictif — comité de validation d'un plan d'urgence météo** : un comité de 7 membres valide un protocole d'alerte pour événements de verglas. Le directeur ouvre la séance en affirmant "c'est la meilleure option possible". Deux membres ont des doutes sur la couverture des zones rurales, mais ne parlent pas — ils pensent que s'ils étaient seuls à s'inquiéter, cela signalerait un manque de compétence. Le vote est unanime. Trois semaines plus tard, un verglas sur une zone rurale non couverte entraîne des incidents évitables.

**Ce qui s'est passé** : autocensure (chacun pensait être le seul à douter), illusion d'unanimité (le silence confirmait l'accord), présence d'un mindguard implicite (le directeur avait clos le débat dès l'ouverture).

### 5.3 Garde-fous opérationnels

1. **Encourager la dissidence structurée** : instituer un rôle de red team ou d'avocat du diable *authentique* (voir §3).
2. **Leader parle en dernier** : le leader énonce ses préférences après les autres, pour ne pas ancrer le groupe.
3. **Vote anonyme** sur les décisions importantes avant la discussion.
4. **Inviter un expert extérieur** qui n'a pas d'investissement dans la cohésion du groupe.
5. **Formuler les doutes par écrit** (individuellement, avant la réunion) — brise l'autocensure.

> **À retenir** : le groupthink prospère dans le silence. La clé est de **rendre le dissensus sûr et structurel**, pas de compter sur le courage individuel.

---

## 6. Intégrer les quatre outils : quand les utiliser ?

| Outil | Quand l'utiliser | Durée typique |
|-------|-----------------|---------------|
| **Pre-mortem** | Avant tout projet ou décision irréversible importante | 20-30 min |
| **Red teaming** | Quand un plan est quasi-finalisé et qu'on veut ses angles morts | 1-2 h |
| **Checklist anti-biais** | Pour toute décision d'équipe avec enjeux significatifs | 10-15 min |
| **Vigilance groupthink** | En continu dès qu'un groupe est sous pression ou très cohésif | Permanente |

Ces outils sont **complémentaires** : le pre-mortem génère des risques, la checklist vérifie que le processus a été rigoureux, le red team attaque le plan, et la vigilance groupthink maintient le canal de parole ouvert.

---

## Flash-cards (Module 13)

**Q1 : Quel mécanisme cognitif le pre-mortem exploite-t-il pour améliorer l'identification des risques ?**
> R : La "rétrospection prospective" (*prospective hindsight*) — se souvenir d'un événement passé active plus d'explications causales qu'en imaginer un futur. En posant l'échec comme déjà advenu, on accède à un meilleur inventaire causal. Klein rapporte ~30 % d'identification de risques supplémentaires.

**Q2 : Quelle différence Nemeth et al. (2001) établissent-ils entre un avocat du diable formel et un dissensus authentique ?**
> R : L'avocat du diable joue un rôle assigné, perçu comme rhétorique — il déclenche la défense du plan. Le dissensus authentique est une opposition sincère — il force une révision réelle et génère davantage d'idées originales.

**Q3 : Citez trois symptômes du groupthink (Janis) et leur effet sur la décision.**
> R : (1) Illusion d'invulnérabilité → prise de risque injustifiée ; (2) Autocensure → les signaux faibles ne remontent pas ; (3) Illusion d'unanimité → le silence est interprété comme accord, renforçant l'autocensure.

**Q4 : Pourquoi une checklist anti-biais est-elle plus efficace que la simple connaissance des biais ?**
> R : Parce que les biais s'exercent pendant le processus de décision, avant que la réflexion consciente intervienne. La checklist externalise le contrôle et force des vérifications procédurales indépendamment de l'état cognitif du décideur.

**Q5 : Quel garde-fou simple contre le groupthink peut être mis en place sans restructurer toute la réunion ?**
> R : Faire rédiger les doutes individuellement et par écrit *avant* la discussion collective — cela brise l'autocensure sans nécessiter de rôle formel supplémentaire.

---

## Points clés à retenir

1. La connaissance des biais ne les réduit pas : il faut des procédures externalisées qui agissent sur le *processus*.
2. Le **pre-mortem** suppose l'échec déjà advenu pour libérer la parole et inventorier les risques (~+30 % d'identification).
3. Le **red teaming** fonctionne avec des opposants sincères, pas des avocats du diable formels.
4. La **checklist anti-biais** garantit que le jugement s'exerce sur des informations complètes — elle ne remplace pas le jugement.
5. Le **groupthink** prospère dans l'autocensure et l'illusion d'unanimité ; le rendre structurellement sûr de dissenter est le seul remède durable.

---

## Pour aller plus loin

- **Pre-mortem** : Klein, G. (2007). *Performing a Project Premortem.* Harvard Business Review, 85(9), 18-19. https://hbr.org/2007/09/performing-a-project-premortem
- **Checklists** : Gawande, A. (2009). *The Checklist Manifesto: How to Get Things Right.* Metropolitan Books. https://en.wikipedia.org/wiki/The_Checklist_Manifesto
- **Checklist de décision** : Kahneman, D., Lovallo, D. & Sibony, O. (2011). *Before You Make That Big Decision...* Harvard Business Review, 89(6), 50-60. https://hbr.org/2011/06/the-big-idea-before-you-make-that-big-decision
- **Groupthink** : Janis, I. L. (1982). *Groupthink: Psychological Studies of Policy Decisions and Fiascoes* (2e éd.). Houghton Mifflin. https://archive.org/details/groupthinkpsycho00jani/
- **Red teaming / dissensus** : Nemeth, C. J. et al. (2001). *Devil's advocate versus authentic dissent: stimulating quantity and quality.* European Journal of Social Psychology, 31(6). DOI 10.1002/ejsp.58. https://onlinelibrary.wiley.com/doi/abs/10.1002/ejsp.58
