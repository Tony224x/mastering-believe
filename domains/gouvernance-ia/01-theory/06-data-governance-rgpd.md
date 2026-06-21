# Data governance & RGPD pour l'IA agentique

## Pourquoi ce module

Un agent qui lit des emails clients, mémorise des préférences et déclenche des remboursements *traite des données personnelles* — souvent à l'insu de l'équipe juridique. Ce module apprend à détecter ce traitement, à le rattacher à une base légale, et à décider si une analyse d'impact (DPIA) s'impose.

---

## 1. L'exemple qui réveille : un agent support « innocent »

Une équipe déploie un agent de support. Pour « personnaliser » ses réponses, il garde en mémoire long-terme : nom du client, historique de tickets, ton préféré, et — par capture accidentelle des logs — des numéros de carte tronqués. Personne n'a rempli de registre de traitement. Personne ne sait que cet agent est, au sens du RGPD, un **traitement de données personnelles** soumis à toutes ses obligations.

Le réflexe naïf : « ce n'est qu'un cache technique ». Le réflexe gouvernance : poser quatre questions.

1. **Y a-t-il des données personnelles ?** (toute info se rapportant à une personne identifiée ou identifiable — Art. 4(1) RGPD). Oui : nom, historique.
2. **Quelle est la finalité ?** Personnaliser le support. Une finalité doit être *déterminée, explicite et légitime* (Art. 5(1)(b)).
3. **Quelle base légale ?** (Art. 6). Sans l'une des six bases, le traitement est **illicite**, point.
4. **Collecte-t-on plus que nécessaire ?** Les cartes tronquées ne servent pas la finalité → violation de **minimisation** (Art. 5(1)(c)).

Le principe abstrait derrière l'exemple : **la data governance d'un agent n'est pas un add-on, c'est une condition de licéité.** Un agent sans base légale identifiée est un agent qu'on doit éteindre, pas documenter après coup. Comme le rappelle la CNIL, déterminer la base légale est « un préalable à la mise en œuvre du traitement », pas une régularisation a posteriori [CNIL, 2024-2025].

> **Key takeaway** — Dès qu'un agent voit, stocke ou transmet une donnée se rapportant à une personne, le RGPD s'applique intégralement : finalité, base légale et minimisation sont des questions *avant* déploiement, pas après incident.

---

## 2. Les 6 bases légales (Art. 6) appliquées aux agents

Le RGPD n'autorise un traitement que s'il repose sur **exactement une** des six bases de l'Art. 6(1) :

| Base légale | Quand l'invoquer pour un agent | Piège agentique |
|-------------|--------------------------------|-----------------|
| **Consentement** (a) | L'utilisateur opte explicitement (ex. agent qui mémorise des préférences perso) | Le consentement doit être libre, éclairé, *révocable* — un agent doit savoir « oublier » |
| **Contrat** (b) | Traitement nécessaire à exécuter un contrat (ex. agent qui traite une commande) | « Nécessaire » : pas tout ce qui est *pratique*, seulement l'indispensable |
| **Obligation légale** (c) | La loi impose le traitement (ex. agent de conformité KYC) | Identifier la norme exacte qui l'impose |
| **Intérêts vitaux** (d) | Vie/santé d'une personne en jeu | Rare hors santé/sécurité |
| **Mission d'intérêt public** (e) | Acteur public / mission déléguée | Réservé au secteur public ou délégataire |
| **Intérêt légitime** (f) | Intérêt de l'organisme, équilibré contre les droits de la personne | Exige un **test de mise en balance** documenté ; ne couvre PAS tout |

L'intérêt légitime est la base la plus convoitée pour l'IA — et la plus risquée. L'EDPB précise qu'il faut un test en trois étapes (intérêt légitime réel, nécessité du traitement, mise en balance avec les droits des personnes) et que ce test doit tenir compte des attentes raisonnables des personnes concernées [EDPB, Opinion 28/2024, 2024]. Un agent qui scrape des données publiques « parce qu'elles sont en ligne » n'a pas, par ce seul fait, d'intérêt légitime valable.

Côté gouvernance agentique, la base légale doit être **attachée à la finalité de l'agent, pas à l'agent en bloc** : un même agent peut traiter des commandes (contrat) *et* envoyer du marketing (consentement) — deux finalités, deux bases.

> **Key takeaway** — Tout traitement repose sur une et une seule base légale par finalité. L'intérêt légitime n'est pas une base « par défaut » : il exige un test de mise en balance documenté tenant compte des attentes des personnes.

---

## 3. Données d'entraînement vs données d'inférence vs mémoire d'agent

Un système agentique manipule des données personnelles à **trois moments distincts**, chacun avec son régime :

1. **Entraînement / fine-tuning** — données utilisées pour façonner le modèle. La base légale du *training* est une question à part entière (souvent intérêt légitime, souvent contesté). L'EDPB note qu'un modèle entraîné sur des données personnelles **traitées illicitement** peut voir son déploiement ultérieur affecté — l'illicéité « remonte » la chaîne [EDPB, Opinion 28/2024, 2024].
2. **Inférence** — données soumises au modèle en production (le prompt, le contexte client). Régime classique : finalité + base légale + minimisation à l'instant de l'appel.
3. **Mémoire d'agent** — l'élément *nouveau* du paradigme agentique. Un agent qui persiste un état entre sessions crée un **stockage de données personnelles** : il faut une durée de conservation (Art. 5(1)(e) — limitation de la conservation), un mécanisme d'effacement, et le rattacher au registre.

L'erreur classique : traiter la mémoire d'agent comme un « buffer technique » exempté. Elle ne l'est pas. Dès qu'elle contient une donnée se rapportant à une personne et qu'elle survit à la requête, c'est un traitement à part entière avec rétention bornée.

L'EDPB ouvre aussi la question de l'**anonymat des modèles** : un modèle peut, sous conditions strictes, être considéré comme anonyme (ne contenant plus de données personnelles), mais cela doit être démontré au cas par cas, pas présumé [EDPB, Opinion 28/2024, 2024].

> **Key takeaway** — Distinguez trois traitements : entraînement, inférence, mémoire. La mémoire d'agent persistante est un stockage de données personnelles à part entière — elle exige une durée de conservation bornée et une voie d'effacement.

---

## 4. Quand déclencher une DPIA / AIPD (Art. 35)

Une **DPIA** (Data Protection Impact Assessment ; en français AIPD, *analyse d'impact relative à la protection des données*) est obligatoire quand un traitement est « susceptible d'engendrer un risque élevé pour les droits et libertés des personnes » (Art. 35(1) RGPD).

L'Art. 35(3) liste trois cas où elle est **explicitement requise** :
- (a) **évaluation systématique et approfondie** d'aspects personnels fondée sur un traitement automatisé (profilage) produisant des effets juridiques ou similaires ;
- (b) traitement **à grande échelle** de catégories particulières (Art. 9 : santé, opinions, biométrie…) ou de données pénales ;
- (c) **surveillance systématique à grande échelle** d'une zone accessible au public.

La CNIL publie en complément une liste de critères ; **dès que deux critères** sont réunis (parmi : profilage, décision automatisée, données sensibles, grande échelle, croisement de jeux, personnes vulnérables, usage innovant, etc.), une DPIA est en pratique attendue [CNIL, 2024-2025].

Un agent autonome coche vite plusieurs cases : « usage innovant » (techno émergente), souvent « décision automatisée », parfois « grande échelle ». **La DPIA est donc la règle plutôt que l'exception pour les systèmes agentiques à impact.** Elle doit être menée *avant* la mise en œuvre et décrire : le traitement et ses finalités, la nécessité/proportionnalité, les risques pour les personnes, et les mesures de mitigation.

> **Key takeaway** — La DPIA est obligatoire en cas de risque élevé (Art. 35) ; en pratique, le critère CNIL « deux critères sur la liste » suffit. Un agent autonome (innovant, souvent décisionnel) tombe presque toujours dedans — menez-la avant déploiement.

---

## 5. Deux régimes cumulatifs : RGPD ↔ EU AI Act

Erreur fréquente en réunion : « on est conforme AI Act, donc le RGPD est couvert ». Faux. Ce sont **deux régimes distincts et cumulatifs** :

| Axe | RGPD (2016/679) | EU AI Act (2024/1689) |
|-----|-----------------|------------------------|
| Protège | les **données personnelles** des personnes | la **sécurité et les droits fondamentaux** face aux systèmes d'IA |
| Déclencheur | présence de données personnelles | présence d'un *système d'IA*, classé par tier de risque |
| Autorité (FR) | CNIL | autorité(s) de surveillance du marché désignée(s) |
| Peut s'appliquer seul ? | Oui (agent sans IA « à risque » mais avec data perso) | Oui (système IA sans data perso) |

Les deux se cumulent : un agent haut-risque traitant des données personnelles doit satisfaire **les deux** corpus simultanément. L'AI Act le reconnaît explicitement — il s'applique « sans préjudice » du RGPD, et certaines obligations (ex. la DPIA RGPD) coexistent avec les obligations propres à l'AI Act (gestion des risques, documentation technique). En gouvernance, on tient donc **deux colonnes de conformité**, pas une.

Point d'articulation utile : une DPIA RGPD bien faite alimente une partie de l'évaluation de risque AI Act (et inversement), mais elles ne se substituent pas l'une à l'autre.

> **Key takeaway** — RGPD et AI Act sont cumulatifs, pas alternatifs. « Conforme AI Act » ne dit rien sur la licéité du traitement de données. Tenir deux colonnes de conformité ; la DPIA nourrit l'analyse de risque mais ne la remplace pas.

---

## 6. Droits des personnes : ce qu'un agent doit pouvoir faire

Le RGPD donne aux personnes des droits *opposables* (Art. 12 à 22). Un agent gouvernable doit être conçu pour les honorer — pas les contourner :

- **Accès** (Art. 15) : restituer les données détenues sur la personne (y compris en mémoire d'agent).
- **Rectification** (Art. 16) : corriger une donnée fausse mémorisée.
- **Effacement / « droit à l'oubli »** (Art. 17) : supprimer sur demande, et propager l'effacement à la mémoire d'agent et aux logs dérivés.
- **Limitation** (Art. 18) et **opposition** (Art. 21), notamment si la base est l'intérêt légitime.
- **Décision automatisée** (Art. 22) : droit de ne pas faire l'objet d'une décision *exclusivement* automatisée à effet juridique/significatif, sauf exceptions — avec, dans ce cas, droit à une **intervention humaine**.

L'Art. 22 est central pour les agents : un agent qui décide seul d'un refus de crédit, d'un licenciement ou d'une résiliation déclenche le droit à intervention humaine. C'est le pont direct avec le **human-in-the-loop** vu côté autonomie : la gouvernance des données *impose* parfois un humain dans la boucle.

Concrètement, l'effacement doit être **propageable** : effacer dans la base mais laisser la donnée dans la mémoire vectorielle de l'agent ou dans un log, c'est ne pas effacer. Un agent gouverné expose une opération « forget(subject) » qui balaie tous ses stores.

> **Key takeaway** — Un agent doit être *conçu* pour servir les droits (accès, effacement propageable, opposition). L'Art. 22 (décision automatisée) peut imposer un human-in-the-loop : la protection des données et l'autonomie se rejoignent ici.

---

## Spaced repetition

1. **Q :** Un agent garde en mémoire long-terme le nom et l'historique d'un client « juste pour un cache technique ». Est-ce un traitement RGPD ?
   **R :** Oui. Toute donnée se rapportant à une personne identifiable (Art. 4(1)) persistée au-delà de la requête est un traitement à part entière : il faut finalité, base légale, et durée de conservation bornée. Le label « cache technique » n'exempte de rien.

2. **Q :** Pourquoi l'intérêt légitime (Art. 6(1)(f)) n'est-il pas une base « par défaut » ?
   **R :** Parce qu'il exige un test de mise en balance documenté en trois étapes (intérêt réel, nécessité, équilibre avec les droits des personnes), tenant compte des attentes raisonnables des personnes [EDPB, Opinion 28/2024]. « C'est en ligne / c'est pratique » ne suffit pas.

3. **Q :** Quels sont les trois moments distincts où un système agentique traite des données personnelles ?
   **R :** Entraînement (base légale du training, l'illicéité peut « remonter »), inférence (le prompt/contexte en production), et mémoire d'agent (stockage persistant → rétention bornée + effacement). La mémoire n'est pas un buffer exempté.

4. **Q :** « On est conforme AI Act, donc on est OK côté données. » Pourquoi est-ce faux ?
   **R :** RGPD et AI Act sont deux régimes cumulatifs. L'AI Act vise les systèmes d'IA (tiers de risque, sécurité, droits fondamentaux) ; le RGPD vise la licéité du traitement de données personnelles. Un agent peut être conforme à l'un et illicite au regard de l'autre. On tient deux colonnes.

5. **Q :** À quel article du RGPD un agent qui décide seul d'un refus de crédit doit-il faire attention, et quelle conséquence en gouvernance ?
   **R :** Art. 22 (décision exclusivement automatisée à effet significatif) : la personne a droit à une intervention humaine. Conséquence : la protection des données peut *imposer* un human-in-the-loop, reliant data governance et niveau d'autonomie de l'agent.
