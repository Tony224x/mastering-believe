# J5 — EU AI Act & gouvernance des tiers/GPAI

> **Temps estime** : 45-60 min | **Prerequis** : J4 (taxonomie des risques, NIST AI RMF)
> **Objectif** : classer un systeme IA dans l'un des 4 tiers de risque de l'EU AI Act, lister ses obligations, situer la deadline applicable, et cadrer une due diligence fournisseur minimale (modeles/agents achetes, GPAI downstream).

## Pourquoi ce module

L'EU AI Act est la **premiere loi horizontale** au monde sur l'IA : elle est **contraignante** (pas une norme volontaire) et son non-respect coute cher. Savoir ou tombe un systeme dans ses 4 tiers — et **quand** ses obligations mordent — est la base de toute conversation de conformite avec un board ou une DSI.

---

## 1. Le cas concret : trois systemes, trois mondes

Une banque deploie trois systemes IA la meme annee :

- **(A)** Un agent qui **trie le scoring de credit** des particuliers et propose une decision d'octroi.
- **(B)** Un **chatbot de support** qui repond aux questions de facturation et se presente comme un assistant virtuel.
- **(C)** Un systeme de **notation sociale** des clients qui ajuste leurs droits d'acces a des services publics selon un comportement general agrege.

Naivement, on les traiterait pareil (« ce sont tous des IA »). L'EU AI Act dit **l'inverse** : il les classe par **niveau de risque pour les droits fondamentaux et la securite**, pas par sophistication technique.

- (C) la notation sociale generalisee par une autorite est **interdite** (risque inacceptable). Pas de mise en conformite possible : on ne le deploie pas.
- (A) le scoring de credit est **haut risque** (Annexe III, acces aux services essentiels) : autorise, mais sous un paquet lourd d'obligations (gestion des risques, qualite des donnees, supervision humaine, documentation technique, journalisation).
- (B) le chatbot est **risque limite** : la seule obligation forte est la **transparence** — l'utilisateur doit savoir qu'il parle a une machine.

**Le principe abstrait** derriere ce cas : l'EU AI Act est une **regulation a etages** (`risk-based`). Plus le systeme peut nuire aux droits ou a la securite des personnes, plus la charge reglementaire est lourde — jusqu'a l'interdiction pure. C'est l'inverse d'une regle uniforme « toute IA = meme traitement » [UE / EUR-Lex, 2024].

> **Key takeaway** : l'EU AI Act ne regule pas « l'IA » en bloc — il regule des **usages** selon leur risque. Le meme modele technique peut etre minimal dans un contexte et haut risque dans un autre. On classe l'**usage**, pas la technologie.

---

## 2. Les 4 tiers de risque

L'EU AI Act range tout systeme IA dans **exactement un** de ces quatre tiers (Reglement (UE) 2024/1689) :

| Tier | Definition | Exemples (illustratifs) | Regime |
|------|-----------|--------------------------|--------|
| **Inacceptable** | Usages contraires aux valeurs de l'UE | Notation sociale par les autorites, manipulation subliminale, scraping non cible de visages pour reco faciale, certaines categories de reco biometrique | **Interdit** (Art. 5) |
| **Haut risque** | Peut nuire serieusement a la sante, la securite, les droits fondamentaux | Composant de securite d'un produit (Annexe I) ; ou usage sensible liste a l'**Annexe III** (emploi, credit, education, justice, infrastructures critiques, migration...) | **Autorise sous obligations lourdes** (Art. 8-15, etc.) |
| **Risque limite** | Risque de manipulation/confusion homme-machine | Chatbots, systemes generant du contenu, deepfakes | **Transparence** (Art. 50) : signaler que c'est une IA / un contenu genere |
| **Risque minimal** | Tout le reste | Filtres anti-spam, IA de jeux video, recommandation produit basique | **Aucune obligation specifique** (codes de conduite volontaires) |

Deux subtilites importantes :

- Le tier **haut risque** a deux portes d'entree distinctes : **Annexe I** (l'IA est un composant de securite d'un produit deja regule — jouets, machines, dispositifs medicaux...) et **Annexe III** (l'IA est utilisee dans un domaine sensible explicitement liste). Elles n'ont pas la **meme deadline** (voir section 4).
- La majorite des systemes du marche sont **minimaux** — l'EU AI Act ne transforme pas tout en paperasse. Le piege inverse existe aussi : sous-classer un systeme reellement haut risque (« ce n'est qu'un assistant »).

> **Key takeaway** : 4 tiers, un seul par systeme. Inacceptable = on ne deploie pas. Haut risque (Annexe I **ou** III) = autorise mais obligations lourdes. Limite = transparence. Minimal = rien d'impose. La question de gouvernance n°1 : « dans quel tier tombe cet usage, et pourquoi ? »

---

## 3. Le cas a part : les modeles a usage general (GPAI)

Les 4 tiers classent des **systemes a finalite definie**. Mais un **modele de fondation** (GPT-x, Llama, Mistral...) n'a pas de finalite unique : il peut servir a tout. L'EU AI Act lui ajoute donc un **regime parallele** : les obligations **GPAI** (`General-Purpose AI`).

Deux niveaux :

1. **Tout GPAI** : documentation technique du modele, information aux fournisseurs en aval (downstream), politique de respect du droit d'auteur, resume public des donnees d'entrainement.
2. **GPAI a risque systemique** (les plus capables, au-dessus d'un seuil de calcul) : en plus, evaluation du modele, evaluation et attenuation des risques systemiques, signalement d'incidents graves, cybersecurite renforcee.

**Pourquoi ca compte pour l'agentique.** Un agent que vous construisez s'appuie presque toujours sur un GPAI **achete a un tiers**. Vous etes alors un **deployeur/fournisseur en aval**. La chaine de responsabilite se dedouble :

- Le **fournisseur du GPAI** porte les obligations GPAI sur le modele lui-meme.
- **Vous**, qui l'integrez dans un systeme a finalite precise, portez les obligations du **tier** de votre usage (limite, haut risque...). Si vous adaptez substantiellement le modele ou le mettez sur le marche sous votre nom, vous pouvez **herite** d'obligations de fournisseur.

> **Key takeaway** : les obligations GPAI sont un regime **distinct** des 4 tiers, qui pese sur les modeles de fondation. Construire un agent sur un modele tiers ne vous exonere pas : vous heritez des obligations du tier de **votre** usage, et vous dependez de la conformite GPAI **de votre fournisseur**.

---

## 4. Le calendrier : ce qui mord, et quand

L'EU AI Act est entre **en vigueur le 1er aout 2024**, mais ses obligations s'appliquent **par vagues** (Art. 113). Confondre « en vigueur » et « applicable » est l'erreur de date la plus courante.

| Date | Ce qui devient applicable |
|------|----------------------------|
| **2 fevrier 2025** | Interdictions (pratiques a risque inacceptable, Art. 5) + obligations de **litteratie IA** |
| **2 aout 2025** | Obligations **GPAI** (modeles a usage general) + gouvernance |
| **2 aout 2026** | Application **generale** + haut risque **Annexe III** (emploi, credit, education...) |
| **2 aout 2027** | Haut risque **Annexe I** (IA composant de securite d'un produit deja regule) |

Lecture pratique a la date d'aujourd'hui (mi-2026) : les **interdictions** et les **obligations GPAI** sont **deja applicables** ; le gros bloc **haut risque Annexe III** mord le **2 aout 2026** — soit la deadline a piloter en priorite pour un scoring de credit ou un tri de CV ; le **haut risque Annexe I** beneficie d'un an de plus (2 aout 2027) [Commission europeenne / AI Act Explorer, 2024].

> **Key takeaway** : « en vigueur 1er aout 2024 » ≠ « applicable ». Calendrier a memoriser : **2 fev. 2025** interdictions ; **2 aout 2025** GPAI ; **2 aout 2026** haut risque Annexe III ; **2 aout 2027** haut risque Annexe I. La deadline qui vous concerne depend de **votre tier et de votre annexe**.

---

## 5. Gouvernance des tiers : la due diligence fournisseur

Vous ne construisez quasi jamais tout vous-meme : vous **achetez** un modele GPAI, un agent SaaS, une brique de vision. La conformite ne s'arrete pas a votre code — elle **traverse la chaine d'approvisionnement**. C'est la **gouvernance des tiers** (`third-party / supplier governance`).

Une due diligence fournisseur minimale, pour un composant IA achete, repond a :

1. **Quel tier ?** L'usage que **vous** en faites tombe dans quel tier de l'EU AI Act ? (c'est **votre** classification, pas celle du vendeur.)
2. **GPAI ?** Le composant est-il (ou repose-t-il sur) un GPAI ? Le fournisseur publie-t-il la documentation technique, le resume des donnees d'entrainement, la politique copyright ?
3. **Documentation transmissible** : le fournisseur fournit-il ce qu'il faut pour **vos** obligations downstream (instructions d'usage, limites connues, logs/traçabilite) ?
4. **Responsabilites contractualisees** : qui porte quoi en cas d'incident ? Le contrat trace-t-il la frontiere fournisseur/deployeur ?
5. **Deadline** : a quelle date vos obligations (selon votre tier) deviennent-elles exigibles ? Le fournisseur sera-t-il conforme a temps ?

Sans inventaire des composants tiers et de leur tier, vous ne pouvez ni prouver votre conformite ni savoir quelle deadline vous menace. La gouvernance des tiers est le prolongement naturel du **registry d'agents** (J3) : chaque agent achete est une ligne du registry avec un fournisseur, un tier EU AI Act, et une deadline.

> **Key takeaway** : la conformite EU AI Act traverse la supply chain. Pour chaque brique IA **achetee**, classez **votre** usage (tier), verifiez le statut GPAI et la doc transmissible du fournisseur, contractualisez les responsabilites, et notez la deadline. C'est la due diligence minimale — et elle s'ancre dans le registry.

---

## 6. Ou se situent les systemes agentiques ?

L'EU AI Act n'a pas de catered « tier agent » : un agent est classe par **l'usage** qu'il sert, comme tout systeme. Mais l'agentique deplace souvent le curseur **vers le haut** :

- Un agent qui **agit** (execute une transaction, modifie un dossier, declenche un workflow) touche plus directement les droits/securite qu'un systeme qui se contente de **suggerer**. Le meme cas d'usage peut basculer de « limite » a « haut risque » selon que l'humain garde ou non la main sur la decision finale.
- L'obligation de **supervision humaine** (Art. 14, pour le haut risque) devient centrale : un agent out-of-the-loop sur un usage Annexe III est difficile a justifier.
- La **transparence** (Art. 50, risque limite) s'applique des qu'un agent conversationnel interagit avec une personne : elle doit savoir que c'est une machine.

En pratique : on classe l'usage, puis on regarde le **niveau d'autonomie** (cf. J10) comme un facteur qui peut faire monter le tier ou alourdir les obligations de supervision.

> **Key takeaway** : pas de tier special « agent ». On classe l'usage — mais le fait qu'un agent **agisse** (et non suggere) pousse souvent vers le haut risque et rend la supervision humaine (Art. 14) incontournable.

---

## Spaced repetition

1. **Q :** Le meme modele de classification d'images est utilise (a) pour recommander des stickers dans une appli de chat et (b) pour trier des candidatures a l'embauche. Tombent-ils dans le meme tier EU AI Act ?
   **R :** Non. On classe l'**usage**, pas la technologie. (a) est minimal (aucune obligation) ; (b) est **haut risque** (Annexe III, emploi). Le modele est identique, le tier non.

2. **Q :** Une entreprise dit « l'EU AI Act est entre en vigueur le 1er aout 2024, donc nos obligations haut risque Annexe III s'appliquent depuis cette date ». Ou est l'erreur ?
   **R :** « En vigueur » ≠ « applicable ». Les obligations haut risque Annexe III ne mordent que le **2 aout 2026**. Entree en vigueur et application sont decalees (Art. 113).

3. **Q :** Vous construisez un agent sur un modele de fondation tiers (GPAI). Le fournisseur gere les obligations GPAI. Etes-vous exonere de toute obligation EU AI Act ?
   **R :** Non. Les obligations GPAI portent sur le **modele** (cote fournisseur). Vous, qui l'integrez dans un usage a finalite definie, portez les obligations du **tier de votre usage** (transparence, voire haut risque). Les deux regimes se cumulent.

4. **Q :** Citez les 4 dates-cles du calendrier d'application et ce qui devient exigible a chacune.
   **R :** 2 fev. 2025 = interdictions + litteratie IA ; 2 aout 2025 = GPAI ; 2 aout 2026 = application generale + haut risque Annexe III ; 2 aout 2027 = haut risque Annexe I.

5. **Q :** Quelles sont les 2 portes d'entree du tier « haut risque », et pourquoi la distinction compte-t-elle concretement ?
   **R :** **Annexe I** (IA = composant de securite d'un produit deja regule) et **Annexe III** (usage sensible explicitement liste). La distinction compte car elles n'ont **pas la meme deadline** : Annexe III = 2 aout 2026, Annexe I = 2 aout 2027.
