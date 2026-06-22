# Solutions (hard) — Module 01 : Cadre, methode & niveaux de preuve

> Corriges modeles. La qualite du raisonnement (calibration, asymetrie, distinction des niveaux de preuve) prime sur la formulation exacte.
>
> **⚠️ Disclaimer medical.** Contenu educatif uniquement. Ne remplace pas un avis medical. Toute decision impliquant un complement, une molecule ou un changement majeur de mode de vie se discute avec un professionnel de sante.

---

## Exercice 1 — Analyse critique : la lecon PREDIMED

### Corrige modele

**1. Mecanisme du biais (randomisation par menage)**
La randomisation est censee garantir que les groupes compares sont **equilibres en moyenne sur tous les facteurs**, connus et inconnus (age, tabac, genetique, alimentation de fond...). C'est ce qui permet d'attribuer une difference de resultats a l'intervention plutot qu'a des confondants.
Randomiser "par menage" (cluster) plutot que par individu casse l'**independance** des observations : les membres d'un meme foyer partagent alimentation, habitudes, environnement. Si des menages entiers sont alloues au meme bras, on peut introduire un **desequilibre systematique** entre groupes (un foyer = un bloc de confondants correles), et la taille effective d'echantillon est plus petite qu'annoncee. La promesse de comparabilite de la randomisation est donc partiellement rompue.

**2. Lecture de la correction**
La bonne reponse est **partiellement rassurant**, et il faut distinguer deux choses :
- *Robustesse de l'estimation* : que l'effet corrige reste "du meme ordre" (~30 %) est rassurant sur la **stabilite numerique** du resultat — il ne s'est pas effondre apres correction.
- *Force de la preuve causale* : elle est, elle, **affaiblie** par le defaut de randomisation initial. Une estimation stable obtenue avec une methode imparfaite reste plus fragile qu'une estimation issue d'une randomisation impeccable. On accorde donc une confiance reelle mais **revue a la baisse** par rapport a un ECR sans defaut. C'est pourquoi le module presente PREDIMED "avec ces reserves".

**3. Refutation du scepticisme generalise**
"Le NEJM se trompe donc tout se vaut" est un sophisme. La retraction puis republication montre exactement l'inverse : la science a des **mecanismes d'auto-correction** (relecture, signalement, re-analyse) qui ont detecte et corrige le defaut. Une croyance non scientifique, elle, n'a aucun mecanisme pour se corriger. La fiabilite ne vient pas de l'infaillibilite d'un article isole mais du **processus** qui revise les conclusions a la lumiere de nouvelles informations. Conclure "je crois ce que je veux" abandonne precisement la seule chose qui distingue le savoir de l'opinion.

**4. Trois reflexes transferables (exemples)**
- Lire systematiquement la **section limites/methodes** avant les conclusions.
- Verifier si l'article a fait l'objet d'une **correction, d'un erratum ou d'une retraction** (PubPeer, Retraction Watch, note de la revue).
- Regarder l'**intervalle de confiance** et la **population incluse**, pas seulement le pourcentage de reduction et le titre.

**Enseignement cle** : une preuve "forte" peut etre faillible ; la maturite epistemique consiste a ajuster sa confiance (ni 0 ni 100 %) en fonction de la methode et des corrections, pas a basculer dans le tout-ou-rien.

---

## Exercice 2 — Arbitrer une decision sous incertitude

### Corrige modele

Principe directeur : **le seuil de preuve exige doit augmenter avec le cout, le risque et l'irreversibilite de l'action.** Une action quasi gratuite et reversible peut etre adoptee sur preuve moderee ; une action couteuse/risquee exige une preuve forte.

**Proposition 1 — Marche apres dejeuner → (a) adopter maintenant**
- Preuve : direction robuste (OMS, mecanismes metaboliques). Effet plausible : modeste mais reel.
- Cout/risque : quasi nuls. Reversibilite : totale.
- Asymetrie : agir a tort coute presque rien (un peu de temps) ; ne pas agir prive d'un petit gain a cout nul. L'asymetrie penche clairement vers l'action.
- Info pivot : quasi aucune ne justifierait d'arreter ; une contre-indication medicale personnelle (rare) serait la seule.

**Proposition 2 — Complement NMN → (c) attendre / (d) ne pas adopter hors recherche**
- Preuve : donnees humaines limitees, courte duree, pas de demonstration sur la longevite humaine (objet de recherche, cf. module 07).
- Cout/risque : cout financier reel, securite long terme inconnue. Reversibilite : arret possible, mais effets long terme inconnus = incertitude residuelle.
- Asymetrie : agir a tort = depenser pour un effet non demontre + exposition a un risque mal caracterise ; ne pas agir = renoncer a un benefice hypothetique non prouve. L'asymetrie penche vers l'abstention.
- Info pivot : un ou plusieurs ECR longue duree chez l'humain montrant un benefice net **et** un profil de securite rassurant feraient basculer vers (b). En attendant, ne pas prendre sans supervision medicale.

**Proposition 3 — Jeune intermittent → (b) adopter prudemment, petit pas reversible (avec reserves)**
- Preuve : peut fonctionner comme outil de restriction calorique pour certains ; preuves longue duree limitees ; contre-indications existantes (cf. module 04).
- Cout/risque : cout faible ; risque variable selon le profil (grossesse, antecedents de troubles alimentaires, diabete traite, etc.).
- Reversibilite : totale.
- Asymetrie : pour une personne sans contre-indication, essayer a tort coute peu et est reversible ; mais pour un profil a risque, agir a tort peut etre nocif → d'ou la posture "prudente, apres verification des contre-indications", pas "adopter pour tous".
- Info pivot : la presence de contre-indications personnelles ferait basculer vers (d) ; un retour positif sur quelques semaines sans effet indesirable conforterait (b).

**Enseignement cle** : decider sous incertitude n'est pas "attendre la certitude" (elle n'arrive jamais) ni "tout essayer". C'est calibrer le seuil de preuve sur les enjeux : faible cout + reversible → on peut agir tot ; cout/risque eleve ou irreversible → on exige une preuve forte. Et pour toute molecule, la regle du domaine reste : objet de recherche, decision avec un medecin.
