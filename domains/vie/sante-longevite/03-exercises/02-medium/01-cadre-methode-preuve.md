# Exercices (medium) — Module 01 : Cadre, methode & niveaux de preuve

> **Prerequis** : avoir lu `01-theory/01-cadre-methode-preuve.md` et complete les exercices `01-easy/01-cadre-methode-preuve.md`
>
> **⚠️ Disclaimer medical.** Ces exercices sont a visee educative uniquement. Ils ne constituent pas un bilan de sante ni un avis medical. Ils visent a entrainer la lecture critique des preuves, pas a guider des decisions de sante personnelles.

---

## Exercice 1 — Convertir un risque relatif en risque absolu

### Objectif

Aller au-dela de l'identification du type d'etude (vu en easy) : manipuler les chiffres pour distinguer un effet impressionnant en relatif d'un effet reel en absolu — la competence qui desamorce la plupart des titres alarmistes.

### Consigne

Voici 3 enonces (fictifs mais realistes). Pour chacun, vous disposez du **risque de base** (groupe non expose) et du **risque relatif** (RR) ou de la reduction relative du risque (RRR).

**Cas A — Depistage**
"Un nouveau biomarqueur reduit de 50 % le risque de developper la maladie X." Risque de base sur 10 ans : 2 personnes sur 1000.

**Cas B — Comportement (proche du DPP)**
"L'intervention mode de vie reduit l'incidence du diabete de 58 %." Risque de base sur 3 ans dans une population prediabetique : 290 personnes sur 1000.

**Cas C — Aliment**
"Consommer cet aliment double le risque de la maladie Y (RR = 2,0)." Risque de base sur 10 ans : 1 personne sur 100 000.

Pour chaque cas, calculez et indiquez :
1. Le **risque absolu** dans le groupe expose (ou intervention).
2. La **reduction (ou augmentation) absolue du risque** (RAR ou difference de risque).
3. Le **NNT** (nombre de personnes a traiter pour eviter 1 cas) pour A et B, ou le **NNH** (nombre a exposer pour 1 cas supplementaire) pour C.
4. Une phrase de conclusion : ce resultat est-il cliniquement important pour une personne moyenne ? Justifiez par les chiffres, pas par l'intuition.

> Rappel : NNT = 1 / RAR (en proportion). Exemple : RAR de 0,05 (5 %) → NNT = 20.

### Criteres de reussite

- [ ] Les 3 risques absolus du groupe expose sont calcules correctement
- [ ] Les 3 differences absolues de risque sont calculees
- [ ] Le NNT (A et B) et le NNH (C) sont calcules avec la bonne formule
- [ ] Pour chaque cas, la conclusion sur l'importance clinique s'appuie sur le chiffre absolu et le risque de base, pas sur le RR seul
- [ ] Vous avez identifie que le cas C (RR = 2,0 sur une base infime) a un impact absolu negligeable malgre un RR "spectaculaire"

---

## Exercice 2 — Trier un dossier de preuves contradictoires

### Objectif

Quand plusieurs etudes se contredisent, hierarchiser au lieu de "moyenner les opinions". On entraine ici l'application de la hierarchie des preuves a un cas conflictuel realiste.

### Consigne

Sur un meme sujet (effet d'un complement Z sur la fatigue), vous trouvez 4 sources. Vous devez produire une **synthese calibree**, pas un verdict tranche.

| # | Source | Design | Resultat rapporte |
|---|--------|--------|-------------------|
| S1 | Article de blog d'un influenceur sante | Temoignages | "Z a transforme mon energie" |
| S2 | Etude transversale, 4000 personnes | Observationnelle, 1 mesure | Les utilisateurs de Z se disent moins fatigues (association) |
| S3 | ECR, 60 sujets, 8 semaines, double aveugle vs placebo | ECR de petite taille | Pas de difference significative (IC large incluant 0) |
| S4 | Meta-analyse de 9 ECR, 1200 sujets | Niveau 1 | Effet nul a tres faible sur la fatigue, IC etroit autour de 0 |

Repondez :
1. Classez les 4 sources de la plus forte a la plus faible en niveau de preuve, avec une justification d'une ligne par source.
2. S3 (un ECR) ne trouve "pas de difference". Cela **prouve-t-il** que Z ne marche pas ? Expliquez la difference entre "absence de preuve d'effet" et "preuve d'absence d'effet", et pourquoi la taille d'echantillon de S3 compte.
3. S4 et S3 vont dans le meme sens, mais leur poids n'est pas le meme. Pourquoi la meta-analyse S4 est-elle plus concluante que l'ECR isole S3 ?
4. Redigez la synthese en 2-3 phrases, telle que vous l'ecririez pour un proche, avec le bon niveau de certitude.

### Criteres de reussite

- [ ] Le classement respecte la hierarchie : meta-analyse d'ECR > ECR > observationnel > temoignages
- [ ] La distinction "absence de preuve d'effet" vs "preuve d'absence d'effet" est correctement expliquee (puissance statistique)
- [ ] Le role de la taille d'echantillon (S3 petit, IC large) est identifie comme limite
- [ ] La synthese finale est calibree : elle conclut a un effet probablement nul/negligeable sans affirmer une certitude absolue, et reste honnete sur le niveau de preuve

---

## Exercice 3 — Auditer un article de presse reel avec la grille des 5 questions

### Objectif

Appliquer la grille des 5 questions du module 01 (source, niveau de preuve, taille d'effet, population incluse, beneficiaires/contre-indications) a un article reel de votre choix.

### Consigne

Choisissez **un article de presse grand public** sur la sante, paru dans les 12 derniers mois (alimentation, sommeil, exercice, complement, etc.). Idealement, retrouvez l'etude citee (titre, revue, annee — souvent en bas de l'article ou via une recherche rapide).

Remplissez la grille suivante :

| Question | Votre reponse |
|----------|---------------|
| 1. **Source** : organisme officiel, ECR, observationnel, expert seul ? | |
| 2. **Niveau de preuve** (1-5) | |
| 3. **Taille d'effet** : RR/RAR/SMD ? L'article la donne-t-il en absolu ? | |
| 4. **Population incluse** : qui ? (age, sexe, etat de sante, taille d'echantillon) | |
| 5. **Beneficiaires & contre-indications** : a qui le resultat s'applique-t-il vraiment ? | |

Puis :
- Identifiez **l'ecart le plus important** entre ce que dit le titre et ce que montre l'etude.
- Reformulez le titre de facon honnete et calibree.

### Criteres de reussite

- [ ] Un article reel et identifiable est choisi (lien ou reference notes dans le workspace)
- [ ] Les 5 questions de la grille sont renseignees (avec "non disponible" si l'info manque — c'est une donnee en soi)
- [ ] L'ecart titre/etude est identifie precisement (causalite surestimee, population non generalisable, effet absolu minuscule, etc.)
- [ ] Le titre reformule est factuel, mentionne le type d'etude et ne pretend pas a une causalite non demontree

> **Note** : Cet exercice se fait dans `03-exercises/workspace/` — il n'a pas vocation a etre commite.
