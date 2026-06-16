# Sante & Longevite — Cursus academiques, litterature primaire & lentille critique

> **Approche complementaire** au fichier `sante-longevite-references.md` (qui couvre les "sources faisant autorite").
> Cet angle-ci ajoute : (1) **cursus academiques** (comment des universites enseignent le sujet),
> (2) **litterature primaire** (ECR & meta-analyses avec tailles d'effet), (3) **lentille critique**
> (limites de l'epidemiologie nutritionnelle, controverses sur le sommeil, hype longevite).
> Toutes les sources ci-dessous ont ete **verifiees via recherche web** (titre, auteur/organisme, annee, URL).

> **⚠️ Disclaimer medical (fort).** Contenu **strictement educatif**. Ne constitue **pas** un avis medical
> et **ne remplace pas** la consultation d'un professionnel de sante. Les molecules dites "de longevite"
> (metformine, rapamycine) sont presentees comme **objets de recherche**, jamais comme recommandations.
> La force de la preuve est signalee explicitement a chaque source ; l'incertitude est nommee quand elle existe.

---

## 1. Cursus academiques trouves

> Note : `shared/external-courses.md` ne contient que des cours d'IA/ML — **aucun reuse** pour ce domaine.
> Les cursus ci-dessous sont specifiques sante/nutrition/longevite.

| Cours / Ressource | Institution | URL | Structure & usage |
|---|---|---|---|
| **Stanford Introduction to Food and Health** | Stanford (School of Medicine, Dr Maya Adam) | https://www.coursera.org/learn/food-and-health (aussi edX & [Stanford Online](https://online.stanford.edu/courses/som-ycme0004-introduction-food-and-health)) | MOOC gratuit, 5 modules. Cadre les epidemies diet-related (obesite, DT2), cuisine maison vs ultra-transforme, competences pratiques. 4.7/5 (~34k avis). Bon module d'ouverture "nutrition appliquee". |
| **The Nutrition Source** | Harvard T.H. Chan School of Public Health (Dept. Nutrition, dir. W. Willett) | https://nutritionsource.hsph.harvard.edu/ | Encyclopedie en ligne (non un cours), maj. continue. "Healthy Eating Plate", revues thematiques sourcees. Reference de synthese nutritionnelle independante de l'industrie. (Deja en tier-1 dans le fichier autorite — sert ici de squelette pedagogique.) |
| **Examine.com — Database & Guides** | Examine (independant, dir. K. Patel, 30+ chercheurs) | https://examine.com/database/ | 800+ supplements/interventions notes A→F selon ECR & meta-analyses. Zero lien financier industrie. Outil de reference pour le module "complements & esprit critique". |

> Pistes a verifier au moment de la construction (non encore confirmees par recherche dans ce dossier) :
> Coursera "Nutrition and Lifestyle in Pregnancy", edX "Science of Exercise" (Univ. Colorado),
> Stanford "Stanford Center on Longevity" ressources publiques. A passer au crible avant citation.

---

## 2. Litterature primaire (ECR & meta-analyses) par sous-theme

### Sommeil
- **Cognitive behavioral therapy for chronic insomnia (CBT-I) — recommandation 1re ligne** — American College of Physicians (Qaseem A et al.), 2016, *Annals of Internal Medicine*. https://www.acpjournals.org/doi/10.7326/M15-2175 — *CBT-I est le traitement de **premiere intention** de l'insomnie chronique (avant les hypnotiques). Meta-analyses : reduction ~19 min de latence d'endormissement, ~26 min d'eveil nocturne, +10 % d'efficacite du sommeil ; effets equivalents aux somniferes mais sans effets secondaires et plus durables. Tailles d'effet sur la severite : ~0.5 (depression comorbide) a 1.4-1.5 (PTSD/addiction).*

### Nutrition fondee sur preuves
- **Primary Prevention of Cardiovascular Disease with a Mediterranean Diet (PREDIMED, version corrigee)** — Estruch R et al., 2018, *NEJM* 378:e34. https://www.nejm.org/doi/full/10.1056/NEJMoa1800389 — *ECR pivot (~7447 sujets a haut risque CV). Regime mediterranéen + huile d'olive extra-vierge ou noix vs regime pauvre en graisses : **~30 % de reduction** du critere composite (IDM, AVC, deces CV). ⚠️ La publication **2013 a ete retractee** puis remplacee en 2018 pour irregularites de randomisation (une partie des sujets randomises par menage, non individuellement) ; les estimations corrigees restent du meme ordre mais la preuve est a presenter **avec ces reserves** (cf. section 3).*
- **Mediterranean-style diet for prevention of cardiovascular disease (Cochrane)** — Rees K et al., 2019, *Cochrane Database Syst Rev* CD009825.pub3. https://www.cochranelibrary.com/cdsr/doi/10.1002/14651858.CD009825.pub3/full — *Revue systematique (30 ECR). Preuve **moderee** pour la baisse des AVC (prevention primaire), plus **limitee** pour la mortalite. (Deja en tier-1 — rappele ici pour calibrer le niveau de preuve.)*

### Activite physique (cardio + force)
- **WHO 2020 guidelines on physical activity and sedentary behaviour** — Bull FC, Al-Ansari S, Biddle S et al. (OMS), 2020, *Br J Sports Med* 54:1451-1462. https://bjsm.bmj.com/content/54/24/1451 — *Base de preuve = revues systematiques mises a jour par un Guideline Development Group. Adulte : 150-300 min/sem modere (ou 75-150 vigoureux) + renforcement ≥ 2 j/sem. Sedentarite associee a la mortalite toutes causes/CV/cancer, mais **seuil non quantifiable** (preuve insuffisante). (En tier-1 — cite ici pour sa **base de preuve** explicite.)*
- **Exercise interventions for depressive symptoms (meta-analyses ECR, 2024)** — multiples, *J Affect Disord* / *Sports Med* / PLOS One. Ex. umbrella review : https://www.sciencedirect.com/science/article/pii/S0165032724015635 — *Effet **modere** sur les symptomes depressifs : SMD ≈ -0.4 a -0.67 globalement ; mais en se limitant aux etudes **a faible risque de biais**, l'effet retombe a **SMD ≈ -0.38** (NNT ~4.7). Illustration honnete de la regression de l'effet quand on filtre la qualite methodologique.*

### Sante metabolique
- **Reduction in the Incidence of Type 2 Diabetes with Lifestyle Intervention or Metformin (DPP)** — Knowler WC et al. (DPP Research Group), 2002, *NEJM* 346(6):393-403. https://www.nejm.org/doi/full/10.1056/NEJMoa012512 — *ECR de reference (3234 sujets prediabetiques). Intervention intensive sur le mode de vie (objectif -7 % de poids + 150 min/sem d'activite) : **-58 %** d'incidence du DT2 (IC95 % 48-66) vs **-31 %** pour la metformine. Demonstration nette que le comportement domine le medicament en prevention. (En tier-1 cote CDC — ici l'ECR fondateur source.)*

### Lien social
- **Social Relationships and Mortality Risk: A Meta-analytic Review** — Holt-Lunstad J, Smith TB, Layton JB, 2010, *PLOS Medicine* 7(7):e1000316. https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1000316 — *Meta-analyse fondatrice (148 etudes prospectives) : fortes relations sociales ≈ **+50 %** de probabilite de survie. (En tier-1 — confirme la robustesse du noyau "lien social".)*

---

## 3. Controverses & statut de la preuve

### 3.1 Epidemiologie nutritionnelle — limites structurelles
- **The Challenge of Reforming Nutritional Epidemiologic Research** — Ioannidis JPA, 2018, *JAMA* 320(10):969-970. https://jamanetwork.com/journals/jama/fullarticle/2698337 — *Critique de reference : les associations observationnelles aliment↔maladie (1) impliquent abusivement la causalite, (2) detectent mal de petits effets, (3) sont massivement confondues, (4) reposent sur des mesures alimentaires peu fiables (questionnaires). Beaucoup ne se repliquent pas en ECR.*
- **Limiting Dependence on Nonrandomized Studies… in Human Nutrition Research** — Ioannidis JPA, Trepanowski JF, 2018, *Adv Nutr* / PubMed 30032218. https://pubmed.ncbi.nlm.nih.gov/30032218/ — *Plaide pour plus d'ECR et moins de dependance a l'observationnel.* **Implication pedagogique** : enseigner a distinguer **correlation observationnelle** vs **ECR**, et a lire une taille d'effet + son IC plutot qu'un titre. PREDIMED (retraction/correction) sert d'etude de cas vivante.

### 3.2 Sommeil — critique de "Why We Sleep"
- **Matthew Walker's "Why We Sleep" Is Riddled with Scientific and Factual Errors** — Alexey Guzey, 2019. https://guzey.com/books/why-we-sleep/ — *Fact-check (130 h sur le seul chapitre 1) documentant exagerations et au moins une **barre de graphique supprimee**. Repris/discute par Andrew Gelman (statmodeling.stat.columbia.edu, 2019).* **Posture** : utiliser *Why We Sleep* comme **porte d'entree de vulgarisation**, mais ancrer les **chiffres** sur le consensus AASM/SRS (≥ 7 h) — pas sur les affirmations alarmistes du livre.

### 3.3 Complements & molecules de longevite — recherche, pas recommandation
- **TAME — Targeting Aging with Metformin** — Barzilai N et al. (AFAR), protocole en attente de financement complet. https://www.afar.org/tame-trial — *ECR propose (3000 sujets 65-79 ans, non diabetiques, metformine vs placebo) ; vise a prouver qu'on peut cibler le **vieillissement** comme endpoint. Encore non realise faute de financement (~30-50 M$). **Statut : hypothese, pas preuve.***
- **mTOR inhibitors to improve immune function in older adults (rapamycine/analogues)** — Mannick JB et al., 2014/2018 (*Sci Transl Med*) ; phase 2b/3 : Lancet Healthy Longevity 2021. https://www.thelancet.com/journals/lanhl/article/PIIS2666-7568(21)00062-3/fulltext — *Signal positif sur la reponse vaccinale/immunosenescence chez les > 65 ans. Mais **aucun ECR long terme chez l'humain sain** ne demontre une extension de duree de vie ; securite chronique inconnue.* **Posture** : presenter rapamycine/metformine comme **front de recherche** ; renvoyer a Examine.com (note A→F) pour calibrer la preuve sur les complements grand public.

---

## 4. Convergences / divergences vs l'approche "sources faisant autorite"

**Convergences (cross-check OK)** — les deux approches s'accordent sur le **noyau Pareto** :
- Sommeil ≥ 7 h (consensus AASM/SRS) ; CBT-I comme levier actionnable confirme cote litterature primaire.
- Activite physique : meme chiffre OMS 150-300 min + force ≥ 2 j/sem, ici **adosse a sa base de preuve**.
- Nutrition : schema mediterranéen/Healthy Eating Plate = **exemple soutenu, non dogme** — les deux fichiers insistent sur le niveau de preuve (Cochrane "moderee").
- Metabolique : DPP -58 % retrouve a l'identique (CDC cote autorite, NEJM 2002 cote primaire).
- Lien social : meme meta-analyse Holt-Lunstad 2010.

**Divergences / valeur ajoutee de cet angle** :
- **Ce fichier insiste davantage sur les FAIBLESSES de preuve** : retraction PREDIMED 2013, regression de l'effet "exercice↔depression" quand on filtre le biais, limites Ioannidis de l'epidemiologie nutritionnelle.
- Sur *Why We Sleep* : meme nuance que le fichier autorite, mais **documentee** (Guzey/Gelman) pour en faire un cas d'esprit critique.
- Sur la longevite pharmacologique : meme posture "objet de recherche", **etayee** par le statut reel des essais (TAME non finance, rapamycine sans ECR long terme).
- **Aucune contradiction de fond** : l'angle critique **renforce** le message central (privilegier le noyau comportemental robuste, se mefier de l'optimisation marginale et du hype).

---

## 5. Sequencement de modules suggere

1. **M0 — Penser la preuve sante** : correlation vs causalite, ECR vs cohorte, lire une taille d'effet + IC ; etude de cas PREDIMED (retraction) & critique Ioannidis. *(Pose la lentille critique d'emblee.)*
2. **M1 — Sommeil** : besoin (≥ 7 h, AASM/SRS), CBT-I comme levier 1re ligne ; critique de *Why We Sleep* en exercice d'esprit critique.
3. **M2 — Activite physique** : OMS 150-300 min + force ; lire la base de preuve ; exercice↔sante mentale (et la regression de l'effet par qualite d'etude).
4. **M3 — Nutrition fondee sur preuves** : Healthy Eating Plate, mediterranéen comme exemple ; forces/limites de l'epidemiologie nutritionnelle.
5. **M4 — Sante metabolique** : prediabete, DPP (-58 % par le mode de vie) ; le comportement avant le medicament.
6. **M5 — Lien social, stress, alcool/tabac** : Holt-Lunstad, position OMS alcool ; gestion du stress.
7. **M6 — Complements & longevite (front de recherche)** : methode Examine.com (A→F) ; metformine/rapamycine = **recherche**, pas conseil ; clore par le disclaimer medical fort.

> Fil rouge pedagogique : **M0 (esprit critique) cadre tout le reste** — chaque module revient a la question "quel est le niveau de preuve et la taille d'effet ?".
