# Finance personnelle — Cursus académiques, littérature primaire & lentille critique

> **Vague 2 de recherche** (complémentaire à `finance-personnelle-references.md`, qui couvre les auteurs/livres faisant autorité). Approche ici : **comment les universités enseignent le sujet**, **les travaux fondateurs vérifiés** derrière les livres grand public, et **le statut de la preuve** sur les débats clés.
>
> ⚠️ **Disclaimer** : contenu **éducatif**, pas un conseil financier personnalisé. Les classes d'actifs (dont la crypto) sont présentées par leurs propriétés mesurées, jamais comme recommandation. Le débat actif/passif est traité **par la donnée (SPIVA, arithmétique de Sharpe)**, pas par dogme.
>
> Toutes les sources ci-dessous ont été vérifiées via recherche + consultation web (titre / auteur·org / année / URL). Date de vérification : 2026-06.

---

## 1. Cursus académiques trouvés

| Cours | Institution | URL | Ce qu'on en retient (structure) |
|---|---|---|---|
| **15.401 Finance Theory I** (Fall 2008) | MIT Sloan (OpenCourseWare) | https://ocw.mit.edu/courses/15-401-finance-theory-i-fall-2008/ | Cœur de la finance moderne. Séquence : marchés & intermédiaires → **valeur temps de l'argent / actualisation** → obligations (fixed-income) → actions → **budgétisation du capital (NPV)** → **diversification & sélection de portefeuille** → **CAPM (équilibre de prix du risque)** → **marchés efficients** → intro dérivés/options. Manuel : Brealey, Myers & Allen, *Principles of Corporate Finance*. Lectures additionnelles : Malkiel *A Random Walk Down Wall Street*, Bernstein *Capital Ideas*. **Leçon de structure : on part de la valeur-temps avant tout, le portefeuille/risque vient après la valorisation des actifs simples.** |
| **Financial Markets** (Coursera, gratuit en audit) | Yale / Robert Shiller (prix Nobel) | https://www.coursera.org/learn/financial-markets-global | 7 modules. Idées, méthodes et institutions qui permettent à la société de gérer le risque. Couvre : risque & assurance, **CAPM**, finance comportementale, fintech, banque commerciale/d'investissement, immobilier, régulation, politique monétaire, « démocratisation de la finance ». **Angle distinctif : la finance comme outil social de gestion du risque + biais comportementaux.** |
| **ECON 252 Financial Markets** (2008 & 2011, transcripts complets) | Yale / Robert Shiller (Open Yale Courses) | https://oyc.yale.edu/economics/econ-252 | Version universitaire complète et libre du précédent. Ordre indicatif : Intro → Risque & crises → Technologie de la finance → **Diversification de portefeuille (Markowitz + CAPM, lec. 4)** → Assurance → invité David Swensen (allocation d'actifs) → **Marchés efficients vs excès de volatilité (lec. 6–7)** → Dette & levier → Actions → … → Investissement institutionnel & limites à l'arbitrage. **Leçon : diversification/CAPM enseignés tôt, l'efficience des marchés juste après comme contrepoint critique.** |
| **Personal Finance / Financial Literacy** (Life skills) | Khan Academy (gratuit) | https://www.khanacademy.org/college-careers-more/personal-finance | Le plus proche d'un cursus « finance perso » pur. Unités : 1) Intro → 2) **Épargne & budget** → 3) **Intérêts & dette** → 4) **Investissements & retraite** (IRA/401k, fonds, ETF, actions) → 5) Revenus → 6) Logement → 7) Auto → 8) Impôts → 9) Études ; + assurance, arnaques/fraude, carrière. **Leçon de séquencement : budget/épargne AVANT dette AVANT investissement — l'investissement ne vient qu'après les fondations.** |

> Convention repo (`shared/external-courses.md`) : référencer la vidéo/ressource officielle, paraphraser sans recopier, pas d'affiliation. Aucun de ces cours n'est encore dans `external-courses.md` (qui est centré IA/ML) — à ajouter si le domaine se concrétise.

---

## 2. Sources de littérature primaire (par sous-thème)

### Théorie du portefeuille & diversification
- **Portfolio Selection** — H. Markowitz, *The Journal of Finance*, 7(1), 77–91, 1952. https://doi.org/10.1111/j.1540-6261.1952.tb01525.x — *Acte de naissance de la théorie moderne du portefeuille : formalise le couple rendement/variance et montre que la diversification entre actifs peu corrélés réduit le risque sans sacrifier le rendement espéré. Fondement « caché » derrière tous les conseils « diversifiez ».*
- **Common risk factors in the returns on stocks and bonds** — E. Fama & K. French, *Journal of Financial Economics*, 33(1), 3–56, 1993. https://doi.org/10.1016/0304-405X(93)90023-5 — *Modèle à trois facteurs (marché + taille SMB + value HML). Base empirique de l'investissement factoriel ; nuance le CAPM « un seul facteur ».*

### Efficience des marchés (pourquoi l'indiciel a un socle théorique)
- **Efficient Capital Markets: A Review of Theory and Empirical Work** — E. Fama, *The Journal of Finance*, 25(2), 383–417, 1970. https://doi.org/10.1111/j.1540-6261.1970.tb00518.x — *Définit l'efficience (faible / semi-forte / forte) : si les prix reflètent l'information disponible, « battre le marché » durablement est très difficile. Argument académique central pour l'investissement passif.*

### Actif vs passif (le cœur arithmétique)
- **The Arithmetic of Active Management** — W. Sharpe (prix Nobel), *Financial Analysts Journal*, 47(1), 7–9, 1991. https://doi.org/10.2469/faj.v47.n1.7 — *Démonstration en quelques pages : avant frais, l'ensemble des gérants actifs = le marché ; après frais, l'actif est en moyenne perdant face au passif. Pur raisonnement comptable, pas une opinion. (Contesté en marge : voir §3.)*

### Crypto (classe d'actifs, lentille neutre)
- **Risks and Returns of Cryptocurrency** — Y. Liu & A. Tsyvinski, *The Review of Financial Studies*, 34(6), 2689–2727, 2021 (NBER w24877, 2018). https://doi.org/10.1093/rfs/hhaa113 — *Étude empirique de référence : les rendements crypto sont pilotés par des facteurs spécifiques au marché crypto (réseau, attention des investisseurs, momentum) et **pas** par les facteurs macro/actions traditionnels → comportement de classe d'actifs distincte, à très forte volatilité.*

### Cadre de littératie financière (référentiel de compétences)
- **G20/OECD INFE Core Competencies Framework on Financial Literacy for Adults** — OCDE/INFE, 2016. https://www.oecd.org/finance/financial-education/ (PDF GPFI : https://www.gpfi.org/sites/default/files/documents/Core-Competencies-Framework-Adults.pdf) — *Référentiel international des compétences adultes, 4 domaines : Argent & transactions · Planifier & gérer ses finances · Risque & récompense · Paysage financier. Sert de **carte des modules** validée institutionnellement.*
- **OECD/INFE Toolkit for Measuring Financial Literacy and Financial Inclusion 2022** — OCDE, 2022. https://www.oecd.org/en/publications/oecd-infe-toolkit-for-measuring-financial-literacy-and-financial-inclusion-2022_cbc4114f-en.html — *Méthodologie de mesure (questionnaire standard) ; utile pour des quiz d'auto-évaluation / active recall dans le domaine.*

### Données empiriques actif vs passif (rapport de marché faisant autorité)
- **SPIVA U.S. Scorecard — Year-End 2024** — S&P Dow Jones Indices, 2025. https://www.spglobal.com/spdji/en/documents/spiva/spiva-us-year-end-2024.pdf — *Suivi semestriel de référence. Voir §3 pour les chiffres clés et la controverse de méthode.*

---

## 3. Controverses & statut de la preuve

| Débat | État de la preuve | Données / nuance |
|---|---|---|
| **Actif vs passif** | **Robuste en faveur du passif sur longue durée**, mais avec une réserve méthodologique reconnue. | SPIVA Year-End 2024 : ~**65 %** des fonds actions large-cap US battus par le S&P 500 sur 1 an, ~**79 %** des fonds actions US battus vs S&P Composite 1500, et ~**92 %** des fonds domestiques battus sur **20 ans**. Socle théorique : Sharpe 1991 (arithmétique). **Contre-point honnête** : Cremers, Fulkerson & Riley (SSRN, *Financial Analysts Journal*) montrent que SPIVA compte les **fonds** (équipondéré) et non les **actifs** ; pondéré par les encours, ~56 % des actifs sous-performent en 2024 → l'écart se réduit mais ne s'inverse pas. **À enseigner par la donnée, pas comme dogme.** |
| **Marchés efficients** | **Théorie dominante mais contestée** (débat Fama ↔ Shiller, co-Nobel 2013). | Fama 1970 : prix ≈ information → battre le marché est dur. Shiller : « excès de volatilité », bulles, finance comportementale → l'efficience n'est pas totale. Statut : **les deux camps sont primés**, la vérité opérationnelle pour l'épargnant (« diversifie, baisse les frais, n'essaie pas de timer ») tient dans les deux cas. |
| **Crypto** | **Preuve = classe d'actifs distincte, très volatile ; PAS une recommandation.** | Liu & Tsyvinski 2021 : facteurs propres au crypto, déconnectés des facteurs traditionnels. Littérature : Bitcoin ~50 % de rendement annualisé pour ~75 % de volatilité (un ordre de grandeur au-dessus des actions). À cadrer comme **actif spéculatif à haut risque**, pas adapté à un besoin de capital à court terme ni à une forte aversion au risque. |
| **Conseils génériques vs situation individuelle** | **Consensus pédagogique** : un référentiel donne le cadre, pas la décision. | Le cadre OCDE/INFE structure les compétences mais insiste sur l'adaptation au contexte (revenu, fiscalité nationale, horizon, tolérance au risque). Tout module doit rappeler que les règles (ex. « fonds d'urgence 3–6 mois », allocations types) sont des **points de départ**, pas des prescriptions. |

---

## 4. Convergences / divergences vs l'approche « auteurs de référence » (cross-check)

> Note de recoupement avec la vague 1 (`finance-personnelle-references.md`, à ne pas modifier).

**Convergences (les deux approches se rejoignent) :**
- **Indiciel à bas frais / diversification** : ce que Bogle, Malkiel, *Random Walk* recommandent côté grand public a son **fondement académique** dans Markowitz 1952 (diversification), Fama 1970 (efficience) et Sharpe 1991 (arithmétique). Les livres = la conclusion ; les papiers = la démonstration.
- **Séquencement budget → dette → investissement** : Khan Academy (cursus) et les best-sellers de finance perso ordonnent le sujet de la même façon (fondations avant marchés).
- **« Ne pas timer le marché »** : conseil populaire ET résultat des cours MIT/Yale sur l'efficience.

**Divergences / apports propres à cette vague :**
- **Rigueur du débat actif/passif** : les livres tranchent souvent « passif gagne » ; l'approche académique ajoute la **nuance méthodologique SPIVA fonds-vs-actifs** → plus honnête.
- **CAPM & facteurs (taille/value)** : peu présents dans les livres grand public, centraux dans MIT 15.401 et Fama-French 1993 → niveau « intermédiaire/avancé » que les auteurs populaires survolent.
- **Crypto** : les livres grand public l'ignorent ou la diabolisent ; la littérature (Liu-Tsyvinski) offre un **cadrage neutre et chiffré**.
- **Lentille comportementale (Shiller)** : tempère l'efficience pure que certains livres présentent comme acquise.

---

## 5. Séquencement de modules suggéré par les cursus

Synthèse de l'ordre commun MIT 15.401 / Yale ECON 252 / Khan Academy / référentiel OCDE :

1. **Fondations & littératie** — argent, transactions, valeur-temps de l'argent (actualisation), budget. *(Khan U1–2 ; OCDE « Argent & transactions » ; MIT démarre par la valeur-temps.)*
2. **Épargne, dette & intérêt composé** — gérer ses flux, dette saine vs toxique. *(Khan U2–3 ; OCDE « Planifier & gérer ».)*
3. **Risque & assurance** — mutualisation, fonds d'urgence, couverture. *(Yale lec. 2 & 5 ; OCDE « Risque & récompense ».)*
4. **Investissement — bases** : actions, obligations, fonds/ETF, comptes retraite. *(Khan U4 ; MIT obligations→actions.)*
5. **Théorie du portefeuille & diversification** — Markowitz, frontière efficiente, allocation d'actifs. *(MIT & Yale lec. 4 ; Swensen sur l'allocation.)*
6. **Valorisation & équilibre du risque** — CAPM, facteurs (Fama-French). *(MIT 15.401 cœur ; niveau intermédiaire.)*
7. **Efficience des marchés & actif vs passif** — Fama vs Shiller, Sharpe, données SPIVA, finance comportementale. *(MIT « efficient markets » ; Yale lec. 6–7.)*
8. **Sujets avancés / paysage financier** — dérivés (intro), crypto comme classe d'actifs, fiscalité, immobilier, régulation. *(MIT intro dérivés ; Khan U6–8 ; OCDE « Paysage financier ».)*

> **Invariant pédagogique des 4 cursus** : on ne touche à l'investissement qu'après avoir posé budget + dette + risque ; la théorie du portefeuille (diversification) précède toujours le débat actif/passif, qui sert de conclusion critique.
