# PLAN — Finance personnelle (curriculum fige, J1–J14)

> Ce fichier est la **reference figee du curriculum**. Cursus complet 14 jours (~45 min/module), grand public, code leger. Ne pas modifier la structure sans consensus.
>
> Issu d'une re-planification challengee par relecture adverse (fusion fiscalite J7+J11, ajout "Revenus & negociation salariale", nuance SPIVA fonds-vs-actifs, recentrage crypto+robo, capstone en gabarit parametrable). Sources : `shared/track-vie/finance-personnelle-references.md`, `-curricula-evidence.md`, `-j14-sources.md`.

---

## Curriculum — 14 modules

### J1 — Interets composes et valeur du temps
**Slug** : `01-interets-composes`
**Objectif** : Comprendre le moteur de toute accumulation + ancrer la distinction taux **nominal vs reel** (inflation).
**Livrables** : `01-theory/01-interets-composes.md` · `02-code/01-interets-composes.py` · easy + solution.
**Concepts** : A = P(1+r/n)^(nt), temps > taux > montant, regle des 72, **inflation & pouvoir d'achat (taux reel ≈ nominal − inflation)**, cout de l'attente.

### J2 — Budget et se payer en premier
**Slug** : `02-budget-se-payer-dabord`
**Objectif** : Fondation pratique : flux mensuels maitrises et epargne automatisee.
**Livrables** : theorie · easy + solution.
**Concepts** : 50/30/20, "pay yourself first", automatisation des virements, suivi des depenses, taux d'epargne.

### J3 — Fonds d'urgence, objectifs et etapes de vie
**Slug** : `03-fonds-urgence-objectifs`
**Objectif** : Securiser (reserve) **et** cadrer : definir objectifs/horizons avant d'investir.
**Livrables** : theorie · easy + solution.
**Concepts** : fonds d'urgence 3-6 mois, liquidite vs rendement, erosion par l'inflation, objectifs SMART par horizon (court/moyen/long), planification par etapes de vie (OCDE/INFE).

### J4 — Dette, credit et score de credit
**Slug** : `04-dette-credit`
**Objectif** : Distinguer bonne/mauvaise dette, calculer le cout reel, comprendre le score de credit.
**Livrables** : theorie · easy + solution.
**Concepts** : TAEG, amortissement, avalanche vs boule de neige, **score/historique de credit (principe general)**, cout d'opportunite.

### J5 — Bases de l'investissement
**Slug** : `05-bases-investissement`
**Objectif** : Poser risque/rendement, classes d'actifs et **principe** de diversification (avant la mise en oeuvre).
**Livrables** : theorie · `02-code/05-bases-investissement.py` (simulation risque/rendement, volatilite) · easy + solution.
**Concepts** : couple risque/rendement, classes d'actifs (actions/obligations/liquidites/immobilier), diversification (Markowitz 1952, SEC asset allocation), horizon, volatilite.
**Garde-fou** : J5 s'arrete au *principe* de diversification ; l'allocation concrete appartient a J6.

### J6 — Fonds indiciels et allocation
**Slug** : `06-fonds-indiciels-allocation`
**Objectif** : Mise en oeuvre : ETF, portefeuille 3 fonds, debat actif/passif par la donnee, impact des frais.
**Livrables** : theorie · `02-code/06-fonds-indiciels-allocation.py` (impact des frais composes + allocation) · easy + solution.
**Concepts** : fonds indiciels/ETF, 3-fund portfolio, frais composes (Bogle), SPIVA.
**Garde-fou HIGH** : enseigner SPIVA **avec sa nuance** (chiffre equipondere fonds vs pondere encours — Cremers et al.) ; "les preuves suggerent", pas "vous devez". **Aucun nom de produit/ticker** (rester generique : "ETF actions monde a bas frais"). Les frais sont ancres ici (pas re-demontres en J7).

### J7 — Fiscalite et enveloppes (principes universels) [fusion ex-J7+J11]
**Slug** : `07-fiscalite-enveloppes`
**Objectif** : Comprendre les mecanismes fiscaux generaux et pourquoi les enveloppes comptent — **sans loi nationale**.
**Livrables** : theorie · easy + solution.
**Concepts** : progressivite, plus-values, deductions, fiscalite differee, role des enveloppes (principes OCDE *Revenue Statistics*), frottement fiscal (distinct de l'erosion par frais vue en J6).
**Garde-fou** : 100% principes OCDE, zero legislation nationale ; renvoyer aux sources officielles locales (AMF/IEFP/equivalents) pour les specificites.

### J8 — Psychologie de l'argent et arnaques
**Slug** : `08-psychologie-arnaques`
**Objectif** : Identifier biais comportementaux **et** reconnaitre/refuser les arnaques financieres (le biais est le vecteur de l'arnaque).
**Livrables** : theorie · easy + solution.
**Concepts** : aversion a la perte, biais de confirmation, FOMO, "ignorer le bruit" (Housel) ; schemas d'arnaque (rendement garanti, urgence, Ponzi, get-rich-quick), reflexes FINRA.

### J9 — Assurance, gestion du risque et transmission
**Slug** : `09-assurance-risque`
**Objectif** : Comprendre le transfert de risque (assurance) vs auto-assurance, et le principe de transmission.
**Livrables** : theorie · easy + solution.
**Concepts** : role de l'assurance, transfert vs retention de risque, types courants (sante, responsabilite, vie), franchise/prime, **principe de transmission/beneficiaire (general, pas de droit national)** (III Insurance Handbook).

### J10 — Immobilier vs location et gros achats
**Slug** : `10-immobilier-location`
**Objectif** : Decider acheter/louer et gros achats par la donnee, sans dogme.
**Livrables** : theorie · easy + solution.
**Concepts** : rent vs buy (cout total de possession, cout d'opportunite, calculateur NYT Upshot), levier/liquidite, "gros achats" (encart : voiture, etc.).
**Garde-fou** : ne pas trancher immobilier vs bourse ; deux outils a profils differents.

### J11 — Revenus et negociation salariale [NOUVEAU — trou comble]
**Slug** : `11-revenus-negociation`
**Objectif** : Travailler le levier #1 sur une vie : le revenu (et savoir le negocier).
**Livrables** : theorie · easy + solution.
**Concepts** : revenu comme levier principal (vs optimisation des frais), capital humain, sources de revenu, preparation d'une negociation salariale (fourchette de marche, BATNA, ancrage ethique), augmentation/evolution.

### J12 — Independance financiere et retrait soutenable
**Slug** : `12-independance-financiere`
**Objectif** : Comprendre le cadre FIRE, la regle des 4% **et ses limites**.
**Livrables** : theorie · `02-code/12-independance-financiere.py` (simulateur taux d'epargne → annees + retrait 4%) · easy + solution.
**Concepts** : taux d'epargne moteur, Trinity Study, taux de retrait soutenable, **limites (sequence de rendements, hypothese 30 ans)**, FIRE comme cadre pas comme promesse.

### J13 — Actifs numeriques et robo-advisors (neutre)
**Slug** : `13-actifs-numeriques-robo`
**Objectif** : Comprendre (pas acheter) deux nouveautes : crypto comme classe d'actifs et la gestion automatisee.
**Livrables** : theorie · easy + solution.
**Concepts** : crypto = classe tres volatile (rendement historique TOUJOURS colle a la volatilite ~75% et "ne mettre que ce qu'on peut perdre"), risques (custody, fraude — SEC/FINRA), robo-advisors (principe, frais, limites — Investor.gov 2017).
**Garde-fou HIGH** : NEUTRE (ni hype ni FUD) ; pas de langage "a mettre dans un portefeuille diversifie" qui sonnerait comme reco d'allocation ; aucun nom de produit.

### J14 — Capstone : simulateur de plan financier
**Slug** : `14-capstone-plan-financier`
**Objectif** : Assembler un **gabarit parametrable** (pas un plan prescriptif) couvrant le cursus.
**Livrables** : theorie (guide) · `02-code/14-capstone-plan-financier.py` (simulateur parametrable) · easy + solution.
**Scope** : budget+taux d'epargne, fonds d'urgence cible, plan dette, allocation simulee, projection composee 20-30 ans (reelle vs nominale), cible FIRE optionnelle.
**Garde-fou HIGH** : livrable = **gabarit ou l'apprenant entre SES chiffres** ; disclaimer "projection educative, hypotheses ajustables, pas une recommandation" **dans le livrable lui-meme** ; aucune sortie prescriptive type "investis X% en actions".

---

## Posture editoriale et regles anti-clivant (TOUS les modules)

- **Disclaimer obligatoire par fichier** : *contenu educatif, pas un conseil financier personnalise ; les performances passees ne prejugent pas des performances futures ; tout investissement comporte un risque de perte en capital.*
- **Actif vs passif** : par la donnee (SPIVA) **avec** la nuance methodologique (fonds vs encours) ; ne pas militer.
- **Crypto** : neutre, volatilite toujours mentionnee, "que ce qu'on peut perdre" ; ni maximalisme ni rejet dogmatique.
- **Fiscalite** : mecanismes generaux (OCDE) uniquement ; renvoyer aux sources officielles a jour pour le national.
- **Aucun produit/emetteur/ticker commercial recommande** (Bogle/Vanguard cites comme source historique = OK).
- **Pas de get-rich-quick** ; le revenu et le temps priment sur l'optimisation marginale.

---

## Sequencement (invariant confirme MIT/Yale/Khan/OCDE)

Valeur-temps → budget → securite+objectifs → dette → bases invest → allocation/indiciel → fiscalite → psycho+arnaques → assurance → gros achats → **revenus** → independance → nouveautes → capstone.
**Invariant** : on ne touche a l'investissement (J5) qu'apres budget+securite+dette (J2-J4). Concret avant abstrait.

## Lots de construction
7 builders (2 modules chacun) : F1 J1-J2 · F2 J3-J4 · F3 J5-J6 · F4 J7-J8 · F5 J9-J10 · F6 J11-J12 · F7 J13-J14.
