# PLAN — Finance personnelle (curriculum fige)

> Ce fichier est la **reference fige du curriculum**. Ne pas modifier la structure sans consensus. Agent A construit les modules 01-03 ; Agent B construit les modules 04-07.

---

## Curriculum — 7 modules

### Module 01 — Interets composes et valeur du temps
**Slug** : `01-interets-composes`
**Objectif** : Comprendre le mecanisme central de toute accumulation de capital a long terme.
**Livrables** :
- `01-theory/01-interets-composes.md`
- `02-code/01-interets-composes.py` (calculateur stdlib)
- `03-exercises/01-easy/01-interets-composes.md` + `solutions/01-interets-composes.md`
**Concepts cles** : capitalisation (annuelle/mensuelle), formule A = P(1+r/n)^(nt), effet du temps vs timing, cout de l'attente.

### Module 02 — Budget, epargne automatique et fonds d'urgence
**Slug** : `02-budget-epargne-fonds-urgence`
**Objectif** : Construire les fondations : budget solide + reserve de securite.
**Livrables** :
- `01-theory/02-budget-epargne-fonds-urgence.md`
- `03-exercises/01-easy/02-budget-epargne-fonds-urgence.md` + `solutions/02-budget-epargne-fonds-urgence.md`
**Concepts cles** : regle 50/30/20, epargne automatique "se payer en premier", fonds d'urgence 3-6 mois, liquidite vs rendement.

### Module 03 — Dette et credit
**Slug** : `03-dette-credit`
**Objectif** : Distinguer bonne et mauvaise dette, calculer le cout reel, decider avec methode.
**Livrables** :
- `01-theory/03-dette-credit.md`
- `03-exercises/01-easy/03-dette-credit.md` + `solutions/03-dette-credit.md`
**Concepts cles** : TAEG, amortissement, avalanche vs boule de neige, score de credit, cout d'opportunite.

### Module 04 — Investir simplement et sur le long terme
**Slug** : `04-investir-long-terme`
**Objectif** : Comprendre la diversification, les fonds indiciels a bas cout et l'horizon.
**Concepts cles** : actions/obligations/ETF, portefeuille 3 fonds, frais de gestion, "buy and hold", SPIVA.

### Module 05 — Psychologie de l'argent et impact des frais
**Slug** : `05-psychologie-argent-frais`
**Objectif** : Identifier les biais comportementaux et l'impact compose des frais.
**Concepts cles** : aversion a la perte, biais de confirmation, cout compose des frais, "ignorer le bruit".

### Module 06 — Independance financiere et retrait soutenable
**Slug** : `06-independance-financiere`
**Objectif** : Comprendre le mouvement FIRE, la regle des 4 % et ses limites.
**Concepts cles** : taux d'epargne, Trinity Study, taux de retrait soutenable, FIRE comme cadre pas comme promesse.

### Module 07 — Capstone : mon plan financier personnel
**Slug** : `07-capstone-plan-financier`
**Objectif** : Produire son propre plan financier chiffre en 6 etapes.
**Scope** :
1. Budget mensuel detaille + taux d'epargne
2. Objectif et calendrier fonds d'urgence
3. Inventaire et plan de remboursement dettes
4. Allocation d'investissement simulee
5. Projection interets composes 20-30 ans
6. Cible optionnelle d'independance financiere

---

## Posture editoriale et regles anti-clivant

Ces regles s'appliquent a TOUS les modules sans exception.

### Sujets a traiter avec soin

**Actif vs passif**
- Presenter la preuve par les donnees (SPIVA Year-End 2024 : ~92 % des fonds domestiques US battus par leur indice sur 20 ans ; ~65 % sur 1 an) et l'arithmetique de Sharpe 1991.
- Nuancer honnêtement : le chiffre SPIVA compte les fonds (equiponderes) ; pondere par encours, l'ecart se reduit mais ne s'inverse pas.
- **Ne pas militer** : "les preuves suggerent que..." pas "vous devez absolument..."

**Stock-picking / day-trading**
- Presenter comme activite a haut risque, chronophage, difficile a rentabiliser durablement pour un particulier.
- Ne pas interdire ni encenser. Cadrer comme hors du "20 % qui compte".

**Crypto**
- Mentionner brievement comme classe d'actifs a tres forte volatilite (Liu & Tsyvinski 2021).
- Eviter tout maximalisme et tout rejet dogmatique.
- Cadrer : n'y investir que ce qu'on peut perdre.

**Immobilier vs bourse**
- Ne pas trancher. Presenter comme deux outils avec profils differents (liquidite, levier, horizon).

**Fiscalite**
- Rester sur les mecanismes generaux (principe de la fiscalite differee, importance des enveloppes).
- Renvoyer aux sources officielles a jour (AMF, IEFP, equivalents locaux) pour les specificites nationales.
- Ne pas figer des chiffres/regles qui changent selon les pays et les annees.

### Disclaimer obligatoire
Rappel dans chaque module pertinent : **ce contenu est educatif, pas un conseil financier personnalise**. Les performances passees ne prejugent pas des performances futures. Tout investissement comporte un risque de perte en capital.

---

## Sequencement pedagogique

Inspire de la convergence MIT 15.401 / Yale ECON 252 / Khan Academy / OCDE-INFE :
1. Valeur du temps et interets composes (moteur fondamental)
2. Budget + epargne + fonds d'urgence (fondations pratiques)
3. Dette et credit (risque courant)
4. Investissement long terme (deployer l'epargne)
5. Psychologie + frais (facteurs souvent decides mais determinants)
6. Independance financiere (horizon ultime)
7. Capstone (synthese appliquee)

**Invariant** : on ne touche a l'investissement qu'apres avoir pose budget + dette + securite. Concret avant abstrait. Chaque module ouvre sur un exemple ou calcul, puis extrait le principe.
