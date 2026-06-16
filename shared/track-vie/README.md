# Track Vie — dossiers de recherche sourcee (ecole de la vie)

Ce dossier rassemble la **recherche sourcee** des 5 prochains domaines du
**Track Vie** (cf. la *Roadmap* dans [`/CLAUDE.md`](../../CLAUDE.md)). Ce sont
des **dossiers preparatoires** : ils ne sont pas encore des domaines. Quand un
domaine sera construit (via le skill `mastering-domain-creator`), son
`REFERENCES.md` et son `PLAN.md` s'appuieront sur ces dossiers.

## Exigences appliquees

Deux contraintes posees par le projet, verifiees pour chaque domaine :
1. **Sources tier-1, reconnues et verifiees** — meta-analyses, revues
   systematiques (Cochrane), institutions (OMS, NIH, Harvard, OCDE, S&P),
   cursus universitaires, auteurs de reference. Chaque source a ete
   **verifiee via recherche web** (titre/auteur/annee/URL). *Note : plusieurs
   sites `.gov`/`.edu`/editeurs (NIH, OMS, Cochrane, investor.gov, SagePub,
   PubMed, SEP) renvoient un 403 au fetch automatise mais sont des URL
   canoniques, corroborees par recoupement de resultats.*
2. **Minimisation des sujets clivants** — chaque dossier contient une section
   dediee qui **cartographie les controverses** et fixe la **posture
   evidence-based / consensus** a adopter (poser la donnee, pas militer).

## Methodologie de triangulation : 2 vagues, 2 approches

Pour chaque domaine, deux recherches **independantes et de methodologie
differente** ont ete croisees :

- **Vague 1 — `*-references.md`** : approche *sources faisant autorite*
  (auteurs/livres de reference, guidelines institutionnelles, canon du domaine).
- **Vague 2 — `*-curricula-evidence.md`** : approche *cursus academiques +
  litterature primaire + lentille critique* (syllabi MIT/Stanford/Yale/MOOCs,
  etudes fondatrices et meta-analyses avec tailles d'effet, statut de
  replication / debats). Chaque dossier vague 2 inclut un **cross-check**
  explicite vs la vague 1.

> Le croisement renforce la fiabilite (les deux approches doivent converger sur
> le noyau) et **expose les fragilites** (la ou la vague 2 nuance une source
> populaire de la vague 1).

## Les 5 domaines (pilier, sources, verdict de cross-check)

| Domaine | Pilier | Sources verifiees (V1 / V2) | Convergence | Apport propre de la vague 2 |
|---------|--------|------------------------------|-------------|------------------------------|
| `finance-personnelle` | **Argent** | 16 / 7 + 4 cursus | Forte | Donnees SPIVA (actif vs passif), Markowitz/Fama-French, debat Fama↔Shiller |
| `sante-longevite` | **Corps** | 13 / 7 + 3 cursus | Totale sur le noyau | Tailles d'effet (PREDIMED, DPP, Cochrane), critiques Ioannidis / Guzey |
| `apprendre-a-apprendre` | **Esprit** | 13 / 5 + cursus LHTL | Tres forte | Effets chiffres (Cepeda, Rohrer 72% vs 38%), Macnamara (limites deliberate practice) |
| `rationalite-decision` | **Jugement** | 12 / + 6 cursus | Noyau commun | Gigerenzer (rationalite ecologique) + lentille replication (OSC 2015 : 36%) |
| `communication-persuasion` | **Relations** | 11 / + meta-analyses | Bonne | Meta-analyses (Van Laer 2014, Shen 2015) + Many Labs 2 (~54% replication) |

## Synthese du cross-check (par domaine)

- **Finance** — Les deux vagues convergent sur la sequence
  *budget → dette → risque → investissement → diversification → portefeuille*.
  La vague 2 transforme le debat **actif vs passif** en **fait chiffre** (rapports
  SPIVA) plutot qu'en opinion. Clivant neutralise : get-rich-quick, hype crypto,
  immobilier vs bourse, fiscalite nationale/politique. *Disclaimer : educatif,
  pas un conseil financier personnalise.*
- **Sante** — Convergence **totale sur le noyau Pareto** (sommeil, alimentation
  peu transformee, exercice force+cardio, lien social, non-tabac). La vague 2
  ajoute la **force de preuve** (tailles d'effet) et une lentille critique
  honnete (faiblesses de l'epidemiologie nutritionnelle, critiques de *Why We
  Sleep*, longevite off-label = **recherche**, pas conseil). *Disclaimer medical
  fort.*
- **Apprendre a apprendre** — Convergence quasi parfaite : *Make It Stick* et
  *Learning How to Learn* sont la **vulgarisation** de la litterature primaire.
  Clivant = **pseudoscience educative**, debunkee avec preuve : styles
  d'apprentissage (Pashler 2008, Dekker 2012), brain-training (Simons 2016),
  neuromythes, sur-simplification "10 000 heures" (Macnamara 2014).
- **Rationalite & decision** — Noyau **Kahneman/Tetlock** commun aux deux vagues ;
  la vague 2 ajoute **Gigerenzer** (contrepoint ecologique) et la **crise de
  replication** (honnetete intellectuelle : ne pas survendre certains effets).
  Posture **anti-clivant centrale** : enseigner *la methode*, pas *les
  conclusions* — exemples neutres (probas, jeux, meteo, sport), jamais
  politique/religion ; verification d'info (SIFT) non partisane.
- **Communication & persuasion** — Convergence sur storytelling, negociation
  raisonnee (Harvard Negotiation Project) et Aristote. La vague 2 **relativise**
  certains effets (Many Labs 2, base de preuve mince pour la CNV et certaines
  tactiques). Cadrage **persuasion ethique** via le garde-fou **CTR**
  (Consentement, Transparence, Reciprocite honnete) — exclut dark patterns, PUA,
  spin politique.

## Principes anti-clivant transverses (a appliquer a tous)

1. **Poser la donnee, pas l'opinion** — privilegier meta-analyses et tailles
   d'effet ; signaler l'incertitude quand elle existe.
2. **Methode > conclusions** — surtout en rationalite : outiller le jugement,
   ne pas imposer de positions.
3. **Exemples neutres** — eviter politique, religion, sujets societaux brulants
   comme terrain d'illustration.
4. **Honnetete sur la replication** — ne pas survendre un resultat fragile ;
   citer les critiques connues.
5. **Disclaimers** — medical (sante) et financier (finance) : contenu educatif,
   ne remplace pas un avis professionnel personnalise.
6. **Ethique explicite** — persuasion honnete (CTR), pas de manipulation.

## Prochaine etape

Quand on construira un domaine, lancer `mastering-domain-creator` : la phase
*recherche* part de ces dossiers (les deux vagues fusionnees → `REFERENCES.md`),
la phase *plan* part des squelettes Pareto-first proposes (challenge adverse),
puis creation + verification. Pour ces domaines de vie, la passe **code-runner**
est optionnelle (code leger) ; **facts-checker** (contre ces sources) et
**pedagogy-reviewer** (incl. controle anti-clivant) restent critiques.

Ordre recommande pour demarrer : **`apprendre-a-apprendre`** (meta-competence
qui accelere tous les autres) ou **`finance-personnelle`** (levier universel).
