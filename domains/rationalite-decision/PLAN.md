# PLAN figé — `rationalite-decision` (cursus complet J1–J14, mode full)

> Contrat lu par les builders de Phase 4. **Ne lisez PAS les autres jours** (ils
> n'existent pas tous encore) — ce fichier est la seule vue d'ensemble.
> Sources : `REFERENCES.md` du domaine + `shared/track-vie/rationalite-decision-references.md`,
> `-curricula-evidence.md`, `-j14-sources.md`.

## Règles transverses (OBLIGATOIRES sur CHAQUE jour)
- **Anti-clivant absolu** : on enseigne LA MÉTHODE, jamais des conclusions.
  **Exemples 100% neutres uniquement** : probas, jeux, météo, sport, dépistage
  médical factuel, logistique, contrôle qualité. **Interdits** : politique,
  religion, débats sociétaux/identitaires, géopolitique, élections.
- **Honnêteté sur la preuve** : signaler la nuance de réplication quand pertinent
  (ne pas survendre un biais ou un effet).
- Langue FR, **concret/chiffré AVANT le principe**. ~45 min/module.
- Format théorie : H1 `# Module N — Titre` ; blockquote `> **Temps estimé** : 45 min | **Prérequis** : Modules 01-0(N-1)` + `> **Objectif** :` ; sections `## 1.` … ; callouts `> **À retenir** :` ; **4-5 flash-cards** `**Q1 : …**` / `> R : …` ; `## Points clés à retenir` ; `## Pour aller plus loin` (sources avec URL).
- Exercices : `03-exercises/01-easy/NN-slug.md` = **3 exercices gradués easy→hard** (`## Exercice N`, `### Objectif`/`### Consigne`/`### Critères de réussite` en cases `- [ ]`). Solution : `03-exercises/solutions/NN-slug.md` (corrigé chiffré modèle) ou `.py` si module à code.
- Code (seulement où indiqué) : `02-code/NN-slug.py`, stdlib pur, sans clé API/réseau, `if __name__=="__main__":` démo, exit 0.

## Carte de réutilisation
Certains modules **upgradent** un fichier existant (le lire d'abord, puis écrire sous le nouveau slug). Les anciens fichiers non repris seront supprimés en Phase 7.

| Nouveau | Réutilise l'existant |
|---------|----------------------|
| 01-systeme-jugement | `01-theory/01-systeme-jugement.md` (upgrade) |
| 03-probabilites-frequences | `02-probabilites-utiles.md` + `02-code/02-probabilites-utiles.py` |
| 04-pensee-bayesienne | `03-pensee-bayesienne.md` + `02-code/03-pensee-bayesienne.py` |
| 05-heuristiques-biais | `04-heuristiques-biais.md` |
| 07-decision-incertitude | `05-decision-incertitude.md` |
| 08-calibration-forecasting | `06-calibration-verification.md` (partie Brier) + `02-code/06-calibration-verification.py` |
| 11-verification-information | `06-calibration-verification.md` (partie SIFT/vérif) |
| 14-capstone-boite-outils-jugement | `07-capstone-boite-outils-jugement.md` (upgrade) |

---

## J1 — Le système du jugement
- Système 1 / Système 2 ; ce qu'est un biais ; rationalité ≠ intelligence (Stanovich).
- Acquis : distinguer intuition rapide vs raisonnement contrôlé ; un biais est systématique, pas aléatoire.
- Sources : Kahneman 2011 ; Stanovich 2011. Slug : `01-systeme-jugement`. Pas de code.

## J2 — Arguments & sophismes (🆕)
- Structure d'un argument (prémisses → conclusion) ; validité vs solidité ; sophismes courants (homme de paille, faux dilemme, pente glissante, ad hominem, appel à l'autorité, corrélation→causation comme sophisme).
- Acquis : disséquer un argument, nommer un sophisme, reconstruire une version charitable.
- Exemples neutres (raisonnements sur chiffres/objets). Slug : `02-arguments-sophismes`. Pas de code.

## J3 — Probabilités en fréquences naturelles
- Penser en fréquences (Gigerenzer) ; probabilité conditionnelle ; taux de base ; faux positifs — **sans la formule de Bayes** (réservée à J4).
- Code : `03-probabilites-frequences.py` (table de fréquences naturelles, VPP/VPN) — réutiliser `02-probabilites-utiles.py`.
- Exemple : dépistage médical factuel. Slug : `03-probabilites-frequences`.

## J4 — Pensée bayésienne
- Théorème de Bayes (intuition + notation) ; priors/posteriors ; **mise à jour itérative** (le posterior devient le prior suivant) — valeur ajoutée vs J3.
- Code : `04-pensee-bayesienne.py` (mise à jour séquentielle) — réutiliser `03-pensee-bayesienne.py`.
- Slug : `04-pensee-bayesienne`.

## J5 — Heuristiques & biais
- Ancrage, disponibilité, cadrage, biais de confirmation. **Encadré réplication** (ne pas survendre ; priming social mal répliqué — OSC 2015).
- Exemples neutres (ancrage numérique, Wason neutre — éviter version sociale et « problème de Linda » stéréotypé).
- Slug : `05-heuristiques-biais`. Pas de code.

## J6 — Rationalité écologique
- Gigerenzer : heuristiques rapides et frugales ; recognition heuristic ; quand une règle simple bat un modèle complexe. Recadrage : les raccourcis ne sont pas que des erreurs.
- Exemple : reconnaissance sur classements de villes/sport. Slug : `06-rationalite-ecologique`. Pas de code.

## J7 — Décision sous incertitude
- Espérance, utilité, aversion au risque, arbres de décision ; paradoxes d'Allais/Ellsberg.
- Code : `07-decision-incertitude.py` (espérance + arbre de décision jouet). Exemples : paris équitables, choix d'assurance illustratif.
- Slug : `07-decision-incertitude`.

## J8 — Calibration & forecasting
- Penser en probabilités ; **score de Brier** ; leçons des superforecasters (Tetlock) ; journal de prévisions.
- Code : `08-calibration-forecasting.py` (Brier + courbe de calibration) — réutiliser `06-calibration-verification.py`.
- Exemples : météo, sport. Slug : `08-calibration-forecasting`.

## J9 — Pensée causale (🆕)
- Corrélation ≠ causalité ; confondants ; contrefactuels ; **pourquoi on randomise** (RCT vs observationnel) — angle prospectif/conceptuel (J10 = critique d'une étude déjà faite).
- **Exemples neutres IMPOSÉS** : glaces↔noyades (chaleur confondante), cigognes↔naissances, engrais↔rendement, A/B test produit. **Bannir** tout confondant à charge sociale.
- Sources : Pearl ; Hernan & Robins. Slug : `09-pensee-causale`. Code optionnel léger (sim de confondant).

## J10 — Lire une étude & stats trompeuses (🆕, recentré)
- Pourquoi un résultat publié peut être faux : **p-hacking**, **taille d'effet > p-value**, réplication. PUIS lecture critique de stats médias (graphiques tronqués, % vs points de %, biais de survie, dénominateur manquant, moyenne vs médiane). *Méta-analyse/hiérarchie complète : juste mentionner.*
- Code : `10-lire-une-etude.py` (mini-sim de p-hacking : tester N hypothèses → faux positif). Sources : Ioannidis 2005 ; Simmons 2011.
- Slug : `10-lire-une-etude`.

## J11 — Vérification de l'information à l'ère de l'IA
- SIFT (Caulfield), lateral reading ; deepfakes ; hallucinations de LLM ; fausse citation/image hors-contexte.
- **Exemples 100% APOLITIQUES IMPOSÉS** : fausse citation attribuée à Einstein, image de catastrophe sortie de son contexte/date, faux remède « miracle » (sans nommer de débat sanitaire actuel), deepfake d'acteur. **Interdire** élections/santé publique controversée/géopolitique.
- Slug : `11-verification-information`. Pas de code.

## J12 — Pensée systémique (🆕, recentré)
- Stocks & flux ; boucles de rétroaction (renforçantes/équilibrantes) ; délais ; effets de second ordre ; points de levier (Meadows). *Latticework de Munger réduit à une mention (sa place = intro du capstone).*
- Exemples neutres : thermostat, baignoire, file d'attente, gestion de stock logistique. Sources : Meadows « Thinking in Systems ». Slug : `12-pensee-systemique`. Code optionnel (sim de boucle simple).

## J13 — Débiaisage en pratique (🆕)
- Pre-mortem (Klein) ; red teaming (Nemeth) ; checklists (Gawande) ; groupthink (Janis) — **mécanisme uniquement, SANS les cas politiques de Janis** (transposer sur comité météo/projet logistique/décision d'achat).
- Acquis : appliquer un pre-mortem et une checklist anti-biais à une décision neutre.
- Slug : `13-debiaisage-pratique`. Pas de code.

## J14 — Capstone : la boîte à outils de jugement
- Livrable portfolio sur **une décision neutre et concrète** : (1) checklist de pré-décision, (2) **mini-arbre de décision** (réinvestit J7), (3) **analyse de second ordre** (réinvestit J12), (4) **journal de prévisions calibré** mesuré au **score de Brier** sur ≥10 prédictions datées (réinvestit J8), (5) protocole de vérification (réinvestit J11).
- Livrables : `01-theory/14-capstone-boite-outils-jugement.md` (brief + grille d'auto-éval) + exercices + solution + `04-projects/README.md` (gabarit). Upgrade de `07-capstone-boite-outils-jugement.md`.
- Slug : `14-capstone-boite-outils-jugement`.
