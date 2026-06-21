# Rationalité & décision — Sources tier-1 pour l'extension J1–J14

> Complément aux dossiers existants `rationalite-decision-references.md` et `-curricula-evidence.md` (qui couvrent : S1/S2, probas/Bayes, heuristiques & biais, rationalité écologique, décision sous incertitude, calibration/forecasting, vérification d'info/SIFT). Ce fichier ne re-source PAS ces thèmes ; il source **uniquement les nouveaux modules** ajoutés pour passer d'un cursus 6 modules à un cursus J1–J14.
>
> **Posture anti-clivant (rappel)** : ce domaine enseigne la **méthode**, pas des conclusions. Tous les exemples doivent rester **100 % neutres** (jeux, probas, météo, sport, santé publique factuelle, logistique). **Jamais** de politique ni de religion comme support d'exercice. Chaque module ci-dessous propose une note « exemples neutres suggérés ».
>
> Toutes les sources ci-dessous ont été **vérifiées via WebSearch** (titre / auteur(s) / année / URL exacts) le 2026-06-16.

---

## Module — Théorie des jeux & décisions stratégiques

**Theory of Games and Economic Behavior** — John von Neumann & Oskar Morgenstern, 1944 (Princeton University Press). https://en.wikipedia.org/wiki/Theory_of_Games_and_Economic_Behavior → *Texte fondateur qui crée le champ de la théorie des jeux ; introduit les jeux à somme nulle, la valeur d'un jeu et (2e éd.) l'axiomatique de l'utilité espérée vNM. Tier-1 absolu : c'est la référence d'origine du domaine. Alimente le module sur les notions de stratégie, gains et utilité.*

**Equilibrium Points in n-Person Games** — John F. Nash, 1950. Proceedings of the National Academy of Sciences 36(1):48-49. DOI 10.1073/pnas.36.1.48. https://www.pnas.org/doi/10.1073/pnas.36.1.48 (PDF libre : https://people.irisa.fr/Nicolas.Markey/PDF/Papers/pnas36(1)-Nash.pdf) → *Article séminal d'une page définissant l'équilibre de Nash (concept central de toute la théorie des jeux non coopérative, prix Nobel d'économie 1994). Tier-1 : la source primaire du concept. Alimente l'équilibre de Nash et le dilemme du prisonnier.*

**The Evolution of Cooperation** — Robert Axelrod, 1984 (Basic Books ; ISBN 0-465-00564-0). https://en.wikipedia.org/wiki/The_Evolution_of_Cooperation → *Synthèse du tournoi du dilemme du prisonnier itéré (victoire de la stratégie Tit-for-Tat) ; explique sous quelles conditions la coopération émerge sans autorité centrale. Auteur et résultats canoniques, massivement cités. Alimente les jeux répétés, la coopération et les jeux de coordination.*

> **Exemples neutres suggérés** : pierre-feuille-ciseaux et matrices de gains chiffrées ; tournoi Tit-for-Tat simulé en Python ; coordination « choisir le même côté de la route » ; partage d'une ressource commune (pêche, bande passante) ; jeux de la poule mouillée stylisés. Aucun acteur politique réel.

---

## Module — Pensée causale (corrélation ≠ causalité)

**The Book of Why: The New Science of Cause and Effect** — Judea Pearl & Dana Mackenzie, 2018 (Basic Books ; ISBN 978-0465097609). https://en.wikipedia.org/wiki/The_Book_of_Why → *Vulgarisation par le pionnier des modèles graphiques causaux (do-operator, échelle de la causalité). Tier-1 : auteur de référence du domaine, accessible au grand public. Alimente corrélation ≠ causation, diagrammes causaux et intuition des confondants.*

**Causal Inference: What If** — Miguel A. Hernán & James M. Robins, 2020 (Chapman & Hall/CRC), révision 2025 — **gratuit en ligne (PDF officiel)**. https://miguelhernan.org/whatifbook → *Manuel de référence rigoureux et libre d'accès (confondants, contrefactuels, DAG, essais randomisés vs observationnel, biais de sélection). Auteurs Harvard, standard académique. Alimente RCT vs observationnel, confondants et contrefactuels.*

**Causal Inference in Statistics: A Primer** — Judea Pearl, Madelyn Glymour & Nicholas P. Jewell, 2016 (Wiley ; ISBN 9781119186847). https://web.cs.ucla.edu/~kaoru/primer-complete-2019.pdf → *Primer débutant le plus accessible sur l'inférence causale formelle (≈160 p., questions d'étude par chapitre). Pont idéal entre vulgarisation et manuel. Alimente la partie « formaliser un confondant » et les contrefactuels.*

> **Exemples neutres suggérés** : glaces vendues vs noyades (confondant = chaleur) ; cigognes vs naissances ; essai clinique randomisé fictif d'un traitement vs étude observationnelle ; effet d'un engrais sur le rendement agricole ; A/B test produit. Pas de causes politiquement chargées.

---

## Module — Méthode scientifique & qualité de la preuve

**Why Most Published Research Findings Are False** — John P. A. Ioannidis, 2005. PLoS Medicine 2(8):e124. DOI 10.1371/journal.pmed.0020124. https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.0020124 → *Essai fondateur de la métascience : montre comment petits échantillons, faibles tailles d'effet et flexibilité analytique gonflent les faux positifs. Tier-1, l'un des articles les plus cités sur le sujet. Alimente la hiérarchie de la preuve et la valeur prédictive d'un résultat.*

**Estimating the reproducibility of psychological science** — Open Science Collaboration (Nosek et al.), 2015. Science 349(6251):aac4716. DOI 10.1126/science.aac4716. https://www.science.org/doi/10.1126/science.aac4716 → *Réplication de 100 études : ~36 % de réplications significatives, tailles d'effet divisées par deux. Pièce maîtresse empirique de la crise de réplication. Alimente réplication, taille d'effet et publication bias.*

**False-Positive Psychology: Undisclosed Flexibility in Data Collection and Analysis Allows Presenting Anything as Significant** — Joseph P. Simmons, Leif D. Nelson & Uri Simonsohn, 2011. Psychological Science 22(11):1359-1366. DOI 10.1177/0956797611417632. https://journals.sagepub.com/doi/10.1177/0956797611417632 → *Démontre par simulation et expériences que les « researcher degrees of freedom » (p-hacking) permettent de rendre presque n'importe quoi « significatif » ; propose des remèdes (pré-enregistrement). Tier-1 sur p-hacking. Alimente p-hacking et le jardin des chemins bifurquants.*

> **Exemples neutres suggérés** : p-hacking simulé en Python (générer du bruit, tester 20 hypothèses, en trouver une « significative ») ; lancers de pièce et faux positifs ; pyramide des niveaux de preuve appliquée à une question de santé publique factuelle (efficacité d'un dépistage) ; méta-analyse de rendements sportifs. Jamais de sujet idéologique.

---

## Module — Mental models & pensée systémique

**Poor Charlie's Almanack: The Essential Wit and Wisdom of Charles T. Munger** — Charles T. Munger, éd. Peter D. Kaufman, 2005 (réédition Stripe Press 2023 ; ISBN 9781953953230). https://press.stripe.com/poor-charlies-almanack → *Source d'origine de la métaphore du « latticework of mental models » (≈80-90 modèles multidisciplinaires reliés). Tier-1 pour le concept de treillis de modèles mentaux. Alimente le latticework et l'approche multidisciplinaire.*

**Thinking in Systems: A Primer** — Donella H. Meadows (éd. Diana Wright), 2008 (Chelsea Green Publishing ; ISBN 9781603580557). https://en.wikipedia.org/wiki/Thinking_In_Systems:_A_Primer → *Introduction de référence à la pensée systémique par une autrice majeure du domaine (stocks/flux, boucles de rétroaction, points de levier, effets de second ordre). Tier-1. Alimente pensée systémique, feedback loops et effets de second ordre.*

> **Exemples neutres suggérés** : boucle de rétroaction d'un thermostat ; modèle stock/flux d'une baignoire ; dynamique proie-prédateur ; effets de second ordre d'une remise commerciale sur les stocks ; cercle vertueux/vicieux d'un système de file d'attente logistique. Pas de système social controversé.

---

## Module — Débiaisage en pratique

**Performing a Project Premortem** — Gary Klein, 2007. Harvard Business Review 85(9):18-19. https://hbr.org/2007/09/performing-a-project-premortem → *Source primaire de la méthode du pre-mortem (supposer l'échec déjà advenu pour libérer la parole ; « prospective hindsight » +30 % d'identification des risques). Tier-1. Alimente le pre-mortem.*

**The Checklist Manifesto: How to Get Things Right** — Atul Gawande, 2009 (Metropolitan Books ; ISBN 9780805091748). https://en.wikipedia.org/wiki/The_Checklist_Manifesto → *Démonstration de l'efficacité des checklists pour réduire les erreurs d'« ineptie » dans les environnements complexes (chirurgie, aviation). Auteur et ouvrage de référence. Alimente la partie checklists.*

**Groupthink: Psychological Studies of Policy Decisions and Fiascoes** (2e éd.) — Irving L. Janis, 1982 (Houghton Mifflin ; ISBN 9780395317044). https://archive.org/details/groupthinkpsycho00jani/ → *Ouvrage fondateur du concept de « groupthink » (symptômes : illusion d'invulnérabilité, pression vers le conformisme, autocensure) et des garde-fous. Tier-1 pour la pensée de groupe. Alimente biais de groupe / groupthink.*

**Devil's advocate versus authentic dissent: stimulating quantity and quality** — Charlan J. Nemeth et al., 2001. European Journal of Social Psychology 31(6). DOI 10.1002/ejsp.58. https://onlinelibrary.wiley.com/doi/abs/10.1002/ejsp.58 → *Étude expérimentale montrant que le dissensus authentique surpasse l'avocat du diable artificiel pour générer des idées originales — nuance essentielle pour le red teaming / la critique organisée. Tier-1 (pair-évalué). Alimente red teaming et avocat du diable.*

**Before You Make That Big Decision...** — Daniel Kahneman, Dan Lovallo & Olivier Sibony, 2011. Harvard Business Review 89(6):50-60. https://hbr.org/2011/06/the-big-idea-before-you-make-that-big-decision → *Checklist de 12 questions pour détecter et neutraliser les biais dans une décision d'équipe (Kahneman, Nobel 2002). Tier-1 ; relie débiaisage et checklists. Alimente le débiaisage appliqué en groupe.*

> **Exemples neutres suggérés** : pre-mortem d'un projet logistique fictif (lancement d'un entrepôt) ; checklist de décision pour un achat matériel ; red team sur un plan de migration technique ; simulation de groupthink dans un comité d'évaluation de risque météo. Aucune décision politique ou religieuse réelle.

---

### Récapitulatif

**16 sources tier-1 vérifiées**, réparties sur 5 nouveaux modules :
- Théorie des jeux : 3 (von Neumann-Morgenstern, Nash, Axelrod)
- Pensée causale : 3 (Pearl & Mackenzie, Hernán & Robins, Pearl-Glymour-Jewell)
- Méthode scientifique : 3 (Ioannidis, Open Science Collaboration, Simmons et al.)
- Mental models & systèmes : 2 (Munger, Meadows)
- Débiaisage : 5 (Klein, Gawande, Janis, Nemeth, Kahneman-Lovallo-Sibony)
