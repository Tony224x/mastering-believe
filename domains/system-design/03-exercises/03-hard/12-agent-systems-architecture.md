# Exercices Hard — Agent Systems Architecture

---

## Exercice 1 : Agents d'operations IT autonomes — actions irreversibles en production

### Objectif
Concevoir un systeme d'agents qui AGIT sur une infrastructure reelle, avec un modele de permissions, d'approbation et de blast radius rigoureux.

### Consigne
Une entreprise (800 microservices, 3 000 alertes/mois) veut un systeme d'agents pour automatiser la reponse aux incidents : diagnostiquer, remedier (restart, rollback, scaling), et escalader aux humains.

**Contraintes chiffrees :**
- 3 000 alertes/mois, dont ~60% resolvables par 5 runbooks standards (restart, rollback, scale-up, purge cache, failover)
- MTTR actuel avec humains : 45 min ; objectif : < 10 min sur les cas standards
- Une action destructrice erronee (rollback du mauvais service, failover non necessaire) peut couter 100x plus cher que l'incident initial
- Budget tokens : < 15 K$/mois ; latence de diagnostic : < 2 min
- Auditabilite totale : chaque action doit etre tracee, attribuable et rejouable
- L'astreinte humaine reste le fallback 24/7 (mais on veut la reveiller 5x moins)

**Livre :**
1. **Architecture de l'agent** : single-agent avec tools, ou pipeline diagnose -> plan -> execute avec des roles separes ? Justifie par le risque (le modele qui diagnostique doit-il etre celui qui execute ?). Definis le state complet d'un incident en cours de traitement.
2. **Modele de permissions a 3 niveaux** : classe les actions (read-only : logs, metrics ; reversibles : restart, scale ; destructrices : rollback, failover) et definis pour chaque niveau le mecanisme : auto-execute, auto-execute avec delai d'annulation, approbation humaine obligatoire. Ou mets-tu la frontiere et POURQUOI (relie au cout 100x) ?
3. **Garde-fous d'execution** : concois les protections code-level (pas prompt-level) : rate limit d'actions par fenetre (combien ?), blast radius max (1 service a la fois ?), dry-run obligatoire, verification post-action (l'agent verifie que sa remediation a marche ? avec quel timeout avant rollback de sa propre action ?).
4. **Boucle d'apprentissage** : comment le systeme s'ameliore : memoire des incidents (post-mortems structures interrogeables), promotion progressive d'un runbook de "approbation obligatoire" vers "auto-execute" (quels criteres chiffres ? combien d'executions reussies ?).
5. **Le scenario noir** : l'agent diagnostique mal un incident reseau, decide un failover de base de donnees non necessaire, qui provoque 20 min d'indisponibilite. Post-mortem : quelles barrieres ont manque ? Redesigne pour que CE scenario precis soit impossible.
6. **Economie du systeme** : budget tokens (3 000 incidents x diagnostic ~50K tokens + planification), cout vs les heures d'astreinte economisees, et le seuil de rentabilite.

### Criteres de reussite
- [ ] L'architecture separe diagnostic et execution (separation of concerns motivee par le risque) avec un state structure complet (incident_id, hypotheses, evidences, actions proposees/executees, confiance, horodatage)
- [ ] Les 3 niveaux de permissions sont definis avec mecanismes distincts, et les actions destructrices exigent une approbation humaine (ou un delai d'annulation long) — justification explicite par l'asymetrie 100x
- [ ] Les garde-fous sont code-level et chiffres : ex. max 3 actions/15 min, 1 service a la fois, verification post-action avec timeout (5 min) et auto-rollback, kill switch global
- [ ] La promotion d'un runbook est un processus chiffre (ex : 50 executions supervisees, 98%+ de succes, 0 faux positif destructeur) — jamais de promotion automatique des actions destructrices
- [ ] Le post-mortem du scenario noir identifie les barrieres manquantes (confiance du diagnostic non requise pour une action destructrice, pas de validation croisee, pas de circuit "doute -> humain") et le redesign les ajoute
- [ ] L'economie est calculee : ~3 000 x 60K tokens ~ 180M tokens/mois -> verifie la coherence avec le budget en choisissant les modeles (petit modele pour le triage, grand pour le diagnostic complexe)
- [ ] 3 tradeoffs explicites (autonomie vs securite, MTTR vs taux d'approbation, cout du double-check)

---

## Exercice 2 : Plateforme d'agents de recherche concurrents — orchestration, budget et qualite

### Objectif
Concevoir un systeme multi-agent a parallelisme massif avec controle strict du cout et de la qualite du resultat final.

### Consigne
Un cabinet d'etudes lance un produit "due diligence automatisee" : pour chaque dossier (une entreprise cible), le systeme enquete en parallele sur 6 axes (financier, juridique, marche, technologie, reputation, equipe) et produit un rapport consolide avec sources.

**Contraintes chiffrees :**
- 200 dossiers/jour ; un dossier complet doit sortir en < 30 minutes
- Chaque axe demande 15-40 appels de tools (recherche web, bases legales, APIs financieres) ; certains tools sont payants (0.05-0.50 $/appel) et rate-limites (l'API financiere : 10 req/s GLOBAL)
- Budget : < 12 $ par dossier tout compris (LLM + tools) ; aujourd'hui le prototype consomme 45 $/dossier
- Qualite : chaque affirmation du rapport doit etre sourcee ; les contradictions entre axes (le juridique dit X, le financier dit non-X) doivent etre detectees et resolues, pas moyennees
- 15% des dossiers concernent des entreprises quasi inconnues (peu de donnees) : le systeme doit le DIRE plutot que broder

**Livre :**
1. **Topologie d'orchestration** : supervisor avec 6 sub-agents paralleles ? Justifie contre les alternatives (sequentiel, swarm). Concois le contrat de chaque sub-agent (input, output structure avec sources et score de confiance, budget alloue) et le mecanisme de consolidation finale.
2. **Gestion du budget 12 $** : decompose le budget par axe et par etape (qui obtient combien ?). Concois le budget enforcement DANS le code (compteur partage ? arret gracieux d'un agent qui depasse ?) et la degradation si le budget s'epuise a 70% du dossier (livrer partiel ? quels axes sacrifier ?).
3. **Le goulot API financiere (10 req/s global)** : 200 dossiers/jour x ~30 appels financiers, avec des bursts de dossiers simultanes. Concois le partage : rate limiter centralise, file de priorite entre dossiers, cache des donnees financieres (TTL ? les donnees boursieres ont quelle fraicheur acceptable ?).
4. **Detection des contradictions** : concois l'etape de consolidation : comment detecter que 2 sub-agents affirment des choses incompatibles (comparaison structuree des claims ? LLM-judge sur les paires ?), et le protocole de resolution (re-investigation ciblee avec budget additionnel borne, ou escalade humaine).
5. **Le cas "entreprise inconnue"** : comment chaque sub-agent quantifie la suffisance de ses sources (nombre, qualite, recence) et comment le rapport final exprime l'incertitude par axe au lieu d'inventer. Quel seuil declenche un "donnees insuffisantes" assume ?
6. **Observabilite et amelioration** : trace complete d'un dossier (graphe d'execution, couts par agent, sources), metriques produit (cout/dossier, taux de contradictions detectees, taux de dossiers "insuffisants"), et la boucle d'eval hebdomadaire.

### Criteres de reussite
- [ ] Le supervisor + 6 paralleles est justifie (axes independants = parallelisme reel, contrat clair) et le contrat de sub-agent est un schema structure (claims[] avec source, confiance, horodatage)
- [ ] Le budget est decompose chiffre (ex : 6 axes x 1.5 $ + consolidation 1.5 $ + marge 1.5 $) avec enforcement code-level (compteur central, arret gracieux avec rapport partiel) et une politique de sacrifice explicite
- [ ] Le goulot 10 req/s est calcule (200 x 30 = 6 000 appels/jour = OK en moyenne (~0.07 req/s) MAIS les bursts paralleles le saturent) -> rate limiter global partage + file + cache avec TTL differencie (fondamentaux : 24h+, cours : minutes)
- [ ] La consolidation compare des claims STRUCTURES (pas du texte libre) et la resolution est bornee (1 re-investigation max, sinon contradiction signalee dans le rapport)
- [ ] Le mecanisme d'incertitude est concret (score de suffisance par axe : nb de sources independantes, recence) avec seuil chiffre et rendu explicite dans le rapport
- [ ] La reduction 45 $ -> 12 $ est expliquee par des leviers identifies (petit modele pour la recherche, gros pour la consolidation, cache, deduplication des appels tools, plafonds par agent)
- [ ] 3 tradeoffs explicites (parallelisme vs cout, completude vs budget, autonomie vs escalade humaine)
