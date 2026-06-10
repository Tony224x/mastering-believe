# Exercices Hard — Observabilite & MLOps

---

## Exercice 1 : Plateforme d'observabilite ML pour 40 modeles en production

### Objectif
Concevoir une plateforme d'observabilite mutualisee couvrant modeles classiques et LLMs, avec un budget et des SLAs internes.

### Consigne
Une scale-up fintech a 40 modeles en production (scoring credit, fraude, churn, 6 features LLM) deployes par 8 equipes. Trois incidents recents : un modele de fraude silencieusement degrade pendant 6 semaines (perte 800 K$), un prompt change sans eval qui a fait chuter le CSAT, et un audit regulateur qui a demande "expliquez cette decision de credit d'il y a 14 mois" (8 jours-homme pour repondre).

**Contraintes chiffrees :**
- Volumetrie : 25M de predictions/jour (modeles classiques) + 800K appels LLM/jour
- Budget plateforme : 35 K$/mois (infra + outils) ; equipe : 3 ingenieurs ML platform
- Exigence regulateur : toute decision de credit rejouable et explicable pendant 5 ans
- SLA internes a definir : detection d'une degradation majeure en < 48 h (vs 6 semaines), reponse a un audit en < 4 h (vs 8 jours)
- Les 8 equipes ont des stacks differentes (sklearn, XGBoost, PyTorch, 3 providers LLM)

**Livre :**
1. **Architecture de la plateforme** : couche de collecte (SDK commun ? sidecar ? log shipping ?), stockage (hot pour les dashboards, cold pour l'audit 5 ans), compute d'analyse (jobs de drift quotidiens), alerting. Dimensionne le stockage : 25M predictions/jour x ~2 Ko de contexte (features + score + version) = combien par an ? Cout S3/objet vs base analytique ?
2. **Detection en < 48 h sans labels** : le modele de fraude degrade 6 semaines : les labels (fraude confirmee) arrivent a 30-60 jours. Concois la batterie de signaux SANS labels (PSI features, drift du score, taux d'alertes fraude, proxy metrics metier) avec seuils et politique d'escalade. Quelle est la limite fondamentale de la detection sans labels (que peut-on rater) ?
3. **Audit regulateur en < 4 h** : concois le "decision record" immuable : que capturer pour CHAQUE decision de credit (features exactes, version du modele, version des transformations, explication SHAP precalculee ou recalculable ?). Tradeoff : precalculer l'explication (cout x N) vs la recalculer (il faut TOUT versionner : modele + features + code). Choisis et chiffre.
4. **Couverture LLM** : les 800K appels LLM/jour : tracing complet (prompts/completions = donnees sensibles bancaires : politique de retention/masquage ?), evals continues (LLM-judge sur echantillon : quel taux d'echantillonnage tient dans le budget ?), et le garde-fou "pas de changement de prompt sans eval" (comment l'IMPOSER techniquement : prompts versionnes dans un registry + CI ?).
5. **Adoption par 8 equipes heterogenes** : la plateforme ne marche que si tout le monde l'utilise. Strategie : SDK leger multi-framework, instrumentation par defaut dans les templates de deploiement, ou obligation par la CI ? Quel est le minimum impose vs optionnel ?
6. **Priorisation a 3 ingenieurs** : tu ne peux pas tout construire. Ordonne les chantiers sur 12 mois en justifiant par les 3 incidents (cout evite vs effort).

### Criteres de reussite
- [ ] Le stockage est calcule : 25M x 2 Ko ~ 50 Go/jour ~ 18 To/an (+ LLM payloads) ; architecture hot (30-90 j, base analytique) / cold (5 ans, stockage objet, couts chiffres) avec hypotheses de prix posees
- [ ] La detection sans labels combine >= 4 signaux avec seuils (PSI > 0.25, derive de la distribution des scores, volume d'alertes, proxy metier) ET la limite est nommee (un concept drift qui ne deforme pas les inputs peut passer : d'ou backtesting des que les labels arrivent)
- [ ] Le decision record est specifie champ par champ avec le tradeoff explication precalculee vs recalculable arbitre et chiffre (ex : SHAP precalcule sur les refus uniquement)
- [ ] La partie LLM impose : prompt registry versionne + eval gate en CI + masquage PII + retention differenciee (metadata long terme, payloads courts terme echantillonnes)
- [ ] La strategie d'adoption definit un minimum NON negociable (decision record pour les modeles regules, via CI/template) et du progressif pour le reste
- [ ] La roadmap 12 mois est priorisee par le rapport cout evite/effort, en commencant par le decision record (regulateur) et les signaux sans labels (incident a 800 K$)
- [ ] 3 tradeoffs explicites avec consequences acceptees

---

## Exercice 2 : Pipeline de retraining haute frequence sous contraintes reglementaires

### Objectif
Concevoir un systeme de retraining quasi continu pour un modele soumis a validation reglementaire et a des boucles de feedback.

### Consigne
Le modele de pricing d'un assureur auto en ligne doit suivre un marche qui bouge vite (concurrents, inflation des sinistres). L'equipe veut retrainer chaque semaine. Probleme : chaque nouveau modele de pricing doit etre valide par l'actuariat ET documente pour le regulateur (aujourd'hui : 3 semaines de validation manuelle par version).

**Contraintes chiffrees :**
- Cycle cible : un modele frais en production chaque semaine ; validation actuarielle actuelle : 3 semaines (incompatible)
- Garde-fous reglementaires : aucune prime ne peut varier de plus de +/-15% d'une version a l'autre pour un meme profil ; le modele doit rester explicable (pas de boite noire integrale)
- 50K devis/jour ; un devis trop cher = client perdu (mesurable en 24h), un devis trop bas = perte actuarielle (mesurable en... 24 mois de sinistralite)
- Boucle de feedback : le prix affiche influence qui souscrit (selection adverse) — les donnees de souscription sont biaisees par le pricing courant
- Incident il y a 8 mois : un bug de feature (inflation x100 sur une variable) a produit des primes absurdes pendant 6 h, 40 K$ de devis sous-tarifes honores

**Livre :**
1. **Le conflit 1 semaine vs 3 semaines** : concois le processus qui les reconcilie : validation automatisee de quoi (suite de tests actuariels codifiee : stabilite par segment, bornes de primes, monotonie des variables metier) vs validation humaine de quoi (changements structurels seulement ?). Definis les criteres qui declenchent une revue humaine complete.
2. **Le garde-fou +/-15%** : ou l'implementer (contrainte dans le training ? post-processing au serving ? gate de validation ?). Compare les 3 options et leurs effets pervers (un clamp au serving fausse la coherence actuarielle ; une contrainte au training peut empecher de suivre le marche). Choisis et justifie.
3. **Asymetrie des feedbacks (24 h vs 24 mois)** : la conversion se mesure vite, la sinistralite tres lentement. Comment empecher l'optimisation court-termiste (le retraining hebdo "apprend" a baisser les prix car la conversion monte, la perte ne se voit que dans 2 ans) ? Concois les metriques de surveillance et les invariants actuariels qui servent de proxy long terme.
4. **Selection adverse et donnees biaisees** : les souscripteurs de la semaine N dependent du pricing N-1. Quelles techniques pour entrainer sans amplifier le biais (ponderation, donnees de marche externes, holdout de pricing stable sur un % du trafic) ? Quel cout pour ce holdout ?
5. **Plus jamais l'incident x100** : concois la defense en profondeur sur les features : tests de distribution en CI du pipeline de donnees, bornes par feature au serving, canary de devis (comparaison N vs N-1 sur un echantillon fixe AVANT mise en prod), kill switch avec retour a la derniere version validee.
6. **Architecture complete** : assemble le tout en un diagramme (data -> training -> suite de validation -> registry -> canary -> prod -> monitoring -> boucle) avec les points de controle humains et les artefacts d'audit produits a chaque etape.

### Criteres de reussite
- [ ] Le processus distingue retraining de routine (memes features, validation 100% automatisee + revue legere) et changement structurel (nouvelles features/architecture -> validation complete 3 semaines) avec criteres de classification objectifs
- [ ] Le +/-15% est arbitre avec les 3 options comparees ; la reponse attendue privilegie le gate de validation (bloquer la version) + clamp au serving en filet de securite, en nommant l'effet pervers accepte
- [ ] L'asymetrie 24h/24 mois est traitee par des invariants actuariels (loss ratio projete par segment, primes planchers techniques) surveilles a chaque version — la conversion seule ne peut jamais justifier une baisse
- [ ] Le holdout de pricing stable (ex : 5% du trafic a pricing gele) est propose avec son cout assume (manque a gagner borne) comme reference non biaisee
- [ ] La defense anti-x100 a >= 4 couches independantes (tests data en CI, bornes au serving, canary sur echantillon fixe, kill switch + rollback) — une seule couche n'aurait pas suffi et c'est explique
- [ ] Le diagramme final montre les artefacts d'audit a chaque etape (rapport de validation, model card, decision record) et les 2 points de controle humains
- [ ] 3 tradeoffs explicites (vitesse vs validation, fidelite marche vs stabilite, cout du holdout vs qualite des donnees)
