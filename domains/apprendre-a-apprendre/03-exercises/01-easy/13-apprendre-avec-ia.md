# Exercices — Module 13 : Apprendre avec l'IA

---

## Exercice 1 — Générer un jeu de retrieval avec un LLM et le critiquer

### Objectif
Expérimenter le rôle de générateur de retrieval d'un LLM, puis évaluer la qualité des questions produites avec un regard critique.

### Consigne

**Étape 1 — Génération (10 min) :**
Prends un concept vu dans ce domaine (ex : la courbe d'oubli, l'interleaving, le mastery learning de Bloom, les étapes de Fitts & Posner). Soumets ce prompt à un LLM de ton choix :

> *"Génère 6 questions de retrieval practice sur [concept choisi], du plus simple au plus complexe. Format : question seule, sans réponse. Je répondrai d'abord de mémoire."*

Réponds aux 6 questions de mémoire, sans aide, avant de lire le corrigé du LLM.

**Étape 2 — Auto-correction (10 min) :**
Demande au LLM :
> *"Voici mes réponses : [copie-colle tes réponses]. Corrige-les et indique pour chacune si c'est juste, partiellement juste ou faux, avec une explication courte."*

**Étape 3 — Critique du quiz (10 min) :**
Évalue les 6 questions générées selon ces critères :
- Est-ce que chaque question force une vraie récupération en mémoire (réponse non triviale) ou est-ce une question de reconnaissance facile ?
- Y a-t-il une progression de difficulté réelle ?
- Une question est-elle ambiguë, trop large, ou factuellement douteuse ?
- Y a-t-il des questions qui testent la compréhension plutôt que la simple mémorisation ?

Note au moins deux forces et deux faiblesses du jeu de questions produit.

### Critères de réussite
- [ ] Les 6 questions ont été répondues de mémoire AVANT de lire le corrigé du LLM
- [ ] L'auto-correction identifie précisément où se situent les lacunes (pas juste "j'ai eu faux")
- [ ] La critique mentionne au moins deux forces et deux faiblesses concrètes du quiz LLM
- [ ] Au moins une faiblesse porte sur la qualité pédagogique (pas seulement sur un fait incorrect)

---

## Exercice 2 — Protocole Feynman augmenté par LLM

### Objectif
Utiliser un LLM comme partenaire de détection de trous dans une explication personnelle, puis mesurer la progression entre deux tours.

### Consigne

**Tour 1 — Explication initiale (10 min) :**
Choisis un des concepts les plus récents que tu as étudiés dans ce domaine (Modules 07 à 13). Écris une explication de ce concept en 150-250 mots, comme si tu l'enseignais à quelqu'un qui n'a pas suivi le cours.

Puis soumets ce prompt au LLM :
> *"Voici mon explication de [concept] : [ta rédaction]. Détecte dans mon explication : (1) les zones floues ou imprécises, (2) les sauts logiques non justifiés, (3) le jargon non défini, (4) les erreurs conceptuelles. Sois précis et concis."*

**Tour 2 — Correction et ré-explication (10 min) :**
Liste les points soulevés par le LLM. Pour chacun :
- Est-il légitime ? (Vérifie dans le module théorique)
- Est-ce une vraie lacune ou une ambiguïté de formulation ?

Réécris les passages problématiques en corrigeant les lacunes identifiées.

**Mesure de progression :**
Compare ta version initiale et ta version corrigée. Pour chaque point soulevé par le LLM, note si ta version corrigée le règle réellement.

### Critères de réussite
- [ ] L'explication initiale fait 150-250 mots sans jargon non défini au premier tour
- [ ] Le LLM est invité à critiquer, pas à expliquer (tu n'as pas demandé "explique-moi ce concept")
- [ ] Au moins deux points soulevés par le LLM sont vérifiés dans le module théorique (pas acceptés sans contrôle)
- [ ] La version corrigée est plus précise et complète que la version initiale sur les points soulevés

---

## Exercice 3 — Construire et évaluer un plan d'espacement généré par LLM

### Objectif
Générer un plan de révision espacée pour les modules de ce domaine, puis l'évaluer à l'aune de ce que tu as appris sur l'espacement (Module 03) et la mesure (Module 09).

### Consigne

**Étape 1 — Génération du plan (10 min) :**
Soumets ce prompt à un LLM :

> *"J'ai étudié les thèmes suivants cette semaine : [liste des modules du domaine que tu as couverts]. Construis un plan de révision espacée pour les 3 prochaines semaines, en appliquant l'idée de la courbe d'oubli : planifie des révisions à J+3, J+7, J+14 et J+21 selon les thèmes. Indique pour chaque révision ce que je devrais faire (quiz, Feynman, blank-page recall, etc.)."*

**Étape 2 — Évaluation critique du plan (15 min) :**
Analyse le plan produit en répondant à ces questions par écrit :

1. Le plan respecte-t-il des intervalles croissants (J+3 < J+7 < J+14 < J+21) ? Sinon, quelle est l'erreur ?
2. Les activités proposées (quiz, Feynman, etc.) sont-elles des techniques à utilité élevée (Dunlosky 2013) ou à utilité faible (relecture, surlignage) ?
3. Le plan prévoit-il un moyen de **mesurer** la rétention à chaque jalon, ou se contente-t-il de planifier des sessions sans indicateur ?
4. Quelles modifications apporterais-tu pour rendre ce plan plus conforme à ce que tu as appris dans les Modules 02, 03 et 09 ?

**Étape 3 — Version améliorée :**
Rédige une version corrigée d'au moins deux jalons du plan, en intégrant tes corrections.

### Critères de réussite
- [ ] Le prompt soumis au LLM mentionne explicitement les intervalles d'espacement voulus
- [ ] L'analyse critique identifie correctement au moins une technique à utilité faible si le LLM en a inclus
- [ ] La question de la mesure de rétention est traitée (pas seulement la planification des sessions)
- [ ] La version corrigée de deux jalons intègre au moins une technique à utilité élevée et un indicateur de mesure
