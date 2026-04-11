# Exercices Hard — Memory & State (J3)

---

## Exercice 1 : MemGPT-lite — agent avec gestion autonome de sa memoire

### Objectif
Implementer un agent qui gere sa propre memoire de maniere autonome : il decide quand sauvegarder, quoi resumer, et quoi oublier. Inspire du paper MemGPT (Packer et al., 2023).

### Consigne
Cree un agent `MemoryAgent` qui dispose de **meta-outils de memoire** en plus de ses outils classiques :

1. **Meta-outils de memoire** (l'agent les appelle comme des outils normaux) :
   - `save_to_long_term(text, category)` : stocker une info dans le vector store avec metadata
   - `search_long_term(query, top_k)` : chercher dans le vector store
   - `update_scratchpad(key, value)` : mettre a jour la working memory
   - `read_scratchpad()` : lire la working memory complete
   - `summarize_context()` : forcer une summarization du context window
   - `forget(memory_id)` : supprimer une memoire devenue obsolete

2. **Contraintes** :
   - Le context window est limite a **2000 tokens** (pas plus — force l'agent a gerer sa memoire)
   - L'agent doit resoudre une tache qui necessite 15+ etapes (impossible sans gestion de memoire)
   - L'agent choisit LUI-MEME quand utiliser ses meta-outils de memoire (pas de logique hardcodee)

3. **Scenario de test** :
   ```
   Tache : "Analyse ces 5 produits, compare-les, et recommande le meilleur pour un developpeur avec un budget de 600 EUR."

   L'agent doit :
   - Chercher les 5 produits (outil search)
   - Analyser chacun (5 appels d'outil — impossible de tout garder en context)
   - Stocker les resultats partiels en working memory ou long-term
   - Resumer son contexte quand il approche des 2000 tokens
   - Produire une recommandation finale qui utilise des infos de toutes les etapes
   ```

4. **Mode simule** : hardcode les decisions de l'agent pour montrer l'architecture. Le LLM est simule mais les meta-outils sont reels (vector store, working memory, checkpointing).

### Criteres de reussite
- [ ] L'agent a acces aux 6 meta-outils de memoire ET aux outils classiques (search, calculate)
- [ ] Le context window ne depasse JAMAIS 2000 tokens (verifier a chaque etape)
- [ ] L'agent utilise `save_to_long_term` pour stocker des infos qu'il ne peut pas garder en context
- [ ] L'agent utilise `search_long_term` pour retrouver des infos stockees precedemment
- [ ] L'agent utilise `summarize_context` quand le context approche la limite
- [ ] L'agent utilise `update_scratchpad` pour maintenir son etat de tache
- [ ] La reponse finale contient des informations de toutes les etapes (prouve que la memoire a fonctionne)
- [ ] Le code est < 400 lignes, bien structure, chaque composant testable independamment

---

## Exercice 2 : Systeme de memoire distribue multi-agent

### Objectif
Construire un systeme de memoire partage entre plusieurs agents qui collaborent sur une tache. Chaque agent a sa propre working memory, mais partage un long-term memory commun avec controle de concurrence.

### Consigne
Cree une architecture multi-agent avec memoire partagee :

1. **SharedMemoryBus** — le bus de memoire partage :
   - Vector store commun pour la memoire long-terme
   - Systeme de **locks** pour eviter les ecritures concurrentes
   - **Event log** : chaque ecriture/lecture est loggee avec l'agent source
   - **Namespace isolation** : chaque agent peut lire tout, mais n'ecrit que dans son namespace
   - **Broadcast** : quand un agent ecrit une info importante, les autres sont "notifies" (ajout dans leur context au prochain tour)

2. **3 agents specialises** :
   - `ResearchAgent` : cherche des informations (outil search), stocke les faits trouves
   - `AnalysisAgent` : lit les faits du ResearchAgent, produit des analyses, stocke les conclusions
   - `ReportAgent` : lit les conclusions de l'AnalysisAgent, genere un rapport final

3. **Orchestration** : les agents tournent en sequence (pas de parallelisme, pour simplifier)
   ```
   Round 1: ResearchAgent cherche → stocke 3 faits dans la memoire partagee
   Round 2: AnalysisAgent lit les faits → analyse → stocke 2 conclusions
   Round 3: ReportAgent lit les conclusions → genere le rapport
   Round 4: ResearchAgent cherche plus d'infos (basees sur les conclusions)
   Round 5: AnalysisAgent re-analyse avec les nouvelles infos
   Round 6: ReportAgent produit le rapport final
   ```

4. **Scenario de test** :
   ```
   Tache : "Analyse comparative des solutions de deploiement cloud (Fly.io vs Render vs Vercel) pour une startup SaaS B2B."

   ResearchAgent → cherche les prix, features, limites de chaque plateforme
   AnalysisAgent → compare sur 5 criteres (prix, DX, scaling, support, ecosystem)
   ReportAgent → genere une recommandation structuree
   ```

5. **Mode simule** : reponses hardcodees pour chaque agent, mais la memoire partagee est reelle.

### Criteres de reussite
- [ ] Le SharedMemoryBus isole les namespaces en ecriture (agent A ne peut pas ecrire dans le namespace de B)
- [ ] Chaque agent peut LIRE les memoires de tous les autres agents
- [ ] L'event log trace qui a ecrit quoi et quand
- [ ] Le broadcast fonctionne : quand ResearchAgent ecrit un fait, AnalysisAgent le voit au tour suivant
- [ ] Le systeme de locks empeche les ecritures simultanees (meme si on est en sequence, le code doit le gerer)
- [ ] Le rapport final contient des informations provenant des 3 agents
- [ ] Le code montre clairement le flux de donnees entre agents via la memoire partagee
- [ ] Chaque agent a sa propre working memory (pas partagee)
- [ ] L'architecture est extensible : ajouter un 4eme agent ne modifie pas le code existant
