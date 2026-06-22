# Exercices Easy — Agent Systems Architecture

---

## Exercice 1 : Single-agent ou multi-agent ?

### Objectif
Savoir choisir le bon pattern d'agent selon le use case.

### Consigne
Pour chacun des produits suivants, indique si tu demarrerais avec :
- **Single-agent** (ReAct loop classique)
- **Supervisor multi-agent**
- **Hierarchical multi-agent**
- **Swarm / peer-to-peer**

Justifie en 2-3 phrases en citant : nombre de tools, diversite des taches, latence cible, debuggabilite requise.

Produits :
1. Un assistant de coding type Cursor qui edite des fichiers d'un repo
2. Un agent de recherche financiere qui doit lire plusieurs rapports, faire des comparaisons et generer un investment memo
3. Un bot Slack qui repond a des questions sur la documentation interne
4. Un agent d'automatisation marketing qui : segmente les users, ecrit des emails personnalises, les envoie via un tool, puis analyse les resultats
5. Un agent "deep research" qui doit ecrire un rapport de 20 pages avec des sources verifiees
6. Un correcteur de grammaire dans un IDE

### Livrables
Un tableau : produit / pattern / justification.

### Criteres de reussite
- [ ] Le coding assistant et le correcteur de grammaire sont single-agent
- [ ] L'agent deep research est hierarchical ou supervisor avec phases
- [ ] Le bot Slack sur doc interne est single-agent (c'est un RAG + outils)
- [ ] L'agent marketing est supervisor (taches heterogenes)
- [ ] Le principe "start simple, scale up" est applique au moins 1 fois
- [ ] La latence est citee comme critere pour au moins 2 produits

---

## Exercice 2 : Dessiner le state et la condition d'arret

### Objectif
Formaliser le state minimum d'un agent et ses conditions d'arret.

### Consigne
Tu construis un agent "assistant de voyage" qui doit :
1. Comprendre la demande (destination, dates, budget)
2. Chercher des vols
3. Chercher des hotels
4. Proposer un itineraire
5. Laisser le user valider avant de booker

**Questions :**
1. Definis le **state** minimum de cet agent (schema TypedDict ou dataclass Python)
2. Liste les 4-5 conditions d'arret possibles de l'agent
3. Que fait l'agent si le budget d'etapes (ex: 15 steps) est depasse avant la completion ?
4. Comment gerer l'attente de validation humaine sans faire tourner l'agent en boucle ?
5. Ou stockes-tu le state entre deux sessions (un user revient 2 jours plus tard) ?

### Livrables
Un doc structure avec : schema du state, conditions d'arret, strategie de human-in-the-loop, persistence.

### Criteres de reussite
- [ ] Le state contient au moins : user_request, destination, dates, budget, candidate_flights, candidate_hotels, itinerary, validation_status, steps_used, messages
- [ ] 4 conditions d'arret citees : task_complete, user_validated, budget_exceeded, error_unrecoverable, human_needed
- [ ] La gestion du budget exceeded retourne un etat intermediaire au user (pas un crash)
- [ ] Le human-in-the-loop utilise un pattern "interrupt + resume" (ex: LangGraph checkpoints, ou stockage externe du state + webhook)
- [ ] La persistence est nommee : DB relationnelle pour le state structure, vector store pour les preferences long-term

---

## Exercice 3 : Diagnostic d'un multi-agent qui hallucine

### Objectif
Identifier les causes courantes d'echec en multi-agent et proposer des mitigations.

### Consigne
Un systeme supervisor avec 5 agents specialistes (search, coder, writer, analyst, critic) hallucine souvent : le critic valide du code qui n'existe pas, le writer cite des sources que le search n'a pas trouvees, le supervisor declare "done" alors qu'une partie de la tache n'est pas faite.

**Travail :**
1. Propose 5 hypotheses pour expliquer ces echecs
2. Pour chacune, decris un test pour la valider
3. Propose une mitigation concrete
4. Classe les mitigations par ratio impact/effort

### Livrables
Une analyse structuree : hypothese / test / mitigation / priorite.

### Criteres de reussite
- [ ] Au moins 1 hypothese sur les handoff messages trop pauvres (context manquant)
- [ ] Au moins 1 hypothese sur le context bleed entre agents (les agents reutilisent des donnees d'autres agents sans verification)
- [ ] Au moins 1 hypothese sur la memoire qui grossit mal (stale data utilisee)
- [ ] Au moins 1 hypothese sur le critic qui est du meme "niveau" que les agents (non-independant)
- [ ] Au moins 1 hypothese sur l'absence de verification groundedness
- [ ] Mitigations incluent : handoff messages structures, groundedness check, memoire horodatee, critic base sur un modele different, stopping criteria explicites
- [ ] La priorisation "quick wins first" est respectee
