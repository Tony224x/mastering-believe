# Exercices Medium — Agent Systems Architecture

---

## Exercice 1 : Choisir le pattern d'orchestration (single vs supervisor vs hierarchical)

### Objectif
Mapper des taches reelles au bon pattern d'agent en raisonnant latence, cout et debuggabilite, et savoir resister a la sur-architecture multi-agent.

### Consigne
Tu dois architecturer 4 systemes d'agents :

| Systeme | Description |
|---|---|
| **S1 — Code assistant** | Repond a des questions sur un repo, lit des fichiers, propose des patchs. Domaine homogene (code). |
| **S2 — Assistant ops** | Recoit une demande qui peut toucher : recherche web, code, ecriture d'email, analyse de donnees. Heterogene. |
| **S3 — Deep research** | Sujet -> plan -> recherche multi-source -> lecture -> verification -> redaction d'un rapport. Multi-phases. |
| **S4 — FAQ bot** | Repond a des questions frequentes a partir d'une base. Simple, 2 tools. |

**Questions :**
1. Pour chaque systeme, recommande un pattern (single ReAct, supervisor, hierarchical, swarm) et justifie en 2 phrases.
2. Pour S2 : pourquoi le supervisor pattern bat un single-agent avec 12 tools ? Quel probleme de contexte resout-il ?
3. Pour S4 : pourquoi un multi-agent serait une erreur ? Estime le surcout de latence et de $ vs un single-agent.
4. Donne, a partir du tableau de tradeoffs du cours, l'ordre de grandeur de latence et de cout (en x) pour single vs supervisor vs hierarchical.
5. Un collegue propose 6 agents pour S1. Quel est le "signal d'alarme multi-agent" ? Que reponds-tu ?
6. Pour S3 : combien de niveaux de hierarchie au maximum recommanderais-tu, et pourquoi pas plus ?

### Criteres de reussite
- [ ] S1 -> single ReAct, S2 -> supervisor, S3 -> hierarchical, S4 -> single (ou simple chaine)
- [ ] Le supervisor pour S2 est justifie par le contexte : chaque specialiste a un prompt focus et un contexte court
- [ ] Le surcout multi-agent pour S4 est chiffre (2-5x cout, 5-15s vs 1-3s) et juge injustifie
- [ ] Les ordres de grandeur reprennent le tableau du cours (single 1x/1-3s, supervisor 2-5x/5-15s, hierarchical 10x+/30s+)
- [ ] Le signal d'alarme est identifie : 6 agents pour ce qui pourrait etre 3 tools sur un seul agent
- [ ] La hierarchie de S3 est bornee (2-3 niveaux) car chaque niveau ajoute latence et cout

---

## Exercice 2 : Concevoir la memoire d'un agent conversationnel

### Objectif
Concevoir une architecture de memoire court-terme + long-terme et gerer le context overflow par le calcul.

### Consigne
Tu construis un assistant personnel qui se souvient des conversations entre sessions.

**Chiffres :**
- Fenetre de contexte du modele : 128K tokens
- System prompt + tools : 4K tokens
- Tu veux garder une marge de securite (ne pas remplir > 60% du contexte)
- Un message moyen : 80 tokens ; une session moyenne : 60 messages
- L'utilisateur revient sur plusieurs jours (memoire long-terme necessaire)

**Questions :**
1. Combien de tokens reels disponibles pour l'historique + la memoire retrieved (apres system prompt et marge 60%) ?
2. Une session de 60 messages fait combien de tokens ? Tient-elle dans le budget ?
3. Au bout de combien de messages le contexte deborde-t-il le budget ? Quel mecanisme declencher ?
4. Concois une memoire a 3 types (episodic, semantic, procedural) : pour chacun, le type de store et un exemple concret.
5. Pourquoi NE PAS tout mettre dans le prompt et faire du "retrieve on demand" sur la memoire long-terme ? Quel pattern reutilises-tu (du J10) ?
6. Decris la strategie de summarization court-terme : a quel seuil, que garde-t-on verbatim, que resume-t-on ?

### Criteres de reussite
- [ ] Le budget reel est calcule (~60% de 128K - 4K ~ 72.8K tokens)
- [ ] La session de 60 messages (~4800 tokens) tient largement dans le budget
- [ ] Le seuil de debordement est calcule en messages ; la summarization est le mecanisme declenche
- [ ] Les 3 types de memoire sont mappes a des stores (episodic -> vector, semantic -> KV/relationnel, procedural -> regles)
- [ ] Le "retrieve on demand" est justifie (eviter le context overflow) et relie au pattern RAG (RAG sur sa propre memoire)
- [ ] La strategie de summarization precise un seuil (ex: garder les N derniers verbatim, resumer les plus anciens)

---

## Exercice 3 : Concevoir un handoff robuste et les conditions d'arret

### Objectif
Concevoir un protocole de handoff multi-agent complet et les garde-fous qui empechent un agent de ne jamais s'arreter.

### Consigne
Dans un systeme supervisor, le superviseur passe la main a un `code_agent` pour refactorer une classe en async, puis a un `test_agent`.

**Contexte :**
- Le `code_agent` a parfois "continue" comme seul message de handoff -> le specialiste ne sait pas quoi faire
- Certaines taches bouclent : code_agent -> test_agent -> code_agent -> ... sans converger
- Budget : 15 steps max, 60K tokens max par run

**Questions :**
1. Liste les 5 champs obligatoires d'un handoff message bien construit (selon le cours) et donne un exemple JSON complet pour le passage supervisor -> code_agent.
2. Pourquoi "continue" est-il un code smell ? Quel echec concret cela provoque ?
3. Concois 4 conditions d'arret independantes pour ce systeme (pas une seule). Pour chacune, le declencheur.
4. Comment detecter et casser une boucle code_agent <-> test_agent qui ne converge pas ?
5. Comment le budget (steps, tokens) doit-il etre propage et decremente a travers les handoffs ?
6. Que doit-il se passer quand le budget est epuise mais la tache pas finie ? (graceful, pas un crash)

### Criteres de reussite
- [ ] Les 5 champs sont presents (context, done, remaining, success_criteria, budget) avec un exemple JSON valide
- [ ] "continue" est identifie comme code smell : le specialiste perd le contexte et refait/ignore du travail
- [ ] Les 4 conditions d'arret couvrent : tache accomplie, budget steps depasse, budget tokens depasse, no-progress / human needed
- [ ] La detection de boucle repose sur un compteur de no-progress (memes actions / pas de nouvel etat) avec un cap
- [ ] Le budget est passe dans le handoff (budget_steps restant) et decremente a chaque step
- [ ] L'epuisement du budget produit une sortie propre (resultat partiel + escalade), pas une exception non geree
