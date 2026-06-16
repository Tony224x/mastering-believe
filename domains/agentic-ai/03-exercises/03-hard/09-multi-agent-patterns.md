# Exercices Hard — Multi-agent patterns (J9)

---

## Exercice 1 : Hierarchie multi-equipes avec routage et agregation remontante

### Objectif
Implementer le pattern **hierarchical** complet du module : un supervisor de haut niveau (CEO) **route** une tache composite vers la ou les bonnes sous-equipes (chacune avec son propre sous-supervisor + workers), chaque manager **condense** la sortie brute de ses workers, et les rapports **remontent et s'agregent** jusqu'au CEO. Tu dois prouver le routage correct (la bonne equipe est sollicitee) ET l'agregation correcte (le CEO ne voit que des resumes, pas le brut).

### Consigne
Construis un systeme a deux niveaux : `CEO → {sous-supervisor, workers}`.

1. **Deux sous-equipes**, chacune avec un manager et 2 workers :
   - equipe `data` : manager `data_manager`, workers `sql_agent` et `stats_agent`.
   - equipe `content` : manager `content_manager`, workers `writer_agent` et `translator_agent`.
2. **Routage CEO** : ecris une methode `route_teams(task: str) -> list[str]` qui decide quelles equipes activer selon le contenu de la tache :
   - mots `donnees`, `sql`, `metrique`, `stats`, `chiffre` → active `data`,
   - mots `redige`, `article`, `traduis`, `resume`, `texte` → active `content`,
   - une tache mixte active **les deux** equipes ; une tache qui ne matche rien active une equipe `fallback` par defaut (au choix : `content`).
3. **Niveau manager** : chaque manager a sa propre boucle de mini-supervisor (plan → delegue a ses workers → **condense** en un rapport court). Le rapport remonte au CEO **ne doit jamais contenir la sortie brute complete des workers** (defense anti-explosion de contexte, section 6.2) — il doit etre strictement plus court que la concatenation des sorties workers.
4. **Agregation CEO** : le CEO assemble une reponse finale **uniquement** a partir des rapports d'equipe (pas des workers). La reponse finale doit citer chaque equipe activee.
5. **Observabilite** : expose dans le resultat `teams_activated`, `team_reports` (par equipe), `worker_calls` (combien de workers ont tourne) et `final_answer`.

Teste **trois** scenarios : une tache purement data, une tache purement content, et une tache mixte (les deux equipes).

### Criteres de reussite
- [ ] `route_teams` active uniquement `data` sur une tache data, uniquement `content` sur une tache content, et **les deux** sur une tache mixte
- [ ] Une tache qui ne matche rien retombe sur l'equipe fallback (jamais d'exception, jamais zero equipe)
- [ ] Chaque manager condense : son rapport est strictement plus court que la concatenation brute de ses workers
- [ ] Le CEO n'agrege que des `team_reports` (jamais la sortie brute d'un worker) — verifie par assertion
- [ ] `final_answer` cite chaque equipe activee dans le scenario mixte
- [ ] `worker_calls` correspond au nombre exact de workers reellement sollicites (2 par equipe activee)
- [ ] Tout est deterministe et tourne offline, sans dependance

---

## Exercice 2 : Hybride supervisor + swarm avec garde de boucle et assembleur final

### Objectif
Combiner deux patterns sur une tache multi-etapes : un **supervisor** lance la tache et fixe le but, mais l'execution se fait en **swarm** (handoffs lateraux entre specialistes) avec une **garde de boucle** robuste (compteur de hops + detection de cycle A↔B). A la fin, un **assembleur de reponse finale** doit prouver que **>= 3 agents distincts** ont reellement contribue, sinon il rejette. C'est le scenario realiste ou tu veux la flexibilite du swarm mais la garantie de couverture du supervisor.

### Consigne
1. **Specialistes** (callables purs) qui s'echangent la main par handoff, chacun contribuant un artefact tague de son nom :
   - `planner` : decompose la tache, handoff vers `researcher`.
   - `researcher` : produit des faits, handoff vers `coder`.
   - `coder` : produit du code, handoff vers `reviewer`.
   - `reviewer` : valide ; s'il trouve un probleme **une seule fois**, il handoff vers `coder` (boucle de correction legitime), sinon il termine.
   Chaque contribution est stockee avec `{"agent": <nom>, "artifact": <texte>}`.
2. **Supervisor + swarm** : `run_hybrid(task)` demarre via le `planner` (decision du supervisor), puis suit les handoffs en mode swarm. Maintiens :
   - `hop_count` avec un plafond dur `max_hops` → `RuntimeError("hop budget exceeded")`,
   - **detection de cycle serre** : si le meme couple `(from, to)` de handoff apparait **2 fois de suite a l'identique** (A→B puis A→B sans rien entre), leve `RuntimeError("tight loop A<->B")`. (La boucle legitime `reviewer→coder→reviewer` une seule fois ne doit PAS declencher la garde.)
3. **Assembleur final** : `assemble_final(contributions) -> str` qui :
   - verifie que `len({c["agent"] for c in contributions}) >= 3`, sinon leve `ValueError("not enough distinct contributors")`,
   - produit un rapport unifie citant chaque agent contributeur **une fois** (dedup par agent, derniere contribution gardee), dans un style homogene (pas une concatenation brute).
4. **Scenarios a tester** :
   - **nominal** : la tache traverse `planner → researcher → coder → reviewer` (avec une passe de correction `reviewer → coder → reviewer`), l'assembleur reussit, >= 4 agents distincts.
   - **degenere boucle** : un `reviewer` buggue qui renvoie **toujours** vers `coder` et un `coder` qui renvoie toujours vers `reviewer` → la garde de cycle serre (ou le hop budget) doit lever, proprement, avant l'infini.
   - **couverture insuffisante** : un run ou seuls 2 agents contribuent → `assemble_final` leve `ValueError`.

### Criteres de reussite
- [ ] Le run nominal suit la sequence attendue, avec exactement **une** passe de correction `reviewer → coder → reviewer` qui ne declenche PAS la garde
- [ ] `hop_count` est borne et leve `RuntimeError` si le plafond est depasse
- [ ] La detection de cycle serre leve `RuntimeError` sur le scenario degenere (boucle infinie evitee)
- [ ] L'assembleur final prouve la presence de **>= 3 agents distincts**, sinon leve `ValueError`
- [ ] Le rapport final dedup par agent (une contribution par agent) et cite tous les contributeurs
- [ ] Le scenario "couverture insuffisante" leve bien `ValueError`
- [ ] Tout est deterministe, gracieux (toujours une terminaison ou une exception claire), et tourne offline
