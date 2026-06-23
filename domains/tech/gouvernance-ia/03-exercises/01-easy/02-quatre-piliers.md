# Exercices Faciles — Les 4 piliers d'un agent gouvernable (J2)

> Stack : Python 3.11+ stdlib uniquement. Point de depart : `02-code/02-quatre-piliers.py`.

---

## Exercice 1 : Decrire un agent avec ses 4 piliers

### Objectif
Savoir instancier un `GovernedAgent` complet et verifier qu'il passe le smell test.

### Consigne
1. Importe (ou recopie) la dataclass `GovernedAgent` et la fonction `check_governance` du code du jour.
2. Cree un agent realiste : un bot qui resume les tickets de support.
   - `agent_id` : un URN unique (ex. `agent://ticket-summarizer/a1`), **pas** l'identite d'un humain.
   - `owner` : une **personne nommee** (prenom + nom), pas « le support ».
   - `permissions` : au moins deux scopes explicites et bornes (ex. `read:tickets`, `write:ticket_summary`) ; **aucun** scope d'ecriture dangereux.
3. Appelle `check_governance(agent)` et affiche le resultat.
4. Affiche les 4 attributs de l'agent sous forme d'une mini « Agent Card » lisible (4 lignes : Identite / Owner / Permissions / Audit).

### Criteres de reussite
- [ ] L'agent est instancie avec les 4 piliers renseignes
- [ ] `check_governance` retourne une liste **vide** (agent gouvernable)
- [ ] L'`agent_id` ne commence pas par `user:` / `human:` / `person:`
- [ ] L'owner est une personne nommee, pas une equipe
- [ ] La mini Agent Card affiche bien Identite, Owner, Permissions, Audit

---

## Exercice 2 : Detecter un agent non gouverne (smell test)

### Objectif
Reconnaitre et qualifier un agent qui echoue a une ou plusieurs des 4 questions.

### Consigne
1. Cree **trois** agents volontairement defectueux, chacun cassant **un seul** pilier :
   - A : pas d'`owner` nomme (owner vide ou `"IT"`).
   - B : `permissions` vides (aucun scope) OU un wildcard (`["*"]`).
   - C : `agent_id` qui usurpe une identite humaine (commence par `user:`).
2. Passe chacun a `check_governance` et affiche, pour chaque agent, **quel** pilier echoue et **pourquoi** (le message renvoye).
3. Ecris une petite fonction `smell_test(agent) -> str` qui retourne `"GOUVERNE"` si la liste d'echecs est vide, sinon `"NON GOUVERNE: <raisons>"`.

### Criteres de reussite
- [ ] Les 3 agents defectueux echouent chacun sur le pilier attendu
- [ ] Le message d'echec nomme explicitement le pilier (IDENTITY / OWNER / PERMISSIONS / AUDIT)
- [ ] `smell_test` distingue correctement gouverne vs non gouverne
- [ ] Un agent complet (celui de l'ex. 1) retourne bien `"GOUVERNE"`

---

## Exercice 3 : Calculer la couverture de gouvernance d'une flotte

### Objectif
Produire un chiffre unique, board-ready : quel pourcentage de la flotte est gouvernable.

### Consigne
1. Construis une liste d'au moins **5 agents** melangeant gouvernables et non gouvernes.
2. Ecris (ou reutilise `governance_coverage`) une fonction qui retourne le **pourcentage** d'agents passant les 4 piliers.
3. Affiche un mini-rapport :
   - nombre total d'agents,
   - nombre d'agents gouvernables,
   - nombre d'agents **orphelins** (sans owner valide) specifiquement,
   - couverture en %.
4. Gere le cas limite : une flotte **vide** doit retourner `0.0` sans planter.

### Criteres de reussite
- [ ] La couverture est un pourcentage correct (verifie a la main sur ton echantillon)
- [ ] Le rapport compte separement les agents orphelins
- [ ] Une flotte vide retourne `0.0` (pas de division par zero)
- [ ] Le rapport est lisible (une metrique par ligne)
