# Exercices Difficiles — Les 4 piliers d'un agent gouvernable (J2)

> Stack : Python 3.11+ stdlib uniquement. Point de depart : `02-code/02-quatre-piliers.py`.

---

## Exercice 1 : Score de gouvernance pondere + niveaux de maturite

### Objectif
Depasser le « passe / passe pas » binaire : produire un **score de gouvernance** par agent et le qualifier par un niveau de maturite defendable devant un board.

### Consigne
1. Definis une ponderation des 4 piliers (ils ne sont pas egaux en risque). Justifie tes poids en commentaire. Suggestion : Permissions et Identite plus lourds (un agent sur-permissionne ou usurpant une identite est le risque OWASP ASI01 le plus aigu).
2. Ecris `governance_score(agent) -> float` qui retourne un score sur 100 :
   - chaque pilier **satisfait** rapporte son poids,
   - un pilier **partiellement** satisfait (ex. owner present mais = « IT ») rapporte la moitie de son poids,
   - un pilier **absent** rapporte 0.
3. Mappe le score sur un niveau : `ABSENT` (<25), `EMERGENT` (25-59), `MANAGED` (60-89), `GOVERNED` (>=90).
4. Sur une flotte d'au moins 6 agents varies, trie-les par score croissant et affiche un tableau `agent_id | score | niveau`. Les pires (a remedier en priorite) en haut.

### Criteres de reussite
- [ ] Les poids somment a 100 et sont justifies en commentaire
- [ ] Un agent parfait score 100 (`GOVERNED`), un agent vide score 0 (`ABSENT`)
- [ ] Le scoring « demi-credit » fonctionne (ex. owner = « IT » ≠ owner absent)
- [ ] Les niveaux sont attribues selon les bons seuils
- [ ] Le tableau est trie par criticite croissante (pires agents en premier)

---

## Exercice 2 : Validateur de chaine — les 4 piliers sont indissociables

### Objectif
Demontrer **par le code** la these centrale du module : retirer un seul pilier casse la valeur des autres.

### Consigne
1. Modelise les 4 piliers comme une chaine de dependances logiques :
   - `audit` n'a de valeur que si `identity` existe (une trace sans qui = inexploitable) ;
   - `permissions` n'ont de valeur prouvable que si `audit` existe (droits bornes mais aucune preuve d'usage) ;
   - `identity` n'est actionnable que si un `owner` existe (on sait qui, mais personne a appeler).
2. Ecris `broken_chain(agent) -> list[str]` qui retourne, pour un agent, la liste des **implications cassees** (ex. `"audit présent mais identity absente -> trace inexploitable"`).
3. Construis 4 agents, chacun privé d'**un seul** pilier, et montre que chaque suppression invalide au moins une implication de la chaine (donc qu'aucun pilier n'est optionnel).
4. Ecris un petit test `assert` qui echoue si un agent complet declenche la moindre implication cassee.

### Criteres de reussite
- [ ] Les dependances entre piliers sont explicitement modelisees
- [ ] Retirer un pilier (parmi les 4) casse au moins une implication
- [ ] Un agent complet ne casse **aucune** implication (assert vert)
- [ ] Les messages expliquent *pourquoi* le maillon manquant invalide la chaine
- [ ] La demo couvre bien les 4 cas de suppression

---

## Exercice 3 : Remediation report — du diagnostic a l'action

### Objectif
Transformer le diagnostic des 4 piliers en un **plan de remediation** priorise, livrable a un owner.

### Consigne
1. Ecris `remediation_plan(agent) -> list[dict]` : pour chaque pilier en echec, produis une action corrective `{"pillar", "issue", "fix", "priority"}`.
   - La priorite reflète le risque : un wildcard de permissions ou une usurpation d'identite = `CRITICAL` ; un owner non nomme = `HIGH` ; etc. (justifie en commentaire).
2. Ecris `fleet_remediation(agents) -> dict` qui agrege la flotte : nombre d'actions par priorite, et l'`agent_id` ayant le plus d'actions critiques.
3. Genere un **rapport markdown** (chaine de caracteres, via `str`/f-strings — pas de lib) : un titre, la couverture de gouvernance globale, puis une section par agent non gouverne avec ses actions triees par priorite.
4. Cas limite : une flotte **100 % gouvernee** doit produire un rapport « Aucune remediation requise » (et `fleet_remediation` ne doit pas planter sur des compteurs vides).

### Criteres de reussite
- [ ] Chaque pilier en echec genere une action `{pillar, issue, fix, priority}`
- [ ] Les priorites reflètent le risque (wildcard / usurpation = CRITICAL)
- [ ] `fleet_remediation` agrege correctement et identifie le pire agent
- [ ] Le rapport markdown est bien forme (titre, couverture, sections par agent)
- [ ] Une flotte saine produit « Aucune remediation requise » sans erreur
