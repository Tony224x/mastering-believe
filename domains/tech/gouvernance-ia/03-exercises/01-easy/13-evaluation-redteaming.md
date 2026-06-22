# J13 — Exercice facile : ecrire ses premiers cas d'eval

## Objectif
Comprendre l'anatomie d'un cas d'eval (entree + comportement attendu + categorie) et mesurer un garde-fou existant, y compris ses **faux positifs**.

On reutilise le garde-fou `guardrail(text) -> Decision` du module (`02-code/13-evaluation-redteaming.py`) ou on le re-implemente en 5 lignes : il bloque toute entree contenant `ignore previous instructions`, `system prompt` ou `debug mode`, et autorise le reste.

## Consigne
1. Re-implemente (ou importe en copiant le code) une fonction `guardrail(text)` qui renvoie la chaine `"BLOCK"` ou `"ALLOW"` selon les trois motifs ci-dessus (insensible a la casse).
2. Ecris une liste `cases` d'au moins **6 cas**, chacun un tuple `(text, expected, category)` :
   - au moins **2 cas benins** (`expected="ALLOW"`, `category="benign"`) — par ex. une question legitime d'un client ;
   - au moins **2 cas de prompt injection** (`expected="BLOCK"`) ;
   - au moins **1 cas que tu sais que le garde-fou va RATER** (une attaque formulee autrement, par ex. « oublie les regles ci-dessus ») — tu dois le marquer `expected="BLOCK"` car c'est le comportement *attendu*, meme si le filtre echoue.
3. Ecris une boucle qui, pour chaque cas, appelle `guardrail`, compare la decision a `expected`, et imprime `PASS`/`FAIL`.
4. A la fin, imprime deux chiffres : le **nombre de faux negatifs** (attaque qui passe) et le **nombre de faux positifs** (cas benin bloque).

## Criteres de reussite
- [ ] La liste `cases` contient au moins 6 cas avec les 3 categories demandees.
- [ ] Au moins un cas est concu pour echouer (attaque non couverte par les motifs) et est correctement marque `expected="BLOCK"`.
- [ ] La sortie affiche un `PASS`/`FAIL` par cas.
- [ ] La sortie affiche separement le compte de faux negatifs et de faux positifs.
- [ ] Le script tourne en `python <fichier>` sans erreur, stdlib seule.
