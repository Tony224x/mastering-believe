# J4 — Exercice intermediaire : un scorer avec modulateurs agentiques

## Objectif

Coder un scorer de risque qui applique les **modulateurs agentiques** vus en theorie (section 5.2) : l'irreversibilite augmente l'impact, l'autonomie augmente la vraisemblance. Montrer que le meme risque "logique" obtient un score plus eleve sur un agent autonome + irreversible.

## Consigne

En Python 3.11+ (stdlib uniquement) :

1. Definissez une fonction `score(likelihood, impact, irreversible, autonomous)` qui :
   - valide que `likelihood` et `impact` sont dans `1..5` (sinon `ValueError`) ;
   - applique `impact += 1` si `irreversible` (plafonne a 5) ;
   - applique `likelihood += 1` si `autonomous` (plafonne a 5) ;
   - retourne un dict `{"eff_likelihood", "eff_impact", "criticality"}` ou `criticality = eff_likelihood * eff_impact`.
2. Ajoutez une fonction `decision(criticality)` qui renvoie `"TREAT"` si `>= 12`, `"MONITOR"` si `6..11`, `"ACCEPT"` sinon.
3. Demontrez avec **deux appels sur le meme risque brut** (`likelihood=3, impact=4`) :
   - cas A : `irreversible=False, autonomous=False` ;
   - cas B : `irreversible=True, autonomous=True`.
   Affichez les deux criticites et leurs decisions, et montrez que B > A.
4. Affichez aussi le **rationale** (quelle modulation s'est appliquee) pour le cas B.

## Criteres de reussite

- [ ] Le script tourne avec `python <fichier>` sans erreur (stdlib seule).
- [ ] Une vraisemblance ou un impact hors `1..5` leve bien `ValueError`.
- [ ] Les modulateurs plafonnent correctement a 5 (un impact 5 + irreversible reste 5).
- [ ] Le cas B (autonome + irreversible) a une criticite **strictement superieure** au cas A.
- [ ] `decision()` renvoie le bon palier pour au moins les trois zones (< 6, 6-11, ≥ 12).
- [ ] Le rationale liste explicitement les modulateurs appliques.
