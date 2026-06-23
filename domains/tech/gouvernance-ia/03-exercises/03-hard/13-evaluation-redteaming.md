# J13 — Exercice avance : boucle red-team -> eval et non-regression

## Objectif
Implementer la **boucle vertueuse** du module : un red-team decouvre une faille, on la fige en eval pour qu'elle ne reapparaisse jamais sans etre detectee (test de non-regression), puis on prouve que la correction tient ET ne casse pas les cas benins. C'est le passage de la mesure ponctuelle a un **regime de mesure continu et reproductible**.

## Consigne
1. Pars d'un garde-fou v1 base sur quelques motifs regex (volontairement incomplet) et d'un dataset d'eval initial (>= 8 cas, benins + attaques tagguees par categorie).
2. Ecris un mini **red-teamer** : une fonction `red_team(guardrail, seeds) -> list[found]` qui, a partir de quelques « graines » d'attaque, genere des **variantes** (par ex. synonymes : `ignore`/`disregard`/`forget`, ajout de bruit, changement de casse) et renvoie celles qui **passent** le garde-fou (donc des failles). La generation doit etre **deterministe** (pas d'aleatoire non seede) pour que l'exercice soit reproductible.
3. Pour chaque faille trouvee, **fige-la** : ajoute automatiquement un nouvel `EvalCase` (`expected=BLOCK`, bonne categorie) au dataset d'eval. C'est le verrou de non-regression.
4. Produis un garde-fou v2 qui corrige les failles trouvees (ajoute les motifs manquants).
5. Re-execute l'eval complet (dataset initial + cas figes) sur v1 **et** v2, et imprime un tableau comparatif : detection_rate et false_positive_rate de chaque version.
6. **Assertions de non-regression** (echec = `raise AssertionError`) :
   - v2 a un `detection_rate` strictement superieur a v1 ;
   - v2 ne degrade aucun cas benin (son `false_positive_rate` n'augmente pas par rapport a v1) ;
   - chaque faille figee a l'etape 3 est desormais `BLOCK` sur v2.
7. **Probe adverse finale** : tente une variante d'attaque *non couverte* par tes graines et montre que v2 peut encore la rater — conclusion : l'eval figee ne remplace pas un red-teaming continu (anti « teaching to the test »).

## Criteres de reussite
- [ ] `red_team` genere des variantes de maniere deterministe et retourne uniquement les entrees qui contournent le garde-fou.
- [ ] Chaque faille trouvee est automatiquement ajoutee au dataset d'eval (non-regression).
- [ ] L'eval est rejoue sur v1 et v2 avec un tableau comparatif des deux taux.
- [ ] Les trois assertions de non-regression sont presentes et passent pour la solution proposee.
- [ ] La probe finale illustre une faille residuelle hors du perimetre des graines (le red-teaming continu reste necessaire).
- [ ] Tout est deterministe et reproductible ; le script tourne sans erreur, stdlib seule.
