# Exercices — Bases de l'investissement : risque, rendement, diversification (Module 05)

> **Niveau** : Debutant | **Temps estime** : 40-55 min
>
> **Matiere premiere** : Theorie du Module 05 + simulation `02-code/05-bases-investissement.py`
>
> **Disclaimer** : exercices educatifs. Les taux et volatilites utilises sont illustratifs ; aucun resultat ne constitue une promesse de rendement. Tout investissement comporte un risque de perte en capital.

---

## Exercice 1 — Classer le couple risque/rendement

### Objectif
Ancrer le lien indissociable entre risque (volatilite) et rendement espere, et reconnaitre un profil de classe d'actifs.

### Consigne

On vous presente quatre placements (chiffres illustratifs) :

| Placement | Rendement espere | Volatilite |
|---|---|---|
| W | 2 % | 1 % |
| X | 4 % | 6 % |
| Y | 7 % | 18 % |
| Z | 9 % | 0 % |

1. Classez W, X, Y du moins au plus risque, et verifiez que leur rendement espere suit le meme ordre.
2. A quelle **classe d'actifs** (liquidites / obligations / actions) chacun de W, X, Y correspond-il le plus vraisemblablement ? Justifiez en une ligne.
3. Le placement Z (9 % espere, 0 % de volatilite) est mis en avant par un demarcheur. Que devez-vous en penser, et pourquoi (un mot-cle du module est attendu) ?

### Criteres de reussite

- [ ] Le classement de W, X, Y est correct et l'ordre rendement/risque est explicitement constate
- [ ] Chaque placement est rattache a une classe d'actifs plausible avec justification
- [ ] La reponse sur Z identifie l'incoherence "rendement eleve + zero risque" comme un signal d'alarme (lien avec la prime de risque)

---

## Exercice 2 — Voir le risque avec la simulation

### Objectif
Comprendre, chiffres en main, qu'une volatilite plus elevee elargit la fourchette des resultats meme a rendement espere egal.

### Consigne

Lancez `python 02-code/05-bases-investissement.py` et observez la **DEMO 1**.

1. Relevez, pour chaque niveau de volatilite (3 %, 10 %, 20 %), la **mediane**, le **p10** (pire 10 %) et le **p90** (meilleur 10 %).
2. Calculez pour chaque ligne l'**amplitude** = p90 − p10. Comment evolue-t-elle quand la volatilite monte ?
3. Le placement le plus volatil a-t-il une mediane plus haute ou plus basse que le moins volatil ? Expliquez pourquoi en une phrase (indice : la composition de rendements tres dispersés "mange" de la performance — pensez a une annee −20 % suivie d'une annee +20 %).
4. En une phrase : pourquoi un epargnant qui aura besoin de son argent dans 1 an devrait-il fuir le profil "20 % de volatilite" ?

### Criteres de reussite

- [ ] Les trois lignes (mediane, p10, p90) sont relevees depuis la sortie reelle du script
- [ ] L'amplitude p90 − p10 est calculee et son augmentation avec la volatilite est constatee
- [ ] La reponse a la question 3 mentionne l'effet de la composition / dispersion
- [ ] La reponse a la question 4 relie volatilite + horizon court = risque de vendre au mauvais moment

---

## Exercice 3 — Le "repas gratuit" de la diversification

### Objectif
Constater que combiner des actifs peu correles reduit le risque sans toucher au rendement espere, et comprendre la limite (risque de marche).

### Consigne

Toujours avec le script, observez la **DEMO 2**.

1. Comparez "1 seul actif" et "30 actifs, faible correlation" : relevez le p10 et le p90 de chacun. La fourchette se resserre-t-elle ? De combien (en euros) le p10 s'ameliore-t-il ?
2. Le rendement espere de chaque actif est le meme partout (7 %). Pourquoi est-ce le point cle de la demonstration de Markowitz ?
3. Comparez "30 actifs, faible correlation" et "30 actifs, forte correlation (0,8)". Pourquoi la diversification aide-t-elle beaucoup moins dans le second cas ? Quel type de risque reste **non diversifiable** ?
4. En une phrase : la diversification a-t-elle protege contre TOUTES les baisses possibles ? Justifiez.

### Criteres de reussite

- [ ] Les p10/p90 des deux configurations sont releves et le resserrement est chiffre
- [ ] La reponse explique que le rendement espere constant isole l'effet "reduction de risque pur"
- [ ] La distinction risque specifique (diversifiable) vs risque de marche (non diversifiable) est correctement faite pour le cas correle
- [ ] La reponse finale rappelle que la diversification reduit mais n'elimine pas le risque
