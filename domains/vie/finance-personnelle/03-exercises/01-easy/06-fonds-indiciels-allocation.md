# Exercices — Fonds indiciels et allocation (Module 06)

> **Niveau** : Debutant | **Temps estime** : 45-60 min
>
> **Matiere premiere** : Theorie du Module 06 + simulation `02-code/06-fonds-indiciels-allocation.py`
>
> **Disclaimer** : exercices educatifs. Aucun produit, emetteur ou ticker n'est recommande. Les taux sont illustratifs ; tout investissement comporte un risque de perte en capital, et les performances passees ne prejugent pas des performances futures.

---

## Exercice 1 — Chiffrer l'impact compose des frais

### Objectif
Mesurer concretement combien un point de frais coute sur un horizon long, et integrer que les frais sont un levier sous votre controle.

### Consigne

Lancez `python 02-code/06-fonds-indiciels-allocation.py` et observez la **DEMO 1** (capital initial 10 000 €, 200 €/mois, 7 % brut, 30 ans).

1. Relevez le capital net final pour 0,1 %, 0,5 % et 1,0 % de frais.
2. De combien d'euros, et de combien de %, le cas 1,0 % est-il en retard sur le cas 0,1 % ?
3. Le total reellement sorti de votre poche est de 82 000 €. Le surcout des frais (0,1 % → 1,0 %) represente combien d'**annees de versements** (a 2 400 €/an) ?
4. En une phrase : pourquoi dit-on que les frais sont "le levier le plus directement sous votre controle", contrairement au rendement futur ?

### Criteres de reussite

- [ ] Les trois capitaux nets sont releves depuis la sortie reelle du script
- [ ] Le surcout 0,1 % → 1,0 % est exprime en euros ET en %
- [ ] Le surcout est converti en nombre d'annees de versements (calcul montre)
- [ ] La reponse distingue ce qui est controlable (frais) de ce qui ne l'est pas (rendement futur)

---

## Exercice 2 — Construire et ajuster une allocation "3 fonds"

### Objectif
Manipuler une allocation, verifier qu'elle somme a 100 % et observer l'effet des poids sur le rendement espere pondere.

### Consigne

Observez la **DEMO 2** (allocation 40 / 40 / 20), puis modifiez les parametres de `demo_allocation_3_fonds()` dans le script.

1. Notez le rendement espere net de l'allocation 40 % actions dom. / 40 % actions int. / 20 % obligations.
2. Creez une allocation **plus prudente** (par ex. 30 / 20 / 50) et relancez : le rendement espere net monte-t-il ou baisse-t-il ? Pourquoi ?
3. Creez une allocation **plus dynamique** (par ex. 50 / 40 / 10) : meme question.
4. Tentez une allocation dont les poids ne somment PAS a 100 % (par ex. 50 / 40 / 20). Que se passe-t-il a l'execution, et pourquoi ce garde-fou est-il utile ?
5. En une phrase : laquelle de ces trois allocations conviendrait a un horizon de 3 ans, et laquelle a un horizon de 30 ans ? (Reliez au Module 05.)

### Criteres de reussite

- [ ] Le rendement espere net de l'allocation de base est releve
- [ ] Les variantes prudente et dynamique sont testees et l'effet sur le rendement espere est correctement explique (plus d'obligations = plus bas)
- [ ] Le declenchement de l'`assert` (poids ≠ 100 %) est observe et son utilite expliquee
- [ ] L'association horizon court → allocation prudente / horizon long → allocation dynamique est faite

---

## Exercice 3 — Lire SPIVA honnetement

### Objectif
Restituer le resultat actif vs passif avec sa nuance methodologique, et adopter le ton "les preuves suggerent" plutot que prescriptif.

### Consigne

Sans script, en vous appuyant sur la section 3 de la theorie :

1. Expliquez en 2-3 phrases l'**argument arithmetique** de Sharpe : pourquoi, *avant frais*, l'ensemble des gerants actifs obtient le rendement du marche, et ce qui en decoule *apres frais*.
2. Le rapport SPIVA indique souvent qu'environ 90 %+ des fonds actifs sous-performent sur ~20 ans. Expliquez la **nuance methodologique** : que mesure ce chiffre (fonds vs encours), et qu'observe-t-on si l'on pondere par les encours ?
3. Reformulez la conclusion du module en une phrase qui commence par "Les preuves suggerent..." (et non "Vous devez...").
4. Un ami affirme : "SPIVA prouve que la gestion active ne sert a rien, point." Corrigez-le en une phrase, en restant factuel et non militant.

### Criteres de reussite

- [ ] L'argument arithmetique (avant frais = marche ; apres frais = en moyenne en dessous) est correctement restitue
- [ ] La nuance fonds (equipondere ~92 %) vs encours (sous-performance reduite mais non inversee) est expliquee
- [ ] La conclusion est reformulee en posture "les preuves suggerent", sans injonction
- [ ] La correction de l'ami evite a la fois la militance pro-passif et le rejet de la nuance
