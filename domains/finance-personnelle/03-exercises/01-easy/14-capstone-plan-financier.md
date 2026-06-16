# Exercices — Module 14 : Capstone, gabarit de plan financier

> Ces exercices vous font manipuler le gabarit `02-code/14-capstone-plan-financier.py` : entrer vos chiffres, lire le reel vs le nominal, et tester les leviers. Le but est de comprendre les consequences, pas de produire un plan "correct".

> ⚠️ Contenu educatif. Le gabarit calcule a partir de vos hypotheses ; il ne recommande rien.

---

## Exercice 1 — Personnaliser le gabarit

### Objectif
Saisir vos chiffres (ou ceux d'un profil fictif) dans le bloc `=== VOS CHIFFRES ===` et lire correctement la sortie.

### Consigne
Utilisez ce profil fictif (ou le votre) en editant les variables en tete du fichier :
- Revenus nets 2 600 €/mois, depenses essentielles 1 500 €, non essentielles 400 €, remboursements dette 150 €.
- Fonds d'urgence actuel 800 €, cible 6 mois.
- Versement investi 350 €/mois, capital deja investi 2 000 €.

1. Lancez le programme. Quel **taux d'epargne** obtenez-vous ?
2. Quel est le **montant cible** du fonds d'urgence, et le manque ?
3. A l'horizon 30 ans, relevez le capital **nominal** ET le capital **reel**. De combien (en %) le reel est-il inferieur au nominal ?
4. Le programme vous a-t-il suggere une allocation ? Expliquez ce que represente le bloc 4.

### Criteres de reussite
- [ ] Taux d'epargne lu correctement depuis la sortie
- [ ] Cible et manque du fonds d'urgence releves (cible = depenses essentielles x 6)
- [ ] Capital nominal et reel a 30 ans releves, ecart en % calcule
- [ ] Reponse correcte : le bloc 4 affiche VOS proportions saisies, ce n'est pas une suggestion

---

## Exercice 2 — Tester un levier a la fois

### Objectif
Pratiquer la "deliberate practice" : isoler l'effet d'une hypothese.

### Consigne
En repartant du profil de l'exercice 1, faites varier **une seule variable a la fois** et notez l'effet sur le capital reel a 30 ans :

- **Variation A** : passez le versement mensuel de 350 € a 500 €.
- **Variation B** : remettez 350 €, mais passez `INFLATION_ANNUELLE_HYP` de 0.02 a 0.04.
- **Variation C** : remettez l'inflation a 0.02, mais baissez le rendement brut de chaque poche de 1 point.

1. Pour chaque variation, indiquez si le capital **reel** a 30 ans monte ou baisse, et commentez l'ampleur.
2. Lequel des trois leviers (versement, inflation, rendement) a, ici, l'effet le plus marquant sur le **reel** ? Que vous apprend-il ?
3. Pourquoi est-il pedagogiquement important de ne changer qu'**une** variable a la fois ?

### Criteres de reussite
- [ ] Les 3 variations testees, sens de variation du reel correct (A monte, B baisse, C baisse)
- [ ] Comparaison des amplitudes et identification du levier dominant dans CE cas
- [ ] Justification de la methode "une variable a la fois" (isoler la cause de l'effet)

---

## Exercice 3 — Lire la cible FIRE et ses limites

### Objectif
Interpreter le bloc 6 (FIRE optionnel) sans le confondre avec une promesse.

### Consigne
Avec `ACTIVER_CIBLE_FIRE = True` et le profil de l'exercice 1 (taux de retrait 4 %) :

1. Quelle est la **cible de capital** FIRE calculee, et d'ou vient le facteur (depenses / 0.04) ?
2. Quel **horizon estime** le programme affiche-t-il ? 
3. Citez les **deux limites** que le programme imprime sous la cible FIRE.
4. En 2-3 phrases : pourquoi cette cible est un "repere" et pas une "garantie" ? (mobilisez le Module 12)

### Criteres de reussite
- [ ] Cible = depenses annuelles / 0,04 = 25 x depenses annuelles, releve correct
- [ ] Horizon estime releve depuis la sortie
- [ ] Les deux limites citees (horizon ~30 ans + retrait indexe inflation ; risque de sequence de rendements non couvert)
- [ ] Distinction repere/garantie correctement formulee
