# J14 - Exercice EASY - Identifier les sources du reality gap

## Objectif

Verifier que tu sais nommer et categoriser les composantes du reality gap, et choisir des parametres a randomiser.

## Consigne

Tu disposes d'un robot bras 6-DOF en simulation. Tu veux le deployer en reel.
Tu connais les phenomenes suivants :

1. La masse de l'effecteur termine a ete mesuree : 850 g en CAD vs 920 g en reel.
2. Le couple commande au moteur arrive avec un retard de 12 ms (boucle de controle ROS).
3. Les encodeurs articulaires ont un bruit gaussien d'ecart-type ~0.05 deg.
4. Les liens du bras flechissent sous charge (non modelise dans la sim qui les considere rigides).
5. La camera a un offset extrinseque inconnu de ±2 mm par rapport au modele CAD.
6. La gravite varie negligeablement (sim utilise 9.81, reel = 9.81).
7. La friction articulaire reelle est environ 3x plus elevee que celle declaree en sim.

**Pour chacun des 7 points** :

1. Indique si c'est un parametre du **dynamics gap** ou du **visual gap** (ou les deux, ou ni l'un ni l'autre).
2. Indique si on peut le couvrir par **domain randomization** (oui/non).
3. Si oui, propose une plage `[min, max]` realiste autour de la valeur nominale (justifie pourquoi pas plus large).
4. Si non, explique brievement comment l'aborder (sysid ? amelioration du modele ? acceptation du risque ?).

Ensuite, donne **l'ordre de priorite** des 7 elements pour un sim-to-real reussi : sur lequel investirais-tu en premier ?

## Criteres de reussite

- Les 7 phenomenes sont classifies dynamics/visual.
- Pour chaque parametre randomisable, plage et justification sont presentes.
- L'ordre de priorite est argumente (pas juste enumere).
- Le point 4 (flexibilite des liens) est correctement identifie comme **non couvert par domain randomization** (modele incomplet, pas un parametre).
- Le point 6 (gravite) est correctement identifie comme negligeable.
