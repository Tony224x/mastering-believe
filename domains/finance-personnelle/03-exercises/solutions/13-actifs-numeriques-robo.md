# Solutions — Module 13 : Actifs numeriques et robo-advisors

> ⚠️ Contenu educatif, neutre. Les solutions exposent des faits (sources SEC/FINRA) et ne recommandent aucun achat ni aucune allocation.

---

## Exercice 1 — Couple risque/rendement et regle de bon sens

**1.** Un rendement passe de +300 % ne dit rien du rendement futur et, surtout, signale une **volatilite extreme** : un actif capable de +300 % est tout aussi capable de chutes profondes. "Argent facile" gomme le risque qui est l'autre face exacte de ce rendement (couple risque/rendement, Module 05).

**2.** Une volatilite ~75 % contre ~15-20 % signifie que les mouvements — donc les **pertes possibles** — sont d'un ordre de grandeur **plusieurs fois superieur** a celui d'un indice actions large. Intuitivement : des baisses de plusieurs dizaines de pourcent en peu de temps sont non seulement possibles mais ont ete observees a repetition. Ce n'est pas un accident, c'est une caracteristique structurelle a ce stade.

**3.** Regle citee par les regulateurs : **ne placer que ce que l'on peut se permettre de perdre entierement**. Concretement : si la perte totale du montant engage serait une catastrophe pour la situation de Lina (dette, fonds d'urgence entame, depenses essentielles), alors ce montant est trop eleve — le scenario de perte totale (custody, fraude, illiquidite) est reel, pas hypothetique.

**4.** **Non.** Le module — et cet exercice — exposent des faits pour comprendre l'instrument, sans dire s'il faut en detenir ni combien. Comprendre n'est pas recommander ; ne jamais en detenir est une position parfaitement legitime.

---

## Exercice 2 — Identifier les risques specifiques

| Cas | Risque principal | Source-type | Commentaire |
|---|---|---|---|
| **A** | **Fraude / Ponzi** | FINRA (et SEC) | "Rendement garanti, eleve, regulier, via parrainage" = signaux classiques d'arnaque. Reflexe FINRA : un rendement garanti et eleve est un signal d'alarme, quel que soit l'habillage technologique. |
| **B** | **Custody — perte de cles** | SEC (custody basics) | Garder ses cles soi-meme : leur perte = perte **definitive** des actifs, pas de "mot de passe oublie". |
| **C** | **Custody — faillite de plateforme** | SEC | Confier la garde a une plateforme tierce : en cas de faillite, on peut devenir simple **creancier** dans une procedure, voire tout perdre. |
| **D** | **Cadre reglementaire mouvant** | SEC / sources officielles locales | Le statut juridique/fiscal varie selon les juridictions et evolue ; renvoyer aux sources officielles locales a jour (Module 07). |

**Distinction cle (B vs C)** : garder ses cles soi-meme transfere le risque de "faillite de tiers" vers le risque de "perte/erreur personnelle" ; confier a une plateforme fait l'inverse. Aucune des deux options n'elimine le risque de custody — elle le deplace.

---

## Exercice 3 — Robo-advisors : benefice et limites

**1.** Un robo-advisor est un service de **conseil en investissement fourni par un algorithme**, avec interaction humaine limitee ou nulle (questionnaire -> allocation geree automatiquement, souvent indicielle, avec rebalancement).

**2.** Avantage principal : des **frais de gestion souvent plus bas** que ceux d'un conseiller humain. Cela compte car, sur le long terme, **les frais composes erodent fortement le rendement net** — la "tyrannie des couts" de Bogle (Module 06). Toutes choses egales par ailleurs, un cout plus bas ameliore le rendement net.

**3.** Deux limites (parmi celles soulignees par la SEC) :
- **Adequation (suitability)** : le conseil repose sur un questionnaire standardise qui peut **ne pas capturer** une situation complexe ou atypique.
- **Absence d'accompagnement humain** : en periode de krach, l'algorithme ne tient pas la main ; la discipline comportementale (Module 08) reste la responsabilite de l'investisseur.
- (Aussi acceptable : **transparence/divulgation** — il faut comprendre comment le service est remunere et quels conflits d'interets existent.)

**4.** **FAUX.** Un robo-advisor automatise une methode (souvent diversifiee, indicielle) a cout reduit, mais il **ne bat pas le marche** et **n'annule pas le risque**. Ses frais plus bas sont un atout mecanique ; ils ne creent pas de rendement gratuit. "Battre le marche sans risque" n'existe pas.
