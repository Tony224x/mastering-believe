# Solutions Medium — Module 04 : Heuristiques & Biais Cognitifs

---

## Exercice 1 — Quantifier et neutraliser l'ancrage

**1. Estimation indépendante (calcul)**

```
Somme des lots (en série) : 3 + 4 + 2 + 5 + 3 = 17 semaines
Marge d'imprévu (+30 %)   : 17 × 1,30 = 22,1 semaines
=> Estimation indépendante ≈ 22 semaines
```

**2. Comparaison à l'ancre de 8 semaines**

```
Écart en semaines   : 22,1 − 8 = 14,1 semaines
Écart en %          : (22,1 − 8) / 22,1 ≈ 64 % en dessous
Direction du biais  : l'ancre (8) tire l'estimation VERS LE BAS (sous-estimation).
```

Si on avait gobé l'ancre, on aurait promis un délai de ~8 semaines pour un projet qui en demande ~22 — recette classique du dépassement de planning.

**3. Protocole de débiaisage (3 étapes, dont consider-the-opposite)**

1. **Estimer en aveugle** : produire son estimation chiffrée (somme des lots + marge historique) **avant** d'entendre toute valeur de référence — idéalement, demander qu'on ne cite aucun chiffre tant qu'on n'a pas posé le sien.
2. **Consider-the-opposite** : prendre l'ancre (« 8 semaines ») et chercher activement les arguments qui la **contredisent** : ce projet est-il plus complexe ? plus de lots ? historique de dépassement à +30 % ? Lister explicitement « pourquoi 8 serait faux ».
3. **Décider sur ses propres données** : ancrer la décision sur l'estimation indépendante (22 semaines) corrigée par la classe de référence, pas sur le chiffre entendu.

**Note épistémique** : l'ancrage est un effet **robuste et répliqué** (roue de loterie de Tversky & Kahneman : 20 points d'écart pour un nombre sans rapport). Ce n'est pas une anecdote — d'où l'intérêt d'un protocole systématique, pas d'un simple « rester vigilant ».

---

## Exercice 2 — Nommer le biais et appliquer la contre-mesure

| Vignette | Biais robuste | Contre-mesure | Justification |
|---|---|---|---|
| A — « il y en a peut-être 500 », réponses proches de 500 | **Ancrage** | Estimer en aveugle puis chercher des références (compter par couches) | Le premier chiffre lancé tire l'estimation vers lui, même arbitraire |
| B — test 90 % fiable, 2 % défectueux, conclusion « 90 % » | **Négligence du taux de base** | Calculer P(base) explicitement | On ignore la prévalence (2 %) au profit de la sensibilité (90 %) ; la vraie proba est bien < 90 % |
| C — « peu fiable » après 3 reportages de pannes | **Disponibilité** | Chercher les statistiques agrégées (taux de retour réels) | La saillance médiatique gonfle la fréquence perçue |
| D — ne retient que les benchmarks où X gagne | **Biais de confirmation** | Chercher l'argument réfutant le plus fort (benchmarks où X perd) | Collecte sélective des preuves favorables à la conviction de départ |
| E — « 95 % remboursés » préféré à « 1 sur 20 refusé » | **Cadrage** | Reformuler dans les deux sens (gain/perte) | 95 % remboursés = 5 % refusés = 1 sur 20 : information identique, présentation différente |

**Point clé** : les 5 vignettes mobilisent les 5 biais **robustes/répliqués** du cours — aucune ne repose sur un effet de priming social fragile. Pour la vignette B, on n'a pas besoin du chiffre exact pour répondre : il suffit de savoir que le taux de base de 2 % rend « 90 % » manifestement trop élevé.

---

## Exercice 3 — Construire les deux cadrages d'une même option

**1. Espérances (cadrage gain)**

```
Plan sûr    : 200 composants sauvés (certain)          => E = 200
Plan risqué : 1/3 × 600 + 2/3 × 0 = 200 + 0 = 200      => E = 200
=> Les deux plans ont la MÊME espérance : 200 composants sauvés.
```

**2. Réécriture en cadrage perte** (sur 600 composants au total)

```
Plan sûr    : 400 composants seront perdus à coup sûr.            (600 − 200 = 400)
Plan risqué : 1/3 de chance qu'AUCUN ne soit perdu,
              2/3 de chance que les 600 soient perdus.
Espérance des pertes : 1/3 × 0 + 2/3 × 600 = 400 perdus  (équivaut à 200 sauvés)
```

Le cadrage gain et le cadrage perte décrivent **exactement la même réalité** : « 200 sauvés » ⇔ « 400 perdus » ; « 1/3 de sauver 600 » ⇔ « 1/3 de ne rien perdre ».

**3. Renversement de préférence**

```
Cadrage GAIN  : la majorité choisit le PLAN SÛR (sauver 200 à coup sûr).
Cadrage PERTE : la majorité bascule vers le PLAN RISQUÉ.
```

**Explication (aversion à la perte)** : en cadrage gain, les gens sont averses au risque face à un gain certain — « un tien vaut mieux que deux tu l'auras ». En cadrage perte, une perte certaine (400 perdus) est très douloureuse ; pour l'éviter, on devient **preneur de risque** et on tente le coup à 1/3. L'espérance est pourtant strictement identique : seule la formulation a changé. C'est le résultat robuste et répliqué du problème des 600 (Tversky & Kahneman, 1981).

**4. Contre-mesure** : avant toute décision chiffrée, **reformuler systématiquement l'option dans les deux cadrages** (gain ET perte). Si la préférence change selon la formulation alors que l'espérance est identique, c'est le signal qu'on se laisse piloter par le cadrage et non par les faits.
