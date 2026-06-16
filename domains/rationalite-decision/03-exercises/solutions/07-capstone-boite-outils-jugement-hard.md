# Solutions Hard — Module 07 : Capstone — La Boîte à Outils du Jugement

> Corrigés-types : exemplaires neutres complets, à imiter. Toute l'arithmétique du Brier ci-dessous est vérifiée.

---

## Exercice 1 — Protocole personnel de calibration sur 30 jours (exemple complet)

### Partie A — Conception

| Classe de référence | Taux de base estimé | Source / raisonnement |
|---------------------|---------------------|------------------------|
| Météo locale à 24h (pluie oui/non) | ~30 % de jours pluvieux | Moyenne saisonnière locale + bulletin la veille |
| Livraison annoncée « sous 48h » respectée | ~75 % | Historique perso des 20 derniers colis |
| Tâche planifiée finie à la date prévue | ~60 % | Mon propre suivi (optimisme de planification connu) |
| Équipe sportive suivie gagne | ~50 % | Bilan victoires/défaites de la saison |

**Cadence de log** : 1 prédiction/jour minimum, notée le matin pour résolution datée ; **probabilité fixée avant la résolution, jamais éditée ensuite** (règle anti-triche). Revue le 30e jour.

### Partie B — Collecte (20 prédictions, échantillon résolu)

| # | Classe | Question binaire | p | o | (p − o)² |
|---|--------|------------------|------|---|----------|
| 1 | Sport | Équipe suivie gagne J1 | 0,55 | 1 | 0,2025 |
| 2 | Sport | Équipe suivie gagne J2 | 0,50 | 0 | 0,2500 |
| 3 | Météo | Pas de pluie demain | 0,60 | 1 | 0,1600 |
| 4 | Tâche | Tâche A finie à temps | 0,55 | 0 | 0,3025 |
| 5 | Tâche | Tâche B finie à temps | 0,65 | 1 | 0,1225 |
| 6 | Livraison | Colis sous 48h | 0,70 | 1 | 0,0900 |
| 7 | Sport | Équipe suivie gagne J3 | 0,65 | 0 | 0,4225 |
| 8 | Météo | Pas de pluie demain | 0,60 | 1 | 0,1600 |
| 9 | Livraison | Colis sous 48h | 0,75 | 1 | 0,0625 |
| 10 | Tâche | Tâche C finie à temps | 0,80 | 0 | 0,6400 |
| 11 | Livraison | Colis sous 48h | 0,75 | 1 | 0,0625 |
| 12 | Météo | Pas de pluie demain | 0,70 | 1 | 0,0900 |
| 13 | Livraison | Colis sous 48h | 0,85 | 1 | 0,0225 |
| 14 | Tâche | Tâche D finie à temps | 0,90 | 0 | 0,8100 |
| 15 | Sport | Équipe suivie gagne J4 | 0,85 | 0 | 0,7225 |
| 16 | Météo | Pas de pluie demain | 0,80 | 1 | 0,0400 |
| 17 | Livraison | Colis sous 48h | 0,95 | 1 | 0,0025 |
| 18 | Météo | Pas de pluie demain | 0,90 | 1 | 0,0100 |
| 19 | Livraison | Colis sous 48h | 0,95 | 1 | 0,0025 |
| 20 | Tâche | Tâche E finie à temps | 0,90 | 0 | 0,8100 |

### Partie C — Revue mensuelle

**1. Score de Brier global.**
Somme des `(p − o)²` par bucket :
- Bucket 50-60 (# 1,2,3,4) : 0,2025 + 0,2500 + 0,1600 + 0,3025 = 0,9150
- Bucket 60-70 (# 5,6,7,8) : 0,1225 + 0,0900 + 0,4225 + 0,1600 = 0,7950
- Bucket 70-80 (# 9,10,11,12) : 0,0625 + 0,6400 + 0,0625 + 0,0900 = 0,8550
- Bucket 80-90 (# 13,14,15,16) : 0,0225 + 0,8100 + 0,7225 + 0,0400 = 1,5950
- Bucket 90-100 (# 17,18,19,20) : 0,0025 + 0,0100 + 0,0025 + 0,8100 = 0,8250

Somme totale = 0,9150 + 0,7950 + 0,8550 + 1,5950 + 0,8250 = **4,9850**.

**Score de Brier = 4,9850 / 20 = 0,249** (≈ baseline 0,25).

**2. Courbe de calibration par tranches** (déclaré moyen vs fréquence observée) :

| Bucket | n | Confiance déclarée (moy.) | Fréquence observée (moy. o) | Écart (déclaré − observé) |
|--------|---|---------------------------|------------------------------|----------------------------|
| 50-60 | 4 | 0,55 | 2/4 = 0,50 | +0,05 (bien calibré) |
| 60-70 | 4 | 0,65 | 3/4 = 0,75 | −0,10 (légère sous-confiance) |
| 70-80 | 4 | 0,75 | 3/4 = 0,75 | 0,00 (bien calibré) |
| 80-90 | 4 | 0,85 | 2/4 = 0,50 | **+0,35 (forte sur-confiance)** |
| 90-100 | 4 | 0,925 | 3/4 = 0,75 | **+0,175 (sur-confiance)** |

**3. Diagnostic.** Le score global ≈ 0,25 n'est pas tiré par le hasard mais par une **sur-confiance nette dans les buckets hauts** (80-90 et 90-100) : annoncer 85-92 % et n'observer que 50-75 %. Les prédictions « tâche finie à temps » à 80-90 % (# 10, 14, 20) sont les plus coûteuses → **optimisme de planification** classique. Les buckets bas/moyens sont sains.

### Partie D — Règle de recalibration (testable au mois suivant)

> « Pour toute prédiction que j'allais coter ≥ 0,80, **je décote de 15 points** (ex. 0,90 → 0,75), surtout sur la classe *tâche finie à temps*, tant que l'écart déclaré−observé du bucket 80-90 dépasse 0,10 à la revue. »

Test au mois M+1 : recalculer l'écart du bucket 80-90 ; si l'écart tombe sous 0,10, lever la décote.

---

## Exercice 2 — Cas intégré : checklist + journal + SIFT (synthèse 1 page)

**Décision** : remplacer un lave-vaisselle ancien par un modèle annoncé « **classe énergétique supérieure, ~40 % de consommation d'eau en moins** » (~500 €), ou conserver l'actuel.

### Outil 1 — Checklist de pré-décision

1. **Clarifier** : acheter le modèle économe (500 €) **vs** garder l'actuel (fonctionne, mais consomme plus). Horizon : amortissement sur la durée de vie restante.
2. **Biais examinés (2)** :
   - *Cadrage* : « 40 % d'eau en moins » sonne énorme ; reformulé en valeur absolue (litres/cycle × cycles/an × prix de l'eau), le gain annuel est modeste → le cadrage relatif gonfle l'attrait.
   - *Confirmation* : je veux justifier un achat neuf → je cherche activement l'argument **contre** : l'ancien marche, et 500 € paient beaucoup d'années de surconsommation.
3. **Probabilités** : `p(la décision se révèle bonne = économies réelles ≥ attendues sur 5 ans)` = **45 %**. Classe de référence : écarts fréquents entre consommation *en labo* (étiquette) et *en usage réel* des électroménagers.
4. **Conséquences / pessimiste** : gain réel < annoncé (usage réel, dureté de l'eau, programmes choisis) → amortissement repoussé au-delà de la durée de vie. Pas de scénario ruineux.
5. **Information clé à vérifier** : l'affirmation « **40 % de consommation d'eau en moins** ».
6. **Décision provisoire** : conditionnée à la vérification ci-dessous. Date de revue : +6 mois (relevé de conso réelle).

### Outil 2 — Journal de prévisions

> Prédiction binaire : « Sur mes 6 prochains mois d'usage, le nouveau lave-vaisselle consommera **au moins 25 % d'eau en moins** que l'ancien (mesuré sur cycles équivalents). »
> `p` = **40 %** | Date de résolution : **2026-12-16** | Indicateur observable : relevé compteur sur 10 cycles comparables avant/après.

(Note : je vise 25 % et non 40 % — décote assumée entre labo et réel, cohérente avec ma règle de recalibration.)

### Outil 3 — Vérification SIFT de l'affirmation « 40 % d'eau en moins »

- **S — Stop** : chiffre marketing sur la fiche produit ; ne pas le prendre pour argent comptant. Noter : source = page commerciale du vendeur.
- **I — Investiguer la source** : la comparaison « 40 % » se fait **par rapport à quoi ?** Lecture des petits caractères : « vs un modèle de référence de 2010 », pas vs mon appareil actuel ni vs la moyenne du marché 2026.
- **F — Find better coverage (lecture latérale)** : croiser avec la **base de données de l'étiquette énergie officielle** et des tests de magazines de consommateurs indépendants → l'écart *entre modèles récents* est bien plus faible que 40 %.
- **T — Trace** : remonter à la mesure source = **conditions de test normalisées (programme éco, charge pleine)**, qui ne correspondent pas forcément à mon usage (cycles courts, charges partielles).
- **Contrôle anti-hallucination** (si la fiche/chiffre vient d'un résumé généré par IA) : la « norme » et le « modèle de référence » cités existent-ils ? → vérifier que la référence réglementaire est réelle et que le pourcentage est sourçable, sinon le traiter comme non étayé.
- **Verdict calibré** : **partiellement étayé** — le gain existe mais l'ampleur « 40 % » dépend d'une base de comparaison favorable et de conditions de test idéalisées ; le gain *dans mon usage réel* est probablement plus proche de 15-25 %.

### Synthèse — rétroaction de la vérification sur la décision

Le verdict SIFT « partiellement étayé » **abaisse** la probabilité que les économies atteignent l'attendu : ma `p(décision bonne)` reste à **45 %** et ma prédiction-journal est cadrée à **25 %** (et non 40 %). Conséquence sur la décision : **ne pas remplacer un appareil fonctionnel sur la seule promesse marketing** ; déclencher l'achat uniquement si (a) l'ancien tombe en panne, ou (b) un test indépendant confirme un gain réel ≥ 25 % en usage type. Modules mobilisés : **probabilités/classe de référence** (3), **biais cadrage + confirmation** (2), **décision sous incertitude / pessimiste** (4), **calibration & Brier / décote labo→réel** (5), **vérification SIFT + anti-hallucination** (6).
