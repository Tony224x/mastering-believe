# Solutions — Module 06 : Calibration & Vérification de l'Information

> Pour le calcul automatisé du score de Brier sur vos prédictions réelles, voir `02-code/06-calibration-verification.py`.

---

## Solution Exercice 1 — Score de Brier

**Calcul de (p − o)² pour chaque prédiction** :

| # | p | o | (p−o) | (p−o)² |
|---|---|---|-------|--------|
| 1 | 0,80 | 1 | −0,20 | **0,0400** |
| 2 | 0,70 | 0 | +0,70 | **0,4900** |
| 3 | 0,55 | 1 | −0,45 | **0,2025** |
| 4 | 0,90 | 1 | −0,10 | **0,0100** |
| 5 | 0,40 | 0 | +0,40 | **0,1600** |
| 6 | 0,20 | 0 | +0,20 | **0,0400** |
| 7 | 0,30 | 1 | −0,70 | **0,4900** |
| 8 | 0,15 | 0 | +0,15 | **0,0225** |

**Score de Brier global** :
```
Brier = (0,0400 + 0,4900 + 0,2025 + 0,0100 + 0,1600 + 0,0400 + 0,4900 + 0,0225) / 8
      = 1,4550 / 8
      = 0,182
```

**Comparaison à la baseline (0,25)** :
Score de **0,182 < 0,25** → l'apprenant est meilleur que le hasard. Mais il reste loin des superforecasters (≈ 0,10-0,15 sur des questions ouvertes).

**Prédiction la plus coûteuse** :
Prédictions #2 et #7, ex-aequo avec (p−o)² = **0,49**.

- **#2** (colis à temps : prédit 70 %, outcome 0) : sur-confiance. L'apprenant était trop certain de la livraison. La classe de référence (taux de livraison à temps pour ce transporteur) aurait dû tempérer la prédiction.
- **#7** (retard ce matin : prédit 30 %, outcome 1 = retard effectif) : sous-confiance sur un risque de retard. La classe de référence (fréquence habituelle des retards le matin pour cet apprenant) aurait suggéré une probabilité plus élevée.

*Leçon* : les erreurs les plus coûteuses en Brier ne sont pas les bonnes décisions (outcomes à 50/50) mais les sur- et sous-confiances fortes (prédir 10 % et avoir 1, ou 90 % et avoir 0).

---

## Solution Exercice 2 — Vérification d'une citation douteuse

**Application de SIFT**

**S — Stop** : avant de partager, marquer une pause. L'affirmation est spectaculaire (+15 points de QI en 3 mois). Cela devrait déclencher l'alerte.

**I — Investigate the source** : l'article est un blog (source non académique). Actions concrètes :
- Rechercher le nom du blog + "À propos" pour identifier l'auteur et ses qualifications.
- Chercher sur Google : `[nom du blog] site:scholar.google.com` pour voir si le blog est cité dans des travaux académiques.
- Quitter la page et ouvrir Google Scholar pour rechercher l'étude.

**F — Find better coverage** :
- Requête Google Scholar : `"meditation" "IQ" "randomized" 2022 Nature Medicine`
- Requête alternative : `"meditation cognitive" "IQ points" meta-analysis 2020 2021 2022 2023`
- Requête sceptique : `"meditation IQ" replication failure OR null result`

**T — Trace to original** :
- L'affirmation dit "Harvard, 2022, Nature Medicine". Vérifier sur doi.org ou PubMed.
- Rechercher exactement : `"Harvard" "meditation" "IQ" "Nature Medicine" 2022` sur PubMed (pubmed.ncbi.nlm.nih.gov).
- Chercher si l'étude existe dans les archives de *Nature Medicine* : nature.com/nm.

**3 signaux d'alerte spécifiques** :
1. **Effet de taille irréaliste** : +15 points de QI en 3 mois est un effet massif, sans précédent dans la littérature. Les méta-analyses sur la méditation et les fonctions cognitives donnent des effets faibles à modérés (d ≈ 0,3-0,6 sur des tests spécifiques, pas le QI global).
2. **"Harvard" sans auteur nommé** : une vraie étude dans *Nature Medicine* aurait des auteurs identifiables et un DOI. L'absence de nom d'auteur et de DOI est un signal fort de fabrication ou de paraphrase inexacte.
3. **"QI augmente"** : la littérature récente est très sceptique sur la possibilité d'augmenter le QI de façon durable par des interventions courtes. Le terme "QI" est souvent confondu avec des mesures d'attention ou de mémoire de travail à court terme.

**Protocole LLM spécifique** :
1. Copier le titre entre guillemets dans Google Scholar : `"meditation 10 minutes IQ 15 points 3 months"`
2. Si aucun résultat : chercher l'auteur ou le DOI mentionné.
3. Vérifier le DOI sur doi.org (entrer le numéro dans la barre de recherche).
4. Si la revue et l'année correspondent : lire le résumé et vérifier que les chiffres cités sont dans l'article.
5. Si rien n'est trouvable en 3 étapes : la citation est probablement hallucinée. Ne pas utiliser.

---

## Solution Exercice 3 — Journal de prévisions

Cet exercice est personnel et n'a pas de solution unique. Le corrigé ci-dessous donne un exemple de 5 prédictions bien formulées.

**Exemple de 5 prédictions bien formulées** :

```
Question : Pleuvra-t-il entre 8h et 10h à mon domicile le [date dans 3 jours] ?
Probabilité : 35 %
Classe de référence : Il a plu 7 des 20 derniers matins similaires en juin → taux de base 35 %.
Date de résolution : [date + 3 jours]

---

Question : Mon prochain colis (commandé aujourd'hui) sera-t-il livré avant [date dans 5 jours] ?
Probabilité : 65 %
Classe de référence : Ce transporteur livre dans les délais 7 fois sur 10 selon mes 10 dernières commandes.
Date de résolution : [date + 5 jours]

---

Question : L'équipe [X] gagnera-t-elle son match le [date dans 7 jours] ?
Probabilité : 55 %
Classe de référence : Cette équipe gagne à domicile 58 % du temps cette saison.
Date de résolution : [date du match]

---

Question : Je terminerai la lecture du livre Y avant [date dans 14 jours] ?
Probabilité : 70 %
Classe de référence : J'ai fini 6 des 8 derniers livres commencés dans ce délai → 75 %, légèrement réduit car livre plus long.
Date de résolution : [date + 14 jours]

---

Question : La réunion prévue le [date] commencera-t-elle dans les 5 premières minutes de l'heure prévue ?
Probabilité : 45 %
Classe de référence : Ces réunions démarrent à l'heure environ 40-50 % du temps dans mon expérience.
Date de résolution : [date de la réunion]
```

**Critères de qualité d'une bonne prédiction** :
- La question est fermée (oui/non, pas "à peu près").
- La date de résolution est fixe et non ambiguë.
- La probabilité est un chiffre, pas une catégorie.
- La classe de référence est citée explicitement (même approximativement).
