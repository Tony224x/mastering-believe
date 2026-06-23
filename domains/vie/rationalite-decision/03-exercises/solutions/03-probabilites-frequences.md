# Solutions — Module 03 : Probabilités en fréquences naturelles

> Les calculs chiffrés sont produits ci-dessous. Le script `02-code/03-probabilites-frequences.py` permet de vérifier tout scénario avec `calculer_frequences(sensibilite, specificite, prevalence, population)`.

---

## Solution Exercice 1 — Capteur de détection de pannes

**Paramètres** : prévalence 3 %, sensibilité 92 %, spécificité 88 %, population 1 000.

**Étape 1 — Effectifs de base :**
- Machines en panne : 1 000 × 3 % = **30**
- Machines en bon état : 1 000 × 97 % = **970**

**Étape 2 — Appliquer sensibilité et spécificité :**
- VP (pannes détectées) : 30 × 92 % = **27,6**
- FN (pannes ratées) : 30 × 8 % = **2,4**
- VN (bons états rejetés) : 970 × 88 % = **853,6**
- FP (fausses alarmes) : 970 × 12 % = **116,4**

**Tableau des fréquences naturelles :**

|                      | En panne | Bon état  | Total  |
|----------------------|----------|-----------|--------|
| Alarme déclenchée    | 27,6     | 116,4     | 144    |
| Pas d'alarme         | 2,4      | 853,6     | 856    |
| **Total**            | 30       | 970       | 1 000  |

**Calculs :**
- **VPP** = 27,6 / 144 = **19,2 %**
- **VPN** = 853,6 / 856 = **99,7 %**

**Interprétation pour le responsable :** « Quand le capteur déclenche une alarme, seulement 1 machine sur 5 environ est réellement en panne. Avant d'arrêter la production, une vérification manuelle rapide est justifiée. En revanche, l'absence d'alarme garantit à 99,7 % que la machine fonctionne. »

*Vérification avec le script :*
```
python 02-code/03-probabilites-frequences.py
# ou en interactif :
# from 03_probabilites_frequences import calculer_frequences
# r = calculer_frequences(0.92, 0.88, 0.03, 1000)
# → vpp ≈ 19.2 %
```

---

## Solution Exercice 2 — Filtre anti-spam

**Paramètres** : prévalence 15 %, sensibilité 98 %, spécificité 97 %, population 1 000.

**Effectifs de base :**
- Spams : 1 000 × 15 % = **150**
- Légitimes : 1 000 × 85 % = **850**

**Tableau (15 %) :**

|                      | Spam  | Légitime | Total |
|----------------------|-------|----------|-------|
| Classé spam          | 147   | 25,5     | 172,5 |
| Classé légitime      | 3     | 824,5    | 827,5 |
| **Total**            | 150   | 850      | 1 000 |

- VP = 150 × 98 % = 147 ; FP = 850 × 3 % = 25,5
- **VPP (15 %)** = 147 / 172,5 = **85,2 %**
- **VPN (15 %)** = 824,5 / 827,5 = **99,6 %**

**Tableau (1 %) sur 1 000 :**

|                      | Spam  | Légitime | Total |
|----------------------|-------|----------|-------|
| Classé spam          | 9,8   | 29,7     | 39,5  |
| Classé légitime      | 0,2   | 960,3    | 960,5 |
| **Total**            | 10    | 990      | 1 000 |

- VP = 10 × 98 % = 9,8 ; FP = 990 × 3 % = 29,7
- **VPP (1 %)** = 9,8 / 39,5 = **24,8 %**

**Conclusion :** Le même filtre, excellent à 15 % de prévalence (VPP 85 %), devient peu fiable à 1 % (VPP 25 %). Sur une adresse à faible volume de spam, les faux positifs dominent et des e-mails légitimes importants risquent d'être bloqués. Il faut adapter les seuils du filtre ou segmenter les boîtes par profil d'usage.

---

## Solution Exercice 3 — Dépistage en deux étapes

**Paramètres Test 1** : prévalence 2 %, sensibilité 85 %, spécificité 90 %, population 10 000.

### Étape A — Test 1

**Effectifs :**
- Atteints : 10 000 × 2 % = **200**
- Sains : 10 000 × 98 % = **9 800**

| | Atteint | Sain | Total |
|--|---------|------|-------|
| Test 1 positif | 170 | 980 | 1 150 |
| Test 1 négatif | 30  | 8 820 | 8 850 |
| **Total** | 200 | 9 800 | 10 000 |

- VP₁ = 200 × 85 % = 170 ; FP₁ = 9 800 × 10 % = 980
- **VPP₁** = 170 / 1 150 = **14,8 %**

### Étape B — Test 2 (sur les 1 150 positifs au Test 1)

Le taux de base pour le Test 2 est la VPP₁ = **14,8 %** (car parmi les 1 150 personnes qui passent le Test 2, 14,8 % sont réellement atteintes).

- Atteints parmi les positifs Test 1 : 1 150 × 14,8 % ≈ **170** (cohérent : ce sont nos VP₁)
- Sains parmi les positifs Test 1 : 1 150 × 85,2 % ≈ **980**

**Test 2 : sensibilité 99 %, spécificité 97 %**

| | Atteint | Sain | Total |
|--|---------|------|-------|
| Test 2 positif | 168,3 | 29,4 | 197,7 |
| Test 2 négatif | 1,7   | 950,6 | 952,3 |
| **Total** | 170 | 980 | 1 150 |

- VP₂ = 170 × 99 % = 168,3 ; FP₂ = 980 × 3 % = 29,4
- **VPP₂** = 168,3 / 197,7 = **85,1 %**

### Étape C — Comparaison

| Après... | VPP |
|----------|-----|
| Test 1 seul | 14,8 % |
| Test 2 (après Test 1 positif) | **85,1 %** |

Le deuxième test multiplie la VPP par ~6 en utilisant le résultat du premier comme nouveau taux de base.

### Étape D — Erreur de taux de base

Si un praticien ignorant le Test 1 recalcule la VPP du Test 2 avec la prévalence générale (2 %) :

- VP = 200 × 99 % = 198 ; FP = 9 800 × 3 % = 294
- VPP naïve = 198 / 492 = **40,2 %**

Écart : **40,2 % vs 85,1 %** — une sous-estimation massive de la probabilité d'être atteint.

**Leçon :** le médecin ou le système qui reçoit le patient doit connaître son historique de tests. Ignorer les tests antérieurs revient à ignorer une information qui a déjà mis à jour le taux de base. La VPP du premier test *est* le nouveau taux de base du second.
