# Exercices — Module 03 : Probabilités en fréquences naturelles

> **Niveau** : Easy → Medium → Hard (progressif)
> **Méthode** : construire le tableau des fréquences naturelles pour chaque exercice avant de calculer quoi que ce soit.

---

## Exercice 1 — Capteur de détection de pannes

### Objectif
Construire un tableau de fréquences naturelles et calculer la VPP dans un contexte industriel neutre.

### Consigne
Une usine surveille ses machines avec un capteur électronique. Sur l'ensemble du parc, **3 % des machines** présentent une panne à un instant donné (taux de base).

Le capteur a une **sensibilité de 92 %** (il détecte 92 % des machines réellement en panne) et une **spécificité de 88 %** (il rejette correctement 88 % des machines en bon état).

1. Sur une population fictive de **1 000 machines**, remplissez le tableau suivant :

   |               | En panne | Bon état | Total |
   |---------------|----------|----------|-------|
   | Alarme déclenchée | ?    | ?        | ?     |
   | Pas d'alarme   | ?       | ?        | ?     |
   | **Total**      | ?       | ?        | 1 000 |

2. Calculez la **VPP** : si le capteur déclenche une alarme, quelle est la probabilité que la machine soit réellement en panne ?

3. Calculez la **VPN** : si aucune alarme n'est déclenchée, quelle est la probabilité que la machine soit vraiment en bon état ?

4. En une phrase, interprétez le résultat pour un responsable de maintenance qui reçoit une alarme.

### Critères de réussite
- [ ] Le tableau est correctement rempli (valeurs arrondies acceptées)
- [ ] VPP calculée avec la formule VP / (VP + FP), résultat entre 15 % et 25 %
- [ ] VPN supérieure à 99 %
- [ ] L'interprétation mentionne que la majorité des alarmes sont des faux positifs et recommande une vérification manuelle

---

## Exercice 2 — Filtre anti-spam

### Objectif
Appliquer les fréquences naturelles à un domaine informatique et observer l'effet du taux de base sur la VPP.

### Consigne
Un filtre anti-spam classe les e-mails entrants. Dans la boîte mail d'un utilisateur professionnel, **15 % des messages** sont des spams (taux de base).

Le filtre a une **sensibilité de 98 %** (il détecte 98 % des vrais spams) et une **spécificité de 97 %** (il laisse passer 97 % des messages légitimes sans les bloquer).

1. Sur **1 000 e-mails**, construisez le tableau des fréquences naturelles.

2. Calculez la VPP : si un e-mail est classé « spam » par le filtre, quelle est la probabilité qu'il soit réellement un spam ?

3. Calculez la VPN : si un e-mail passe le filtre, quelle est la probabilité qu'il soit légitime ?

4. **Question de comparaison** : si le même filtre était déployé sur une boîte où seulement **1 % des messages** seraient des spams (ex. adresse interne d'entreprise), que deviendrait la VPP ? Calculez-la.

5. Que concluez-vous sur l'importance du taux de base pour choisir où déployer un filtre ?

### Critères de réussite
- [ ] Tableau correct pour le scénario à 15 %
- [ ] VPP (15 %) correctement calculée, résultat entre 85 % et 90 %
- [ ] VPP recalculée pour 1 % correctement, résultat entre 20 % et 30 %
- [ ] Conclusion indique que le même filtre donne des résultats très différents selon le contexte de déploiement

---

## Exercice 3 — Dépistage en deux étapes

### Objectif
Comprendre comment chaîner deux tests utilise la VPP du premier test comme nouveau taux de base pour le second.

### Consigne
Un programme de dépistage utilise deux tests successifs pour détecter une condition touchant **2 % de la population**.

**Test 1 (screening)** — rapide et peu coûteux :
- Sensibilité : 85 %
- Spécificité : 90 %

**Test 2 (confirmation)** — plus précis, utilisé uniquement sur les personnes positives au Test 1 :
- Sensibilité : 99 %
- Spécificité : 97 %

**Étape A** : Sur 10 000 personnes, calculez le tableau du Test 1. Quelle est la VPP après le Test 1 ?

**Étape B** : Parmi les personnes positives au Test 1 (total de la colonne « Test positif »), appliquez le Test 2. La prévalence pour cette deuxième analyse est la **VPP du Test 1** (car seuls les positifs au Test 1 subissent le Test 2). Calculez le tableau du Test 2 sur cette sous-population.

**Étape C** : Quelle est la VPP finale après les deux tests ? Comparez avec la VPP après le Test 1 seul.

**Étape D** : Si un médecin voyait directement un patient positif au Test 2 sans connaître l'histoire du Test 1, il utiliserait « 2 % » (prévalence générale) comme taux de base pour interpréter le Test 2. Quel serait l'écart par rapport à la VPP réelle calculée en Étape C ? Que nous apprend cet écart ?

### Critères de réussite
- [ ] Tableau Test 1 correct sur 10 000 personnes ; VPP Test 1 entre 13 % et 17 %
- [ ] Tableau Test 2 utilise bien la VPP du Test 1 (et non 2 %) comme taux de base
- [ ] VPP finale (après deux tests) supérieure à 80 %
- [ ] L'écart calculé en Étape D est identifié comme une erreur de taux de base, et le candidat nomme que l'information sur les tests antérieurs doit être transmise au praticien suivant
