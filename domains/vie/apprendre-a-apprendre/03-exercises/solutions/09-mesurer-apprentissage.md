# Solutions — Module 09 : Mesurer son apprentissage

---

## Exercice 1 — Calculer et interpréter un delta pré/post

### Étape 1 — Calcul

**Alice :**
- Delta pré/post = 9 − 2 = **+7 questions** = **+70 pp** (de 20 % à 90 %)
- Taux de rétention J+7 = 8/10 × 100 = **80 %**
- Chute post → J+7 = 90 % − 80 % = **−10 pp**

**Bob :**
- Delta pré/post = 9 − 2 = **+7 questions** = **+70 pp** (de 20 % à 90 %)
- Taux de rétention J+7 = 3/10 × 100 = **30 %**
- Chute post → J+7 = 90 % − 30 % = **−60 pp**

### Étape 2 — Interprétation

Les deux apprenants ont un delta pré/post identique (+70 pp). Pourtant, leurs profils sont radicalement différents à J+7.

**Alice** : rétention de 80 % à J+7. Ce chiffre est supérieur aux 61 % mesurés par Roediger & Karpicke (2006) pour le groupe "test". Alice a probablement effectué une révision espacée ou du retrieval practice entre le post-test et J+7, consolidant la mémoire à long terme.

**Bob** : rétention de 30 % à J+7. Chute de 60 pp en une semaine. C'est cohérent avec la courbe d'Ebbinghaus pour du contenu non révisé (oubli de ~67 % en 7 jours). Bob a très probablement utilisé uniquement la relecture ou n'a pas révisé du tout après le post-test. Le post-test élevé reflétait la **retrieval strength** (mémoire de travail fraîche), pas la **storage strength** (ancrage à long terme).

**Enseignement central** : le delta pré/post mesure l'acquisition immédiate lors de la session. Un score post-test élevé peut disparaître en quelques jours sans révision espacée. Le delta pré/post seul est insuffisant — il doit être couplé au taux de rappel à J+7 (et au-delà) pour valider l'apprentissage.

### Étape 3 — Recommandation à Bob

Action concrète : programmer une révision espacée du Module 03 dans les 24 heures (J+1 à partir du post-test), puis une deuxième révision à J+6. Lors de chaque révision, utiliser du retrieval practice (blank-page recall ou flashcards) — pas de relecture passive. Mesurer le taux de rappel à chaque révision et noter le résultat dans un journal.

---

## Exercice 2 — Construire et calibrer son journal de métriques

### Réponses modèles aux cinq questions du pré-test

1. **Taux de rappel vs fluency** : le taux de rappel mesure la capacité à restituer sans aide (test objectif, mesure de storage strength). La fluency est la facilité ressentie pendant la relecture (subjectif, mesure de retrieval strength à court terme — trompeuse car elle ne prédit pas la rétention à long terme).

2. **Ce que mesure le delta pré/post** : le gain d'apprentissage immédiat lors d'une session d'étude. Il indique ce que la session a apporté en termes d'acquisition, mais ne garantit pas la rétention consolidée à J+7 ou J+30.

3. **Trois éléments d'un bon feedback formatif (Black & Wiliam, 1998)** : (a) rapide (dans la session ou le lendemain), (b) spécifique (indique précisément ce qui est faux et pourquoi), (c) actionnable (pointe vers une correction concrète). *Note : le module cite aussi "fréquent" comme quatrième caractéristique — 3 des 4 éléments suffisent pour la réponse complète.*

4. **Score de calibration simple** : |score prédit (%) − score réel (%)| en points de pourcentage. Plus l'écart est proche de 0, meilleure est la calibration. Un écart > 15 pp indique une surestimation ou sous-estimation forte.

5. **Goodhart's Law appliquée aux métriques** : quand une mesure devient un objectif, elle cesse d'être une bonne mesure. En apprentissage : optimiser son taux de rappel en ne révisant que les questions faciles donne un score élevé mais un apprentissage superficiel. La métrique sert d'outil de feedback, pas de finalité.

### Exemple de journal rempli (données fictives pour illustration)

```
Contenu      : Module 09 — Mesurer son apprentissage
Date         : J0
Pré-test     : 2/5  (40 %)
Score prédit : 3/5  (60 %)
Post-test    : 4/5  (80 %)
Delta        : +2 questions  (+40 pp)
Calibration  : |60 % − 40 %| = 20 pp  (surestimation modérée du pré-test)
Prédiction J+7 : 65 %
```

**Ce qu'illustre cet exemple** : l'apprenant a surestimé son pré-test de 20 pp — signe courant de fluency illusion (il pensait reconnaître le contenu d'une lecture précédente mieux qu'il ne le retenait vraiment). Le delta post-test est fort (+40 pp), mais la prédiction à J+7 devra être vérifiée.

---

## Exercice 3 — Analyser une courbe d'oubli et concevoir un plan de révision

### Étape 1 — Analyse

**Chute la plus forte** : entre J+0 et J+1 : 85 % → 72 % = **−13 pp**. C'est le premier intervalle — la chute initiale la plus rapide, classique dans les courbes d'oubli sans révision.

*Vérification sur tous les intervalles :*
- J+0 → J+1 : −13 pp (sur 1 jour)
- J+1 → J+3 : −14 pp (sur 2 jours) ← légèrement plus forte en absolu
- J+3 → J+7 : −17 pp (sur 4 jours) ← plus forte si on compare les valeurs brutes
- J+7 → J+14 : −11 pp (sur 7 jours)
- J+14 → J+30 : −9 pp (sur 16 jours)

La chute absolue la plus forte est entre J+3 et J+7 (−17 pp). En termes de vitesse de chute (pp/jour), c'est la période J+0 → J+1 qui est la plus rapide. Les deux réponses sont acceptables si la justification est cohérente.

**Comparaison avec Roediger & Karpicke (2006)** : à J+7, l'apprenant a 41 %, soit **20 pp sous les 61 %** mesurés pour le groupe "test" (qui avait fait du retrieval practice actif). L'écart suggère que cet apprenant n'a pas utilisé de retrieval practice lors de son apprentissage initial — ou qu'il a simplement relu sans se tester, ce qui produit une rétention bien plus faible.

**Comparaison avec Ebbinghaus théorique** :
- J+1 théorique ≈ 58 % ; mesuré = 72 % → **au-dessus** de la courbe théorique.
- J+7 théorique ≈ 33 % ; mesuré = 41 % → **au-dessus** de la courbe théorique.

Inférence : cet apprenant retient mieux que la moyenne d'Ebbinghaus, probablement parce que la courbe d'Ebbinghaus a été établie sur des syllabes sans sens (nonsense syllables, 1885) alors que le Module 02 est du contenu conceptuel significatif — la mémoire sémantique est plus résistante que la mémoire de syllables arbitraires.

### Étape 2 — Plan de révision

Objectif : maintenir ≥ 70 % jusqu'à J+30.

| Révision | Intervalle | Métrique à mesurer après |
|----------|-----------|--------------------------|
| R1 | J+1 (dans les 24h) | Taux de rappel sur 10 questions sans aide (cible : ≥ 75 %) |
| R2 | J+4 (3 jours après R1) | Taux de rappel (cible : ≥ 72 %) |
| R3 | J+10 (6 jours après R2) | Taux de rappel (cible : ≥ 70 %) |
| R4 | J+24 (14 jours après R3) | Taux de rappel (cible : ≥ 70 %) |

*Règle d'ajustement* : si le taux de rappel lors d'une révision est < 60 %, avancer la révision suivante d'un facteur 0.5 (ex : intervalle prévu 7 jours → le ramener à 3-4 jours). Si ≥ 85 %, allonger de 1.2 (logique proche de SM-2, Module 03).

**Méthode de mesure** : sur chaque révision, se tester en blank-page recall (écrire ce qu'on sait sans notes) sur les 5 concepts-clés du module, puis comparer avec le cours. Compter les réponses complètes et correctes. Ne pas confondre avec la relecture — la mesure doit être en mode test, pas en mode reconnaissance.

### Étape 3 — Facteurs personnels de variabilité

Un facteur parmi les suivants est attendu :
- **Niveau initial** : un apprenant avec des connaissances préalables solides en psychologie cognitive aura une courbe de déclin plus lente que quelqu'un qui découvre le domaine.
- **Qualité du sommeil** entre l'apprentissage et les mesures : le sommeil est un consolidateur de mémoire — une mauvaise nuit entre J+0 et J+1 accélère la chute (Schmid et al., 2022 — lien débattu, mais convergence de la littérature).
- **Profondeur de l'encodage initial** : si l'apprenant a fait de l'élaboration (Module 05) ou du retrieval practice (Module 02) pendant l'apprentissage, sa courbe de départ sera plus haute et déclinera moins vite.
- **Nature du contenu** : contenu très conceptuel (SM-2, formules) → chute plus rapide. Contenu narratif ou lié à une expérience personnelle → chute plus lente (effet de la mémoire épisodique).

**Ce que ce corrigé illustre** : les métriques ne sont pas des vérités universelles — elles doivent être lues dans le contexte de l'apprenant, de son historique de révision, et de la nature du contenu. C'est la valeur de mesurer sa propre courbe plutôt que de se fier à la courbe théorique d'Ebbinghaus.
