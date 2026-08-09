# Solution — Partenaire de réflexion

> Module `04-partenaire-reflexion` · Badge : **Miroir** · Lisible **sans coder**.

Les fichiers `.py` du même nom sont des **clés techniques optionnelles** (smoke tests).

## Easy (chemin principal)

Voici un exemple qui marche : l’IA pose des questions, **toi** qui décides.

### Session socratique (~15 min)

Prompt type (à coller) :

```
Rôle : tuteur socratique.
Contexte : je hésite entre [option A] et [option B] (anonyme, pas de noms réels).
Tâche : pose-moi UNE question à la fois (max 8). Ne donne pas ta recommandation finale.
Contraintes : pas de jargon ; pas de stats inventées.
```

**Livrable :** 6–8 échanges + **ta** décision en 3 phrases (pas celle de l’IA).

Critère de réussite : le résumé final est **corrigé à la main** ; aucun « plan de vie » imposé par le modèle.


## Medium (bonus)

### Avocat du diable — 7 axes

Demande 7 critiques dures, une par axe :

| Axe | Exemple de critique |
|-----|---------------------|
| marché | Qui paie vraiment ? |
| temps | Combien d’heures par semaine ? |
| compétences | Que dois-tu apprendre d’abord ? |
| argent | Quel budget mini réaliste ? |
| éthique | Y a-t-il un conflit d’intérêt ? |
| exécution | Quelle est la première action cette semaine ? |
| clarté | Peux-tu l’expliquer en 20 secondes ? |

Ensuite **toi** qui réponds à 3 critiques : `accepte` · `mitige` · `rejette` (+ une phrase).


## Hard (bonus)

### Cadre de décision (avant la suggestion IA)

1. Options (2–3) en une ligne chacune  
2. Critères pondérés (ex. : temps 40 %, impact 30 %, risque 30 %)  
3. Scores honnêtes (1–5)  
4. **Décision écrite**  
5. *Ensuite seulement* : relire / challenger avec l’IA  

Protocole type : cadre 5 lignes → 8+ questions → résumé IA → synthèse humaine → métriques.

Anti-pattern : accepter un plan de carrière tout fait à la 1ʳᵉ réponse.


## Clés structurées (rappel)

### easy

- **prompt_core** : une question a la fois ; pas de conseil avant Q5
- **success** : resume corrige manuellement ; pas de plan de vie impose par l’IA

### medium

- **seven_axes** :
  - marche
  - temps
  - competences
  - argent
  - ethique
  - execution
  - clarte
- **response_labels** :
  - accepte
  - mitige
  - rejette

### hard

- **protocol** :
  - cadre 5 lignes
  - 8+ questions
  - resume IA
  - synthese humaine
  - metriques
- **anti_pattern** : accepter un plan de carriere tout fait a la 1re reponse
