# Solution — Écrire avec l’IA

> Module `05-ecrire-avec-ia` · Badge : **Auteur** · Lisible **sans coder**.

Les fichiers `.py` du même nom sont des **clés techniques optionnelles** (smoke tests).

## Easy (chemin principal)

Voici un exemple qui marche : l’IA accélère le brouillon, **toi** qui restes l’auteur.

### Pipeline éthique (4 étapes)

1. **Outline** sans IA (ou avec, puis tu valides le plan)
2. Brouillon IA d’**une seule** section
3. Réécriture **≥ 30 %** à la main (surligne ou marque les passages touchés)
4. Sources : `[A_VERIFIER]` partout où tu n’as pas la preuve

### Outline type (PME fictive, innovation 30 jours)

1. Intro — enjeu PME  
2. Diagnostic  
3. Options d’innovation  
4. Plan 30 jours  
5. Risques & mesures  

**Interdit :** « écris mon devoir entier » en un seul prompt.


## Medium (bonus)

### Section 300–400 mots

Même pipeline ; contrainte explicite `[A_VERIFIER]` dans le prompt ; preuve de réécriture humaine visible (fichier barré / couleurs / commentaire « réécrit »).

| Règle | Valeur |
|-------|--------|
| Longueur max | 400 mots |
| Réécriture mini | 30 % |
| Marqueur facts | `[A_VERIFIER]` |


## Hard (bonus)

### Page complète + journal IA

Structure attendue :

- intro  
- 2 sections  
- conclusion  
- journal IA (prompts utilisés, ce qui a été gardé / refusé)

Sujet type : test d’innovation 30 jours pour une PME **fictive**.  
Réécriture ≥ 40 % · aucune source inventée.


## Clés structurées (rappel)

### easy

- **outline_example** :
  - Intro enjeu PME
  - Diagnostic
  - Options d’innovation
  - Plan 30 jours
  - Risques & mesures
- **forbidden** : devoir entier en un prompt

### medium

- **max_words** : 400
- **rewrite_min_pct** : 30
- **marker** : [A_VERIFIER]

### hard

- **structure** :
  - intro
  - 2 sections
  - conclusion
  - journal IA
- **topic** : test innovation 30 jours PME (fictif)
