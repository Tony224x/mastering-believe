# Exercices — Module 14 : Capstone — La Boîte à Outils de Jugement

> Ces trois exercices constituent le livrable complet du capstone. Ils se font **sur la même décision neutre et concrète**, choisie par l'apprenant dès l'Exercice 1. Les solutions commentées sont dans `03-exercises/solutions/14-capstone-boite-outils-jugement.md`.

---

## Exercice 1 — Checklist + mini-arbre de décision

### Objectif

Appliquer la checklist de pré-décision (Module 07) et construire un mini-arbre de décision (Module 07) sur une décision personnelle neutre.

### Consigne

**Étape 1 — Choisir votre décision.**
Choisissez une décision réelle et neutre à prendre dans les 2 prochaines semaines. Elle doit comporter au moins deux options et un événement incertain.
Exemples valides : choisir entre deux itinéraires, planifier une tâche sur deux créneaux possibles, sélectionner entre deux équipements, décider d'une formation parmi deux.
Exemples à éviter : décisions politiques, religieuses, identitaires.

**Étape 2 — Remplir la checklist complète.**

```
DÉCISION : ___________  |  DATE : ___________

1. CLARIFIER
   Décision exacte : ___________
   Options réelles (inclure « ne rien faire ») : ___________
   Horizon de temps : ___________

2. BIAIS
   Ancrage — estimation indépendante : ___________
   Disponibilité — taux de base historique : ___________
   Cadrage — reformulation en gains/pertes : ___________
   Confirmation — argument contre l'option favorite : ___________

3. PROBABILITÉS
   Scénario principal : _____%  |  Classe de référence : ___________
   Ajustements : ___________

4. CONSÉQUENCES
   Optimiste (___%) : ___________
   Central   (___%) : ___________
   Pessimiste(___%) : ___________
   Scénario ruineux : ___________

5. VÉRIFICATION
   Fait clé 1 vérifié via : ___________
   Fait clé 2 vérifié via : ___________

6. DÉCISION & SUIVI
   Décision : ___________  |  Probabilité succès : _____%  |  Date revue : ___________
```

**Étape 3 — Construire le mini-arbre de décision.**
Pour chaque option principale, dessiner (ou écrire en texte) les branches probabilistes et calculer l'espérance.

```
Option A ── [p = ___] ── Résultat A+ (valeur : ___)
         └─ [1-p = ___] ── Résultat A- (valeur : ___)
   Espérance(A) = ___ × ___ + ___ × ___ = ___

Option B ── [p = ___] ── Résultat B+ (valeur : ___)
         └─ [1-p = ___] ── Résultat B- (valeur : ___)
   Espérance(B) = ___ × ___ + ___ × ___ = ___

Option retenue d'après l'arbre : ___________
Nuance (aversion au risque, scénario ruineux) : ___________
```

### Critères de réussite

- [ ] Décision réelle, neutre et concrète (ni politique, ni identitaire).
- [ ] Toutes les sections de la checklist remplies.
- [ ] Au moins 2 biais examinés avec un argument concret (pas juste cochés).
- [ ] Probabilité chiffrée avec classe de référence explicite.
- [ ] Mini-arbre avec 2 options, probabilités et espérances calculées.
- [ ] Date de revue fixée dans le futur.

---

## Exercice 2 — Analyse de second ordre + journal de prévisions (≥ 10 entrées)

### Objectif

Prolonger la décision de l'Exercice 1 par une analyse de second ordre (Module 12) et ouvrir un journal de prévisions calibré mesuré au score de Brier (Module 08).

### Consigne

**Étape 1 — Analyse de second ordre sur la décision de l'Exercice 1.**

Remplir le template suivant :

```
ANALYSE DE SECOND ORDRE — [même décision]

Effets de 1er ordre (< 1 semaine) :
  →

Effets de 2e ordre (1 semaine – 3 mois) :
  →

Effets de 3e ordre (> 3 mois, boucles) :
  →

Boucle renforçante principale :
  →

Boucle équilibrante principale (ce qui va freiner l'effet) :
  →

Point de levier identifié :
  →
```

**Étape 2 — Formuler et enregistrer ≥ 10 prédictions.**

Ouvrir un journal de prévisions sur votre décision et sur des événements neutres proches. Les 10 prédictions peuvent porter sur : la décision elle-même (« est-ce que cette option aura l'effet escompté dans 3 mois ? »), des événements quotidiens (météo, livraisons, délais de tâches, scores de compétitions sportives), ou des tendances mesurables (retards, taux de complétion).

| # | Date | Question binaire + date résolution | p (%) | Classe de référence | Outcome | (p−o)² | Note |
|---|------|------------------------------------|--------|---------------------|---------|--------|------|
| 1 | | | | | | | |
| 2 | | | | | | | |
| … | | | | | | | |
| 10| | | | | | | |

**Étape 3 — Calculer le score de Brier (après résolution des 10 prédictions).**

```
Score de Brier = Σ (p−o)² / 10 = _____ / 10 = _____

Zone de sur-confiance identifiée (prévu X %, réalité Y %) : ___________
Zone de sous-confiance identifiée : ___________
```

### Critères de réussite

- [ ] Analyse de second ordre : 3 niveaux d'effets remplis (1er, 2e, 3e ordre).
- [ ] Au moins 1 boucle renforçante et 1 boucle équilibrante identifiées.
- [ ] 10 prédictions formulées avec questions binaires et dates de résolution.
- [ ] Classes de référence renseignées pour ≥ 7 prédictions sur 10.
- [ ] Score de Brier calculé après résolution.
- [ ] Zone de sur- ou sous-confiance identifiée et commentée.

---

## Exercice 3 — Protocole SIFT + synthèse portfolio

### Objectif

Vérifier une information clé de la décision avec le protocole SIFT (Module 11) et rassembler les cinq pièces de la boîte à outils dans une synthèse lisible par une tierce personne.

### Consigne

**Étape 1 — Appliquer le protocole SIFT sur 1 information clé.**

Identifiez une information externe qui a pesé dans votre décision (un taux, une statistique, une affirmation, une recommandation). Documentez chaque étape :

```
Information à vérifier : ___________
Source d'origine : ___________

S — STOP : ai-je pris une pause avant de l'accepter ?  OUI / NON
  Observation : ___________

I — INVESTIGATE THE SOURCE
  Qui publie ? ___________
  Lecture latérale (2-3 onglets) : ___________
  Résultat : source fiable / douteuse / mixte → ___________

F — FIND BETTER COVERAGE
  Autres sources indépendantes consultées : ___________
  Convergence ou divergence ? ___________

T — TRACE TO ORIGINAL
  Source primaire retrouvée ? OUI / NON
  DOI ou URL primaire : ___________
  Date, auteur, contexte correspondent ? ___________

Verdict final : information confirmée / nuancée / non vérifiable
→ Impact sur la décision : ___________
```

**Étape 2 — Rédiger la synthèse portfolio (1 page).**

Rassemblez les cinq pièces en une page structurée :

```
=== BOÎTE À OUTILS DE JUGEMENT — [Décision] — [Date] ===

1. CHECKLIST (résumé 3 lignes) :
   Biais détectés : ___________
   Probabilité retenue : _____% | Classe de référence : ___________
   Décision : ___________

2. ARBRE DE DÉCISION (résumé) :
   Option A : Espérance = ___  |  Option B : Espérance = ___
   Choix retenu : ___________

3. SECOND ORDRE (résumé) :
   Effet 2e ordre principal : ___________
   Boucle clé : ___________

4. JOURNAL BRIER (résumé) :
   N prédictions : ___  |  Score de Brier : ___
   Zone à recalibrer : ___________

5. SIFT (résumé) :
   Information vérifiée : ___________
   Verdict : ___________  |  Impact sur la décision : ___________

LATTICEWORK — modèles activés ce capstone :
  → ___________
```

### Critères de réussite

- [ ] Protocole SIFT : 4 étapes (S, I, F, T) documentées avec résultats.
- [ ] Verdict SIFT explicite (confirmé / nuancé / non vérifiable) et impact sur la décision.
- [ ] Synthèse portfolio : les 5 pièces présentes et résumées.
- [ ] Synthèse lisible et comprise par quelqu'un n'ayant pas fait l'exercice.
- [ ] Mention des modèles mentaux activés (latticework personnel).
