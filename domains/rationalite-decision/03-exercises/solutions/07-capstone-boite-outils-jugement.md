# Solutions — Module 07 : Capstone — La Boîte à Outils du Jugement

## Exercice 1 — Checklist de pré-décision (exemple rempli)

```
DÉCISION : Choisir entre formation Python (150 €) et formation SQL (120 €)
DATE : 2026-06-16

1. CLARIFIER
   Décision exacte : Laquelle acheter en premier pour progresser en data.
   Options réelles : Python (150 €, 20h), SQL (120 €, 15h), ne rien acheter.
   Horizon de temps : 3 mois pour finir.

2. BIAIS
   Ancrage : J'ai vu Python en premier → estimer indépendamment la valeur de SQL.
   Disponibilité : Un ami a adoré SQL → 1 témoignage ≠ représentatif. Taux de
     complétion MOOC data : 10-20 %.
   Cadrage : "150 € de formation" vs "compétence valorisée ~200 €/mois freelance".
   Confirmation : Argument contre Python en 1er → SQL est prérequis à 80 % des
     offres data entry-level.

3. PROBABILITÉS
   Finir la formation choisie dans 3 mois : 40 %
   Classe de référence : taux de complétion MOOC (MIT, 2019) = 5-15 %, ajusté +25 %
     car formation courte et payante.

4. CONSÉQUENCES
   Optimiste (50 %) : finie, compétence acquise, +valeur marché.
   Central (35 %) : à moitié faite, compétence partielle.
   Pessimiste (15 %) : abandonnée, 150 € perdus.
   Scénario ruineux : aucun (150 € = acceptable).

5. VÉRIFICATION
   Fait 1 : offres data entry-level qui demandent SQL → LinkedIn Jobs (lecture latérale).
   Fait 2 : avis formation Python → chercher "[plateforme] completion rate study".

6. DÉCISION & SUIVI
   Décision : SQL en premier.  |  Probabilité succès : 40 %  |  Date revue : 2026-09-16
```

---

## Exercice 2 — Journal de prévisions (10 prédictions, exemple)

Score de Brier cible illustratif : **0,150** (meilleur que baseline 0,25)

| Date | Question | p (%) | Résultat | (p−o)² | Note |
|------|----------|--------|---------|--------|------|
| J+0 | Pluie lundi matin | 75 | 1 | 0,0625 | OK |
| J+0 | Réunion annulée lundi | 20 | 0 | 0,04 | OK |
| J+1 | Finir tâche avant 17h | 65 | 1 | 0,1225 | OK |
| J+1 | Colis livré mardi | 80 | 0 | 0,64 | Sur-confiant |
| J+2 | Match gagné mercredi | 55 | 1 | 0,2025 | OK |
| J+2 | Retard métro > 5 min | 40 | 0 | 0,16 | OK |
| J+3 | Rapport relu avant vendredi | 70 | 1 | 0,09 | OK |
| J+4 | Appel client reporté | 30 | 0 | 0,09 | OK |
| J+5 | Météo > 25°C samedi | 60 | 1 | 0,16 | OK |
| J+6 | Finir livre ce week-end | 35 | 0 | 0,1225 | OK |

Somme (p−o)² = 1,69 | **Score de Brier = 1,69 / 10 = 0,169**

Zone de sur-confiance identifiée : prédictions à 70-80 % se réalisent moins souvent → recalibrer à 55-65 %.

---

## Exercice 3 — Protocole complet (exemple synthèse)

**Cas** : S'inscrire à un MOOC de machine learning à 200 €.

**Outil 1 — Checklist (court)** :
1. Décision : acheter ou non le MOOC ML Coursera.
2. Biais vérifié : disponibilité (ami enthousiaste = N=1). Taux complétion = 10 %.
3. Probabilité succès : 35 %. Classe de référence : MOOCs payants cours < 30h.
4. Scénario ruineux : aucun (200 € remboursable 30 jours).
5. Information à vérifier : "95 % des diplômés trouvent un emploi en 6 mois" (Coursera).
6. Décision : oui. Date revue : +3 mois.

**Outil 2 — Journal** : "Finirai ce MOOC avant le 2026-09-16" → p = 35 %.

**Outil 3 — Vérification** : "95 % des diplômés trouvent un emploi"
- S : pause, chiffre marketing.
- I : chercher "Coursera placement rate independent study" → résultats contradictoires.
- F : blog Wired + rapport CNBC → taux basés sur auto-déclarations, biais de survie.
- T : source originale = enquête interne non auditée.
- Conclusion : chiffre non vérifiable indépendamment → pondérer en conséquence.
