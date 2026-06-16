# Solutions — Module 14 : Capstone — La Boîte à Outils de Jugement

> Exemple de boîte à outils complète sur une décision **neutre** : choisir entre deux créneaux pour planifier une révision de projet logistique en équipe. La décision est banale — c'est intentionnel : la méthode s'applique à toutes les décisions, pas seulement aux grandes.

---

## Exercice 1 — Checklist + mini-arbre de décision (exemple rempli)

### Checklist

```
DÉCISION : Planifier la réunion de révision de projet en équipe
DATE : 2026-06-16

1. CLARIFIER
   Décision exacte : Choisir entre le créneau mardi 9h ou jeudi 14h pour la réunion
     de révision du projet logistique Q3 (3 personnes, 90 min).
   Options réelles : Mardi 9h / Jeudi 14h / Reporter à la semaine suivante.
   Horizon de temps : Décision à prendre avant vendredi 17h.

2. BIAIS
   Ancrage : j'ai vu « mardi 9h » en premier dans la proposition. Estimation
     indépendante : lequel a le meilleur taux de présence historique dans l'équipe ?
     → Vérifier l'agenda partagé sur les 3 derniers mois.
   Disponibilité : la dernière réunion du mardi matin a été annulée → ça ne veut pas
     dire que c'est systématique. Taux d'annulation historique : 1/8 réunions mardi =
     12,5 %, vs 1/12 réunions jeudi = 8,3 %.
   Cadrage : « le mardi libère la semaine » (gain) vs « le jeudi donne plus de temps
     pour préparer » (gain différent). Les deux se reformulent positivement → préférence
     révélée au taux de présence, pas au cadrage.
   Confirmation : argument contre mardi 9h → un membre clé a des transports en commun
     peu fiables le mardi matin (information vérifiée directement).

3. PROBABILITÉS
   Tous les membres présents si créneau mardi 9h : 70 %
   Classe de référence : taux de présence complète pour les réunions mardi 9h sur 3 mois
     = 5/8 = 62,5 %, ajusté +7,5 % car calendrier moins chargé ce mois-ci.
   Tous les membres présents si créneau jeudi 14h : 85 %
   Classe de référence : taux de présence complète pour les réunions jeudi 14h = 10/12
     = 83 %, stable.

4. CONSÉQUENCES
   Optimiste (50 %) : réunion complète, révision finalisée, livrable Q3 validé.
   Central   (35 %) : un absent, réunion partielle, validation reportée de 2 jours.
   Pessimiste(15 %) : 2 absents ou annulation, retard sur le livrable Q3.
   Scénario ruineux : aucun scénario ruineux absolu. Pire cas = retard de 1 semaine.

5. VÉRIFICATION
   Fait clé 1 : taux de présence historique mardi 9h → vérifié sur l'agenda partagé
     (export Google Calendar, 3 mois, 8 réunions).
   Fait clé 2 : contrainte transport du membre clé → vérifiée directement par message.

6. DÉCISION & SUIVI
   Décision : créneau jeudi 14h.
   Probabilité de présence complète : 85 %
   Date de revue : jeudi 16 juin 17h (confirmation de présences).
```

### Mini-arbre de décision

```
Option A — Mardi 9h
  ├─ Présence complète [p = 0,70] → Réunion productive (+10 pts)
  └─ Absent(s)         [p = 0,30] → Report partiel (−3 pts)
  Espérance(A) = 0,70 × 10 + 0,30 × (−3) = 7,0 − 0,9 = 6,1

Option B — Jeudi 14h
  ├─ Présence complète [p = 0,85] → Réunion productive (+10 pts)
  └─ Absent(s)         [p = 0,15] → Report partiel (−3 pts)
  Espérance(B) = 0,85 × 10 + 0,15 × (−3) = 8,5 − 0,45 = 8,05

Option C — Reporter à S+1
  ├─ Disponibilités alignées [p = 0,90] → Réunion productive (+10 pts)
  └─ Conflit S+1              [p = 0,10] → Report supplémentaire (−8 pts)
  Espérance(C) = 0,90 × 10 + 0,10 × (−8) = 9,0 − 0,8 = 8,2

Option retenue d'après l'arbre : Option C (E = 8,2) légèrement supérieure.
Nuance : le délai d'une semaine a un coût implicite non valorisé (−2 pts de pression
calendrier) → avec ce coût : E(C) = 8,2 − 2 = 6,2. Option B (E = 8,05) devient
préférable. Décision finale : jeudi 14h.
```

---

## Exercice 2 — Analyse de second ordre + journal de prévisions (exemple)

### Analyse de second ordre

```
ANALYSE DE SECOND ORDRE — Réunion de révision planifiée jeudi 14h

Effets de 1er ordre (< 1 semaine) :
  → Réunion tenue avec 3/3 membres. Révision Q3 complète. Livrable validé.

Effets de 2e ordre (1 semaine – 3 mois) :
  → La validation rapide libère du temps pour la phase Q4. L'équipe développe
    l'habitude de planifier le jeudi 14h → coordination plus fluide à terme.
    Risque : si le créneau jeudi 14h est systématisé, il peut saturer ce slot pour
    d'autres projets.

Effets de 3e ordre (> 3 mois, boucles) :
  → Boucle renforçante : chaque réunion réussie renforce la confiance dans ce créneau,
    qui attire plus de réunions critiques → disponibilité de l'équipe jeudi 14h
    devient une contrainte forte. Le créneau se "fossilise".
  → Boucle équilibrante : la saturation du jeudi 14h finit par générer des conflits,
    ce qui force l'équipe à diversifier les créneaux → retour à un équilibre.

Boucle renforçante principale :
  → Succès du créneau → répétition → dépendance au créneau → risque de saturation.

Boucle équilibrante principale :
  → Saturation → conflits → diversification forcée → créneaux alternatifs adoptés.

Point de levier identifié :
  → Documenter la décision de créneau et la revoir tous les 2 mois plutôt que de
    laisser la routine s'installer sans feedback. Point de levier : la revue périodique.
```

### Journal de prévisions (10 entrées, exemple)

| # | Date | Question binaire + date résolution | p (%) | Classe de référence | Outcome | (p−o)² | Note |
|---|------|------------------------------------|--------|---------------------|---------|--------|------|
| 1 | 2026-06-16 | Présence 3/3 à la réunion jeudi 14h (avant 17h) | 85 | Taux historique 83 % jeudi 14h | 1 | 0,0225 | OK |
| 2 | 2026-06-16 | Pluie demain matin avant 10h | 40 | Prévision météo locale | 0 | 0,16 | OK |
| 3 | 2026-06-17 | Livraison colis avant 18h | 70 | Délai annoncé 24h, 75 % à l'heure | 1 | 0,09 | OK |
| 4 | 2026-06-17 | Finir la relecture du rapport avant midi | 60 | Auto-estimation, 3 tâches similaires passées | 0 | 0,36 | Sur-confiant |
| 5 | 2026-06-18 | Réunion suivante planifiée sans report | 75 | Taux de maintien historique 70 % | 1 | 0,0625 | OK |
| 6 | 2026-06-19 | Score de l'équipe A ≥ score de l'équipe B (match vendredi) | 55 | Classement : A 3e, B 7e | 1 | 0,2025 | OK |
| 7 | 2026-06-20 | File d'attente logiciel < 100 tâches samedi soir | 65 | Moyenne semaine = 85, weekends −20 % | 1 | 0,1225 | OK |
| 8 | 2026-06-21 | Température max ≥ 22°C dimanche | 80 | Prévision météo 7 jours | 0 | 0,64 | Sur-confiant |
| 9 | 2026-06-22 | Réponse client reçue avant lundi 17h | 50 | Délai moyen réponse : 3 jours ouvrés | 0 | 0,25 | OK |
|10 | 2026-06-23 | Tâche de code complétée avant fin de journée | 45 | Tâches similaires : 40 % finies J0 | 0 | 0,2025 | OK |

**Somme (p−o)²** = 0,0225 + 0,16 + 0,09 + 0,36 + 0,0625 + 0,2025 + 0,1225 + 0,64 + 0,25 + 0,2025 = **2,112**

**Score de Brier = 2,112 / 10 = 0,211** (meilleur que la baseline 0,25 ; objectif 3 mois : < 0,20)

**Zone de sur-confiance** : prédictions à 70-80 % (entrées 3, 8) → réalisées seulement 50 % → recalibrer à 55-65 %.

**Zone neutre** : prédictions à 50-55 % → résultats conformes (0 sur 2 prédictions à ~50 %, score ≈ 0,25 = baseline).

---

## Exercice 3 — Protocole SIFT + synthèse portfolio (exemple)

### Protocole SIFT — information vérifiée

```
Information à vérifier : « Le taux de présence aux réunions d'équipe le mardi matin
  est de 62,5 % en moyenne selon l'agenda partagé. »
Source d'origine : export manuel du calendrier Google partagé (3 mois, 8 réunions).

S — STOP : oui — vérification de l'export avant d'utiliser le chiffre.
  Observation : l'export est limité à 3 mois ; l'échantillon est petit (N=8).

I — INVESTIGATE THE SOURCE
  Qui publie ? Données internes, produites par nous-mêmes.
  Lecture latérale : pas de source externe — vérifier la logique de l'export (filtre
    correct ? jours fériés exclus ?).
  Résultat : source directe, mais échantillon faible → cote la fiabilité à « mixte ».

F — FIND BETTER COVERAGE
  Autres sources : comparer avec les comptes-rendus écrits (9 réunions sur 5 mois →
    taux légèrement différent : 61 %).
  Convergence : les deux méthodes donnent 61-63 % → convergence acceptable.

T — TRACE TO ORIGINAL
  Source primaire : agenda Google et comptes-rendus. Pas de DOI. Données directement
    accessibles et vérifiables.
  Date, auteur, contexte : confirmés (période juin 2025 – mai 2026, équipe de 3).

Verdict : information **confirmée avec nuance** (N faible, mais deux méthodes
  convergent). Décision : utiliser 62 % comme base, noter l'incertitude (±5 %).
Impact sur la décision : la marge d'incertitude ne change pas le choix (jeudi 14h
  reste nettement supérieur à 62 % vs 83 %).
```

### Synthèse portfolio

```
=== BOÎTE À OUTILS DE JUGEMENT — Planifier la réunion Q3 — 2026-06-16 ===

1. CHECKLIST (résumé)
   Biais détectés : ancrage sur mardi (vu en premier) ; disponibilité (annulation
     récente). Classe de référence utilisée pour chaque option.
   Probabilité retenue : 85 % de présence complète jeudi 14h.
   Décision : créneau jeudi 14h.

2. ARBRE DE DÉCISION (résumé)
   Option A (mardi 9h) : Espérance = 6,1
   Option B (jeudi 14h) : Espérance = 8,05
   Option C (reporter S+1) : Espérance = 8,2 → réduite à 6,2 avec coût du délai.
   Choix retenu : jeudi 14h (B), robuste face au délai.

3. SECOND ORDRE (résumé)
   Effet 2e ordre : systématisation du créneau jeudi → boucle renforçante de
     dépendance, compensée par saturation future (boucle équilibrante).
   Point de levier : revue bimestrielle du créneau habituel.

4. JOURNAL BRIER (résumé)
   N prédictions : 10 | Score de Brier : 0,211 (< 0,25 : baseline battue)
   Zone à recalibrer : prédictions à 70-80 % → ramener à 55-65 %.

5. SIFT (résumé)
   Information vérifiée : taux de présence mardi 9h = 62,5 %.
   Verdict : confirmée avec nuance (N faible, deux méthodes convergent ±5 %).
   Impact : ne change pas le choix (jeudi 14h nettement préférable).

LATTICEWORK — modèles activés ce capstone :
  → Classe de référence (taux de base) — Module 03
  → Espérance + arbre de décision — Module 07
  → Score de Brier + journal — Module 08
  → Vérification SIFT — Module 11
  → Boucles de rétroaction + second ordre — Module 12
  → Checklist anti-biais — Module 13
```
