# Module 02 — Prompts qui marchent

> **Temps estimé** : 45 min | **Prérequis** : Module 01
>
> **Objectif** : Remplacer les questions floues par une structure de prompt réutilisable (rôle, contexte, tâche, format, contraintes).

---

![Grille RCCFC : Rôle, Contexte, Tâche, Format, Contraintes](../assets/02-rccfc-prompt.svg)

> **En une phrase :** regarde ce schéma avant de lire le reste.

### Écran exemple — prompt en 3 blocs

![Capture pedagogique : chat ChatGPT avec prompt Contexte Demande Resultat et formule Excel](../assets/screens/screen-chatgpt-prompt-3-blocs.png)

> Maquette pedagogique (contenu fictif). Tu peux t en inspirer pour coller le même style de prompt.

![Doc officielle OpenAI — prompt engineering](../assets/screens/screen-docs-openai-prompting.png)

> Capture publique : [OpenAI Prompt engineering](https://platform.openai.com/docs/guides/prompt-engineering).


```mermaid
flowchart LR
  A[Idee floue] --> B[RCCFC]
  B --> C[1er jet]
  C --> D[Iteration]
  D --> E[Livrable utile]
```



## Version simple (a utiliser d abord)

Oublie les acronymes 2 minutes. Ecris seulement :

1. **Contexte** — mon tableau / mon sujet  
2. **Demande** — ce que je veux  
3. **Resultat attendu** — format (liste, formule, 5 puces…)

La grille **RCCFC** plus bas est la version complète (optionnelle le jour 1 des prompts).

## 1. Scène concrete : deux prompts, deux mondes

**Prompt A (flou) :**
> « Aide-moi pour mon PowerPoint. »

**Prompt B (structure) :**
> Tu es coach de présentation pour un formation en entrepreneuriat.
> Contexte : pitch de 7 minutes pour un projet de PME fictive de livraison locale en ville.
> Tâche : propose une structure de 10 slides (titre + 1 phrase d'intention par slide).
> Format : liste numérotee.
> Contraintes : français soutenu mais clair ; pas de jargon startup inutile ; aucune donnée inventée presentee comme réelle.

Le prompt B produit quelque chose d'*actionnable*. Le A produit du generique.

> **À retenir :** La qualité de sortie suit la qualité d'entrée. Le modèle n'est pas telepathe.

---

## 2. La grille RCCFC

Memorise :

| Lettre | Signifie | Exemple |
|--------|----------|---------|
| **R** | Rôle | "Tu es comptable pedagogue" |
| **C** | Contexte | "Je suis débutante, tableau mensuel" |
| **T** | Tâche | "Propose 5 formules Excel..." |
| **F** | Format | "Tableau markdown" / "liste" / "JSON simple" |
| **C** | Contraintes | "Pas de VBA" ; "données fictives" ; "FR-CA" |

La doc officielle OpenAI sur le *prompt engineering* insiste sur des **instructions claires**, des **exemples**, et la precision du format de sortie. [OpenAI Prompting Guide]

---

## 3. Few-shot : montrer un exemple

Si tu veux un ton precis, donne **1 exemple** :

```
Exemple de bonne slide titre :
"Titre : Le probleme des 3 retards de paiement / Intention : faire sentir l'urgence en 10 secondes"

Maintenant, ecris les 9 autres selon le meme format.
```

---

## 4. Iteration (le vrai super-pouvoir)

Rarement le 1er jet est le bon. Enchaine :

1. "Raccourcis de 30 %."
2. "Supprime le jargon."
3. "Donne 2 alternatives plus concretes pour la slide 4."
4. "Qu'est-ce qui est faible dans cette structure ? Sois direct."

> **À retenir :** Un bon usage = conversation courte et dirigee, pas un monologue magique.

---

## 5. Trois prompts modèles a copier

### Réflexion carrière
```
Role : coach de carriere neutre.
Contexte : formation entrepreneuriat ; je explore un pivot.
Tache : pose-moi 8 questions (une par une) pour clarifier mes contraintes avant de proposer des pistes.
Contraintes : ne propose pas de plan avant la question 8 ; pas de cliches motivationnels.
```

### Excel
```
Role : formateur Excel pour non-developpeurs.
Contexte : colonnes Date | Libelle | Categorie | Montant | Type (entree/sortie).
Tache : donne les formules pour (1) total entrees (2) total sorties (3) solde.
Format : pour chaque formule : nom, formule FR-CA, explication en 1 phrase.
Contraintes : Excel Microsoft 365 ; pas de macros.
```

### Slides
```
Role : designer de presentation sobriete (Presentation Zen).
Tache : transforme ce plan en 10 titres de slides + 3 bullets max chacun.
Contraintes : une idee par slide ; pas de mur de texte.
```

---

## Spaced repetition

**Q1.** Que signifie RCCFC ?
**R1.** Rôle, Contexte, Tâche, Format, Contraintes.

**Q2.** Pourquoi un prompt flou donne une réponse mediocre ?
**R2.** Le modèle comble les trous par du generique "moyen".

**Q3.** Qu'est-ce qu'un few-shot ?
**R3.** Fournir 1+ exemples du format/ton attendu dans le prompt.

**Q4.** Quelle est la meilleure suite après un 1er jet correct mais long ?
**R4.** Iterer : raccourcir, preciser, challenger — pas recommencer de zero.

**Q5.** Ou trouver des patterns officiels de prompting ?
**R5.** [OpenAI Prompting Guide]
