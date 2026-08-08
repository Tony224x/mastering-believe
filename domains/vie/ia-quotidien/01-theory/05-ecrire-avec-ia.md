# Module 05 — Ecrire avec l'IA (Word / rapports)

> **Temps estime** : 45 min | **Prerequis** : Modules 02–04
>
> **Objectif** : Produire un brouillon utile (plan + page) pour un rendu type formation, puis le reapproprier a la main.

---

## 1. Scene concrete : le rapport de 3 pages pour hier

Tu as des notes de cours eparses et un enonce. ChatGPT peut :

1. transformer tes notes en **plan** ;
2. proposer un **premier jet** section 1 ;
3. te signaler les trous logiques.

Il ne doit **pas** devenir l'auteur cache du devoir entier (ethique + apprentissage + detection).

Mollick (2023) insiste sur des usages ou l'etudiant *assigne un role* a l'IA tout en restant responsable du rendu. [Mollick, 2023]

> **Key takeaway :** Pipeline honnete = **tes idees → structure IA → brouillon IA → reecriture humaine majoritaire**.

---

## 2. Pipeline en 5 etapes

| Etape | Toi | IA |
|-------|-----|----|
| 1. Brutes | Notes, consignes, idees en vrac | — |
| 2. Plan | Valides les titres | Propose outline |
| 3. Brouillon | Choisis section prioritaire | Genere 300–500 mots |
| 4. Reecriture | Reecris ~30–50 % minimum | — |
| 5. Controles | Faits, consignes, ton | "Liste les faiblesses" |

Prompts de format : preciser longueur, public, niveau de langue. [OpenAI Prompting Guide]

---

## 3. Prompt modele "rapport formation"

```
Role : assistant redaction academique (niveau certificat, pas these).
Contexte : [colle l'enonce + tes 10 puces de notes].
Tache : (1) outline en 5 parties (2) redige UNIQUEMENT la partie 2 (max 400 mots).
Contraintes :
- francais canadien professionnel ;
- aucune source inventee — si besoin de source, ecrire [A_VERIFIER] ;
- ne pas flatter ; signaler les trous dans mon raisonnement en fin de reponse.
```

---

## 4. Signes que tu as trop delegue

- Tu ne peux pas expliquer un paragraphe a voix haute.
- Le vocabulaire n'est pas le tien.
- Des references "ScienceDirect 2019" non verifiees.
- L'intro pourrait servir a n'importe quel sujet voisin.

**Remede :** fermer le chat, reecrire la section de memoire, puis rouvrir pour polish seulement.

---

## Spaced repetition

**Q1.** Quelle est l'etape non negociable apres un brouillon IA ?
**R1.** Reecriture humaine substantielle + verification des faits.

**Q2.** Que mettre dans le prompt pour eviter les fausses sources ?
**R2.** Interdiction d'inventer ; marqueur [A_VERIFIER].

**Q3.** Pourquoi generer une seule section d'abord ?
**R3.** Controler la qualite et rester proprietaire du fond.

**Q4.** Cite un usage ethique vs non ethique.
**R4.** Ethique : plan + relecture. Non ethique : devoir entier non relu/non compris.

**Q5.** Reference utile sur les roles IA en education ?
**R5.** Mollick & Mollick, Assigning AI (2023). [Mollick, 2023]
