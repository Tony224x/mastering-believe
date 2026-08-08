# Module 03 — Hallucinations & verification

> **Temps estime** : 45 min | **Prerequis** : Modules 01–02
>
> **Objectif** : Detecter les inventions plausibles, proteger les donnees sensibles, et appliquer une checklist avant usage scolaire ou pro.

---

![Checklist V-A-I-R : Verifiable, Ancree, Invention, Risque](../assets/03-vair-checklist.svg)

> **En une phrase :** regarde ce schema avant de lire le reste.



## 1. Scene concrete : la fausse statistique

Prompt : *« Donne-moi le taux moyen de defaillance des PME au Quebec en 2024 avec source. »*

Reponse possible (exemple pedagogique) : un pourcentage precis + un "Rapport Statistique Canada 2024" qui **n'existe pas sous cette forme**.

Le ton est confiant. Les chiffres sont ronds. **C'est exactement le danger.**

Le *GPT-4 System Card* reconnait les hallucinations comme risque central. [GPT-4 System Card, 2023]

> **A retenir :** Traite toute stat, citation, loi ou "etude" comme **non prouvee** jusqu'a verification externe.

---

## 2. Checklist V-A-I-R (simple, memorable)

Avant d'utiliser une reponse pour l'ecole ou le travail :

| | Question | Si non → |
|--|----------|----------|
| **V** | **V**erifiable ? Y a-t-il une source que *je* peux ouvrir ? | Ne pas citer |
| **A** | **A**ncre dans *mes* faits ? (mes chiffres, mon enonce) | Ne pas coller tel quel |
| **I** | **I**nvention possible ? (date precise, % trop propre, auteur obscur) | Chercher confirmation |
| **R** | **R**isque si faux ? (note, client, legal) | Verification obligatoire |

---


![Feu tricolore des donnees : vert fictif, orange anonymiser, rouge ne pas coller](../assets/03b-donnees-feu.svg)

> **En une phrase :** si tu ne mettrais pas l info sur un ecran de bus, ne la mets pas dans le chat.

**Avant / apres :** `Marie Dupont, salaire 2450` → `Employe A, montant supprime`.

## 3. Donnees : ce qui ne se colle jamais

**Interdit dans le chat (cours + vraie vie) :**

- noms de clients, clients, beneficiaires, collegues ;
- salaires, numéros de compte, documents fiscaux reels ;
- donnees de sante, dossiers RH ;
- mots de passe, captures d'ecran internes.

Prefere : **jeux de donnees fictifs** (comme le projet tresorerie J9).

Cadre utile : guides **CNIL** sur l'IA et principes de minimisation ; au Canada, principes de l'**OPC**. [CNIL IA] Le NIST AI RMF rappelle de penser *risque* meme pour un usage "simple". [NIST AI RMF]

> **A retenir :** Si tu n'afficherais pas cette info sur un ecran de bus, ne la mets pas dans un LLM grand public.

---

## 4. Technique anti-hallucination dans le prompt

Ajoute systematiquement :

```
Si tu n'es pas sur d'un fait, ecris "INCERTAIN" et propose comment verifier.
N'invente aucune source. Si tu cites, donne un type de source a chercher (ex. "site gouvernemental") sans inventer le titre exact.
Utilise uniquement les chiffres que je fournis : [coller chiffres fictifs].
```

---

## 5. Mini-protocole 60 secondes

1. Surligne dans la reponse tout chiffre / nom propre / loi.
2. Pour chaque : V-A-I-R.
3. Verifie 1 item critique sur le web officiel.
4. Reecris la phrase finale **avec tes mots**.

---

## Spaced repetition

**Q1.** Qu'est-ce qu'une hallucination LLM ?
**R1.** Une affirmation confiante fausse ou non fondee generee par le modele.

**Q2.** Que faire d'une statistique sans source ouvrable ?
**R2.** Ne pas la citer ; chercher une source officielle ou l'oter.

**Q3.** Donne 2 exemples de donnees a ne jamais coller.
**R3.** Identifiants clients / beneficiaires ; salaires / documents fiscaux reels.

**Q4.** A quoi sert la contrainte "ecris INCERTAIN" ?
**R4.** Forcer le modele a signaler le doute au lieu d'inventer.

**Q5.** Cite un cadre institutionnel mentionne pour le risque IA.
**R5.** NIST AI RMF et/ou guides CNIL. [NIST AI RMF] [CNIL IA]
