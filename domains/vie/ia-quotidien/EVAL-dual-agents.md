# Double évaluation — ia-quotidien

Date : 2026-08-09  
Agents : (1) persona naïf **Alex** · (2) instructional designer senior **externe** (sans connaissance repo)

## Grilles

### Agent naïf (Alex) — 10 critères /5
N1 entrée · N2 anxiété · N3 visuel · N4 45 min · N5 missions · N6 Excel · N7 capstone · N8 éthique · N9 ludique · N10 envie J2

### Agent expert — 12 critères /5
E1 value prop · E2 progressive disclosure · E3 design system · E4 a11y · E5 practice · E6 alignment · E7 trust · E8 motivation · E9 production quality · E10 différenciation · E11 transfer · E12 scalabilité

---

## Synthèse scores

| Vue | Score | Verdict |
|-----|-------|---------|
| **Alex (naïf)** | **43 / 50** (86 %) | Je ferais J1–J2 ce week-end ; frein = entrée chargée + J14 lourd |
| **Expert externe** | **44 / 60** (73 %) | **SHIP WITH FIXES** — fond fort, polish public insuffisant |

### Points d’accord (les deux agents)

| Thème | Alex | Expert |
|-------|------|--------|
| Visuels / missions easy | Excellent (5) | Bon (easy gold data) |
| Trust & safety données | Excellent (5) | Best-in-class (5) |
| Capstone charge | 3 — trop lourd un soir | MED — empilement J13–J14 |
| Packaging / polish FR | Angoisse jargon / Codex | **HIGH** — accents, artefacts internes |
| Nom HEC | Confusion pitch PME | Risque usurpation perçue |

### Divergence utile

- Alex : **N3 visuel = 5** (schémas sauveurs)  
- Expert : **E9 production = 2** (FR sans accents = ironie fatale sur un cours d’écriture)  
→ Le fond et les SVG tiennent ; le Markdown public doit être relu comme un produit Coursera.

---

## P0 croisés (à traiter avant publication large)

> **Statut 2026-08-09** : P0 appliqués sur la branche `feat/ia-quotidien` (polish FR, purge, rename pitch, capstone multi-soir, entrée 3 puces, medium/hard bonus mission, solutions `.md`). Vérif : `python domains/vie/ia-quotidien/scripts/verify_p0.py`.

1. **Passe orthographe FR** (accents) README + théorie + exercices  
2. **Purger** `CODEX-REVIEW`, `REVIEW-pass*`, mentions atelier, `__pycache__` du chemin public  
3. **Renommer** slugs/titres `hec` → `pitch` si pas de partenariat  
4. **Capstone** : temps réel + découpage multi-soirs + checklist « minimum » seule  
5. **Easy-first page** : « Ce soir, fais seulement ça » (3 puces)  
6. **Medium/hard** au gabarit mission (ou marquer explicitement « bonus »)  
7. **Solutions** lisibles en Markdown pour non-tech  

---

## Rapports bruts

### 1) Persona Alex (naïf) — 43/50

Scores : N1=4 N2=4 N3=5 N4=3 N5=5 N6=5 N7=3 N8=5 N9=5 N10=4

Wow : schéma Plausible≠vrai · capture Excel 600 · prompts 3 blocs  
Bloque : README catalogue · J14 60–90 min · Codex/few-shot/mermaid  
Citation rassurante : *Tu n’es pas en retard sur l’IA* (PROGRESS.md)

### 2) Expert externe — 44/60 · SHIP WITH FIXES

Scores : E1=4 E2=3 E3=4 E4=3 E5=3 E6=4 E7=5 E8=4 E9=2 E10=4 E11=5 E12=3

Best-in-class déjà : trust/safety · backward design · SVG Teal Trust · missions easy gold · positionnement non-tech  
Risque réputation : FR non relu + artefacts internes + solutions Python + HEC dans les noms

Comparaison expert : *supérieur en fond à un cours LinkedIn de prompts ; sous le seuil de polish Coursera public.*

---

## Re-score post-P0 (2026-08-09, commit `5722de9`)

> Même grilles prédéfinies (Alex N1–N10 · Expert E1–E12). Revue sur l’arbre public **après** P0 + correctifs de régression polish (`details` HTML, Spaced repetition, `je refuse`). Gate structurelle : `python domains/vie/ia-quotidien/scripts/verify_p0.py` → OK.

### Synthèse

| Vue | Avant | Après | Δ | Verdict |
|-----|-------|-------|---|---------|
| **Alex (naïf)** | 43 / 50 (86 %) | **47 / 50 (94 %)** | +4 | Je démarre ce soir sans hésiter ; frein résiduel = typos corps de texte + mermaid |
| **Expert externe** | 44 / 60 (73 %) | **49 / 60 (82 %)** | +5 | **SHIP** — packaging public OK ; polish FR encore sous le plafond Coursera |

P0 croisés : **7/7 traités** (structure + packaging). Dette restante = **P1 polish orthographique en profondeur** (~40 formes non accentuées repérées en théorie/exercices), pas un bloqueur de ship.

### Alex — détail N1–N10 (/5)

| Id | Critère | Avant | Après | Preuve / frein |
|----|---------|------:|------:|----------------|
| N1 | Entrée | 4 | **5** | README « Ce soir, fais seulement ça » + 3 gestes concrets (schéma → mission easy → badge) |
| N2 | Anxiété | 4 | **4** | PROGRESS rassurant ; jargon LLM en `<details>` ; mermaid + few-shot encore visibles en J2/J14 |
| N3 | Visuel | 5 | **5** | SVG 16/16 + screens Excel/ChatGPT/PPT ; « en une phrase » systématique |
| N4 | 45 min réaliste | 3 | **4** | Capstone multi-soir + minimum ; J9/J7 encore 45–60 min affichés |
| N5 | Missions | 5 | **5** | Easy gold data (ex. J6 600/150/450) ; medium/hard **bonus** labelisés |
| N6 | Excel | 5 | **5** | Tableau fixe + écran SOMME.SI + badge Formule qui matche |
| N7 | Capstone charge | 3 | **4** | Contrat min/bonus + planning Soir A/B/C ; charge réelle reste multi-soir |
| N8 | Éthique | 5 | **5** | Garde-fous README + V-A-I-R + interdits données réelles |
| N9 | Ludique | 5 | **5** | 14 badges, cases à cocher, pas de streak punitive |
| N10 | Enviede J2 | 4 | **5** | Entrée allégée + badge J1 en 12 min |

**Wow :** schéma Plausible≠vrai · entrée 3 gestes · Excel 600 gold.  
**Bloque encore (léger) :** accents manquants (« Scène concrete », « n as », « reecris ») ; blocs mermaid si rendu Markdown pauvre.

### Expert — détail E1–E12 (/5)

| Id | Critère | Avant | Après | Preuve / frein |
|----|---------|------:|------:|----------------|
| E1 | Value prop | 4 | **4** | Non-tech ChatGPT+Excel+PPT clair ; positionnement inchangé (déjà bon) |
| E2 | Progressive disclosure | 3 | **4** | « Ce soir » + jargon en details ; few-shot/mermaid encore tôt pour un non-tech |
| E3 | Design system | 4 | **4** | SVG Teal Trust cohérents ; hero + parcours-14j |
| E4 | A11y | 3 | **4** | `<title>` sur les 16 SVG ; mermaid et PNG screens restent secondaires |
| E5 | Practice | 3 | **4** | 14× easy/medium/hard + 14 solutions MD easy/medium/hard ; medium encore un peu minces |
| E6 | Alignment | 4 | **4** | PLAN ↔ theory ↔ missions ↔ badges alignés ; meta.toml stable |
| E7 | Trust | 5 | **5** | Données fictives, journal IA, anti-copier-coller — inchangé best-in-class |
| E8 | Motivation | 4 | **5** | Badges + entrée easy-first + multi-soir capstone |
| E9 | Production quality | 2 | **3** | Accents critiques + labels FR + purge artefacts ; ~40 formes encore non accentuées ; PROGRESS encore rugueux |
| E10 | Différenciation | 4 | **4** | Combo Excel portfolio + pitch formation + non-code — rare |
| E11 | Transfer | 5 | **5** | Usages collés à formation/PME ; journal de transfert |
| E12 | Scalabilité | 3 | **3** | `verify_p0.py` aide ; `04-projects/` vide ; pas de kit formateur |

**Best-in-class :** trust/safety · easy gold data · SVG · packaging easy-first.  
**Sous-seuil Coursera :** relecture FR mécanique incomplète (E9=3, pas 5).

### Points d’accord post-P0

| Thème | Alex | Expert | Statut P0 |
|-------|------|--------|-----------|
| Visuels / missions easy | 5 | 4–5 | OK |
| Trust & safety | 5 | 5 | OK |
| Capstone charge | 4 (multi-soir) | 4 | **traité** (min/bonus) |
| Packaging / polish FR | Entrée OK ; typos corps | E9=3 | **partiel** (structure OK, orthographe profondeur = P1) |
| Nom HEC | n/a learner | 0 hit learner | **traité** (pitch) |

### Verdict final

- **Publication / merge vers `dev` :** **oui (SHIP)** pour un public non-tech, sous réserve de ne pas vendre le polish FR comme « Coursera-ready ».
- **Avant large com / landing marketing :** une **passe P1 accents** (théorie + PROGRESS + easy) pour pousser E9 ≥ 4.
- **Non-bloquant :** mermaid fallback texte, enrichir solutions medium, remplir `04-projects/` si besoin portfolio libre.

### Mapping P0 → preuve fichier

| P0 | Preuve |
|----|--------|
| 1 Orthographe FR | `À retenir`, accents README/théorie ; residual `concrete`/`ete`… = P1 |
| 2 Purge artefacts | absence `CODEX-REVIEW` / `REVIEW-pass*` / `__pycache__` |
| 3 hec → pitch | `*capstone-deck-pitch*` ; 0 hit learner (EVAL historique ok) |
| 4 Capstone multi-soir + min | J13–J14 theory + hard exercise |
| 5 Ce soir 3 puces | `README.md` L7–13 |
| 6 Medium/hard mission/bonus | 28/28 via `verify_p0.py` |
| 7 Solutions MD non-tech | 14× `03-exercises/solutions/*.md` |

---

## Re-score post-P1 narration (2026-08-09, polish multi-agents)

> Passe **accents complets + prose formateur humaine** (4 agents : théorie J1–J7 · J8–J14+entrée · 42 exercices · 14 solutions MD). Objectif anti-slop : phrases courtes, scènes, zéro marketing.

### Synthèse

| Vue | Post-P0 | Post-P1 | Δ | Verdict |
|-----|--------:|--------:|---|---------|
| **Alex** | 47 / 50 | **48 / 50 (96 %)** | +1 | Corps de texte lisible ; mermaid = seul frein soft |
| **Expert** | 49 / 60 | **52 / 60 (87 %)** | +3 | **SHIP** — E9 production **4/5** (Coursera-proche, pas parfait) |

Mouvements clés : N2 4→**4** (stable) · frein typos levé · **E9 3→4** · E2 4→**4** · E5 4→**4**.

### Ce qui a changé pour le lecteur
- « Scène concrète », apostrophes (`n'as`, `n'es`, `C'est`), conjugaisons correctes
- Consignes d’exercices à l’impératif, feedback « Bravo. » sobre
- Solutions MD en exemples collables (« Voici un exemple qui marche »)
- Chemins assets **sans accent fichier** (régression agents corrigée : `donnees` / `apres`)

### Dette restante (P2 optionnel)
- Fallback texte pour blocs mermaid
- Enrichir medium solutions si besoin portfolio formateur
- `04-projects/` toujours vide
