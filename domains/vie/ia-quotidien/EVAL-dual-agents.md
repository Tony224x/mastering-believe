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
