# PLAN — Apprendre a apprendre (curriculum fige)

> Ce plan est **fige** : il ne doit pas etre modifie sans consensus. Il sert de contrat entre les agents de creation et de verification.

---

## Note anti-pseudoscience transversale

**Chaque module contient au moins un encart "Pseudoscience ?" quand pertinent.** L'objectif est de former un reflexe, pas de faire un module isole.

Neuromythes a aborder explicitement :
- **Styles d'apprentissage VAK** (module 01) — refutes par Pashler et al. 2008 (*PSPI*)
- **Brain-training / Lumosity & co.** (module 01 ou 02) — pas de transfert generalise, Simons et al. 2016 (*PSPI*)
- **"10 000 heures"** (module 06) — nuance forte via Macnamara et al. 2014 et Ericsson lui-meme (*Peak*)
- **Cerveau gauche/droit, "on n'utilise que 10 % du cerveau"** (module 05) — Dekker et al. 2012 (*Frontiers in Psychology*)

**Posture constante :** factuelle, non condescendante — expliquer pourquoi le mythe est seduisant avant de le refuter.

---

## Curriculum module par module

### Module 01 — Pourquoi tu oublies (et comment le savoir te change)
**Objectif** : comprendre que la memoire est un systeme de recuperation (pas de stockage passif) et identifier l'illusion de competence.  
**Sources** : Dunlosky 2013 ; Bjork/Dunlosky/Kornell 2013 ; Brown/Roediger/McDaniel 2014 (*Make It Stick*) ; Roediger & Karpicke 2006  
**Neuromythe** : styles d'apprentissage VAK (Pashler 2008) + brain-training (Simons 2016)  
**Flash-cards** : 4-5 sur courbe d'oubli, fluency illusion, memoire de travail vs long terme

### Module 02 — Retrieval practice : se tester, pas se relire
**Objectif** : maitriser la technique #1 de Dunlosky — active recall, flashcards, blank-page recall.  
**Sources** : Roediger & Karpicke 2006 (*Psychological Science*) ; Karpicke & Roediger 2008 (*Science*) ; Dunlosky 2013 (utilite "elevee")  
**Chiffre cle** : rappel a 1 semaine — 61 % (groupe testing) vs 40 % (groupe relecture) — Roediger & Karpicke 2006  
**Flash-cards** : 4-5 sur testing effect, spacing effect, active recall vs re-encoding

### Module 03 — Spaced repetition : espacer pour ancrer
**Objectif** : comprendre la distributed practice, les intervalles croissants, et utiliser un systeme SRS (Anki/SM-2).  
**Sources** : Cepeda et al. 2006 (*Psychological Bulletin*) ; Cepeda et al. 2008 (*Psychological Science*) ; SuperMemo/SM-2 (Wozniak 1987-1990)  
**Regle chiffree** : gap optimal ≈ 10-20 % du delai avant test (Cepeda 2008)  
**Code** : `02-code/03-spaced-repetition.py` — planificateur SM-2 simplifie (stdlib pur)  
**Flash-cards** : 4-5 sur distributed vs massed practice, intervalles, facteur "easiness"

### Module 04 — Difficultes desirables : interleaving & variation
**Objectif** : comprendre pourquoi "plus dur sur le moment = mieux ancre" et appliquer l'interleaving.  
**Sources** : Bjork & Bjork 2011 ; Rohrer & Taylor 2007 (*Instructional Science*) ; Rohrer, Dedrick & Stershic 2015 (*Journal of Educational Psychology*)  
**Chiffre cle** : 72 % vs 38 % de reussite (entrelace vs bloque) — Rohrer et al. 2015

### Module 05 — Attention & deep work : encoder en profondeur
**Objectif** : comprendre la charge cognitive, le chunking, et structurer ses sessions d'etude sans distraction.  
**Sources** : Newport 2016 (*Deep Work*) ; Cowan 2001 (*Behavioral and Brain Sciences*, ~4 chunks en memoire de travail)  
**Neuromythe** : cerveau gauche/droit, "10 % du cerveau" — Dekker et al. 2012

### Module 06 — Pratique deliberee & metacognition
**Objectif** : passer de "retenir" a "devenir bon" ; piloter son propre apprentissage avec le feedback.  
**Sources** : Ericsson & Pool 2016 (*Peak*) ; Bjork/Dunlosky/Kornell 2013 (*Annual Review of Psychology*) ; Macnamara et al. 2014  
**Neuromythe** : "10 000 heures" — nuance forte (la qualite prime le volume, Macnamara 2014 : 26 % de variance seulement)

### Module 07 — Capstone : apprendre avec l'IA
**Objectif** : assembler tout le domaine en un systeme personnel augmente par un LLM.  
**Sources** : Bloom 1984 (2 sigma problem) ; Brown/Roediger/McDaniel 2014 (*Make It Stick*)  
**Livrable** : README de systeme personnel (retrieval + spacing + deliberate practice + tuteur LLM)

---

## Contraintes pedagogiques globales

1. **Concret avant abstrait** — exemple d'abord, principe ensuite
2. **Pareto-first** — les 3 premiers modules couvrent 80 % du gain
3. **Flashcards a chaque module** — 4-5 cartes au format Q/R
4. **Encarts pseudoscience** — minimum 1 par module ou les modules 01, 02, 05, 06 en portent la charge principale
5. **Code standalone** — le script SM-2 tourne avec `python 02-code/03-spaced-repetition.py` sans dependance externe
