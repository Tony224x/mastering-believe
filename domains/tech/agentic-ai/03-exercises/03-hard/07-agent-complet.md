# Exercices Hard — Agent complet (J7, capstone semaine 1)

---

## Exercice 1 : Executor parallele avec fan-out / fan-in

### Objectif
Le J7 execute les etapes du plan une par une (sequentiel). Beaucoup d'etapes sont pourtant **independantes** (chercher l'aire et chercher la population n'ont aucune dependance). Implemente un executor qui execute en parallele les etapes independantes et merge leurs resultats.

### Consigne
Construis un mini-orchestrateur "fan-out / fan-in" :

1. Le planner produit, en plus des steps, un **graphe de dependances** : chaque step declare ses `depends_on` (liste d'ids de steps). Ex :
   ```python
   plan = [
       {"id": "s1", "action": "search:africa area",        "depends_on": []},
       {"id": "s2", "action": "search:africa population",  "depends_on": []},
       {"id": "s3", "action": "compute:density",           "depends_on": ["s1", "s2"]},
       {"id": "s4", "action": "format:final_answer",       "depends_on": ["s3"]},
   ]
   ```
2. L'orchestrateur calcule les **vagues d'execution** (topological levels) : `s1` et `s2` sont dans la vague 0 (executables en parallele), `s3` dans la vague 1, `s4` dans la vague 2
3. Simule le parallelisme avec un `ThreadPoolExecutor` (les tools sont I/O-bound -> les threads sont legitimes) et un `time.sleep` de 0.05s par tool pour rendre le gain mesurable
4. Verifie 2 choses :
   - **Correction** : le resultat final est identique a l'execution sequentielle
   - **Gain** : le temps total est proche de `max(vague)` et non de `sum(steps)` (mesure les deux et compare)
5. Detecte et rejette un graphe cyclique (`s1 depends_on s2`, `s2 depends_on s1`) avec une erreur claire

### Criteres de reussite
- [ ] Le plan porte un graphe de dependances explicite (`depends_on`)
- [ ] Le tri topologique en vagues est correct (s1/s2 ensemble, puis s3, puis s4)
- [ ] L'execution parallele donne le MEME resultat que le sequentiel
- [ ] Le temps mesure montre un gain reel (parallele < sequentiel)
- [ ] Un graphe cyclique leve une erreur explicite (pas de boucle infinie)
- [ ] Le code est generaliste : ajouter une 3e recherche independante l'integre automatiquement dans la vague 0

---

## Exercice 2 : Agent self-correcting avec verification de plausibilite

### Objectif
Un agent production-credible ne fait pas que collecter des chiffres : il **verifie leur plausibilite** et se corrige tout seul quand un resultat est aberrant (mauvaise unite, ordre de grandeur impossible, division par zero).

### Consigne
Ajoute une couche de verification entre l'analyzer et le synthesizer :

1. Ecris un `PlausibilityChecker` avec des regles declaratives par type de fait :
   ```python
   RULES = {
       "area_km2":   {"min": 1, "max": 200_000_000},      # rejette 0 et les valeurs delirantes
       "population": {"min": 1, "max": 10_000_000_000},
       "density":    {"min": 0.01, "max": 50_000},
   }
   ```
2. Si un fait viole sa regle (ex: une `area_km2` extraite vaut `30` parce que le regex a chope le mauvais nombre), le checker :
   - Marque le fait comme `suspect`
   - Declenche une **re-extraction** via un outil de fallback (`read_doc` du rapport detaille au lieu du snippet de search)
   - Re-verifie ; si toujours suspect apres 1 retry, le fait est marque `unreliable` et le synthesizer l'annonce explicitement
3. Verifie aussi les **derivations** : la densite doit etre coherente avec `population / area` a +/- 5%. Si l'agent calcule une densite incoherente avec ses propres faits, il la recalcule.
4. Teste avec :
   - Un cas nominal (tout plausible, 0 correction)
   - Un cas ou l'aire extraite est aberrante (le checker declenche une re-extraction qui corrige)
   - Un cas ou aucune source ne donne une valeur plausible (l'agent annonce `unreliable` sans halluciner)

### Criteres de reussite
- [ ] `PlausibilityChecker` applique des regles declaratives par type de fait
- [ ] Un fait aberrant declenche une re-extraction via fallback
- [ ] La coherence de la derivation (densite vs population/aire) est verifiee a +/- 5%
- [ ] Apres 1 retry infructueux, le fait est marque `unreliable` (pas d'hallucination)
- [ ] Les 3 scenarios produisent respectivement 0, 1 et >=1 corrections, tous sans crash
- [ ] La reponse finale distingue les faits fiables des faits `unreliable`
