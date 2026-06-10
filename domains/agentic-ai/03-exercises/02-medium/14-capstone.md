# Exercices Medium — Capstone (J14)

---

## Exercice 1 : Mode conversationnel avec resolution de suivi

### Objectif
Transformer le capstone one-shot en assistant conversationnel : les questions de suivi ("And its revenue?") doivent etre resolues contre le contexte de session avant d'entrer dans le pipeline.

### Consigne
En partant de `02-code/14-capstone.py` :

1. Cree une classe `Session` : `history: list[tuple[query, report]]`, `entities: list[str]` (entites detectees dans les queries precedentes, ex: "Acme"), `last_topic: str | None` (ex: "revenue", "team")
2. Ecris un resolveur local `resolve_followup(query, session) -> tuple[str, bool]` :
   - "And its revenue?" / "What about its team?" -> substitue le possessif par la derniere entite -> ("What is the revenue of Acme?", True)
   - "Tell me more" -> reformule avec le dernier topic ET la derniere entite ("More details about the revenue of Acme")
   - "Same question for Globex" -> reapplique le dernier topic a la nouvelle entite
   - Une question autonome passe inchangee -> (query, False)
3. Cree `ConversationalResearcher` qui enveloppe `AcmeResearcher` :
   - Resout la query, l'affiche si une resolution a eu lieu (`resolved: ...`)
   - Apres chaque run, met a jour `entities` et `last_topic` (extraction par mots-cles deterministe)
   - Le state de chaque run reste independant (le pipeline n'est pas modifie), seule la session persiste
4. Scenario de demo (4 tours) : "What is the revenue of Acme?" -> "And its team size?" -> "Tell me more" -> "Same question for Globex" (hors corpus -> reponse honnete)
5. Garde-fou : une question de suivi SANS session prealable ("And its revenue?" au tour 1) -> reponse demandant une precision, sans appel au pipeline
6. Asserts : le tour 2 mentionne la team d'Acme ; le tour 4 ne contient pas de chiffre invente ; le garde-fou ne consomme aucun appel LLM

### Criteres de reussite
- [ ] Les 3 formes de suivi sont resolues localement et affichees
- [ ] La session accumule entites et topics au fil des tours
- [ ] Le pipeline sous-jacent n'est pas modifie (composition, pas invasion)
- [ ] Le suivi sans contexte est gere proprement sans appel LLM
- [ ] Les 4 tours de la demo passent leurs asserts

---

## Exercice 2 : Boucle de qualite — judge gate avec retro-feedback au writer

### Objectif
Fermer la boucle qualite du capstone : si le judge note le rapport sous un seuil, le writer retravaille avec le feedback du judge, dans la limite du budget et de 2 retries.

### Consigne
1. Etends `MockLLM.judge` pour retourner, en plus du score, un feedback actionnable : `{"score": 0.55, "missing": ["revenue figure"], "advice": "include the exact figure from findings"}` (mock deterministe : detecte les keywords attendus absents du rapport)
2. Etends `MockLLM.write` avec un parametre optionnel `feedback` : quand il est fourni, le mock produit une version amelioree qui integre les elements manquants (et marque `"[rev N]"` dans le rapport)
3. Cree un `QualityGate(threshold=0.7, max_retries=2)` insere apres le writer :
   - Judge le rapport ; si score >= threshold -> livre
   - Sinon -> re-appelle le writer avec le feedback, re-judge, jusqu'a max_retries
   - Chaque cycle est trace (span `quality.retry`) et consomme du budget normalement
   - Si le seuil n'est jamais atteint : livre la MEILLEURE version avec un avertissement `"[QUALITY: below threshold after 2 revisions]"`
4. Calibre le mock pour la demo : la query 1 passe du premier coup ; la query 2 echoue au premier jet (score 0.55), reussit en rev 1 (score 0.85) ; une query 3 plafonne sous le seuil -> avertissement
5. Verifie l'interaction avec le budget : avec un budget volontairement reduit, la boucle qualite s'arrete proprement sur `BudgetExceeded` et livre la derniere version disponible avec l'avertissement budget
6. Affiche pour chaque query : scores successifs, revisions, verdict final, cout total

### Criteres de reussite
- [ ] Le judge produit un feedback exploitable (missing + advice)
- [ ] Le writer integre le feedback et le score remonte au retry
- [ ] La limite de 2 retries et le seuil sont respectes
- [ ] Le cas "jamais au niveau" livre la meilleure version avec avertissement
- [ ] Le depassement de budget pendant un retry est gere sans crash
- [ ] Les scores et couts par query sont affiches

---

## Exercice 3 : Tableau de bord operationnel du capstone

### Objectif
Construire la vue "ops" du capstone : agreger les metriques de tous les runs (cout, latence, guardrails, verdicts) et produire un dashboard ASCII + un export JSONL.

### Consigne
1. Cree un `OpsCollector` branche sur le tracer et le budget : apres chaque run, il enregistre un `RunRecord` : query (tronquee a 40 chars), verdict, score judge, duree totale ms, cout USD, tokens in/out, nb de spans, guardrail_flags (liste), erreurs
2. Ecris `dashboard(records) -> str` qui rend en ASCII :
   - Un encadre "TOTALS" : runs, cout cumule, tokens cumules, duree moyenne, pass rate
   - Un tableau par run : `# | query | verdict | score | cost | duration | flags`
   - Une mini-barre horizontale du cout par run (`#` proportionnels, normalises sur le max)
   - Le "Top 3 spans les plus lents" tous runs confondus (nom + duree)
3. Ajoute `export_jsonl(records, path)` et `load_jsonl(path)` (round-trip verifie par assert) — chaque ligne est un RunRecord serialise avec un champ `schema_version: 1`
4. Alimente le dashboard en lancant les 3 queries du demo + 1 query qui declenche un guardrail (injection dans l'input) + 1 query qui depasse un mini-budget (BudgetExceeded attrape et enregistre comme erreur)
5. Le dashboard doit montrer : 5 runs, au moins 1 flag guardrail, au moins 1 erreur budget, et des couts varies
6. Bonus : une fonction `alerts(records)` qui retourne les runs anormaux (cout > 2x la mediane, ou erreur) pour affichage en tete de dashboard

### Criteres de reussite
- [ ] Le collector capture toutes les metriques sans modifier la logique metier
- [ ] Le dashboard ASCII est lisible et contient les 4 sections
- [ ] Le run guardrail et le run budget apparaissent avec leurs marqueurs
- [ ] Le round-trip JSONL est exact (assert)
- [ ] Les top spans lents sont corrects par rapport aux traces
- [ ] (Bonus) `alerts` identifie les bons runs
