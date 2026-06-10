# Exercices Medium — RAG Architecture

---

## Exercice 1 : Pipeline d'ingestion incrementale a l'echelle

### Objectif
Concevoir le cote "indexation" d'un RAG d'entreprise avec mises a jour continues.

### Consigne
Tu construis le RAG interne d'une entreprise : **2 millions de documents** (Confluence, Google Drive, tickets), dont **50 000 modifies ou crees par jour**. Les utilisateurs doivent voir les mises a jour dans l'index en **moins de 15 minutes**.

1. **Architecture d'ingestion** : dessine le pipeline complet (detection des changements -> extraction -> chunking -> embedding -> upsert vector DB). Push (webhooks) ou pull (polling) pour chaque source ?
2. **Mise a jour d'un document modifie** : le document a change, ses anciens chunks sont dans l'index. Decris la strategie d'upsert (comment retrouver et remplacer les anciens chunks ? quel identifiant stable ?).
3. **Dimensionnement embedding** : 50K docs/jour, ~20 chunks/doc, le modele d'embedding traite 500 chunks/sec par instance. Combien d'instances pour absorber un backfill complet (2M docs) en 24h ? Et pour le flux quotidien ?
4. **Permissions** : les documents ont des ACLs (un commercial ne doit pas retrouver les docs RH). Ou appliquer le filtrage : a l'ingestion, au retrieval, ou apres ? Quel est le piege du post-filtering ?
5. **Detection des regressions** : apres un changement de modele d'embedding, faut-il re-indexer tout le corpus ? Pourquoi ?

### Criteres de reussite
- [ ] Le pipeline est event-driven (webhooks quand dispo, polling en fallback) avec une queue entre detection et processing
- [ ] L'upsert s'appuie sur un doc_id stable + version : delete des chunks de l'ancienne version puis insert (ou upsert par chunk_id deterministe doc_id#chunk_n)
- [ ] Backfill : 2M x 20 = 40M chunks / 86400 s = ~460 chunks/s -> ~1 instance ne suffit pas avec marge ; ~2-3 instances ; flux quotidien : 1M chunks/jour = ~12 chunks/s -> 1 instance largement
- [ ] Le filtrage ACL au retrieval (metadata filter dans la query vector DB) ; post-filtering = risque de fuite et de top-k vide
- [ ] Re-indexation totale obligatoire : les espaces d'embedding de deux modeles ne sont pas comparables

---

## Exercice 2 : Optimiser la qualite de retrieval mesurable

### Objectif
Mettre en place une demarche d'evaluation chiffree et ameliorer un pipeline de retrieval etape par etape.

### Consigne
Ton RAG support client repond a cote de la plaque. Tu as construit un gold set de **200 questions** avec les documents attendus. Mesures actuelles avec dense retrieval seul (top-10) : **recall@10 = 0.62**, **MRR = 0.41**.

1. Definis precisement recall@10 et MRR, et explique ce que la combinaison (recall correct mais MRR bas) revele.
2. Les questions contiennent beaucoup de references exactes (codes produit "XR-200", messages d'erreur). Pourquoi le dense retrieval echoue la-dessus, et que proposes-tu ? Decris la fusion RRF (formule incluse).
3. Apres hybrid retrieval : recall@10 = 0.78. Tu ajoutes un reranker cross-encoder sur le top-50. Explique pourquoi on recupere top-50 pour ne garder que top-5, et l'impact latence (reranker : 15 ms par paire, en batch de 50 : ~80 ms).
4. Le chunking actuel est fixe a 512 tokens et coupe les procedures en deux. Propose une strategie mieux adaptee a de la doc de support (structure title/section/steps).
5. Apres chaque changement, que re-mesures-tu et quel garde-fou poses-tu pour eviter les regressions silencieuses en prod ?

### Criteres de reussite
- [ ] recall@10 = part des questions dont un doc attendu est dans le top-10 ; MRR = moyenne de 1/rang du premier doc pertinent ; MRR bas = les bons docs sont trouves mais mal classes
- [ ] BM25/sparse ajoute pour le lexical exact + RRF : score = somme des 1/(k + rang) sur chaque retriever (k~60)
- [ ] Le reranker re-classe finement un pool large (recall du retriever + precision du cross-encoder) ; latence ajoutee ~80-100 ms a budgeter
- [ ] Chunking structurel (par section/heading, procedures entieres, overlap ou parent-document retrieval)
- [ ] Re-evaluation systematique sur le gold set + eval continue en prod (echantillonnage + LLM-judge ou feedback users) avec seuil de regression

---

## Exercice 3 : RAG multi-tenant — isolation et couts

### Objectif
Adapter une architecture RAG a un produit SaaS multi-clients.

### Consigne
Ton produit SaaS offre "chat avec vos documents" a **500 entreprises clientes** : de 100 docs (PME) a 500K docs (grand compte). Confidentialite stricte entre tenants. Budget infra a optimiser.

1. **Isolation vector DB** : compare 3 options — (a) un index/collection par tenant, (b) un index partage avec filtre tenant_id, (c) un cluster dedie par tenant. Recommande une strategie (eventuellement par tier de client).
2. **Risque majeur** : decris le scenario de fuite cross-tenant le plus probable dans l'option (b) et la defense en profondeur (3 couches minimum).
3. **Couts d'embedding** : un grand compte arrive avec 500K docs (10M chunks). A 0.02 $/M tokens et ~400 tokens/chunk, combien coute son onboarding ? Faut-il refacturer ?
4. **Noisy neighbor** : un tenant lance 50 req/s (les autres font < 1 req/s). Quelles protections pour que les autres tenants ne soient pas degrades ?
5. **Metriques par tenant** : quelles 4 metriques suivre par tenant pour piloter qualite et marge ?

### Criteres de reussite
- [ ] Recommandation par tier : index partage + filtre pour les petits, collection ou cluster dedie pour les grands/regules ; justification cout vs isolation
- [ ] Scenario de fuite : oubli/bug du filtre tenant_id dans UNE query ; defenses : filtre impose par middleware (pas par l'appelant), tests automatiques cross-tenant, namespacing/scoped API keys, audit logs
- [ ] Calcul : 10M chunks x 400 tokens = 4B tokens x 0.02 $/M = 80 $ — cout faible, mais le re-embedding recurrent et le stockage vector (10M vecteurs) dominent
- [ ] Protections : rate limiting par tenant, quotas, pools de capacite separes ou priorites, degradation par tenant
- [ ] Metriques : latence p95, taux de reponses avec citation/groundedness, cout par requete (tokens), volume de requetes (+ hit rate cache le cas echeant)
