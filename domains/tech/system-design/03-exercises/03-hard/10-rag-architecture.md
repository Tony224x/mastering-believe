# Exercices Hard — RAG Architecture

---

## Exercice 1 : Concevoir un RAG multi-tenant a l'echelle entreprise

### Objectif
Concevoir un RAG SaaS multi-tenant resilient : isolation des donnees, fraicheur de l'index, latence sous contrainte, cout maitrise. C'est un design d'entretien senior complet.

### Consigne
Tu es l'architecte d'un produit "RAG-as-a-Service" vendu a des entreprises (type assistant sur leur Confluence/Notion/Drive).

**Chiffres :**
- 500 tenants (clients), du petit (10K docs) au gros (5M docs)
- Total : 200M chunks tous tenants confondus
- Embedding : 1536 dimensions
- Trafic : 5M requetes/jour, fortement asymetrique (20% des tenants = 80% du trafic)
- Les documents changent : ~2% du corpus de chaque tenant modifie par jour
- SLA : latence p95 < 2s, fraicheur < 5 min (un doc modifie doit etre retrievable en < 5 min)
- Contraintes : isolation stricte (le tenant A ne doit JAMAIS voir un chunk du tenant B), conformite (un doc supprime doit disparaitre de l'index sous 1h)

**Livre :**

1. **Isolation multi-tenant** :
   - 3 strategies possibles : collection par tenant, namespace partage avec filtre, cluster dedie par gros tenant. Compare-les (isolation, cout, ops).
   - Quel modele recommandes-tu, et comment traites-tu differemment les gros et les petits tenants ?
   - Comment garantis-tu qu'un bug de filtre ne fait jamais fuiter un chunk cross-tenant ?

2. **Pipeline d'indexation incrementale** :
   - Comment detecter les 2% de docs modifies sans tout re-indexer ?
   - Quelle architecture pour tenir la fraicheur < 5 min ? (CDC ? webhook ? polling ?)
   - Comment gerer la suppression (doc retire -> chunks retires de l'index) sous 1h ?

3. **Sizing et placement** :
   - Estime la memoire totale de l'index dense (200M * 1536 * 4 bytes) et le nombre de noeuds.
   - Comment places-tu les tenants pour eviter qu'un gros tenant noie un petit (noisy neighbor) ?
   - Faut-il garder tout l'index en RAM ? Propose une strategie hot/cold.

4. **Latence sous contrainte** :
   - Decompose le budget p95 de 2s. Ou mettre un cache ?
   - Un cache semantique est-il pertinent en multi-tenant ? Quels pieges ?

5. **Cout et tarification** :
   - Quels sont les 3 gros postes de cout ? (storage vecteurs, embedding, generation)
   - Comment factures-tu equitablement un petit vs un gros tenant ?

6. **Failure modes** :
   - Que se passe-t-il si la re-indexation prend du retard (backlog) ? Comment degrades-tu proprement ?
   - Comment evites-tu qu'une requete d'un tenant sature l'index d'un autre ?

### Criteres de reussite
- [ ] Les 3 strategies d'isolation sont comparees ; recommandation : namespace partage + filtre pour les petits, collection/cluster dedie pour les gros
- [ ] La garantie anti-fuite repose sur le filtre tenant_id applique au niveau index (pas seulement applicatif) + tests automatises
- [ ] L'indexation incrementale utilise CDC ou webhooks (pas du re-embedding complet) avec une file de travail
- [ ] La suppression est traitee (tombstone / hard delete) avec un SLA explicite
- [ ] Le sizing memoire est coherent (~1.2 To brut, hot/cold pour ne pas tout garder en RAM)
- [ ] Le noisy neighbor est traite (quota par tenant, isolation des gros tenants, rate limiting)
- [ ] Les failure modes couvrent le backlog d'indexation (servir stale + alerter) et l'isolation des charges

---

## Exercice 2 : Post-mortem — Le RAG qui a leak des donnees et hallucine

### Objectif
Analyser un incident composite (fuite de donnees + regression qualite), reconstituer la chaine causale, et concevoir les garde-fous.

### Consigne
Voici le rapport d'incident (resume) d'un assistant RAG interne RH.

**Contexte** : Un assistant RAG repond aux questions des employes sur leurs avantages, conges, fiches de paie. Le corpus contient des documents publics (politique RH) ET des documents personnels (fiches de paie nominatives), tous indexes dans la meme collection avec un champ `acl` (liste des user_id autorises).

**Timeline de l'incident :**

| Heure | Evenement |
|---|---|
| J-7 | Une nouvelle version de l'embedding model est deployee (upgrade de v2 vers v3, meilleur recall en benchmark). |
| J-7 | L'equipe re-indexe le corpus avec le nouveau modele. Le job tourne 6h. Par gain de temps, **seuls les documents publics sont re-embeddes** ; les docs personnels gardent les embeddings v2. |
| J-1 | Une refonte du retrieval ajoute un filtre `acl` cote applicatif (apres le retrieval), pour "simplifier la requete vector DB". Avant, le filtre etait pousse dans la requete vector DB. |
| J0 10:00 | Un employe demande "quel est mon salaire ?". Le retrieval melange des vecteurs v2 et v3 (espaces incompatibles) : les scores de similarite sont incoherents. |
| J0 10:00 | Le top-k retourne des chunks de fiches de paie d'AUTRES employes (scores aberrants a cause du mismatch v2/v3). |
| J0 10:01 | Le filtre `acl` applicatif a un bug : il compare `user_id` (int) a `acl` (liste de strings) -> le filtre ne matche jamais et **laisse tout passer**. |
| J0 10:01 | Le LLM recoit des fiches de paie d'autres employes dans son contexte et **les inclut dans la reponse**. |
| J0 10:05 - 11:30 | ~40 employes voient des donnees salariales d'autrui. Plusieurs captures d'ecran circulent. |
| J0 11:30 | L'incident remonte. L'equipe coupe l'assistant. |
| J0 14:00 | Post-mortem : fuite RGPD majeure, obligation de notification CNIL, perte de confiance. |

**Questions :**

1. **Root cause analysis** :
   - Reconstitue la chaine causale complete (ce n'est pas un seul bug).
   - Pour chaque maillon, identifie le garde-fou manquant.
   - Classe les causes : processus, architecture, securite, qualite.

2. **Le mismatch d'embeddings** :
   - Pourquoi melanger des vecteurs v2 et v3 dans le meme index est-il une erreur grave ?
   - Quelle est la bonne procedure pour migrer un embedding model en prod (zero downtime, zero mismatch) ?

3. **Le filtre ACL** :
   - Pourquoi pousser le filtre cote applicatif (post-retrieval) est-il dangereux ici ?
   - Concois un mecanisme d'isolation defense-in-depth (plusieurs couches independantes).
   - Comment un test automatise aurait-il attrape le bug de type ?

4. **La qualite degradee** :
   - Comment l'absence de gold set re-evalue apres le changement de modele a contribue ?
   - Propose un gate de CI/CD qui aurait bloque ce deploiement.

5. **Resilience et reponse** :
   - Concois un "circuit breaker de confidentialite" : si le retrieval retourne des scores aberrants ou des docs hors-ACL, on coupe plutot que de servir.
   - Propose un runbook de 8 etapes pour un incident de fuite de donnees dans un RAG.

### Criteres de reussite
- [ ] La chaine causale complete est reconstituee : migration partielle v2/v3 -> espaces incompatibles -> scores aberrants -> filtre ACL deplace + bug de type -> fuite -> LLM recopie les donnees
- [ ] Le mismatch d'embeddings est identifie comme cause racine #1 (les distances entre v2 et v3 n'ont aucun sens)
- [ ] La migration zero-mismatch est decrite : re-embedder TOUT le corpus, indexer en parallele (blue/green), basculer atomiquement
- [ ] Le filtre ACL doit etre pousse DANS la requete vector DB (pre-filter), pas applique apres ; defense-in-depth (pre-filter + post-check + audit)
- [ ] Le bug de type aurait ete attrape par un test d'isolation (user A ne doit jamais voir un doc de user B) dans la CI
- [ ] Un gate CI/CD est propose (re-eval gold set + test d'isolation obligatoires avant deploiement)
- [ ] Le circuit breaker de confidentialite coupe sur scores aberrants / docs hors-ACL ; le runbook commence par "couper l'assistant"
