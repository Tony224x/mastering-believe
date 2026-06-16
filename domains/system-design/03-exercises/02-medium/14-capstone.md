# Exercices Medium — Capstone (extensions)

> Ces exercices ETENDENT les 2 designs de reference du J14 (Dropbox et LLM Support Assistant).
> On ne re-concoit pas le systeme : on ajoute/modifie une brique et on chiffre l'impact.

---

## Exercice 1 : Ajouter un tier de cache semantique au LLM Support Assistant

### Objectif
Etendre le design J14 du support LLM en inserant un semantic cache, et chiffrer son impact sur le cout et la latence.

### Consigne
Rappel des chiffres du J14 (LLM Support Assistant) :
- 500K conversations/jour, 5 tours/conversation -> 2.5M LLM calls/jour
- ~3000 tokens in + 500 tokens out par call
- Cible de cout : < $0.10/conversation
- Le design J14 a deja routing + RAG + guardrails, mais le semantic cache est mentionne sans etre dimensionne

Tu ajoutes un **tier de semantic cache** devant le RAG/LLM.

**Questions :**
1. Ou places-tu exactement le semantic cache dans le flow du J14 (avant ou apres le RAG ? avant ou apres les guardrails d'entree) ? Justifie.
2. Quel hit rate realiste pour un support assistant (selon le cours J11) ? Distingue les types de requetes.
3. Avec un hit rate de 40%, combien de LLM calls economises par jour ? Quel % de cout LLM en moins ?
4. Quel threshold et quel scope choisis-tu pour eviter de servir une mauvaise reponse ou de fuiter du PII ?
5. Quel piege specifique au support (vs un chatbot generique) doit te rendre prudent sur le cache ? (donnees personnelles : statut de commande, etc.)
6. Comment mesures-tu que le cache ne degrade pas la qualite (faux positifs) ?

### Criteres de reussite
- [ ] Le cache est place APRES les guardrails d'entree (PII scrub) et AVANT le RAG/LLM (on court-circuite le pipeline cher)
- [ ] Le hit rate distingue FAQ generique (40-60%) et requetes personnelles (~0%)
- [ ] L'economie est chiffree : 40% de 2.5M = 1M calls/jour evites -> ~40% de cout LLM en moins
- [ ] Le threshold est eleve (0.92-0.97) et le scope exclut les requetes avec donnees personnelles
- [ ] Le piege "statut de MA commande" est identifie : ces requetes ne doivent JAMAIS etre servies depuis un cache global
- [ ] La qualite est mesuree par echantillonnage LLM-as-a-judge sur les hits (false positive rate)

---

## Exercice 2 : Dimensionner le Dropbox-like pour 10x le trafic

### Objectif
Reprendre les estimations du J14 (Dropbox) et les re-derouler a 10x, en identifiant quel composant casse en premier.

### Consigne
Rappel des chiffres du J14 (Dropbox) :
- 50M users actifs/jour, 2 uploads + 10 downloads par user/jour, fichier moyen 2 Mo
- Storage net ~200 To/jour (avec replication x3 : ~219 Po/an)
- Bandwidth download ~93 Gbps

La direction annonce une croissance : **x10 sur les users actifs** (50M -> 500M), tout le reste proportionnel.

**Questions :**
1. Recalcule a 10x : uploads/s, downloads/s, bandwidth download (Gbps), storage net/jour et /an.
2. Quel composant casse en PREMIER a 10x ? (bandwidth, storage, metadata DB, notification WebSocket ?) Justifie.
3. Le storage croit lineairement. Quelle strategie pour ne pas exploser les couts ? (tiering, dedup — rappelle le gain dedup du J14)
4. Les downloads passent par le CDN : qu'est-ce qui change a 10x cote CDN ? (hit rate, cout)
5. La metadata DB (Postgres partitionne par user_id) : tient-elle a 10x ? Quel ajustement ?
6. Le service de notification (WebSocket) : combien de connexions concurrentes a 10x ? Quelle architecture pour tenir ?

### Criteres de reussite
- [ ] Les chiffres 10x sont coherents (downloads ~58K/s, bandwidth ~950 Gbps, storage ~1.9 Po/jour net)
- [ ] Le composant qui casse en premier est identifie avec justification (bandwidth download et/ou WebSocket connections)
- [ ] La strategie storage rappelle le tiering (hot/warm/cold) ET la dedup (30-40% economises) du J14
- [ ] L'effet CDN a 10x est traite (le hit rate sur les fichiers populaires limite la croissance de la bande passante origine)
- [ ] La metadata DB est traitee (read replicas, sub-partitioning, cache du folder tree)
- [ ] Les connexions WebSocket a 10x sont estimees et l'archi proposee (pool de serveurs WS + pub/sub Redis)

---

## Exercice 3 : Appliquer le framework 6-etapes a une extension multi-langue

### Objectif
Reutiliser le framework de resolution du J14 pour concevoir une extension precise (multi-langue) du support LLM, sans tout re-concevoir.

### Consigne
Le LLM Support Assistant du J14 doit maintenant supporter **12 langues** (il etait monolingue). Tu dois concevoir CETTE extension en suivant le framework.

**Questions (suis les etapes du framework) :**
1. **Clarifier** : quels nouveaux requirements fonctionnels et non-fonctionnels la multi-langue ajoute-t-elle ?
2. **Estimer** : la knowledge base (100K articles) doit-elle etre traduite et re-indexee ? Quel impact sur le vector index (taille, cout d'embedding) ?
3. **Design** : 2 approches — (A) traduire la query vers l'anglais puis RAG en anglais puis re-traduire ; (B) indexer la KB dans chaque langue. Compare-les (qualite, cout, latence).
4. **Deep dive** : comment geres-tu le retrieval cross-lingue ? (embeddings multilingues ? reranker multilingue ?)
5. **Bottlenecks** : ou la qualite risque-t-elle de chuter en multi-langue ? (retrieval, generation, guardrails PII par langue)
6. **Extension** : comment detectes-tu automatiquement la langue et route-tu ?

### Criteres de reussite
- [ ] Les nouveaux requirements distinguent fonctionnel (repondre dans la langue de l'user) et non-fonctionnel (qualite homogene par langue)
- [ ] L'impact sur le vector index est estime (approche B = ~12x les chunks, ou embeddings multilingues = 1x)
- [ ] Les 2 approches sont comparees ; les embeddings multilingues + reranker multilingue sont identifies comme l'option moderne (1 index, retrieval cross-lingue)
- [ ] Le deep dive aborde le retrieval cross-lingue (modele d'embedding multilingue type BGE-m3, Cohere multilingue)
- [ ] Les bottlenecks couvrent la qualite par langue (recall plus faible sur langues rares, PII detection par langue)
- [ ] La detection de langue + routing est proposee (classifieur leger en premiere etape)
