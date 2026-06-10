# Exercices Hard — Design classiques (entretiens)

---

## Exercice 1 : Design Stripe (payment processing) — entretien senior 60 min

### Objectif
Mener un design complet de niveau senior sur un systeme transactionnel critique, en defendant chaque chiffre et chaque tradeoff.

### Consigne
Concois le coeur d'un processeur de paiement (type Stripe) : les marchands integrent une API, les paiements sont autorises aupres des reseaux bancaires, l'argent est reconcilie et reverse.

**Contraintes chiffrees :**
- 10 000 paiements/sec en pic (Black Friday), 1 500 en moyenne
- Latence d'autorisation : < 1 s au p99 (dont 300-600 ms incompressibles cote reseaux bancaires)
- Zero perte : un paiement accepte DOIT aboutir (ou etre rembourse de maniere tracable)
- Zero double charge (incident reglementaire + perte de confiance)
- Les reseaux bancaires externes : disponibilite 99.5%, latence erratique, rate limits
- Reconciliation quotidienne : 100% des paiements doivent matcher les settlements bancaires
- Disponibilite de l'API : 99.99% (52 min de downtime/an)

**Deroule le framework complet :**
1. **Estimation** (5 min) : QPS, stockage transactionnel sur 5 ans (1 paiement ~ 2 Ko + 10 events de 500 octets), bande passante. 
2. **High-level design** : API gateway, payment service, machine a etats, ledger, connecteurs reseaux bancaires, reconciliation. Justifie le choix de DB pour le ledger (et pourquoi PAS un simple UPDATE de solde).
3. **Deep dive — double charge** : idempotency keys + machine a etats + transactions. Deroule le scenario "le client clique 2x" ET le scenario "le reseau bancaire timeout, on ne sait pas si l'autorisation est passee" (le cas le plus dur : quelle strategie ? statut incertain, requete de verification, void preventif ?).
4. **Deep dive — ledger** : pourquoi append-only en double entree ? Schema des tables. Comment calculer le solde d'un marchand avec des millions d'ecritures (snapshots/materialized balances) ?
5. **Disponibilite 99.99% avec des dependances a 99.5%** : montre le calcul du SLA compose naif, puis l'architecture qui s'en affranchit (acceptation asynchrone, multi-acquirer routing, file d'attente persistante).
6. **Bottlenecks & failure modes** : les 3 pires scenarios (panne acquirer en pic, lag de reconciliation, hot merchant) et leurs mitigations.

### Criteres de reussite
- [ ] Estimation posee : 1 500/s x 86 400 = ~130M paiements/jour, ~47B/an ; stockage avec events ~7 Ko/paiement -> ~330 To/an, ~1.6 Po sur 5 ans — les ordres de grandeur doivent etre coherents
- [ ] Le ledger est append-only, double entree (debit + credit par mouvement), avec soldes materialises + invariant verifiable (somme des ecritures = 0)
- [ ] Le scenario timeout-acquirer a une reponse explicite : statut "unknown" + reconciliation/inquiry API + politique de retry avec la MEME cle d'idempotence transmise a l'acquirer
- [ ] Le SLA compose naif est calcule (0.9999 impossible si dependance synchrone a 0.995) et l'architecture decouple l'acceptation de l'execution (accepted -> queued -> authorized)
- [ ] Multi-acquirer routing presente comme reponse a la fois dispo et cout
- [ ] Au moins 4 tradeoffs explicites et 3 failure modes avec mitigation chiffree (timeouts, seuils de circuit breaker)

---

## Exercice 2 : Design YouTube (upload + streaming) — entretien senior 60 min

### Objectif
Couvrir un design read-heavy extreme avec pipeline de traitement lourd et distribution mondiale.

### Consigne
Concois la plateforme video : upload, transcoding, stockage, distribution, et compteurs (vues/likes).

**Contraintes chiffrees :**
- 500 heures de video uploadees PAR MINUTE
- 1 milliard d'heures regardees PAR JOUR, 95% du trafic sur 5% des videos
- Une video uploadee doit etre disponible en < 5 minutes (premiere qualite), toutes qualites en < 30 min
- Qualites : 240p a 4K (6 renditions), codec moderne (H.264 + AV1 pour les videos populaires)
- Demarrage de lecture : < 200 ms au p95 partout dans le monde
- Le stockage doit rester maitrise : renditions froides supprimees ou re-encodees a la demande

**Deroule le framework complet :**
1. **Estimation** : volume d'upload (1h de video brute ~ 3 Go), stockage/jour et /an apres transcoding (les 6 renditions ~ 2x la taille source), bande passante de sortie (1h regardee ~ 1 Go en moyenne).
2. **High-level design** : upload service (resumable !), pipeline de transcoding (queue + workers), object storage, CDN multi-tier, metadata DB, compteurs.
3. **Deep dive — transcoding pipeline** : 500h/min a transcoder en 6 renditions. Si un worker transcode 1h de video en 30 min de calcul par rendition, combien de workers ? Comment paralleliser MIEUX (decoupage en chunks/scenes) pour tenir "< 5 min pour la premiere qualite" ?
4. **Deep dive — distribution** : pourquoi le CDN absorbe 95%+ du trafic ici (correlation avec le 95/5) ; strategie multi-tier (edge/regional/origin) ; que fait-on pour la longue traine (5% du trafic, 95% des videos) ?
5. **Compteur de vues a 1B+ vues/jour** : pourquoi l'increment SQL direct meurt ; architecture d'agregation (buffer -> flush periodique) et tolerance d'approximation.
6. **Bottlenecks** : identifie les 3 principaux et les extensions (live streaming, copyright matching).

### Criteres de reussite
- [ ] Estimation coherente : upload 500h/min = 720K heures/jour ~ 2.2 Po/jour brut, ~4-5 Po/jour avec renditions ; egress : 1B heures x 1 Go ~ 1 Eo/jour soit ~90 Tbps en moyenne — chiffre astronomique qui IMPOSE le CDN ; l'essentiel est la coherence des ordres de grandeur et l'identification que l'egress domine tout
- [ ] L'upload est resumable (chunks + offsets, reprise apres coupure) avec URLs pre-signees vers l'object storage
- [ ] Le transcoding est dimensionne : 500h/min x 6 renditions x 0.5 (ratio calcul/duree) = 1 500 heures de calcul par minute -> ~90 000 workers-minute/min ; le decoupage en chunks paralleles est propose pour la latence < 5 min
- [ ] La distribution exploite le 95/5 : hot au edge CDN, warm regional, longue traine servie depuis l'origin avec cache pull
- [ ] Les compteurs sont agreges (Redis/compteurs shardes + flush batch vers la DB), exactitude relachee assumee
- [ ] Au moins 4 tradeoffs explicites (codec AV1 seulement sur le hot content = cout CPU vs egress, etc.)
