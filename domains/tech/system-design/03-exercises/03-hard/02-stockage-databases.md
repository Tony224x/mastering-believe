# Exercices Hard — Stockage & Databases

---

## Exercice 1 : Design du stockage pour un systeme de paiement multi-region

### Objectif
Appliquer sharding, replication, consistency et choix de DB dans un contexte critique ou les erreurs coutent de l'argent.

### Consigne
Tu concois le layer de stockage pour un systeme de paiement (type Stripe simplifie) avec les contraintes suivantes :

**Requirements fonctionnels :**
- Creer un paiement (amount, currency, merchant_id, customer_id)
- Consulter l'etat d'un paiement (pending -> confirmed -> settled / failed)
- Lister les paiements d'un merchant (avec pagination et filtres par date/statut)
- Refund (partiel ou total)
- Dashboard merchant temps reel (volume/jour, montant total, taux de succes)

**Requirements non-fonctionnels :**
- 10K transactions/sec en pic
- Disponibilite > 99.99% (52 min de downtime/an max)
- Consistance forte obligatoire pour les soldes et les transitions d'etat
- Latence d'ecriture < 50ms au p99
- Multi-region (EU + US) pour la compliance GDPR
- Retention : 7 ans (obligation legale)
- Auditabilite : chaque changement d'etat doit etre tracable

**Livre :**

1. **Choix de DB** :
   - Quelle(s) DB pour les transactions de paiement ? Justifie.
   - Quelle DB pour le dashboard analytics ? Justifie.
   - Quelle DB pour l'audit log immutable ? Justifie.
   - Pour chaque choix, donne le consistency model et explique pourquoi.

2. **Schema de donnees** :
   - Tables/collections pour les paiements, refunds, audit log
   - Justifie les types et les contraintes
   - Comment modeliser les transitions d'etat (state machine) ?

3. **Sharding strategy** :
   - Shard key pour la table des paiements. Justifie.
   - Comment gerer "lister les paiements d'un merchant" si la shard key est `payment_id` ?
   - Combien de shards pour 10K TPS ? (montre le calcul)

4. **Replication multi-region** :
   - Comment garantir la consistance forte entre EU et US ?
   - Quel compromis de latence acceptes-tu ?
   - Comment gerer le GDPR (donnees EU ne quittent pas l'EU) ?

5. **Disponibilite 99.99%** :
   - Identifie les single points of failure
   - Propose les mecanismes de failover
   - Comment tester la resilience ? (chaos engineering)

6. **Estimation de stockage** :
   - Taille d'une transaction (tous les champs)
   - Stockage pour 7 ans a 10K TPS
   - Strategie d'archivage pour les donnees anciennes

### Criteres de reussite
- [ ] Au moins 3 DB differentes choisies pour les 3 workloads (OLTP, analytics, audit)
- [ ] Schema complet avec state machine pour les transitions de paiement
- [ ] Shard key justifiee avec analyse des access patterns
- [ ] Le compromis latence vs consistance multi-region est explicite
- [ ] La strategie GDPR est coherente (data residency par region)
- [ ] Le calcul de 99.99% est detaille avec les SPOFs identifies
- [ ] Estimation de stockage sur 7 ans (attendu : ~200-500 To)

---

## Exercice 2 : Post-mortem — La migration qui a casse la production

### Objectif
Analyser les causes racines d'un incident lie au stockage et concevoir les protections pour eviter la recidive.

### Consigne
Voici le rapport d'incident (resume) :

**Contexte** : Une equipe a migre la table `user_sessions` (800M lignes) de PostgreSQL vers Redis pour ameliorer les performances (latence de 15ms a < 1ms).

**Timeline de l'incident** :

| Heure | Evenement |
|---|---|
| 14:00 | Debut de la migration. Double-write active : chaque session est ecrite dans PostgreSQL ET Redis. |
| 14:30 | Bascule des reads vers Redis. PostgreSQL garde les writes comme backup. |
| 15:00 | Tout semble OK. L'equipe supprime le double-write vers PostgreSQL. |
| 17:00 | Redis Sentinel detecte un probleme : le master Redis est OOM (Out of Memory). Redis commence a evicter des cles. |
| 17:05 | 30% des sessions sont evictees. 30% des utilisateurs sont deconnectes. |
| 17:10 | L'equipe tente de rollback vers PostgreSQL, mais les sessions creees depuis 15:00 n'y sont pas (double-write supprime a 15:00). |
| 17:30 | Decision de restart Redis avec plus de memoire. Mais les sessions evictees sont perdues. |
| 18:00 | Redis redemarre avec 2x la RAM. Les utilisateurs doivent se re-authentifier. |
| 18:30 | Incident clos. ~100K utilisateurs impactes, 90 min de degradation. |

**Questions :**

1. **Root cause analysis** :
   - Identifie la cause racine (pas les symptomes)
   - Identifie au moins 3 facteurs contributifs
   - Pour chaque facteur, explique COMMENT il aurait pu etre detecte avant l'incident

2. **Analyse des decisions** :
   - A quel moment l'equipe aurait du s'arreter ?
   - Pourquoi le "tout semble OK" a 15:00 etait dangereux ?
   - Le rollback a 17:10 etait voue a l'echec — pourquoi ?

3. **Corrections** :
   - Pour chaque facteur contributif, propose une correction technique concrete
   - Concois un plan de migration qui aurait evite l'incident (etapes, checkpoints, rollback plan)
   - Quel monitoring aurait detecte le probleme AVANT l'impact utilisateur ?

4. **Architecture cible** :
   - Propose une architecture Redis resiliente pour les sessions (HA, persistence, eviction policy)
   - Quel maxmemory-policy utiliser et pourquoi ?
   - Comment dimensionner la RAM Redis correctement ? (montre le calcul : 800M sessions * taille moyenne)
   - Faut-il garder PostgreSQL comme fallback ? Si oui, comment ?

5. **Checklist de migration** :
   - Ecris une checklist generique (10+ points) applicable a toute migration de DB en production

### Criteres de reussite
- [ ] La cause racine est identifiee : dimensionnement memoire insuffisant + suppression prematuree du fallback
- [ ] Au moins 3 facteurs contributifs realistes (pas de monitoring, pas de load test, pas de rollback plan)
- [ ] Le plan de migration corrige est en phases avec checkpoints (canary, pourcentage progressif)
- [ ] Le monitoring propose detecte le probleme AVANT l'impact (alertes sur RAM usage, eviction rate)
- [ ] L'architecture Redis est complete (Sentinel/Cluster, persistence AOF, eviction noallkeys-lru)
- [ ] Le calcul de dimensionnement RAM est present et correct
- [ ] La checklist est generique, actionable, et couvre : load test, rollback, monitoring, canary, data validation
