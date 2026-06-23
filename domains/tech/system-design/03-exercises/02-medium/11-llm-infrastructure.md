# Exercices Medium — LLM Infrastructure

---

## Exercice 1 : Optimiser la facture LLM par le routing et le cache

### Objectif
Chiffrer l'impact combine du routing par tier, du semantic cache et du prompt caching sur une facture LLM reelle.

### Consigne
Ton produit consomme actuellement un seul modele "standard" pour tout.

**Chiffres actuels :**
- 3M requetes/jour
- Tokens moyens : 2500 in + 400 out par requete
- Prix du modele standard : $2.50 / 1M tokens in, $10.00 / 1M tokens out
- System prompt fixe : 1200 tokens (inclus dans les 2500 in)

**Mix de taches observe :**
- 50% classification/extraction (servable par un modele nano : $0.05/1M in, $0.40/1M out)
- 30% Q&A/resume (servable par un modele mini : $0.15/1M in, $0.60/1M out)
- 20% reasoning/code (necessite le modele standard)

**Questions :**
1. Calcule le cout mensuel actuel (tout en standard).
2. Calcule le cout mensuel apres routing par tier (chaque tache sur le bon modele).
3. Un semantic cache atteint 25% de hit (les hits ne coutent rien en LLM). Quel cout apres routing + cache ?
4. Le prompt caching natif reduit de 90% le cout des 1200 tokens de system prompt repetes (lecture a 10% du prix). Quel gain en plus ?
5. Donne le cout final et le facteur de reduction total vs le point de depart.
6. Quel est le poste de cout dominant a la fin (in vs out, et quel tier) ? Que recommandes-tu ensuite ?

### Criteres de reussite
- [ ] Le cout actuel est calcule correctement (de l'ordre de plusieurs centaines de milliers de $/mois)
- [ ] Le routing fait chuter le cout : la majorite du trafic va sur des modeles 10-50x moins chers
- [ ] Le cache de 25% retire 25% du cout LLM restant (sur les requetes, pas les tokens individuels)
- [ ] Le prompt caching s'applique au system prompt repete (pas a tout l'input)
- [ ] Le facteur de reduction total est annonce (typiquement 5-10x)
- [ ] Le poste dominant final est identifie (souvent les tokens out du tier standard) avec une reco

---

## Exercice 2 : Concevoir la chaine de fallback et le circuit breaker

### Objectif
Dimensionner une strategie de fiabilite multi-provider pour atteindre un SLA, en raisonnant sur les disponibilites composees.

### Consigne
Ton produit a un SLA de **99.9%** de disponibilite (max ~43 min de downtime/mois). Tu dependais d'un seul provider.

**Disponibilites observees (sur 90 jours) :**
- Provider A (primaire) : 99.5%
- Provider B (secondaire, autre vendor) : 99.0%
- Provider C (self-hosted, last resort) : 98.0%

**Comportement des pannes :**
- Les pannes de A et B sont quasi-independantes (vendors differents, regions differentes)
- Un timeout agressif est fixe a 8s ; au-dela on bascule
- Le circuit breaker : apres 5 echecs consecutifs, un provider est bypass 60s

**Questions :**
1. Avec A seul, le SLA 99.9% est-il atteignable ? Justifie.
2. Calcule la disponibilite composee de la chaine A -> B (au moins un des deux disponible). Le SLA est-il tenu ?
3. Calcule la disponibilite de la chaine A -> B -> C. Quelle marge ?
4. Pourquoi un timeout de 8s est-il critique pour que le fallback "compte" comme de la disponibilite ? Que se passe-t-il avec un timeout de 60s ?
5. A quoi sert le circuit breaker ici ? Donne un scenario ou son absence aggrave une panne.
6. Quels 2 pieges de portabilite faut-il anticiper quand on bascule de A vers B (prompts, formats) ?

### Criteres de reussite
- [ ] A seul (99.5%) est insuffisant pour 99.9% (justifie par le calcul de downtime)
- [ ] La compo A->B est calculee comme 1 - (1-0.995)*(1-0.99) ~ 99.995% -> SLA tenu
- [ ] La compo A->B->C ajoute une marge supplementaire (~99.9999%)
- [ ] Le timeout agressif est justifie : sans lui, une requete lente bloque l'utilisateur et "consomme" le budget de latence/dispo
- [ ] Le circuit breaker evite de marteler un provider mort (sinon chaque requete attend le timeout avant de fallback)
- [ ] Les pieges de portabilite sont concrets (prompt qui marche sur A pas sur B, format JSON, system role)

---

## Exercice 3 : Regler le semantic cache (threshold, scope, TTL)

### Objectif
Comprendre les tradeoffs de configuration d'un semantic cache et eviter ses modes d'echec (faux positifs, fuite de donnees).

### Consigne
Tu deploies un semantic cache devant ton assistant. Tu observes ces situations :

**Donnees :**
- Volume : 1M requetes/jour
- Threshold de similarite actuel : 0.85
- Scope actuel : cache global (toutes requetes confondues, tous users)
- TTL actuel : 24h
- Profil produit : assistant qui mele des questions generiques ("comment fonctionne X ?") et des questions personnelles ("quel est le statut de MA commande 12345 ?")

**Incidents remontes :**
- (I1) Un user a demande "comment annuler mon abonnement ?" et recu une reponse a "comment activer mon abonnement ?" (faux positif).
- (I2) Un user B a recu une reponse en cache contenant le numero de commande d'un user A.
- (I3) Le hit rate sur les prix produits est eleve, mais des users voient des prix d'il y a 2 jours.

**Questions :**
1. Pour I1 : quel parametre est en cause et comment le corriger ? Quel est le risque inverse si tu sur-corriges ?
2. Pour I2 : pourquoi le cache global est-il dangereux ? Quelle politique de scope adopter ?
3. Pour I3 : quel parametre est mal regle pour les prix ? Faut-il un TTL uniforme ?
4. Propose une politique de cache differenciee selon 3 categories : Q&A generique stable, donnees personnelles, donnees time-sensitive.
5. Estime le hit rate realiste attendu pour chacune des 3 categories (en t'appuyant sur les ordres de grandeur du cours).
6. Comment mesurer la qualite du cache (pas juste le hit rate) pour detecter les faux positifs en prod ?

### Criteres de reussite
- [ ] I1 -> threshold trop bas (0.85) -> monter a 0.92-0.97 ; risque inverse : hit rate s'effondre
- [ ] I2 -> le cache global fuit du PII ; politique : pas de cache (ou cache par user) pour les requetes avec donnees personnelles
- [ ] I3 -> TTL trop long pour des prix ; TTL court (minutes) ou invalidation event-driven pour le time-sensitive
- [ ] La politique differenciee couvre les 3 categories avec threshold/scope/TTL distincts
- [ ] Les hit rates estimes sont coherents (generique 30-60%, personnel ~0%, time-sensitive variable selon TTL)
- [ ] La mesure de qualite inclut un echantillonnage LLM-as-a-judge sur les hits pour detecter les faux positifs
