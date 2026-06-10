# Exercices Medium — Agent Systems Architecture

---

## Exercice 1 : Concevoir un agent support client avec tools

### Objectif
Specifier completement un agent single-agent production-ready : state, tools, boucle, conditions d'arret, garde-fous.

### Consigne
Tu concois un agent de support pour un SaaS de facturation. Il doit pouvoir : repondre aux questions produit (docs), consulter le compte du client, emettre un avoir (credit note) jusqu'a 200 €, et escalader a un humain.

1. **Tools** : definis les 4-5 tools (nom, parametres types, retour, effets de bord). Lesquels sont read-only et lesquels sont des actions irreversibles ?
2. **State** : definis le schema du state de l'agent (messages, donnees client chargees, actions effectuees, compteurs). Pourquoi un state structure plutot que "tout dans l'historique de messages" ?
3. **Boucle et arret** : ecris le pseudo-code de la boucle agentique avec TOUTES les conditions d'arret (succes, max iterations, budget tokens, escalade). Quelles valeurs choisis-tu et pourquoi ?
4. **Garde-fou sur l'avoir** : l'agent ne doit JAMAIS emettre un avoir > 200 € ni 2 avoirs pour le meme ticket. Ou implementer ces regles : dans le prompt, dans le code du tool, ou les deux ? Justifie.
5. **Human-in-the-loop** : quels declencheurs d'escalade automatique definis-tu (en plus de la demande explicite du client) ?

### Criteres de reussite
- [ ] Les tools sont types et classes : search_docs / get_account (read-only), issue_credit_note (irreversible, borne), escalate_to_human
- [ ] Le state contient au minimum : messages, ticket_id, actions log, iteration count, tokens used ; justification : controle programmatique des limites et audit
- [ ] La boucle a >= 4 conditions d'arret avec valeurs chiffrees (ex : max 10 iterations, budget 50K tokens)
- [ ] Les regles de l'avoir sont appliquees DANS LE CODE du tool (le prompt aide mais ne garantit rien) — defense en profondeur
- [ ] Escalade automatique sur : sentiment negatif/menace de churn, depassement des limites, confiance basse, sujet hors perimetre, 2 echecs consecutifs de tool

---

## Exercice 2 : Refactorer un multi-agent qui sur-coute

### Objectif
Auditer une architecture multi-agent et la simplifier en justifiant chaque suppression.

### Consigne
Une equipe a construit ce pipeline pour generer des fiches produit e-commerce :

```
Supervisor Agent
  -> Research Agent (cherche les specs produit, 3-5 tool calls)
  -> Writing Agent (redige la fiche)
  -> Critique Agent (relit et critique)
  -> Revision Agent (applique les critiques)
  -> SEO Agent (optimise les mots-cles)
  -> Compliance Agent (verifie les mentions legales)
```

Constats : 45-90 s par fiche, ~120K tokens par fiche, 12% des fiches partent en boucle critique/revision > 3 cycles, et la qualite n'est pas meilleure qu'un prototype single-agent fait en 2 jours.

1. Pour chaque agent, decide : garder comme agent, fusionner dans un autre, remplacer par un appel LLM simple (sans boucle), ou remplacer par du code deterministe. Justifie.
2. La boucle critique/revision diverge dans 12% des cas. Pourquoi ce pattern boucle-t-il facilement, et quels sont les 2 mecanismes pour la borner ?
3. Propose l'architecture cible et estime les nouveaux couts (tokens, latence) avec des hypotheses raisonnables.
4. Le handoff entre agents perd du contexte (le Writing Agent ignore des specs trouvees par le Research Agent). Quelle est la cause structurelle dans un supervisor pattern, et les 2 remedes ?
5. Quels criteres OBJECTIFS aurais-tu poses AVANT de choisir multi-agent vs single-agent ?

### Criteres de reussite
- [ ] Compliance -> regles deterministes (regex/checklist) ; SEO -> fusion dans le prompt de redaction ; critique/revision -> un seul appel "self-review" ou une seule passe ; research garde des tools mais pas besoin d'etre un agent separe
- [ ] La boucle diverge car les critiques LLM trouvent TOUJOURS quelque chose ; mitigation : max 1-2 cycles + critere d'acceptation explicite (rubrique scoree, seuil)
- [ ] L'architecture cible tient en 1 agent + tools (ou pipeline lineaire), avec une estimation chiffree (~3-5x moins de tokens, latence divisee par 2+)
- [ ] Cause : chaque handoff resume/tronque le contexte ; remedes : state partage structure (artifacts) plutot que resumes, ou contexte complet transmis
- [ ] Criteres avances : parallelisme reel necessaire ? expertises/tools incompatibles dans un seul contexte ? benchmark single-agent d'abord ?

---

## Exercice 3 : Memoire long terme d'un agent personnel

### Objectif
Concevoir l'architecture memoire (court terme + long terme) d'un agent avec contraintes de contexte.

### Consigne
Ton agent assistant personnel discute avec chaque utilisateur sur des mois (300+ conversations). Fenetre de contexte du modele : 128K tokens, mais au-dela de ~40K tokens de contexte la qualite et la latence se degradent. L'agent doit se souvenir : des preferences (style, contraintes), des faits durables (metier, projets en cours), et des conversations passees pertinentes.

1. **Court terme** : la conversation courante depasse 40K tokens. Decris ta strategie (summarization progressive, fenetre glissante, les deux ?) et ce que tu perds dans chaque cas.
2. **Long terme** : concois le store (quels types de memoires, quel schema, vector + relationnel ?). Comment une memoire est-elle ECRITE (qui decide qu'un fait merite d'etre memorise ?) et RELUE (quand injecter quoi dans le contexte) ?
3. **Conflits** : l'utilisateur disait "je travaille chez Acme" il y a 6 mois, et dit aujourd'hui "je viens de rejoindre Beta Corp". Comment gerer la contradiction (versioning, recence, confiance) ?
4. **Budget contexte** : pour une nouvelle requete, repartis un budget de 8K tokens d'injection memoire entre : preferences, faits durables, souvenirs pertinents recuperes. Justifie la repartition et le mecanisme de selection.
5. **Vie privee** : quelles memoires NE PAS stocker, et quel controle donner a l'utilisateur ?

### Criteres de reussite
- [ ] Court terme : fenetre glissante des N derniers messages + resume cumulatif du debut ; perte identifiee : details fins du milieu de conversation
- [ ] Long terme : memoires typees (preference, fait, episode) avec schema (contenu, type, timestamp, source, confiance) ; ecriture par extraction LLM en fin de conversation ; lecture par retrieval semantique + regles (preferences toujours injectees)
- [ ] Conflit : pas d'ecrasement aveugle — versioning avec timestamps, la recence prime pour les faits, confirmation a l'utilisateur si doute
- [ ] La repartition est chiffree et argumentee (ex : 1K preferences, 2K faits, 5K retrieval) avec selection top-k par similarite
- [ ] Vie privee : pas de donnees sensibles non sollicitees (sante, etc.), commandes "oublie ca" / consultation des memoires, TTL ou revue periodique
