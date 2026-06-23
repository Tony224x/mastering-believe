# Exercice (medium) — Un PDP per-request : expiration & déprovisionnement

## Objectif

Passer du simple test de scope à un véritable **Policy Decision Point** Zero Trust : une fonction qui décide `allow`/`deny` **à chaque requête** en combinant plusieurs vérifications. Tu ajoutes deux propriétés runtime essentielles — l'**éphémérité** (jetons expirables) et le **déprovisionnement** (couper l'accès instantanément).

## Consigne

Étends ton code de l'exercice easy.

1. Modélise un **jeton d'accès** distinct de l'identité, avec au moins : `agent_id`, `scopes` (ensemble), `expires_at` (un timestamp epoch en secondes).
2. Tiens un petit **registre d'identités** (dict `agent_id -> identité`) pour pouvoir vérifier qu'une identité est connue et `active`.
3. Écris `authorize(registry, token, requested_scope, now) -> (allowed: bool, reason: str)` qui applique, **dans l'ordre**, ces vérifications et s'arrête au premier échec (*fail-closed*) :
   1. identité connue et `active` (sinon deny « unknown/deprovisioned ») ;
   2. jeton non expiré : `now < expires_at` (sinon deny « expired ») ;
   3. scope demandé accordé (sinon deny « scope not granted »).
   Chaque `deny` doit porter un **motif lisible** (il alimentera plus tard un journal d'audit).
4. Démontre les trois comportements avec un `now` simulé (pas besoin d'attendre réellement) :
   - une requête légitime → ALLOW ;
   - la **même** requête avec `now` au-delà de `expires_at` → DENY (expired) ;
   - après avoir passé l'identité à `active=False` (déprovisionnement), même un jeton encore « valide » → DENY (deprovisioned).
5. Ajoute une **probe adversariale** : tente d'autoriser un `agent_id` jamais enregistré → DENY (unknown identity).

## Critères de réussite

- [ ] Le jeton est une structure **séparée** de l'identité (identité = « qui », jeton = scopes + expiration).
- [ ] `authorize()` applique les trois checks dans l'ordre et est **fail-closed** (un seul « non » suffit à refuser).
- [ ] Chaque décision `deny` renvoie un **motif distinct et lisible** (unknown, deprovisioned, expired, scope manquant).
- [ ] La démo prouve l'**expiration** (même requête, `now` différent → résultat différent) et le **déprovisionnement** (token encore non expiré mais identité inactive → deny).
- [ ] La probe sur un `agent_id` inconnu renvoie DENY sans lever d'exception.
