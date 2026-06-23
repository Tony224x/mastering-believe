# Exercice (medium) — Persistance, Agent Card & reconciliation

## Objectif

Faire passer ton registry du statut de « liste en memoire » a celui de **controle vivant et durable** : il survit a l'arret du processus (persistance JSON), il s'alimente via des **Agent Cards** declaratives validees, et il sait detecter les **agents fantomes** par reconciliation entre ce qui est declare et ce qui agit.

## Consigne

1. Reprends (ou recree) un `AgentRegistry` avec les champs des 4 piliers + un champ `status` (`"active"` par defaut).
2. **Enrolement par Agent Card** : ecris une fonction `validate_agent_card(card: dict) -> list[str]` qui retourne la liste des problemes (vide = valide). Regles minimales :
   - `agent_id` present et non vide ;
   - `risk_tier`, s'il est fourni, appartient a l'ensemble autorise ;
   - `permissions`, s'il est fourni, est bien une liste.
   Puis `enrol_from_card(card)` qui **valide d'abord** et refuse une carte invalide (lever une erreur).
3. **Persistance** : implemente `save(path)` (ecrit le registry en JSON sur disque) et `load(path)` (recree un registry depuis le fichier). Verifie le **round-trip** : `load(save(...))` redonne le meme nombre d'agents et les memes owners.
4. **Reconciliation** : ecris `reconcile(observed_ids: list[str]) -> list[str]` qui retourne les `agent_id` **observes** (telemetrie simulee) mais **absents** du registry — les fantomes a investiguer.
5. **Couverture** : ajoute `coverage()` qui retourne un `dict` avec `total_agents`, `governed_agents` (agents dont les 4 piliers sont remplis et non vides), `orphans`, et `coverage_pct`.
6. Dans `if __name__ == "__main__":` : enrole 4-5 agents (au moins 1 orphelin), montre un refus de carte invalide, fais un round-trip de persistance, affiche le resultat de `reconcile([...])` et de `coverage()`.

## Criteres de reussite

- [ ] `validate_agent_card` detecte au moins : `agent_id` manquant, `risk_tier` invalide, `permissions` non-liste.
- [ ] `enrol_from_card` **refuse** une carte invalide (test adversarial : une carte sans `agent_id` leve une erreur).
- [ ] `save` + `load` font un round-trip fidele (meme nombre d'agents, memes owners) — le fichier JSON est relisible.
- [ ] `reconcile` renvoie exactement les ids observes absents du registry.
- [ ] `coverage()` calcule un `coverage_pct` correct (ex. 4 gouvernes sur 5 => 80.0).
- [ ] Script en **stdlib uniquement** (`json` autorise), `python <fichier>` exit 0, et `python -m py_compile` passe.
