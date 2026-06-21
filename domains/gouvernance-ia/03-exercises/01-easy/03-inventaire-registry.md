# Exercice (easy) — Mon premier registry d'agents

## Objectif

Construire un registry d'agents minimal en memoire et repondre, **par requete**, a la question fondatrice du domaine : « combien d'agents tournent, et qui les possede ? ». Tu manipules ici les 4 piliers de gouvernance (identite, owner, permissions, risk_tier) sous forme de donnees interrogeables — pas un tableur.

## Consigne

1. Definis une structure d'agent (dataclass ou simple `dict`) avec au minimum les champs : `agent_id` (str unique), `owner` (str ou `None`), `permissions` (liste de str), `risk_tier` (un de `"minimal"`, `"limited"`, `"high"`).
2. Cree une classe `AgentRegistry` qui stocke les agents dans un `dict` indexe par `agent_id`. Implemente :
   - `add(agent)` — ajoute un agent ; **leve une erreur** si l'`agent_id` existe deja (l'identite doit etre unique).
   - `count()` — retourne le nombre total d'agents.
   - `by_owner(owner)` — retourne la liste des agents possedes par `owner`.
   - `orphans()` — retourne la liste des agents dont `owner` est `None` (ou vide).
3. Enregistre **au moins 4 agents**, dont **exactement un orphelin** (sans owner).
4. Dans un bloc `if __name__ == "__main__":`, affiche : le nombre total d'agents, les agents d'un owner donne, et la liste des orphelins.

## Criteres de reussite

- [ ] La structure d'agent porte les 4 champs (`agent_id`, `owner`, `permissions`, `risk_tier`).
- [ ] `add()` refuse un `agent_id` deja present (test : tenter d'ajouter deux fois le meme id leve une erreur).
- [ ] `count()` renvoie le bon total.
- [ ] `by_owner()` ne renvoie que les agents du bon owner.
- [ ] `orphans()` renvoie exactement l'agent sans owner (ni plus, ni moins).
- [ ] Le script s'execute sans erreur en **stdlib uniquement** (`python <fichier>`), exit 0.
