# Exercice (easy) — Détecter les données personnelles d'un agent

## Objectif

Écrire un détecteur minimal qui, à partir des champs déclarés par un agent, dit s'il traite des **données personnelles** (Art. 4(1) RGPD) et s'il touche des **données sensibles** (Art. 9). C'est la première porte de tout assessment de data governance : pas de données personnelles → le cœur du RGPD ne se déclenche pas ; données sensibles → le risque grimpe d'un cran.

## Consigne

1. Crée un fichier Python (stdlib seule) dans ton `workspace/`.
2. Définis deux ensembles : `PERSONAL_FIELDS` (ex. `name`, `email`, `ip_address`, `customer_id`) et `SENSITIVE_FIELDS` (ex. `health`, `biometric`, `religion`).
3. Écris une fonction `classify(declared_fields: list[str]) -> dict` qui retourne :
   - `"personal"` : la liste triée des champs personnels présents,
   - `"sensitive"` : la liste triée des champs sensibles présents,
   - `"processes_personal"` : un booléen (vrai si l'un des deux est non vide).
4. Dans un `if __name__ == "__main__":`, teste avec au moins deux agents : un agent support (`["name", "email", "ticket_history"]`) et un agent métriques anonymes (`["cpu_load", "request_count"]`). Affiche le résultat lisiblement.

## Criteres de reussite

- [ ] Le script tourne avec `python <fichier>` sans erreur (stdlib seule).
- [ ] `classify(["name", "email", "ticket_history"])` renvoie `processes_personal = True`.
- [ ] `classify(["cpu_load", "request_count"])` renvoie `processes_personal = False`.
- [ ] Un champ sensible (ex. `health`) apparaît bien dans `"sensitive"` ET dans la détection de données personnelles.
- [ ] La sortie distingue clairement « données personnelles » et « données sensibles (Art. 9) ».
