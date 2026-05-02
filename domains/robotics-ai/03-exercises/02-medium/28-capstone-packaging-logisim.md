# Exercice (medium) — Concevoir un Makefile + scripts/ pour orchestrer le capstone

## Objectif

Construire l'infrastructure d'execution du capstone : un `Makefile` (ou un equivalent `tasks.py` invoke) plus un dossier `scripts/` qui permettent de lancer chaque phase du pipeline avec une seule commande, de maniere reproductible.

## Consigne

Tu dois livrer **3 fichiers minimum** :

1. Un `Makefile` a la racine du capstone, exposant **au moins 6 cibles** : `install`, `data`, `train`, `eval`, `demo`, `clean`. Cibles supplementaires bienvenues : `lint`, `test`, `figures`, `format`.

2. Un dossier `scripts/` contenant **3 scripts Python** invoques par les cibles du Makefile :
   - `scripts/generate_demos.py` — genere les demos PushT (CLI : `--n_demos`, `--output`, `--seed`).
   - `scripts/train.py` — entraine la Diffusion Policy (CLI : `--config`, `--checkpoint_out`, `--epochs`).
   - `scripts/eval.py` — evalue la policy (CLI : `--config`, `--checkpoint`, `--n_rollouts`, `--results_out`).

3. Un fichier `configs/train.yaml` (ou JSON) decrivant les hyperparametres d'entrainement de maniere lisible, override-able via CLI.

## Contraintes

- **Aucun chemin en dur** dans les scripts. Tout chemin passe par CLI ou config YAML.
- Le Makefile doit declarer toutes ses cibles dans une ligne `.PHONY` (convention).
- Chaque cible Makefile **affiche ce qu'elle va faire** avant de le faire (ex : `@echo ">>> Training Diffusion Policy..."`).
- Les scripts doivent renvoyer un **exit code non-zero** en cas d'erreur (utile pour CI).
- Le script `scripts/eval.py` doit ecrire un `eval/results.json` parsable contenant au minimum : `success_rate_pct`, `mean_episode_length`, `mean_step_latency_ms`, `n_rollouts`.
- Compatible Windows : si tu utilises `Makefile` strict, indique en commentaire la compatibilite (`make` natif sur Linux/macOS, GnuWin32 sur Windows). Alternative recommandee : `tasks.py` avec [invoke](https://www.pyinvoke.org/) qui marche cross-OS.

## Etapes guidees

1. Squelette du Makefile : declarer les 6 cibles vides + `.PHONY`.
2. Implementer `scripts/generate_demos.py` avec `argparse` ou `click`. Sortie : un fichier `data/pusht_demos.zarr` (ou `.npz` si zarr non dispo).
3. Implementer `scripts/train.py` avec un parsing YAML simple (lib : `pyyaml`). La logique d'entrainement peut etre stub si tu n'as pas le temps : juste prouver que le contrat (input config, output checkpoint) tient.
4. Implementer `scripts/eval.py` qui ecrit `eval/results.json`. La logique de rollout peut reutiliser le code du J28.
5. Cabler les 3 scripts depuis le Makefile.
6. Tester la chaine : `make data && make train && make eval && make demo` doit s'enchainer sans intervention manuelle.

## Criteres de reussite

- `make demo` execute en moins d'une minute apres `make install`.
- Les 6 cibles minimales sont presentes et fonctionnent.
- Aucun script ne contient de chemin en dur (`/home/anthony/...`).
- `eval/results.json` est genere avec les 4 cles requises.
- Le `train.yaml` contient au moins : `learning_rate`, `batch_size`, `n_epochs`, `action_horizon` (action chunking k).
- Les scripts retournent un exit code != 0 sur erreur (testable : passer un `--config inexistant.yaml` doit echouer proprement).

## Pour aller plus loin

- Ajouter une cible `make ci` qui chaine `lint -> test -> data -> train -> eval` pour un check end-to-end.
- Pinner les versions dans un `requirements.txt` ou `pyproject.toml` avec un lock file (`uv lock` ou `pip-compile`).
- Ajouter un `make figures` qui regenere les plots a partir de `eval/results.json` sans relancer l'eval.
- Migrer vers [Hydra](https://hydra.cc/) pour la gestion de config (standard de fait dans la communaute Diffusion Policy : voir le repo `real-stanford/diffusion_policy`).
