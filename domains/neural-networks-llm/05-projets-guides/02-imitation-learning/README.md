# Projet 02 — Imitation learning depuis traces d'experts

## Contexte metier

MASA a un corpus de traces d'exercices ou des **officiers experts** ont commande des unites. L'idee : apprendre a une IA a commander "comme eux" par imitation, pour enrichir les unites autonomes de SWORD avec des comportements plus subtils que ceux de Direct AI (qui est base sur des regles).

C'est du **behavioral cloning** : on traite les decisions des experts comme un dataset supervise (etat -> action), et on entraine un modele a predire l'action donnee l'etat.

Limite connue : BC souffre du **distribution shift** (le modele voit des etats jamais vus a l'inference et diverge). On l'acceptera dans la v0 et on discutera DAgger / RL comme extension.

## Objectif technique

Entrainer un modele sequentiel (LSTM ou petit Transformer) qui, etant donne l'historique recent d'un peloton (positions, contact ennemi, sante), predit la **prochaine action a prendre** parmi un set discret : `MOVE_FORWARD`, `TAKE_COVER`, `ENGAGE`, `WITHDRAW`, `HOLD`, `REPORT`.

## Dataset

Generateur `solution/generate_traces.py` qui produit des sequences d'etats + actions, avec un "expert rule-based" simple (heuristique). C'est artificiel mais permet de se concentrer sur le ML.

Format :
```python
state_t = (pos_x, pos_y, enemy_distance, enemy_count, health, ammo, time_since_last_order)  # 7 floats
action_t = int in [0..5]

sequence = [(s_0, a_0), (s_1, a_1), ..., (s_T, a_T)]
```

## Consigne

1. Genere 1000 sequences de longueur variable (20-100 steps)
2. Split train/val/test 70/15/15
3. Modele LSTM : embed state (Linear 7 -> 32), LSTM(32, 64), Linear(64, 6)
4. Training : sequence-level, `torch.nn.utils.rnn.pack_padded_sequence` pour les longueurs variables
5. Loss : cross-entropy moyennee sur tous les steps
6. Eval : accuracy per-step sur test, accuracy per-sequence (toutes les actions justes)

## Criteres de reussite

- Accuracy per-step > 75% sur test (l'expert est stochastique, pas de 100% possible)
- Pas de gap train/test massif
- Une sequence de demo jouable : plot des etats, actions expert vs actions modele

## Piege distribution shift

Si tu utilises ton modele pour **generer** une sequence (jouer un peloton), les erreurs s'accumulent : au step 1 l'etat est "normal", mais une mauvaise prediction emmene a un etat legerement off-distribution, puis de plus en plus, et le modele diverge completement. C'est le cas classique de BC.

Solutions a discuter dans la section extensions :
- **DAgger** : iterer avec l'expert a disposition pour annoter les etats visites par le modele
- **RL avec reward shaping** : passer au RL une fois qu'on a une baseline BC
- **Action smoothing** : lisser les predictions pour eviter les switches brusques

## Solution

Voir `solution/generate_traces.py` et `solution/train_bc.py`.

## Pour aller plus loin

- Remplacer LSTM par un petit **Transformer** decoder (causal mask), comparer les perfs
- Ajouter un **value head** pour passer a du actor-critic light
- Publier le modele en onnx pour inference C++ cote SWORD
