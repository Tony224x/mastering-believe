# Exercice HARD — J26 : mixed precision + gradient checkpointing + benchmark

## Objectif

Optimiser le training loop pour tirer parti d'un GPU consumer : **mixed precision (fp16/bf16)** pour la vitesse, **gradient checkpointing** pour la memoire, et **benchmark systematique** pour mesurer ce que chaque optim apporte vraiment.

## Consigne

Tu vas creer 4 variantes du training loop et les benchmarker.

### Tache 1 — Augmenter la taille du modele pour rendre le benchmark realiste

Le `TinyDiffusionPolicy` du jour est trop petit pour observer les gains. Cree une variante `MediumDiffusionPolicy` avec :
- `hidden = 1024` (au lieu de 128)
- 8 couches Linear + SiLU au lieu de 3
- Action horizon = 16, action_dim = 4 (~plus realiste)
- Ajoute aussi une "pseudo image cond" : un tensor `(B, 3, 64, 64)` passe par 3 couches Conv2d + Flatten avant d'etre concatene a la cond. Ca simule l'encoder visuel ResNet18 du capstone reel.

### Tache 2 — Implementer 4 variantes

Pour chaque variante, mesure :
- **Throughput** : `steps / second`
- **Peak memory** : via `torch.cuda.max_memory_allocated()` (si CUDA dispo) ou `tracemalloc` (CPU)
- **Loss finale** : moyenne des 100 derniers steps

Variantes :
1. **baseline** : fp32, pas de checkpointing
2. **fp16** : `torch.amp.autocast` + `GradScaler`, pas de checkpointing
3. **bf16** : `torch.amp.autocast(dtype=torch.bfloat16)`, sans GradScaler (bf16 a la dynamic range de fp32)
4. **bf16 + grad ckpt** : ajoute `torch.utils.checkpoint.checkpoint` sur 2 couches MLP cles

### Tache 3 — Tableau de resultats + analyse

Produis un tableau Markdown :

```
| Variante         | Throughput (st/s) | Peak mem (MB) | Loss finale |
|------------------|-------------------|---------------|-------------|
| baseline fp32    | ...               | ...           | ...         |
| fp16 + scaler    | ...               | ...           | ...         |
| bf16             | ...               | ...           | ...         |
| bf16 + ckpt      | ...               | ...           | ...         |
```

## Criteres de reussite

- Les 4 variantes passent `python -m py_compile`.
- Le benchmark tourne sans NaN (attention : fp16 sans GradScaler diverge presque toujours, c'est attendu — utilise toujours le scaler avec fp16).
- Les poids EMA restent en fp32 dans toutes les variantes (la theorie l'exige : EMA en fp16 derive).
- L'analyse identifie au minimum :
  1. fp16/bf16 reduit la peak memory de ~30-40%
  2. bf16 a un overhead inferieur a fp16 sur les GPUs Ampere+ (pas de scaler)
  3. Gradient checkpointing reduit encore la memoire mais coute ~30% en throughput
  4. La loss finale doit rester comparable (a 5% pres) entre toutes les variantes — sinon il y a un bug

## Indices

- Pour autocast : `from torch.amp import autocast, GradScaler`. La nouvelle API prend `device_type="cuda"` ou `"cpu"`.
- Sur CPU, l'autocast bf16 fonctionne mais le gain de vitesse est limite. C'est OK pour valider la correction, mais le vrai benchmark a du sens sur GPU.
- Pour `torch.utils.checkpoint.checkpoint` : il faut activer `use_reentrant=False` (deprecation warning sinon). Le checkpointing trade du compute (recomputation forward) contre de la memoire (pas de stockage des activations intermediaires).
- Si tu n'as pas de GPU : tu peux mocker le benchmark en utilisant `time.perf_counter()` et `tracemalloc` sur CPU. La structure de l'analyse reste la meme, et le code sera reutilisable sur GPU.
- Attention : EMA stockee en fp32 implique que pendant `update`, il faut faire `live_param.detach().float()` avant la combinaison.
