# Exercice J20 — niveau easy

## Objectif

Maitriser les **shapes** et la **logique de tokenization** d'OpenVLA. Tu dois savoir, sans regarder le code, ou se trouvent les 256 patches, comment on combine DINOv2+SigLIP, et comment on transforme un token en valeur d'action.

Reference : `domains/robotics-ai/01-theory/20-openvla-architecture.md` sections 2 et 4.

## Consigne

Ecris une fonction Python `openvla_shape_report(image_size, patch_size, dinov2_dim, siglip_dim, llm_hidden, action_dim, action_bins, vocab_size)` qui renvoie un `dict` contenant :

1. `num_patches` : nombre de patches que produisent les ViT pour une image carree de cote `image_size` avec des patches de cote `patch_size`.
2. `concat_dim` : dimension feature apres concatenation DINOv2+SigLIP.
3. `projector_in_out` : tuple `(input_dim, output_dim)` du MLP projector qui amene les features fusionnees vers la dimension cachee du LLM.
4. `action_token_range` : tuple `(start, end)` des indices du vocabulaire reserves aux 256 bins d'action.
5. `bin_for_value` : helper qui prend `(value, dim_idx, low=-1.0, high=1.0)` et retourne l'**indice de bin** entre 0 et `action_bins-1` correspondant a la valeur (clipping inclus aux extremes).

Ajoute une mini-doctring expliquant pourquoi la concatenation DINOv2+SigLIP se fait sur l'axe **feature** (et pas sur l'axe patch).

Verifie ta fonction sur les valeurs reelles d'OpenVLA :

```python
report = openvla_shape_report(
    image_size=224, patch_size=14,
    dinov2_dim=1024, siglip_dim=1152,
    llm_hidden=4096,
    action_dim=7, action_bins=256, vocab_size=32000,
)
# attendu :
# num_patches = 256
# concat_dim = 2176
# projector_in_out = (2176, 4096)
# action_token_range = (31744, 32000)
```

## Criteres de reussite

- [ ] Les 5 valeurs sont correctes pour la config OpenVLA reelle.
- [ ] `bin_for_value(0.0, 0)` retourne `127` ou `128` (centre de l'intervalle [-1, 1]).
- [ ] `bin_for_value(2.0, 0)` retourne `255` (clipping).
- [ ] `bin_for_value(-3.0, 0)` retourne `0` (clipping).
- [ ] La docstring mentionne explicitement que DINOv2 capte la **geometrie** et SigLIP la **semantique**, donc on veut les deux **par patch** (pas l'un derriere l'autre).
