# J18 - Exercice MEDIUM - Probe et collapse : auditer une mini-JEPA

## Objectif

Comprendre pratiquement les deux pieges centraux d'une JEPA : (i) **representation collapse** (le modele finit par predire un vecteur constant), et (ii) la **qualite latente** vs reconstruction pixel — mesurer empiriquement que la JEPA fournit une representation downstream meilleure malgre l'absence de decoder.

Tu reutilises et etends `02-code/18-jepa-cosmos.py`.

## Consigne

### Partie 1 - Detecter le collapse

1. Modifie le code pour **desactiver l'EMA** : remplace `update_target_encoder` par une copie hard-sync `encoder_tgt.load_state_dict(encoder_ctx.state_dict())` apres chaque step (ce qui revient a `ema_decay = 0`, donc target = context exactement).

2. Re-entraine la mini-JEPA dans cette configuration. Observe la latent loss : que se passe-t-il ?

3. Calcule sur un batch de 256 echantillons :
   - La **norme moyenne** des latents `||z_ctx||_2`.
   - La **variance** sur chaque dimension du latent (`z.std(dim=0).mean()`).

4. Refais la meme mesure avec EMA actif (`ema_decay = 0.99`). Compare.

5. Conclusion attendue : sans EMA + stop-grad, la JEPA tend a collapse vers `z = constante` parce que c'est la solution triviale qui minimise la loss latente. **Pourquoi l'EMA evite ce collapse ?** Reponds en 2-3 phrases.

### Partie 2 - Linear probe robustness

6. Ajoute du **bruit** dans la generation des images de la moitie droite (target) : multiplie par un facteur `noise_amp ∈ {0.0, 0.5, 1.0, 2.0}`. Pour chaque niveau de bruit :
   - Re-entraine PixelAE et MiniJEPA from scratch.
   - Mesure la pixel-MSE de PixelAE sur le target half.
   - Mesure la linear probe accuracy sur les deux latents.

7. Trace ou tabule les 4 lignes (4 niveaux de bruit) x 3 colonnes (pixel-MSE PixelAE, probe acc PixelAE, probe acc JEPA).

8. Reponds : a mesure que le bruit augmente, **quelle metrique se degrade le plus** ? Pourquoi cela illustre-t-il l'argument LeCun "99% des pixels sont du bruit" (REFERENCES.md #21) ?

### Partie 3 - Tester un latent_dim variable

9. Repete l'experience principale (sans bruit) avec `latent_dim ∈ {2, 8, 32, 128}`. Quelle dimension donne la meilleure linear probe accuracy ? Au-dela de quelle valeur le rendement marginal s'effondre-t-il ?

## Criteres de reussite

- Le collapse sans EMA est observe : norme stable mais variance dimensionwise ≈ 0 ou loss qui descend trivialement vers ~0 sans signal utile (probe accuracy proche du hasard).
- L'EMA actif maintient une variance significative et une probe accuracy nettement superieure au hasard.
- La justification du role EMA mentionne le **stop-gradient** + le fait que le target bouge lentement, empechant le predictor de "tricher" en collapsant.
- Le tableau de bruit montre que **PixelAE pixel-MSE explose** avec le bruit (il essaie de reconstruire le bruit lui-meme), tandis que **JEPA probe acc est plus stable** (il a appris la structure, pas le bruit).
- L'analyse latent_dim identifie le sweet spot (typiquement 16-64 pour cette tache toy) et explique l'effondrement aux deux extremites (trop petit = perte d'info, trop grand = surapprentissage / capacite gaspillee).
- Bonus : citer le mecanisme equivalent dans BYOL ou DINO (meme strategie EMA + stop-grad pour eviter le collapse en self-supervised).
