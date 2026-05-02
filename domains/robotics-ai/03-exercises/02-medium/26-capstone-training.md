# Exercice MEDIUM — J26 : EMA from scratch avec halflife configurable

## Objectif

Comprendre EMA en profondeur en l'implementant from scratch, et mesurer son impact en variant le **halflife** plutot que le `decay` directement.

## Consigne

L'EMA fournie dans le script du jour utilise un parametre `decay` (typiquement 0.999). Mais ce qui a un sens physique pour un humain, c'est le **halflife** : le nombre de steps apres lesquels un poids "ancien" perd 50% de son influence.

Relation : `decay = 0.5 ** (1 / halflife)`

### Tache 1 — Implementer une classe `EMAHalflife`

Ecris une classe avec la signature :

```python
class EMAHalflife:
    def __init__(self, model: nn.Module, halflife_steps: int, warmup: int = 100):
        ...
    def update(self, model: nn.Module, step: int) -> None:
        ...
    def copy_to(self, model: nn.Module) -> None:
        ...
```

Contraintes :
- `halflife_steps` est en **steps d'optimizer** (pas en epochs).
- Le `decay` effectif est calcule a l'init.
- Pendant les `warmup` premiers steps, on synchronise les poids EMA avec les poids live (sans moyenne).
- Apres `warmup`, on applique la formule classique `ema = decay * ema + (1 - decay) * live`.
- Les poids EMA doivent rester en fp32 (meme si le modele est en fp16 plus tard).

### Tache 2 — Comparer 3 halflifes

Relance le training (1500 steps suffisent) avec :
- `halflife = 50` (EMA tres reactive, presque = poids live)
- `halflife = 500` (EMA moderee, environ 1/3 du run)
- `halflife = 5000` (EMA tres lisse, plus longue que le run lui-meme)

A la fin, **echantillonne 16 actions** depuis le modele EMA (en utilisant `copy_to`) pour 4 etats fixes, et plotte-les. Compare visuellement la "qualite" / le bruit residuel des trajectoires generees.

## Criteres de reussite

- La classe `EMAHalflife` passe `python -m py_compile`.
- Le calcul `decay = 0.5 ** (1 / halflife)` est present et commente.
- Les poids EMA sont stockes en fp32 (`.float()` ou `dtype=torch.float32` explicite).
- Le warmup synchronise les poids sans appliquer la formule EMA (verifie avec un assert ou un test simple).
- Le rapport texte/plot final identifie : halflife trop court = EMA inutile (= poids live), halflife trop long = EMA traine sur du bruit initial, halflife optimal = sweet spot pres de la longueur de convergence.

## Indices

- Pour stocker en fp32 systematiquement : `self.shadow = {k: v.detach().float().clone() for k, v in model.state_dict().items()}`.
- Pour la mise a jour : attention au dtype quand le modele est en fp16 — il faut faire `.float()` sur le tenseur live avant d'additionner.
- Pour echantillonner depuis le modele EMA, il faut **sauver les poids live**, copier l'EMA dedans, faire le sampling, puis restaurer les poids live. Sinon tu casses le training si tu le reprends.
