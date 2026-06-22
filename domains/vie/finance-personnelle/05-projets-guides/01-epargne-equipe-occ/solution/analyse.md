# Analyse — Projet 01 (épargne d'une équipe OCC)

> Lecture commentée de la solution. Chiffres issus de l'exécution de
> `epargne_simulator.py` (taux **illustratifs**, non garantis — projet éducatif).

## Résultats (horizon 35 ans, 200 €/mois pendant 10 ans = 24 000 € versés)

| Opérateur | Variable changée vs baseline | Capital final | Versé perso |
|---|---|---:|---:|
| Diallo (abondement +50 %) | + abondement employeur | **194 465 €** | 24 000 € |
| Amina (baseline) | — | 129 644 € | 24 000 € |
| Carla (frais 2 %) | frais 0,4 % → 2 % | 79 920 € | 24 000 € |
| Bruno (début +10 ans) | commence 10 ans plus tard | 74 150 € | 24 000 € |

Les quatre versent **exactement la même somme de leur poche** (24 000 €). Tout
l'écart vient de **trois leviers**, isolés un par un.

## Les trois leçons, chacune en isolant une seule variable

1. **Le temps est le levier n°1.** Amina et Bruno ont le *même* comportement
   d'épargne ; Bruno commence juste 10 ans plus tard. Résultat : **−55 493 €**,
   soit ~43 % du capital d'Amina. Le retard ne se rattrape pas, parce que les
   années perdues sont précisément celles où les intérêts composés auraient eu
   le plus de temps pour agir.

2. **Les frais sont un impôt silencieux.** Carla = Amina, mais 2 % de frais au
   lieu de 0,4 %. Coût : **−49 723 €** (~38 %). On ne « voit » jamais ces frais
   passer (ils sont prélevés sur l'encours), mais sur des décennies ils
   amputent un tiers du résultat. C'est le cœur du module 05.

3. **L'abondement employeur, c'est du rendement immédiat et sans risque.**
   Diallo = Amina, mais l'employeur abonde +50 %. Il finit avec **+64 822 €**
   pour le *même* effort personnel. Refuser un abondement, c'est refuser une
   augmentation déguisée.

## Choix de modélisation (et pourquoi)

- **Boucle mensuelle plutôt que formule fermée.** La formule de valeur future
  d'une annuité gère bien « je verse X €/mois pendant N ans ». Elle gère mal,
  *en même temps*, l'arrêt des versements suivi d'une phase passive **et** des
  frais prélevés sur l'encours. La boucle rend ces deux mécanismes explicites
  et auditables ligne à ligne — au prix d'un coût négligeable (420 itérations).

- **Ordre dans la boucle : croissance *puis* versement.** Le versement du mois
  arrive en fin de période (annuité ordinaire), donc il ne « gagne » pas
  d'intérêts le mois où il est versé. C'est le choix conservateur et standard.

- **Rendement net = brut − frais, appliqué à l'encours.** On ne prélève pas les
  frais sur les versements (erreur fréquente) mais sur le capital accumulé,
  comme un vrai frais de gestion annuel.

- **Baseline + une variable à la fois.** C'est un principe de méthode autant
  que de pédagogie : si Bruno cumulait « plus tard » *et* « plus de frais », on
  ne saurait pas attribuer l'écart. Une bonne démonstration isole ses causes.

## Limites honnêtes

- **Rendement constant.** La réalité est volatile (cf. projet 03, risque de
  séquence). Un rendement lissé sur-simplifie : il sert à isoler l'effet des
  trois leviers, pas à prédire un résultat.
- **Pas d'inflation ni de fiscalité.** Les montants sont nominaux et bruts
  d'impôt. En euros réels, divise mentalement par ~1,8 sur 35 ans à 2 %
  d'inflation. La fiscalité dépend du pays et de l'enveloppe.
- **Abondement plafonné dans la vraie vie.** Les employeurs plafonnent
  l'abondement ; le modèle l'applique sans plafond, ce qui sur-estime Diallo.
  L'ordre de grandeur de la leçon tient malgré tout.

## Vérification

```
$ python epargne_simulator.py      # tourne, sortie déterministe
$ ruff check .                      # All checks passed!
```
Probe adversariale : `annee_debut` au-delà de l'horizon → capital 0 et versé 0,
sans exception (la fenêtre de versement est simplement vide).
