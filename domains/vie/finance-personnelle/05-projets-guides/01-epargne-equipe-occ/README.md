# Projet guide 01 — Le plan d'épargne d'une équipe OCC

> **Disclaimer.** Projet **purement éducatif**. Rien ici n'est un conseil
> financier, fiscal ou en investissement personnalisé. Les taux sont
> illustratifs et non garantis ; tout investissement comporte un risque de
> perte en capital. Contexte métier partagé :
> [`shared/logistics-context.md`](../../../../../shared/logistics-context.md).

**Niveau** : facile → moyen · **Modules mobilisés** : 01 (intérêts composés),
02 (épargne automatique), 05 (impact des frais).

## 1. Contexte metier

Dans la salle de contrôle (OCC, *Operations Control Center*) d'un client
FleetSim, quatre opérateurs se relaient pour superviser la flotte robotisée.
Même métier, même grille de salaire, même âge. Pourtant, dans trente ans, leur
patrimoine sera radicalement différent — non pas à cause de leur salaire, mais
à cause de **trois décisions invisibles** prises aujourd'hui : *quand* ils
commencent à épargner, *combien de frais* ils acceptent de payer, et s'ils
saisissent l'*abondement* offert par l'employeur.

Tu es la personne à qui l'équipe demande : « explique-nous, chiffres en main,
ce qui fait vraiment la différence. » Pas de discours — un simulateur.

## 2. Objectif technique

Construire un **simulateur d'épargne** qui projette le capital de chaque
opérateur à un horizon donné, en modélisant proprement trois mécanismes que la
formule d'annuité fermée gère mal *ensemble* :

1. l'**arrêt des versements** après N années suivi d'une phase de croissance passive ;
2. les **frais de gestion** prélevés sur l'encours (pas sur les versements) ;
3. l'**abondement employeur** (un pourcentage ajouté à chaque versement).

## 3. Consigne

Implémente une fonction de simulation et un jeu de profils :

```python
@dataclass
class Operateur:
    nom: str
    versement_mensuel: float
    annees_versement: int
    annees_totales: int        # horizon (versement + croissance passive)
    annee_debut: int = 0       # délai avant le 1er versement
    abondement_employeur: float = 0.0
    frais_annuels: float = 0.0
    rendement_brut: float = 0.06

def simuler(op: Operateur) -> Operateur:
    """Remplit op.capital_final, op.total_verse_perso, op.total_abonde."""
    ...
```

Contraintes :
- Simulation **mensuelle** (croissance puis versement, dans cet ordre) — pas la formule fermée.
- Le rendement qui fait croître l'encours est le rendement **net de frais** (`brut − frais`).
- **stdlib uniquement**, déterministe (aucun aléa : deux exécutions = mêmes chiffres).
- Affiche un **classement** par capital final + un **détail** par opérateur.

## 4. Etapes guidees

1. **Modèle** — un `dataclass Operateur` avec les paramètres + les champs calculés (`field(init=False)`).
2. **Boucle mensuelle** — pour chaque mois : (a) `capital *= 1 + taux_mensuel` ; (b) si le mois est dans la fenêtre `[annee_debut, annee_debut + annees_versement[`, ajoute le versement **et** l'abondement.
3. **Le piège du net** — `taux_mensuel = (rendement_brut − frais_annuels) / 12`. C'est *toute* la leçon du module 05 : 1,6 point de frais ampute un tiers du capital sur 35 ans.
4. **Conçois tes profils en isolant UNE variable** — pars d'une baseline (Amina), puis ne change *qu'un seul* paramètre par opérateur (Bruno = +10 ans de délai ; Carla = +frais ; Diallo = +abondement). Sinon tu ne sauras pas attribuer l'écart à sa cause.
5. **Restitue** — un classement trié + trois « leçons chiffrées » qui comparent chaque opérateur à la baseline.

## 5. Criteres de reussite

- [ ] Le script tourne sans dépendance externe : `python solution/epargne_simulator.py`
- [ ] Deux exécutions donnent **exactement** les mêmes chiffres (déterminisme)
- [ ] À effort d'épargne **identique** (24 000 € versés), le *début précoce* (Amina) bat le *début tardif* (Bruno) — l'écart est dû **uniquement** au temps
- [ ] L'opérateur à 2 % de frais (Carla) finit **nettement** sous la baseline à 0,4 %, tout le reste égal
- [ ] L'abondement +50 % (Diallo) domine le classement, pour le même effort personnel
- [ ] Les cas limites ne plantent pas : `annee_debut` au-delà de l'horizon → capital 0, pas d'exception

> **Piège à comprendre** : on croit spontanément que « verser plus longtemps »
> bat « commencer tôt ». C'est faux *à montant égal* : ici Amina et Bruno
> versent la même somme (24 000 €), mais Bruno commence 10 ans plus tard et
> finit avec **~43 % de moins**. Le levier n°1 n'est ni le salaire ni la
> discipline mensuelle — c'est le **temps laissé aux intérêts composés**.

## 6. Corrige

Voir [`solution/epargne_simulator.py`](./solution/epargne_simulator.py) (commenté)
et [`solution/analyse.md`](./solution/analyse.md) pour la lecture des résultats,
les choix de modélisation et les limites honnêtes du modèle.

## 7. Pour aller plus loin

- **Inflation** — raisonne en euros *réels* : soustrais ~2 %/an au rendement et compare.
- **Versements progressifs** — fais croître le versement de 2 %/an (augmentations de salaire). Qui rattrape Amina ?
- **Sensibilité au rendement** — rejoue à 4 %, 6 %, 8 % brut. À quel rendement l'effet frais devient-il dramatique ?
- **Visualisation** — exporte un CSV (année, capital par opérateur) et trace la divergence des courbes.
