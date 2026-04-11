# 05 — Projets guides (System Design)

> Voir `shared/masa-context.md` pour le contexte metier de MASA Group.

Trois problemes d'architecture que tu peux rencontrer chez MASA. Chaque projet est fourni sous forme de **cahier des charges** + **solution detaillee avec diagrammes ASCII**, parce que le livrable d'un exercice system design n'est pas du code mais un document d'architecture.

## Projets

| # | Projet | Concepts cles | Difficulte |
|---|---|---|---|
| 01 | **Simulation distribuee tick-based** | partitioning spatial, tick sync, state replication, determinisme | hard |
| 02 | **Pipeline AAR (After-Action Review)** | event sourcing, time-series, indexation, replay | medium |
| 03 | **Plateforme multi-tenant air-gapped** | isolation, deploiement offline, auth, supply chain | hard |

## Methodologie

Pour chaque projet :
1. Lire le contexte et les **contraintes non fonctionnelles** (latence, debit, determinisme, air-gap)
2. Esquisser ton architecture sur papier, meme rapidement
3. Ouvrir la solution et confronter ton design au "reference design"
4. Lire la section **trade-offs** : pourquoi ces choix et pas d'autres
5. Repondre aux **questions de revue** a la fin (ce sont les questions que ton architecte tech ferait en review)
