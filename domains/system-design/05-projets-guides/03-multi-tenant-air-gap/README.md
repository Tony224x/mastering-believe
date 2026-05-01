# Projet 03 — Plateforme multi-tenant air-gapped

> **Note de scope** : ce projet est plus **platform engineering / supply chain security** que "system design" au sens strict (architecture runtime). Il est inclus dans cette section parce qu'il couvre un vrai probleme specifique a LogiSim (livraison chez 20+ clients on-premise / air-gap) que l'AI Engineer rencontrera des son arrivee. Attends-toi a un angle devops/securite plutot qu'a un design de service classique.

## Contexte metier

LogiSim vend FleetSim a **plus de 20 grands operateurs logistiques** (3PL, retailers, industriels). Chaque client est un tenant qui :
- Refuse categoriquement que ses donnees de flux sortent de son reseau (contrats clients, sensibilite competitive)
- A ses propres standards de classification interne, parfois incompatibles entre groupes
- A ses propres docs (SOP, layout d'entrepot, historique de shifts) qui ne doivent **jamais** se melanger avec ceux d'un autre client
- Deploie en on-premise, souvent dans un datacenter du groupe ou en edge sur site

Problematique : comment LogiSim, depuis son siege, livre une plateforme qui tourne identiquement chez 20 clients sans jamais acceder a leurs donnees, et sans que le client n'ait besoin de staff devops pointu pour l'installer ?

## Objectif technique

Designer :
1. Le **packaging** (comment on livre le software chez un client air-gap / quasi air-gap)
2. L'**architecture multi-tenant** (si un deploiement sert plusieurs sites du meme client, isolation garantie)
3. L'**auth et la gestion d'identite** (pas de SSO cloud, integration avec les IdP du client : AD, LDAP, OIDC interne)
4. La **strategie de mise a jour** (comment patcher sans internet)
5. La **supply chain security** (comment le client est sur que le binaire vient bien de LogiSim)

## Contraintes

| Contrainte | Valeur |
|---|---|
| Connectivite | air-gap total ou quasi-air-gap, pas de telemetrie sortante |
| Update cadence | une release / 3 mois, hotfix possibles |
| Install time | < 1 jour par un operateur client (pas un consultant LogiSim sur place) |
| Tenants par deploiement | 1 a 10 (typiquement : differents entrepots, differentes BU) |
| OS cible | RHEL, Ubuntu LTS |
| Architecture | x86_64 et eventuellement ARM (edge) |
| Certif | ISO 9001, SOC 2 type II envisageable |

## Etapes guidees

1. **Packaging** — image Docker + compose ? Package RPM/DEB classique ? Ostree ? Binaire statique ? Appliance virtuelle OVA ?
2. **Delivery** — USB signe ? Repository apt/yum local ? Avec quelle chaine de signature ?
3. **Tenants** — deploiement par tenant (1 stack / tenant) ou stack unique avec isolation logique ?
4. **Auth** — comment s'integrer a l'IdP client (ActiveDirectory, FreeIPA, LDAP) sans exiger une config speciale ?
5. **Updates** — signed bundle, verif locale, rollback en cas d'echec
6. **Supply chain** — SBOM (Software Bill of Materials), reproducibility, SLSA level ?

## Questions de revue

- Un client decouvre que le package a ete altere entre l'editeur et son datacenter : comment il le detecte ?
- Un admin local casse la conf en prod : comment il rollback en 5 minutes ?
- Un tenant A et un tenant B sur le meme hardware : comment tu prouves qu'A ne peut pas voir les donnees de B, meme si l'OS est compromis ?
- Un scanner de vulnerabilite trouve une CVE high dans une dependance. Tu n'as pas de release prevue avant 2 mois. Quel est le chemin de hotfix ?
- Le client veut un audit complet de la chaine de build. Qu'est-ce que tu lui donnes ?

## Solution

Voir `solution/platform.md` pour le reference design.

## Pour aller plus loin

- **Reproducible builds** — prouver qu'une source donne exactement un binaire donne
- **Hardware attestation** — TPM + measured boot pour refuser de demarrer si le systeme a ete altere
- **Dual-sign** — LogiSim signe avec une cle HSM + un verificateur client signe apres audit local
