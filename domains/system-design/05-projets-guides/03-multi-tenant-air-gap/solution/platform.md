# Reference design — Plateforme multi-tenant air-gapped

## Packaging — OCI images + bundle signe

**Choix : images OCI (Docker) dans un tarball signe, deploye via compose ou k3s single-node.**

Pourquoi :
- OCI est le standard portable, supporte partout
- Air-gap friendly : on peut `docker save` tout un graph d'images dans un tarball, le transferer par USB, `docker load` cote client
- Pas besoin d'un registry prive chez le client (simplifie l'infra)
- Compose ou k3s au choix du client selon sa maturite : compose pour < 5 tenants, k3s pour plus

Alternatives ecartees :
- **RPM/DEB** : fragile entre distros, trop de branches a maintenir (RHEL 8, RHEL 9, Ubuntu 22.04, 24.04)
- **Appliance OVA** : lourd, duplique l'OS, mais bon fallback pour les clients qui refusent le container
- **Binaire statique** : pas possible car on a des services multiples (simu, AAR store, UI)

## Bundle signe

Livraison : un seul fichier `sword-release-YYYY-MM.tgz`

Contenu :
```
sword-release-2026-04.tgz
├── MANIFEST.yaml           # liste images + hashes SHA-256
├── images/
│   ├── sim.tar             # image docker de la simu
│   ├── aar.tar
│   ├── ui.tar
│   └── auth.tar
├── compose.yaml            # deploiement ref
├── install.sh              # script d'installation (verifie signature d'abord)
├── sbom.spdx.json          # Software Bill of Materials
├── SIGNATURE.sig           # signature Ed25519 sur MANIFEST.yaml
└── PUBKEY.pem              # cle publique (contremarque par fingerprint papier)
```

Verification cote client :
1. `install.sh` check que `SIGNATURE.sig` matche `MANIFEST.yaml` avec la cle publique fournie hors-bande (papier ou email direct)
2. Pour chaque image dans `MANIFEST`, verifie le hash SHA-256 du tar file
3. `docker load` chaque image, verifie que le digest OCI matche aussi
4. Puis lance compose

Toute alteration (USB compromis, MITM) fait echouer a l'etape 1 ou 2.

## Multi-tenancy — isolation par namespace + PV

**Choix : un seul deploiement physique, plusieurs tenants logiques separes au niveau k3s namespace.**

Pour 10 tenants sur le meme hardware :
- 1 namespace k3s par tenant
- NetworkPolicy stricte : aucun trafic cross-namespace (sauf vers ingress)
- PV dedie par tenant, chiffre (LUKS au niveau host)
- Auth : chaque tenant a son propre IdP / realm, pas de compte cross-tenant

Pour des niveaux de classification differents sur la meme machine, **non** : il faut des machines physiques distinctes (air-gap intra-classification). Le design ici couvre la multi-unite au sein d'une meme classification.

**Isolation "meme si l'OS est compromis"** : impossible sans hardware. On documente la limite : le design repose sur la securite de l'OS host. Pour cross-classification, deployer sur des serveurs physiquement separes.

## Auth — OIDC avec adapter IdP

**Choix : service d'auth interne au deploiement, qui parle OIDC a un IdP client (LDAP/AD/FreeIPA via gateway).**

Flux :
```
User -> SWORD UI -> SWORD auth service (OIDC) -> LDAP/AD client (Kerberos/LDAPS)
                         |
                         v
                   Mapping roles locaux
```

Le service d'auth local (Dex, Keycloak, ou custom) est la couche d'abstraction qui permet de ne pas coder N integrations IdP dans l'application. Le client configure son realm a l'install, rien d'autre a toucher.

## Updates — A/B rollout avec rollback

**Choix : blue-green au niveau namespace.**

Procedure :
1. Nouveau bundle signe livre en USB
2. `install.sh upgrade --target blue` cree un nouveau deploiement dans un namespace `sword-blue` (ancien : `sword-green`)
3. Smoke test automatique (lancer un micro-scenario, verifier healthcheck)
4. Bascule du ingress vers blue
5. Si probleme dans les 24h, rollback = bascule ingress vers green
6. Apres 7 jours, garbage collect green

Pas de mise a jour in-place : trop risque sur un systeme critique ou la prod est en cours d'exercice.

## Supply chain

Livrables documentaires au client :
- **SBOM SPDX** — toutes les deps avec versions et licences
- **Build attestation** — SLSA v1 niveau 3 (build isole, non-falsifiable, traceable a la source)
- **Vuln scan** — rapport grype/trivy au moment de la release
- **Changelog signe** — liste des changes depuis la version precedente, avec reference au bug tracker

Pour les hotfix CVE : meme processus, bundle plus petit, signature cle HSM de la meme racine.

## Reponses aux questions de revue

**Detection d'un bundle altere ?**
Signature Ed25519 sur MANIFEST + hash SHA-256 sur chaque fichier + digest OCI sur chaque image. Triple verification. La cle publique est contremarquee par canal distinct (papier, email direct, fingerprint au telephone).

**Rollback en 5 min ?**
Blue-green : bascule de l'ingress, 30 secondes. L'ancien namespace est encore vivant et sain.

**Isolation cross-tenant en cas d'OS compromis ?**
Non garantie par design logiciel. Doit etre adressee par separation physique. C'est une limite explicite du design, a documenter dans le threat model.

**CVE high sans release prevue ?**
Chemin hotfix : bundle nomme `hotfix-CVE-XXXX-YYYY`, ne contient que l'image impactee, installe par-dessus. Delay : typiquement 1 semaine (build + sign + valid + ship).

**Audit chaine de build ?**
SLSA attestation + SBOM + dockerfiles source-controlled. L'auditeur peut reproduire le build a partir du tag git.
