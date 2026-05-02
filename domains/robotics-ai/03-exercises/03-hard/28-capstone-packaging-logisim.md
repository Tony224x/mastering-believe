# Exercice (hard) — Roadmap : du capstone PushT vers le picking AGV temps reel dans FleetSim

## Objectif

Concevoir un mini-roadmap technique credible pour porter le capstone Diffusion Policy (PushT 2D) vers une **vraie tache de picking** dans le contexte LogiSim/FleetSim : un AGV de picking 7-DOF dans un rack, avec contraintes temps reel, safety, et un Work Order textuel issu du WMS.

Le livrable est un document `roadmap.md` qui sert de base de discussion avec un staff engineer ou un PO chez LogiSim. Il doit etre **executoire** : chaque etape doit identifier un risque, un livrable concret, un budget temps, et une metrique de succes.

## Contexte metier (rappel)

Voir `shared/logistics-context.md`. Les elements cles :

- **AGV de picking** : robot mobile + bras 7-DOF (Franka-like) qui pick des colis dans un rack et les depose dans un panier embarque.
- **Work Order** : ordre WMS standardise (origine, destination, deadline, contraintes).
- **OCC** : salle de controle qui supervise. Veut des metriques traceables.
- **Safety Policy** : regle d'engagement. Notamment : `Pcollision` (probabilite de collision sur un mouvement) doit etre filtree.
- **Determinisme** : meme seed = meme resultat (pour reconstitution incident + certification).
- **Connectivite limitee** : on-premise, parfois quasi air-gap.
- **Schema event canonique** : top-level stable (`id`, `shift_id`, `seq`, `t_sim`, `tick`, `unit_id`, `zone`, `kind`, `payload`).

Reference frontiere : Helix Logistics Figure AI 2025 (REFERENCES.md #16) — premier VLA deploye reellement en logistique.

## Consigne

Produire `roadmap.md` structure en **5 phases** ordonnees, chacune avec :

- **Titre + duree estimee** (en semaines-ingenieur).
- **Objectif technique** (1 phrase).
- **3 a 5 livrables concrets** (datasets, scripts, models, docs).
- **Risque principal** (technique, business, ou humain) + mitigation.
- **Metrique de succes binaire** (atteint / pas atteint) avec un seuil chiffre.

Les 5 phases attendues (tu peux affiner les titres mais respecter la progression) :

1. **Phase 1 — Reproduction sur bras 6/7-DOF en simulation (3-4 semaines)**
   Passer de PushT 2D a une tache `lift` ou `square` du benchmark Diffusion Policy sur robosuite + Franka (REFERENCES.md #19, #26). Contrainte : aucun changement metier, juste valider que le pipeline marche en 3D haute-DOF.

2. **Phase 2 — Adaptation au contexte LogiSim avec dataset interne (4-6 semaines)**
   Collecter des demos teleoperees par operateurs LogiSim (joystick ou VR), formatter en LeRobotDataset v0.4 (REFERENCES.md #27), entrainer une Diffusion Policy sur 3-5 SKU canoniques.

3. **Phase 3 — Conditioning textuel via Work Order + safety filter (3-4 semaines)**
   Ajouter le tokenizer de texte (CLIP ou T5) pour conditionner sur le Work Order. Implementer un safety filter classique qui wrap la sortie : reject si `Pcollision` > seuil + emission d'un event `FAULT` au schema canonique LogiSim. Garantir la traceabilite (chaque action emise est loggee avec son hash de checkpoint).

4. **Phase 4 — Optimisation latence + mode degrade (3-4 semaines)**
   Passer de DDPM 100 steps a DDIM 10 steps, puis a flow matching (cf. π0-FAST, REFERENCES.md #14). Cible : latence par step < 30 ms pour rester compatible loop bras 25 Hz. Ajouter un mode degrade : si la policy renvoie une trajectoire incoherente, fallback sur le planner classique RRT existant chez LogiSim.

5. **Phase 5 — Deploiement pilote sur 1 site client (6-10 semaines)**
   Packager pour deploiement on-premise (binaire + ONNX si possible, pas de cloud). Pilote 2 semaines sur un seul site client avec 1 AGV, OCC monitoring complet, comparaison cote-a-cote avec le planner classique sur 500 picks reels. Decision Go/NoGo basee sur la metrique de succes.

## Contraintes specifiques

- Les phases doivent etre **sequentielles** (pas de parallelisation des risques techniques).
- Chaque phase doit citer **au moins une reference** du REFERENCES.md (ne pas tout reinventer, s'appuyer sur l'etat de l'art).
- La metrique de succes de la Phase 5 doit etre une **comparaison directe** avec le baseline LogiSim actuel (planner classique), pas une metrique abstraite type "the model converges".
- Le risque "**humain**" doit apparaitre au moins une fois (acceptation par les operateurs OCC, formation, change management). Ce n'est pas que technique.
- Tu dois mentionner explicitement ce qu'il **NE FAUT PAS FAIRE** : par exemple, ne pas tenter le multi-task generaliste avant la Phase 5, ne pas deployer sans safety filter, ne pas utiliser de cloud, etc.

## Criteres de reussite

- 5 phases presentes, ordonnees, chacune avec les 5 elements demandes (titre + duree, objectif, livrables, risque + mitigation, metrique).
- Au moins 3 references du REFERENCES.md citees.
- Au moins une mention explicite du schema event canonique LogiSim (`shared/logistics-context.md`).
- Au moins une mention des contraintes on-premise / determinisme / certification.
- Une section finale **"What we are NOT doing in v1"** qui liste 4-6 choses explicitement hors scope (multi-embodiment, cloud, generalisation zero-shot, humain-robot collaboration sans cage, etc.).
- Une derniere ligne : un seuil binaire Go/NoGo apres Phase 5.

## Pour aller plus loin

- Ajouter une **vue financiere** : combien de demos faut-il collecter (cout), combien d'heures GPU pour training, cout du pilote site.
- Ajouter une **carte des decisions reversibles vs irreversibles** (ex : choisir LeRobot v0.4 = reversible ; signer un contrat pilote = peu reversible).
- Ajouter un schema Mermaid de l'architecture cible apres Phase 5 (Diffusion Policy + safety filter + planner classique fallback + OCC monitoring).
- Aligner avec la roadmap industrielle reelle : Helix Logistics 2025 (REFERENCES.md #16) et TRI LBM (REFERENCES.md #18) pour la vision long-terme.
