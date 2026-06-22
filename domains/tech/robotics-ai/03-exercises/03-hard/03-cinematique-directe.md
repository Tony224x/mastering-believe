# Exercice HARD — FK Panda et detection d'erreur

## Objectif

Confronter ta FK PoE manuelle au `mj_forward` de MuJoCo sur le Franka Panda 7-DOF (Menagerie), puis introduire **volontairement** une erreur d'1 mm dans un offset et detecter le decalage produit par cette erreur sur la pose effecteur.

## Consigne

Prerequis : avoir clone https://github.com/google-deepmind/mujoco_menagerie et avoir `mujoco` installe.

1. **Phase A — verification de coherence** : reproduit le pattern de `02-code/03-cinematique-directe.py` (`extract_panda_screws` + `fk_poe`) et execute la comparaison sur **20 configurations aleatoires** dans `q ∈ [-π/2, π/2]⁷`. Calcule l'erreur Frobenius `||T_poe - T_mj||` et affiche `(mean, max, std)`.

2. **Phase B — sabotage controle** : recupere un des screws spatiaux `Sᵢ` extraits, puis modifie son point d'ancrage `pᵢ` en lui ajoutant `(0.001, 0, 0)` (1 mm sur x). Reconstruis `vᵢ = -ωᵢ × pᵢ_perturbe` et reexecute la comparaison sur les memes 20 configurations.

3. **Phase C — analyse** : 
   - Quelle est l'erreur moyenne sur la position effecteur (`||p_poe - p_mj||`) avant et apres sabotage ?
   - Le decalage en x de 1 mm sur **un seul** offset cree-t-il un decalage final superieur, inferieur, ou egal a 1 mm ? Pourquoi ?

## Criteres de reussite

- Phase A : erreur Frobenius moyenne `< 1e-4` (ordre de grandeur). Si tu obtiens plus, c'est un bug d'extraction (signe de `ω`, ordre des produits, body parent oublie).
- Phase B : erreur clairement > 1e-3 (le sabotage est detecte).
- Phase C : tu argumentes geometriquement. **Indice** : un offset proximal se propage par la chaine cinematique et est multiplie par les rotations des joints en aval ; selon la configuration, l'amplification peut etre > 1 ou < 1 selon que les rotations en aval « replient » ou « depilent » l'erreur.

## Bonus

Ajoute un mode « robust extraction » qui parcourt tous les bodies entre la base et l'effecteur et recompose `M` non pas comme `data.xpos['hand']` directement, mais comme le produit des transforms statiques entre bodies. Compare avec ta version simple : si elles different, c'est qu'il y a des transforms statiques entre bodies que `data.xpos` masque (Menagerie en utilise pour les flanges, joints fictifs, etc.).
