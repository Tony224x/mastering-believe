# J22 — Exercice EASY : carte mentale du paysage VLA frontier

## Objectif

Verifier que tu sais positionner les 3 papiers cles (GR00T N1, Helix, LBM TRI) face aux VLAs vus avant (RT-2/Octo/OpenVLA/pi0), et que tu identifies correctement le pattern System1/System2.

## Consigne

1. **Tableau comparatif (5 lignes)**. Remplis un tableau Markdown avec ces colonnes : `Modele`, `Annee`, `Architecture (mono ou dual-system)`, `Frequence controle effective`, `DoF supportes`, `Open-weights ?`. Une ligne par : RT-2, Octo, OpenVLA, pi0, GR00T N1, Helix, LBM TRI. (7 lignes au total.)
2. **Identifier le decouplage temporel**. Pour GR00T N1 et Helix, indique :
   - la frequence de System2 (en Hz, valeur approximative donnee dans la theorie ou les sources)
   - la frequence de System1 (en Hz)
   - le ratio entre les deux
3. **Une phrase justificative**. En une phrase de 2 lignes max, explique pourquoi un VLA monolithique 7B (type OpenVLA) ne peut pas piloter un humanoide whole-body 35 DoF a 200 Hz.

## Criteres de reussite

- Le tableau a bien 7 lignes (les 4 monolithiques + les 3 dual-system).
- Tu maries chaque modele a son label "mono" ou "dual-system" sans erreur.
- Les frequences System1/System2 pour GR00T et Helix sont citees (~7-9 Hz et ~120-200 Hz respectivement).
- La phrase justificative mentionne explicitement la contrainte de **latence du VLM** (sortie tokens/s insuffisante pour 200 Hz).

## Format de rendu

Cree un fichier `mes-reponses-22-frontier-humanoid-easy.md` (gitignore-friendly, dans ton dossier perso) avec :

```markdown
# J22 — Easy

## 1. Tableau
| Modele | Annee | Archi | f_ctrl | DoF | Open ? |
| --- | --- | --- | --- | --- | --- |
| ... | ... | ... | ... | ... | ... |

## 2. Decouplage temporel
- GR00T N1 : System2 ~ ? Hz, System1 ~ ? Hz, ratio ?
- Helix    : System2 ~ ? Hz, System1 ~ ? Hz, ratio ?

## 3. Justification monolithique impossible
> ...
```

Pas de code attendu. Compare ensuite avec la solution dans `solutions/22-frontier-humanoid.py` (la solution embarque le tableau de reference en commentaire).
