# Mission J6 — Total des entrées (easy)

> **Temps :** ~15 min · **Badge :** Formule qui matche

## Mission du jour

**But :** obtenir une formule dont le total **égale** ton calcul manuel.

### Données fixes (copie dans Excel)

| Date | Libellé | Montant | Type |
|------|---------|---------|------|
| 2026-01-02 | Vente A | 200 | entrée |
| 2026-01-03 | Vente B | 250 | entrée |
| 2026-01-04 | Vente C | 150 | entrée |
| 2026-01-05 | Loyer | 80 | sortie |
| 2026-01-06 | Fournitures | 40 | sortie |
| 2026-01-07 | Pub | 30 | sortie |

**Calcul manuel attendu :** entrées **600** · sorties **150** · solde **450**

## Écran de référence

![Capture d'Excel montrant une formule SOMME.SI et un total d'entrées de 600](../../assets/screens/screen-excel-somme-si.png)

## À faire

1. Colle le tableau. **Données fictives de la mission uniquement** — ne colle pas de fichier réel.
2. Demande à ChatGPT une formule pour totaliser les entrées (précise les colonnes **et** la langue de ton Excel : FR, FR-CA ou EN).
3. Vérifie les trois résultats : entrées = **600**, sorties = **150**, solde = **450**.
4. Si Excel affiche `#NOM?`, note la langue des fonctions de ta version et demande une formule adaptée. Ne remplace pas la formule au hasard.
5. Si un total diffère : copie le message d’erreur ou le total obtenu dans ChatGPT et corrige.

## Indice
Cherche `SOMME.SI` (ou `SUMIF` en anglais). Attention aux guillemets et à la plage. Le solde (entrées − sorties) est un **contrôle indépendant**, pas un bonus.

## Réussite
- [ ] Formule visible
- [ ] Total entrées = 600
- [ ] Total sorties = 150
- [ ] Solde = 450
- [ ] Je peux expliquer quelles colonnes la formule utilise

## Feedback
**Badge Formule qui matche** si les trois totaux collent au premier ou second essai. Pas de stress si le 2e essai a suffi.
