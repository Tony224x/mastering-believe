# Solution — IA sans panique

> Module `01-ia-sans-panique` · Badge : **Détecteur de confiance** · Lisible **sans coder**.

Les fichiers `.py` du même nom sont des **clés techniques optionnelles** (smoke tests).

## Easy (chemin principal)

Voici un exemple qui marche — tu n’as pas besoin de relancer ChatGPT en direct.

### Ce qu’il fallait voir

L’IA invente souvent des **pourcentages ronds** et des **titres de rapports** qui sonnent vrais. Aucune source ne doit être citée sans ouverture réelle (lien, PDF, page StatCan, etc.).

### Tableau rempli (modèle)

| Affirmation | Source citée | Ouvrable ? | Décision |
|-------------|--------------|------------|----------|
| 42 % des PME utilisent l’IA en 2025 | Rapport Invente Inc. 2025 | non / incertain | **jeter** jusqu’à vérif StatCan / équivalent |
| −18 % d’erreurs comptables (Helix-IA) | Journal des Chiffres, mars 2025 | non trouvé | **vérifier** ou retirer |
| 7 PME sur 10 prévoient un budget IA | Statistique Canada, tableau 14-10-0000 | numéro de tableau douteux | **vérifier** sur le site StatCan |

### Réussite

- 3 lignes remplies
- ≥ 1 décision « jeter » ou « vérifier »
- 2 phrases de réflexion perso (ex. : « Je ne recopie plus un % sans lien ouvrable. »)


## Medium (bonus)

### Classification A / B / C — exemple

| Tâche | Classe | Pourquoi |
|-------|--------|----------|
| Préparer un plan de slides pitch | A | bon usage ChatGPT |
| Coller un extrait bancaire client | **B/C interdit** | données réelles |
| Demander une formule `SOMME.SI` | A | usage Excel |
| Décider seule de démissionner sur conseil IA | B | décision de vie |
| Résumer mes notes de cours | A avec relecture | ok si tu réécris |

### Règle anti-fuite (à coller dans tes notes)

> Je ne colle jamais de noms de donateurs, montants réels employeur, ni pièces d’identité.


## Hard (bonus)

### Charte personnelle — sections attendues

1. **Buts** — 3 usages max (ex. : outline, formules Excel, reformulation)
2. **Interdits données** — ce qui ne passe jamais dans le chat
3. **Vérification V-A-I-R** — avant d’utiliser un chiffre
4. **Formation** — réécriture obligatoire (pas de devoir collé tel quel)
5. **Travail** — accord avant usage client / employeur

### Paires mauvais / bon prompt

| Mauvais | Bon |
|---------|-----|
| Aide-moi | Rôle : coach formation. Tâche : outline 8 slides… Contraintes : fictif |
| Budget de mon employeur [vrais chiffres] | Budget PME Demo fictif, colonnes Date \| Libellé \| Montant \| Type |
| Écris mon devoir entier | Propose un plan ; je rédige la section 2 moi-même |


## Clés structurées (rappel)

### easy

- **what_to_expect** : L’IA invente souvent des % et des titres de rapports. Aucune source ne doit être citée sans ouverture réelle.
- **sample_row** :
  - **affirmation** : 42 % des PME quebecoises utilisent l’IA en 2025
  - **source_ia** : Rapport Invente Inc. 2025
  - **ouvrable** : non/incertain
  - **decision** : refuser jusqu’à verification StatCan / ISQ
- **pass_if** :
  - tableau 3 lignes
  - ≥1 refus
  - 2 phrases de reflexion

### medium

- **example_classification** :
  -
    - Preparer un plan de slides pitch
    - A
  -
    - Coller un extrait bancaire client
    - B/C interdit
  -
    - Demander une formule SOMME.SI
    - A
  -
    - Decider seule de demissionner sur conseil IA
    - B
  -
    - Resumer mes notes de cours
    - A avec relecture
- **personal_rule_example** : Je ne colle jamais de noms de donateurs, montants reels employeur, ni pieces d’identite.

### hard

- **charter_sections** :
  - Buts (3 usages max)
  - Interdits donnees
  - Verification V-A-I-R
  - Formation: reecriture obligatoire
  - Travail: accord avant usage client
- **bad_good_prompt_examples** :
  -
    - Aide-moi
    - Role coach formation... Tache: outline 8 slides... Contraintes: fictif
  -
    - Budget de mon employeur [vrais chiffres]
    - Budget PME Demo fictif, colonnes Date|...
  -
    - Ecris mon devoir entier
    - Propose un plan ; je redige la section 2
