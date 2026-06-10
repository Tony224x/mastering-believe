# Exercices Medium — Jour 8 : Pre-training & Tokenization

---

## Exercice 4 : Entrainer un tokenizer BPE complet

### Objectif

Passer du BPE "a la main" (exercice easy) a un trainer BPE complet et reutilisable : apprentissage des merges, encodage de mots inconnus, decodage.

### Consigne

1. Implementer une classe :

```python
class BPETokenizer:
    def train(self, corpus: list[str], n_merges: int): ...
    def encode(self, word: str) -> list[str]: ...
    def decode(self, tokens: list[str]) -> str: ...
```

   - `train` : decouper chaque mot en caracteres + marqueur de fin `</w>`, puis repeter : compter les paires adjacentes (ponderees par frequence des mots), merger la paire la plus frequente (tie-break deterministe : ordre lexicographique), enregistrer le merge.
   - `encode` : decouper le mot en caracteres puis appliquer les merges appris **dans l'ordre d'apprentissage** (pas par frequence dans le mot).
   - `decode` : concatener et retirer les `</w>`.

2. Entrainer sur le corpus `["low"]*5 + ["lower"]*2 + ["newest"]*6 + ["widest"]*3` avec 10 merges (le corpus canonique du papier BPE).

3. Verifier :
   - le premier merge est `("e", "s")` (frequence 9 : newest x6 + widest x3)
   - `encode("lowest")` reutilise les merges appris (doit contenir le token `est</w>` ou `est` selon vos merges — l'expliquer)
   - round-trip : `decode(encode(w)) == w` pour tous les mots du corpus ET pour "lowest" (jamais vu)
   - un mot avec un caractere jamais vu (ex : "boy") s'encode sans crash en tokens caracteres

4. Afficher la table des 10 merges dans l'ordre et la taille du vocabulaire final (caracteres initiaux + merges).

### Criteres de reussite

- [ ] Le premier merge est ("e", "s") et la sequence des merges est deterministe (2 runs identiques)
- [ ] `encode` applique les merges dans l'ordre d'apprentissage (commente pourquoi c'est important : c'est ce que fait le vrai BPE)
- [ ] Round-trip exact sur corpus + mot inconnu
- [ ] Les caracteres hors vocabulaire ne crashent pas
- [ ] La taille du vocab final = nb caracteres uniques + n_merges (verifiee)

---

## Exercice 5 : Masquage MLM 80/10/10 — implementer et verifier les statistiques

### Objectif

Implementer la strategie de masquage exacte de BERT et la valider statistiquement — un masquage subtilement faux ruine un pre-training entier.

### Consigne

1. Implementer :

```python
def mlm_mask(tokens, vocab_size, mask_id, special_ids, rng, p_select=0.15):
    """Retourne (input_ids, labels).
    - 15% des tokens NON speciaux sont selectionnes
    - parmi eux : 80% -> [MASK], 10% -> token aleatoire, 10% -> inchanges
    - labels = token original aux positions selectionnees, -100 ailleurs."""
```

2. Verifier la mecanique sur un petit exemple deterministe (seed fixe) :
   - les tokens speciaux ([CLS], [SEP]) ne sont JAMAIS selectionnes
   - `labels != -100` exactement aux positions selectionnees
   - aux positions "10% inchanges", `input_ids == tokens` mais `labels != -100` (le modele doit quand meme predire — c'est le piege)

3. Validation statistique : sur 20 000 sequences de 50 tokens, mesurer :
   - proportion de tokens selectionnes : 15% ± 0.5 pt
   - parmi les selectionnes : 80% ± 1 pt en [MASK], 10% ± 1 pt remplaces, 10% ± 1 pt inchanges
   - les remplacements aleatoires sont ~uniformes sur le vocab (chi-2 visuel : min/max des frequences dans un facteur 2 pour un vocab de 20)

4. Question (repondre en commentaire, puis verifier l'intuition par un mini-calcul) : pourquoi ne pas masquer 100% en [MASK] ? (Mismatch train/inference : [MASK] n'existe jamais en aval ; les 10%/10% forcent le modele a garder une representation de TOUS les tokens.)

### Criteres de reussite

- [ ] Les tokens speciaux sont proteges (verifie sur 20k sequences : zero violation)
- [ ] La convention labels=-100 hors selection est respectee
- [ ] Le cas "inchange mais a predire" est correctement gere et teste
- [ ] Les 4 statistiques tombent dans les tolerances
- [ ] La reponse sur le 80/10/10 est correcte et argumentee

---

## Exercice 6 : Calculateur Chinchilla — diagnostiquer des modeles reels

### Objectif

Transformer les scaling laws en outil : calculer l'optimum compute, et diagnostiquer si un modele donne est sur- ou sous-entraine.

### Consigne

Rappels : `C ≈ 6 * N * D` (FLOPs), optimum Chinchilla : `D/N ≈ 20` (tokens par parametre).

1. Implementer :
   - `compute_flops(N, D)` → C
   - `chinchilla_optimal(C)` → `(N_opt, D_opt)` en resolvant `C = 6 * N * 20N` → `N_opt = sqrt(C/120)`
   - `diagnose(N, D)` → ratio `D/N` et verdict : "sous-entraine" (< 10), "~optimal" (10-40), "sur-entraine" (> 40, au sens Chinchilla)

2. Verifier le calculateur sur Chinchilla lui-meme : pour le budget `C = 6 * 70e9 * 1.4e12`, `chinchilla_optimal(C)` doit redonner N ≈ 70B et D ≈ 1.4T (± 2%).

3. Construire la table d'optimums pour `C ∈ {1e21, 1e22, 1e23, 1e24}` : N_opt, D_opt, et verifier la loi de puissance : multiplier C par 10 multiplie N_opt et D_opt par sqrt(10) ≈ 3.16 (± 1%).

4. Diagnostiquer ces modeles (valeurs publiques approximatives) :
   - GPT-3 : N=175B, D=300B
   - Chinchilla : N=70B, D=1.4T
   - LLaMA-1 7B : N=7B, D=1T
   - LLaMA-3 8B : N=8B, D=15T
   Pour chacun : ratio D/N, verdict, et le N_opt qu'il aurait fallu pour ce budget de FLOPs.

5. Question (commentaire) : pourquoi LLaMA-3 s'entraine-t-il 100x au-dela de l'optimum Chinchilla ? (Indice : l'optimum Chinchilla minimise la loss a budget de TRAINING fixe — il ignore le cout d'INFERENCE. Un petit modele sur-entraine est moins cher a servir.)

### Criteres de reussite

- [ ] Le round-trip Chinchilla (C → N_opt, D_opt) retombe sur 70B/1.4T a ±2%
- [ ] La loi de puissance en sqrt(10) est verifiee numeriquement
- [ ] Les 4 diagnostics sont corrects (GPT-3 sous-entraine ratio ~1.7 ; LLaMA-3 ratio ~1875, tres au-dela de Chinchilla)
- [ ] Pour chaque modele, le N_opt a iso-FLOPs est calcule
- [ ] La reponse sur le trade-off training/inference est presente et correcte
