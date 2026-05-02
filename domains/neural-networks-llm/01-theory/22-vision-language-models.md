# Jour 22 — Vision-language models : ViT, CLIP, LLaVA

> **Temps estime** : 5h | **Prerequis** : J5-J9 (attention, transformer, LLMs modernes), J12 (multimodalite intro)

---

## 1. ViT : un transformer pour les images (Dosovitskiy 2020)

Avant 2020, les CNN dominaient la vision (ResNet, EfficientNet). Les transformers etaient cantonnes au texte. ViT (Vision Transformer) a casse cette frontiere avec une idee aussi simple que radicale : **traiter une image comme une sequence de patches, exactement comme un texte est traite comme une sequence de tokens**.

### Le pipeline ViT en une image

```
Image 224x224x3
   |
   |  decoupage en patches 16x16
   v
Grille 14x14 = 196 patches (chacun 16x16x3 = 768 dims)
   |
   |  flatten + projection lineaire (768 -> d_model, ex d=768)
   v
196 vecteurs de dim d
   |
   |  + positional encoding (un par patch)
   |  + token [CLS] en tete
   v
Sequence de 197 tokens
   |
   |  Transformer encoder (12 couches, self-attention)
   v
197 representations
   |
   |  on prend le token [CLS] -> classification head
   v
Logits classes
```

### Pourquoi ca marche

1. **Self-attention sur patches** : chaque patch peut "regarder" tous les autres des la premiere couche. Un CNN met 10+ couches a relier le coin haut-gauche au coin bas-droit ; ViT le fait en 1.
2. **Inductive bias faible** : ViT n'a pas la translation invariance integree d'un CNN. C'est une faiblesse a petite echelle (< 1M images) mais une force a grande echelle (300M+ images) car le modele apprend les biais utiles a partir des donnees.
3. **Scaling** : ViT scale comme un LLM. Plus de params + plus de data = meilleure perf, sans plateau visible.

### Positional encoding 2D

Une image n'est pas 1D. Le patch (3, 5) n'a pas la meme relation au patch (3, 6) qu'au patch (8, 5). Trois choix :
- **PE 1D appris** (ViT original) : on numerote les patches 0..195 et on apprend un vecteur par index. Marche etonnamment bien.
- **PE 2D appris** : un vecteur par (row, col). Plus correct conceptuellement.
- **RoPE 2D** (modeles 2024+) : extension de RoPE en deux dimensions, gere mieux la resolution variable.

En pratique, pour les modeles modernes (DINOv2, SigLIP, Pixtral), c'est du RoPE 2D ou des variantes.

---

## 2. CLIP : l'idee contrastive et l'espace partage (Radford 2021)

CLIP (Contrastive Language-Image Pretraining) a invente un nouveau paradigme : **apprendre image et texte ensemble dans un meme espace vectoriel** sans labels supervisés.

### L'astuce data : le web

OpenAI a scrape 400M paires (image, alt-text) du web. Pas de classification, pas de bounding box, juste : "cette image va avec cette legende". C'est presque gratuit a obtenir.

### L'architecture : dual encoders

```
Image  -> [ViT image encoder]   -> vecteur i  (dim 512)
Texte  -> [Transformer text enc]-> vecteur t  (dim 512)
                                          |
                                          v
                            similarite cosinus s = i . t / (|i| |t|)
```

Aucun cross-attention. Les deux tours sont independantes. C'est ce qui rend l'inference **rapide a l'echelle** : on encode 1M images une fois, et chaque query texte est un simple produit scalaire.

### La loss contrastive (InfoNCE / softmax cross-entropy)

Sur un batch de N paires (image_i, texte_i) :

```
Construire matrice S (N x N) : S[i,j] = sim(image_i, texte_j)
Multiplier par temperature inverse (1/T, T appris)
Loss image->texte : softmax cross-entropy sur les lignes (la bonne paire est en diagonale)
Loss texte->image : softmax cross-entropy sur les colonnes
Loss totale = moyenne des deux
```

**L'intuition** : pousser la diagonale (paires correctes) vers le haut, pousser tout le reste vers le bas. C'est de la classification N-way ou les N-1 negatifs sont les autres exemples du batch.

### Ce qu'on obtient

- **Zero-shot classification** : pour classer une image dans 1000 categories ImageNet, on encode chaque categorie en texte ("a photo of a cat", "a photo of a dog"...) et on prend le plus proche en cosinus de l'image. Performance proche d'un ResNet supervise sans avoir vu un seul label ImageNet.
- **Search image <-> texte** : moteur de recherche d'images natif.
- **Espace partage** : on peut faire l'arithmetique vectorielle (king - man + woman = queen) en cross-modal.

CLIP est devenu le **squelette visuel** de presque tous les VLMs modernes (LLaVA utilisait CLIP comme image encoder ; **Stable Diffusion 1.x utilisait le text encoder de CLIP** comme conditionneur du diffuseur, pas l'image encoder).

---

## 3. SigLIP : pourquoi sigmoid > softmax a grande echelle (Zhai 2023)

CLIP a un probleme structurel a grande echelle : la **softmax loss est globale**. Pour normaliser, il faut tout le batch en memoire. Plus le batch est grand, mieux ca marche (plus de negatifs durs), mais le cout memoire explose.

### Le switch SigLIP

SigLIP remplace la softmax par une **sigmoid loss par paire** :

```
Pour chaque paire (image_i, texte_j) :
  label = 1 si i==j sinon 0
  logit = sim(image_i, texte_j) * temperature + biais
  loss_ij = binary_cross_entropy(sigmoid(logit), label)
Loss = somme sur i,j
```

Chaque paire est une classification binaire independante. **Pas de normalisation globale.**

### Pourquoi c'est mieux

1. **Scale arbitraire en batch** : on peut shard le batch sur plusieurs GPUs sans synchronisation pour la normalization. SigLIP a ete entraine avec des batches de 1M.
2. **Bias term apprenable** : compense le desequilibre 1 positif vs N-1 negatifs (sans ce biais, le modele predit toujours "non match").
3. **Performance** : a batch egal, SigLIP egale CLIP. SigLIP est equivalent ou legerement superieur sous contraintes memoire identiques (Zhai 2023, fig 3).
4. **Stable** : moins de sensibilite a la temperature, convergence plus reguliere.

En 2024-2026, **SigLIP (et SigLIP 2) est devenu le standard** pour les nouveaux VLMs (PaliGemma, Pixtral utilise un derive).

---

## 4. Architectures VLM : trois familles

Une fois qu'on a un encoder visuel solide (CLIP/SigLIP/ViT pretrain), comment le coller a un LLM ?

### Famille A — LLaVA-style : projecteur + concatenation

```
Image -> ViT encoder -> N tokens visuels (dim_vit)
                              |
                              v
                         MLP projecteur
                              |
                              v
                    N tokens (dim_llm)
                              |
                              | concat
                              v
[texte tokens] [image tokens] [texte tokens] -> LLM
```

**Recette** :
1. ViT (souvent SigLIP-So400m) genere une grille de tokens visuels.
2. Un petit MLP (2 couches) projette chaque token visuel dans l'espace du LLM.
3. On concatene avec les tokens de texte, on insere un placeholder `<image>` que le projecteur remplit.
4. Le LLM voit tout ca comme une sequence de tokens normaux.

**Avantages** : simple, peu de params nouveaux (~50M pour le projecteur), exploite a fond un LLM pre-entraine.
**Inconvenients** : les tokens visuels prennent de la place dans le contexte, l'image est encodee en bloc, pas de cross-attention dynamique.

C'est ce que font **LLaVA, LLaVA-NeXT, MiniCPM-V, InternVL, Qwen2-VL, Pixtral**. La famille dominante en 2024-2026. (Llama 3.2 Vision, par contraste, est Flamingo-style — voir Famille B.)

### Famille B — Flamingo-style : cross-attention insere (DeepMind 2022)

```
Texte tokens -> LLM (frozen) avec couches de cross-attention inserees
                                            ^
                                            |
Image -> Perceiver Resampler -> K tokens visuels
```

**Recette** :
1. Le LLM est gele.
2. On insere des couches de **gated cross-attention** entre les couches du LLM.
3. Chaque couche cross-attendi sur les tokens visuels, avec un gate appris (initialise a 0 pour ne pas casser le LLM au depart).
4. Un Perceiver Resampler reduit la grille visuelle a un nombre fixe (ex 64) de tokens.

**Avantages** : LLM intact, image n'occupe pas de places dans le contexte texte, supporte naturellement l'interleaved (image, texte, image, texte...).
**Inconvenients** : architecture plus complexe, plus de params nouveaux, moins flexible pour le scaling.

Flamingo a ouvert la voie en 2022. Idefics (HuggingFace), Otter et quelques modeles l'ont suivi. **Aujourd'hui, LLaVA-style a largement gagne** parce que c'est plus simple et que les LLMs modernes ont des contextes longs (128k+).

### Famille C — Native multimodal (Fuyu, Chameleon, GPT-4o, Gemini 2.5)

```
Image -> patches bruts -> projection lineaire -> tokens
                                                    |
                                                    | concat
                                                    v
[texte tokens] [image tokens] -> Transformer unique
```

**Recette** :
1. **Pas d'encoder visuel separe**. Pas de ViT pre-entraine.
2. Les patches d'image sont projectes directement dans l'espace du transformer.
3. Le transformer est entraine **from scratch** ou continual-trained sur du texte ET des images.
4. Selon la variante : Fuyu reste **discriminatif** (ne genere pas d'images, lit seulement). En revanche, **Chameleon (Meta) et GPT-4o image gen** vont plus loin et generent egalement des tokens visuels via une tete adaptee — c'est cette variante qui unifie lecture ET generation d'image.

**Avantages** :
- **Resolution arbitraire** : Fuyu peut prendre n'importe quelle taille d'image.
- **Generation native** : un meme modele lit ET genere les images (GPT-4o image gen, Gemini Nano Banana).
- **Apprentissage joint** : pas de bottleneck de l'encoder visuel.

**Inconvenients** : extremement gourmand en compute pour pretraining, n'exploite pas un encoder visuel pretrain solide, plus difficile a entrainer correctement.

C'est la voie des **labs frontier en 2025-2026** : GPT-4o, Gemini 2.5 ultra, Chameleon (Meta), Fuyu (Adept). Pour les modeles open-source produits, LLaVA-style reste majoritaire.

---

## 5. Image tokenization : combien de tokens, combien ca coute

### Le calcul de base

Pour une image H x W et un patch de taille P :

```
nb_patches = (H/P) * (W/P)
nb_tokens_apres_resampler = R (souvent egal a nb_patches dans LLaVA-style)
```

| Resolution | Patch 14 | Patch 16 | Patch 32 |
|---|---|---|---|
| 224 x 224 | 256 tokens | 196 tokens | 49 tokens |
| 336 x 336 | 576 tokens | 441 tokens | 110 tokens |
| 512 x 512 | 1 296 tokens | 1 024 tokens | 256 tokens |
| 1024 x 1024 | 5 329 tokens | 4 096 tokens | 1 024 tokens |

**Une image 1024x1024 avec patch 14 = 5 329 tokens** (73x73 = 5329 patches). C'est l'equivalent d'environ 4 pages de texte dense. Pour un VLM avec 128k de contexte, c'est ~25 images max (et encore, sans laisser de place pour le texte).

### Le cout cache

L'attention est **quadratique en sequence length**. Une image dense bouffe la memoire et le compute :
- Compute attention : O(N^2) avec N = nb_tokens.
- KV cache : O(N) par couche -> pour 32 couches d'un LLM 70B, ca chiffre vite.

**C'est pourquoi GPT-4 vision facture l'image plus cher que le texte equivalent : sous le capot, c'est plus de tokens.**

### Strategies de reduction

1. **Resolution fixe basse** (CLIP 224, LLaVA 1.5 336) : simple mais on perd le detail (OCR fine impossible).
2. **Pooling apres ViT** : reduire 24x24 patches a 12x12 par pooling 2x2. Divise le cout par 4.
3. **Perceiver Resampler** (Flamingo) : forcer un nb fixe de tokens (ex 64) quel que soit l'input.
4. **Q-Former** (BLIP-2) : un mini-transformer apprend a compresser N tokens en K << N tokens.
5. **Resolution dynamique / AnyRes** (LLaVA-NeXT et la suite) : voir section suivante.

---

## 6. Resolution dynamique et AnyRes

LLaVA 1.5 etait fixe a 336x336. Inutile pour l'OCR ou les details. **LLaVA-NeXT (janvier 2024) a popularise AnyRes**.

### Le principe

Au lieu de redimensionner l'image en 336x336, on la decoupe en grille de tuiles de 336x336 et on encode chaque tuile separement.

```
Image originale 1024 x 768
   |
   |  choix de la grille la plus proche (2x2, 1x4, 4x1...)
   |
   v
Grille 3x2 de tuiles 336x336 (avec padding)
   |
   |  + 1 thumbnail 336x336 de l'image entiere (pour le contexte global)
   |
   v
7 tuiles -> 7 * 576 tokens = 4 032 tokens
```

### Pourquoi le thumbnail global

Sans lui, le modele perd la vue d'ensemble (chaque tuile ignore les voisines). Le thumbnail donne le contexte ; les tuiles donnent les details.

### Strategies modernes (2025-2026)

- **Qwen2-VL** : NaViT (native resolution ViT) avec RoPE 2D, accepte tout aspect ratio sans padding.
- **InternVL 2.5** : tuilage adaptatif jusqu'a 4K.
- **Pixtral** : grille variable + RoPE 2D, position d'un patch encode son (row, col) absolu.
- **Idefics 3** : packing de plusieurs images sans tuiling explicite.

**La frontiere 2026** : ne plus avoir de patch fixe. Les modeles natifs (GPT-4o, Gemini 2.5) gerent la resolution comme une dimension fluide.

---

## 7. Etat de l'art 2025-2026 (rapide tour)

| Modele | Origine | Architecture | Specificite |
|---|---|---|---|
| **GPT-4o / 4.1** | OpenAI | Native multimodal | Voix + vision + texte unifies, generation image native |
| **Claude 4.5 Sonnet vision** | Anthropic | LLaVA-style + propre encoder | OCR fort, agentic computer use (screenshots) |
| **Gemini 2.5** | Google | Native multimodal | Video native (jusqu'a 1h), 2M context |
| **Llama 3.2 Vision** | Meta | Flamingo-style (cross-attention layers inserees dans le LLM) | 11B et 90B, open-weights |
| **Qwen2-VL / Qwen3-VL** | Alibaba | NaViT + LLaVA-style | Tres bon en OCR multilingue, video |
| **Pixtral 12B / Large** | Mistral | LLaVA-style + RoPE 2D + SigLIP | Performance/poids excellent |
| **InternVL 2.5 / 3** | Shanghai AI Lab | LLaVA-style avec tuilage 4K | Open, OCR, doc understanding |
| **Molmo** | AllenAI | LLaVA-style | Donnees curees humaines, pointage 2D |
| **LLaVA-OneVision** | LLaVA team | LLaVA-style multi-image+video | Open, recherche |

**Tendance 2026** : convergence vers natif multimodal pour les frontier (GPT, Gemini, peut-etre Claude 5), persistance de LLaVA-style pour open-source produit.

---

## 8. Limites actuelles (encore en 2026)

### OCR fine

Lire une note manuscrite sur un schema architectural a basse resolution reste difficile. Meme avec AnyRes, les caracteres < 8 pixels passent mal. Solution actuelle : tools externes (Tesseract, dedicated OCR API) en complement.

### Video longue

Encoder une video d'1h en 1 fps = 3 600 images = millions de tokens. Strategies actuelles : sampling adaptatif, summarization recursive, tokens video specialises (Gemini, Qwen2-VL). Mais le **comprehensive long-form video understanding** reste un probleme ouvert.

### Compositional reasoning

Question type : "combien d'objets rouges sont a gauche du chat ?". Les VLMs progressent mais restent en dessous de l'humain. C'est un probleme d'**ancrage spatial** plus que de perception.

### Counting

Compter precisement (>10 elements identiques) est un point faible structurel. Le modele "voit" l'ensemble globalement, ne fait pas le decompte serie par serie.

### Generation precise

Generer une image ou "le texte affiche est exactement HELLO WORLD en police Arial" est encore peu fiable. Les modeles natifs (GPT-4o image, Gemini Nano Banana) progressent vite mais ne sont pas parfaits.

---

## 9. Idees fausses repandues

**Idee fausse #1** : "ViT a tue les CNN."
Faux. Pour de la vision dense (segmentation, detection mobile), les CNN restent competitifs. ConvNeXt v2, EfficientFormer, etc. ViT a gagne pour le pretraining a grande echelle et le multimodal.

**Idee fausse #2** : "CLIP comprend les images."
Faux. CLIP encode des representations utiles pour le retrieval. Il n'a aucune capacite generative ni de raisonnement. Pour comprendre, il faut connecter a un LLM.

**Idee fausse #3** : "Plus de tokens visuels = meilleure perf."
Pas lineairement. Au-dela d'un seuil (~4k tokens pour la plupart des taches), les gains stagnent et le cout explose. La qualite de l'encodeur (SigLIP-So400m vs CLIP ViT-L) compte plus que la quantite de tokens.

**Idee fausse #4** : "Native multimodal est toujours mieux."
Pas en 2026. Native est plus puissant en theorie mais demande un compute pretraining x10. Pour 90% des cas d'usage, un bon LLaVA-style avec SigLIP-So400m + Llama 3.2 70B Instruct rivalise avec GPT-4o pour 1/100 du cout.

**Idee fausse #5** : "Le LLM frozen est juste pour economiser le compute."
Aussi pour preserver les capacites. Si on fine-tune trop le LLM dans LLaVA, on degrade ses capacites text-only. La sequence classique : 1) entrainer le projecteur seul, 2) fine-tuner LLM + projecteur ensemble avec un LR tres bas.

---

## Key takeaways (flashcards)

**Q1** — Quelle est l'idee centrale de ViT ?
> Decouper l'image en patches, traiter chaque patch comme un token, et appliquer un transformer encoder. La self-attention relie tous les patches des la premiere couche.

**Q2** — Pourquoi SigLIP est-il preferable a CLIP a tres grande echelle ?
> SigLIP utilise une sigmoid loss par paire au lieu d'une softmax globale. Ca elimine la normalization cross-batch, permet des batches massifs (1M+) et facilite le sharding multi-GPU. A batch egal, SigLIP egale CLIP ; a tres grand batch, SigLIP depasse.

**Q3** — Quelle est la difference fondamentale entre LLaVA-style et Flamingo-style ?
> LLaVA concatene les tokens visuels dans le contexte du LLM via un MLP projecteur (LLM voit l'image comme du texte). Flamingo insere des couches de cross-attention dans le LLM, qui restent attentives aux features visuelles externes (LLM ne consomme pas son contexte texte). LLaVA est plus simple et a gagne.

**Q4** — Combien de tokens pour une image 1024x1024 avec patch 14 ?
> 5 329 tokens (1024/14 = 73, 73*73 = 5329 patches). C'est equivalent a 4-5 pages de texte. C'est pour cette raison que les images coutent cher en API et que la resolution dynamique a ete inventee.

**Q5** — Que resout AnyRes / la resolution dynamique ?
> Le compromis entre voir le detail (haute res necessaire) et tenir dans le budget tokens (basse res necessaire). On decoupe l'image en tuiles de resolution fixe + un thumbnail global, on encode chacune separement. Permet une resolution effective elevee sans modifier l'encodeur ViT pretrain.

---

## Sources

- Dosovitskiy et al. (2020) — *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* (ViT). https://arxiv.org/abs/2010.11929
- Radford et al. (2021) — *Learning Transferable Visual Models From Natural Language Supervision* (CLIP). https://arxiv.org/abs/2103.00020
- Zhai et al. (2023) — *Sigmoid Loss for Language Image Pre-Training* (SigLIP). https://arxiv.org/abs/2303.15343
- Liu et al. (2023) — *Visual Instruction Tuning* (LLaVA). https://arxiv.org/abs/2304.08485
- Liu et al. (2024) — *Improved Baselines with Visual Instruction Tuning* (LLaVA-1.5). https://arxiv.org/abs/2310.03744
- Alayrac et al. (2022) — *Flamingo: a Visual Language Model for Few-Shot Learning*. https://arxiv.org/abs/2204.14198
- Adept (2023) — *Fuyu-8B: A Multimodal Architecture for AI Agents*. https://www.adept.ai/blog/fuyu-8b
- Li et al. (2023) — *BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models*. https://arxiv.org/abs/2301.12597


---

## Pour aller plus loin

Lectures couvrant ce sujet (playlists dans [`shared/external-courses.md`](../../../shared/external-courses.md)) :

- **Stanford CS231N — Lec. 16 (Vision and Language)** — fondamentaux ViT, CLIP, captioning.
- **CMU 11-711 (Welleck) — Lec. 20 (Multimodal Modeling I), Lec. 22 (Multimodal Modeling II)** — VLM modernes (LLaVA, Flamingo, Fuyu, fusion native) en 2025.
