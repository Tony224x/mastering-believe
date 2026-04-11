# Jour 14 — Mock Interviews : Comment Simuler un Vrai Entretien Tech

> **Temps estime** : 60-75 min de lecture + 2-3 heures de pratique | **Objectif** : maitriser la structure d'un entretien FAANG, t'entrainer comme en conditions reelles, et eliminer les pieges de communication qui coulent les bons candidats

---

## 1. Pourquoi les mocks sont le levier ultime

Tu peux connaitre tous les patterns de la semaine 1 et 2. Sans pratique en conditions reelles, **tu vas geler le jour J**. La raison : un entretien tech n'est pas un probleme LeetCode, c'est une **performance** — tu dois coder ET parler ET gerer le temps ET reagir aux hints de l'interviewer.

**Les 4 facteurs qui font la difference** :
1. **Communication** : penser a voix haute, clarifier les edge cases, negocier les constraintes
2. **Structure** : suivre un process plutot qu'improviser
3. **Gestion du temps** : 45 min, c'est court. Il faut savoir rythmer.
4. **Reaction aux hints** : un interviewer qui te donne une piste ne le fait pas pour etre sympa — il attend que tu la saisisses rapidement.

---

## 2. La structure d'un entretien FAANG typique (45 min)

```
┌─────────────────────────────────────────────────┐
│ 0-5   min : Introductions + enonce du probleme  │
│ 5-10  min : Clarification + exemples + plan      │
│ 10-35 min : Codage + test                        │
│ 35-40 min : Complexite + edge cases              │
│ 40-45 min : Questions candidat                   │
└─────────────────────────────────────────────────┘
```

**Point critique** : si tu es encore en train de coder a 40 min, tu es en retard. Le code final doit tourner ou presque, et tu dois AVOIR discute de la complexite.

---

## 3. Le process en 6 etapes — a faire DANS L'ORDRE

### Etape 1 — Clarifier (2-3 min)

**Ne JAMAIS commencer a coder tant que tu n'as pas pose ces questions** :

- Quels sont les types d'entree exacts ? (int, string, list, tree...)
- Quelles sont les contraintes de taille ? (n = 10 ? 10^6 ? 10^9 ?)
- Les valeurs peuvent-elles etre negatives ? Zero ? Tres grandes ?
- L'input est-il valide ? Vide ? Avec doublons ?
- Quel est l'output attendu ? Valeur, index, booleen, structure ?
- Y a-t-il une contrainte de complexite ? (O(n log n) ? O(1) espace ?)

> **Exemple** : probleme "find the kth largest element". Questions : k est-il toujours <= n ? Les valeurs sont-elles uniques ou avec doublons ? Faut-il retourner le k-ieme en valeur ou le k-ieme distinct ?

### Etape 2 — Exemples et edge cases (2-3 min)

Donne ton propre exemple pour confirmer ta comprehension. Pense **dehors du cas evident** :

- Cas vide (`[]`, `""`, `None`)
- Taille 1
- Tous identiques
- Deja trie / trie inverse
- Tres grand (peut casser la recursion, overflow)

```
Moi : "Je prends l'exemple [2, 7, 11, 15] target=9. Je pense que la sortie
       est [0, 1] car 2 + 7 = 9. Est-ce que c'est ca que vous attendez ?"
Interviewer : "Oui."
Moi : "Ok. Et si le tableau est vide, ou si aucune paire ne marche, je
       retourne [] ?"
Interviewer : "Oui."
```

### Etape 3 — Proposer une approche brute force (2 min)

Meme si tu connais directement la solution optimale, **enonce la brute force** :

> "Une premiere approche serait deux boucles imbriquees qui testent toutes les paires. Ca donne O(n^2). Est-ce que c'est acceptable ?"

Pourquoi : ca montre que tu comprends le probleme, et ca sert de baseline pour discuter de l'optimisation. L'interviewer dira "peux-tu faire mieux ?" — c'est ton cue pour passer a la solution optimale.

### Etape 4 — Optimiser (2-3 min)

Explique l'optimisation **AVANT de coder** :

> "Je peux faire O(n) en utilisant un hash map pour stocker les elements deja vus. Pour chaque element, je cherche `target - current` dans le map. Time O(n), Space O(n)."

Si tu ne sais pas comment optimiser, reste en brute force et commence a coder. Mieux vaut du code correct et lent que du code optimal et cassé.

### Etape 5 — Coder (15-20 min)

**Regles d'or** :
- Parle pendant que tu codes (un silence > 30s = signal negatif)
- Nomme tes variables explicitement (`left`, `right`, pas `a`, `b`)
- Commente les parties non triviales avec une phrase
- Si tu t'embrouilles, dis-le et reprend avec un exemple

Exemple :
> "Je parcours le tableau. Pour chaque element, je calcule le complement.
> Si le complement est deja dans le map, je retourne les indices. Sinon, j'ajoute
> l'element courant au map. Je stocke APRES le check pour eviter d'utiliser
> le meme element deux fois."

### Etape 6 — Tester et analyser (3-5 min)

Apres avoir ecrit le code, **teste-le a la main** avec un exemple simple. Parcours chaque ligne et maintiens l'etat.

Puis :
- Complexite temps : "O(n) car une seule passe"
- Complexite espace : "O(n) pour le hash map"
- Edge cases : "si le tableau est vide, la boucle ne s'execute pas et je retourne []"
- Optimisations possibles : "je pourrais trier et utiliser two pointers pour O(1) espace, mais ca perd les indices originaux"

---

## 4. Les commandements de la communication

1. **Pense a voix haute en permanence**. Meme si tu doutes : "Je pense que... hmm, mais alors ca ne marche pas si..."
2. **Dis ce que tu es en train de faire**. Pas "je fais ca" mais "je construis un dict pour mapper chaque element a son index".
3. **Pose des questions rhetoriques et reponds-y**. "Quelle est la complexite de cette operation ? C'est O(1) parce que..."
4. **Reagis aux hints IMMEDIATEMENT**. Si l'interviewer dit "peut-on faire mieux en espace ?", c'est une piste directe. Explore-la.
5. **Demande du feedback**. "Est-ce que c'est la bonne direction ?" est parfaitement acceptable et montre que tu es collaboratif.

---

## 5. Les pieges qui coulent les bons candidats

| Piege | Consequence | Solution |
|-------|-------------|----------|
| Commencer a coder sans clarifier | Code faux, -1 en embauche | Clarifier 2-3 min minimum |
| Coder en silence | Interviewer pense que tu patauges | Parler en permanence |
| S'entete sur une approche qui ne marche pas | Burn 15 min, stuck | Si apres 5 min la brute force ne sort pas, demander un hint |
| Ignorer les hints | Burn 10 min a re-decouvrir | Ecouter et rebondir sur chaque hint |
| Sauter les tests | Bugs non detectes | Toujours derouler l'exemple a la main |
| Ne pas parler de complexite | Absence de rigueur | Enoncer time + space + tradeoffs |
| Stresser et bloquer | Rush + erreurs | Prendre 5 secondes, respirer, reprendre a l'exemple |

---

## 6. Les 3 mocks d'aujourd'hui

Aujourd'hui tu vas faire **3 mocks complets** (easy, medium, hard). Pour chacun :

1. Chronometre 45 minutes
2. Lis l'enonce, ne regarde pas la solution
3. Fais le process en 6 etapes a voix haute (enregistre-toi si possible)
4. Code la solution
5. Teste avec 2 exemples
6. Analyse la complexite
7. PUIS et seulement puis, regarde la solution fournie et compare

---

### Mock 1 (Easy) — Valid Parentheses

**Enonce** : etant donne une string `s` contenant uniquement les caracteres `'('`, `')'`, `'{'`, `'}'`, `'['`, `']'`, determine si la string est valide (chaque ouvrant a un fermant correspondant dans le bon ordre).

**Exemples** :
- `"()"` → `True`
- `"()[]{}"` → `True`
- `"(]"` → `False`
- `"([)]"` → `False`
- `"{[]}"` → `True`

**Walkthrough attendu** :
1. Clarifier : string vide = valide ? (oui par convention) ; uniquement ces 6 caracteres ? (oui)
2. Exemple : `"([)]"` illustre pourquoi on ne peut pas juste compter les ouvrants et fermants.
3. Brute force : ... il n'y en a pas de pertinente ici, c'est stack direct.
4. Solution : stack. On push les ouvrants, et a chaque fermant on verifie que le top est l'ouvrant correspondant.
5. Code (voir solutions).
6. Complexite : O(n) temps, O(n) espace.
7. Edge cases : string vide (True), un seul caractere (False), string avec que des ouvrants (False).

---

### Mock 2 (Medium) — Longest Substring Without Repeating Characters

**Enonce** : etant donne une string `s`, trouve la longueur de la plus longue substring **sans caracteres qui se repetent**.

**Exemples** :
- `"abcabcbb"` → 3 (`"abc"`)
- `"bbbbb"` → 1 (`"b"`)
- `"pwwkew"` → 3 (`"wke"`)
- `""` → 0

**Walkthrough attendu** :
1. Clarifier : string vide = 0 ? Majuscules/minuscules distinctes ? Caracteres Unicode ou ASCII ?
2. Exemple : dans `"pwwkew"`, la reponse est 3 (`"wke"`), pas 4 (`"pwke"` n'est pas une sub STRING contigue mais `"pwwke"` contient deux `w`).
3. Brute force : tester toutes les substrings, verifier l'unicite. O(n^3). Inacceptable pour n = 10^5.
4. Solution optimale : sliding window avec un set/dict qui tracke les caracteres de la fenetre courante. Quand on rencontre un duplicate, on avance `left` jusqu'a retirer le duplicate.
5. Code (voir solutions).
6. Complexite : O(n) temps (chaque caractere visite au plus 2 fois), O(min(n, alphabet)) espace.

---

### Mock 3 (Hard) — Merge Intervals

**Enonce** : etant donne un tableau d'intervalles `intervals` ou `intervals[i] = [start_i, end_i]`, fusionne tous les intervalles qui se chevauchent et retourne la liste des intervalles non chevauchants couvrant toutes les entrees.

**Exemples** :
- `[[1,3],[2,6],[8,10],[15,18]]` → `[[1,6],[8,10],[15,18]]`
- `[[1,4],[4,5]]` → `[[1,5]]`
- `[[1,4],[0,4]]` → `[[0,4]]`
- `[[1,4],[2,3]]` → `[[1,4]]`

**Walkthrough attendu** :
1. Clarifier : intervalles peuvent-ils etre vides ou single-point ? Est-ce que `[1,4]` et `[4,5]` se chevauchent (endpoint touch) ? (conventionnellement, oui — on les fusionne).
2. Exemple complique : `[[1,4],[2,3]]` → le deuxieme est entierement dans le premier, resultat `[[1,4]]`.
3. Intuition : si on trie par start, alors les chevauchements sont forcement entre intervalles consecutifs dans l'ordre trie.
4. Algo : trier par start, puis iterer. Pour chaque intervalle, soit il chevauche le dernier du resultat (fusionner), soit pas (append).
5. Code (voir solutions).
6. Complexite : O(n log n) temps (tri), O(n) espace (resultat).
7. Edge cases : tableau vide → `[]`. Un seul intervalle → retourne tel quel. Tous chevauchent → un seul intervalle final.

---

## 7. Comment t'entrainer apres aujourd'hui

- **1 mock par jour** jusqu'a l'entretien, alternance easy/medium/hard
- **Utilise Pramp, Interviewing.io, ou LeetCode Mock Interview** pour des vrais mocks avec de vraies personnes
- **Enregistre-toi** : reecoute pour reperer les hesitations, silences, mots de remplissage
- **Fais des post-mortems** : qu'est-ce qui a coince ? Pourquoi ? Quelle lecon ?

---

## 8. Checklist avant l'entretien reel

- [ ] Je sais reciter les 6 etapes du process
- [ ] Je peux coder les 10 patterns de base (day 1-13) en < 15 min sans erreur
- [ ] J'ai fait au moins 5 mocks chronometres
- [ ] Je peux expliquer une solution a voix haute sans lire le code
- [ ] Je sais gerer un hint sans perdre ma confiance
- [ ] J'ai prepare 3 questions pour l'interviewer a la fin
- [ ] J'ai dormi 8h la veille

---

## 8bis. Entretiens tech en 2026 : l'impact des LLM

Depuis 2024, la majorite des FAANG et des scale-ups utilisent une de ces postures :

1. **LLM interdit** (Meta, Google onsite) : live coding sans assistance, tu dois vraiment savoir coder
2. **LLM autorise, evalue sur le thinking** (Anthropic, OpenAI) : tu as acces a Claude/ChatGPT, l'entretien evalue ta capacite a critiquer, debugger, et diriger l'IA — pas a copier-coller
3. **Take-home + review en live** : tu codes chez toi (LLM ok), tu defends tes choix en live

**Consequences pour ta preparation** :
- **Entraine-toi SANS IA pour les mocks** : tu dois avoir les patterns en memoire musculaire, sinon tu es handicape si le LLM est interdit
- **Apprends a critiquer du code LLM** : entraine-toi a reviewer du code genere par Claude/GPT, a detecter les bugs subtils (off-by-one, complexite cachee, edge cases manques)
- **Soigne ton thinking out loud** : en 2026, expliquer POURQUOI tu choisis cette approche vaut plus que la coder
- **Prepare ton "pourquoi pas l'IA"** : si on te demande "pourquoi pas demander a Claude ?", ta reponse doit etre precise (contexte manquant, verification, ownership)

## 8ter. System design lite pour senior (IC4+)

Pour les entretiens senior, on attend de toi que tu passes d'un algo LeetCode a une discussion systeme en 2-3 minutes. Entraine-toi aux transitions suivantes :

**Exemple** : two-sum resolu en 5 min → "Et si l'array fait 10^9 elements ?"
- Reponse : sharding par hash(value), reduce en parallele, puis merge
- Trade-off : memoire distribuee vs. latence reseau, consistance eventuelle ok car read-only

**Autres questions typiques** :
- "Ton algo est O(n log n), on a besoin de O(n) streaming" → count-min sketch, HyperLogLog, reservoir sampling
- "Tu as 100 QPS, scale a 100k QPS" → cache (Redis), batch, async, read replica
- "Ton heap tient en RAM, mais la data est 10x plus grosse" → external sort, top-k approxime, quickselect sur sample

**Patterns a connaitre pour le "design lite"** :
- Streaming algorithms (Bloom filter, count-min, HLL, reservoir)
- Sharding (hash, range, consistent)
- Approximations (top-k avec erreur borne)
- External algorithms (merge sort externe, B-trees)
- Async + backpressure (quand le producer depasse le consumer)

Pas besoin de maitriser la distribuee, mais savoir "comment tu pivotes" est ce qui distingue IC3 de IC4+.

---

## 9. Flash Cards — Revision espacee

**Q1** : Quelle est la premiere chose a faire quand on recoit un probleme d'entretien ?
> **R1** : **Clarifier l'enonce**. Jamais coder en premier. Pose des questions sur les types, les bornes, les edge cases, les contraintes de complexite. Cela prend 2-3 minutes mais evite de partir dans la mauvaise direction.

**Q2** : Pourquoi enoncer une brute force avant la solution optimale ?
> **R2** : Pour montrer que tu comprends le probleme, et pour avoir une baseline commune avec l'interviewer. Meme si tu connais la solution optimale, le fait de verbaliser la brute force et ses limites est attendu et valorise.

**Q3** : Que faire si tu es completement bloque pendant 5 minutes ?
> **R3** : Dis-le clairement. "Je suis coince sur X, est-ce que vous pouvez me donner un hint ?" C'est BEAUCOUP mieux que de patauger en silence. L'interviewer attend de voir comment tu reagis aux difficultes, pas si tu es une machine infaillible.

**Q4** : Quels sont les signes qu'un interviewer te donne un hint implicite ?
> **R4** : "Peux-tu faire mieux ?", "Et si la taille etait 10^6 ?", "Que se passe-t-il si l'input est deja trie ?", "Est-ce que cette operation est O(1) ?". Chaque question de ce type est une piste directe — saisis-la immediatement.

**Q5** : Comment gerer le temps pour ne pas finir en panique ?
> **R5** : 5 min clarification, 2-3 min brute force + optimisation, 15-20 min code, 3-5 min test + complexite. Si a 25 min tu n'as pas commence a coder, passe en mode "solution suffisante" et code ce que tu as. Un code correct qui resout 70% > un code parfait non termine.

---

## Resume — Key Takeaways

1. **Un entretien tech est une performance** — pratique-le en conditions reelles
2. **Les 6 etapes** : clarifier → exemples → brute force → optimiser → coder → tester
3. **Parle en permanence**, pose des questions rhetoriques, reagis aux hints
4. **Le time management est critique** : 45 min, c'est court
5. **Les pieges** : commencer a coder trop tot, coder en silence, s'entete, ignorer les hints
6. **Apres 2 semaines de theorie et d'exos**, seul l'entrainement en conditions reelles te fera passer le cap
