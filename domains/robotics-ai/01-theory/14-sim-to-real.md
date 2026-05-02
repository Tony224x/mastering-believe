# J14 — Sim-to-real : domain randomization

> **Objectif du jour** : comprendre pourquoi une policy entrainee parfaitement en simulation echoue souvent sur le robot reel, et comment la domain randomization transforme la simulation en un generateur d'environnements assez varies pour que le reel devienne "juste un cas de plus".

---

## 1. Le probleme, en concret

> Un pendule simule dans MuJoCo. On entraine PPO avec `mass = 1.0 kg`, `length = 1.0 m`, friction articulaire = 0. Apres 200k steps, la policy stabilise le pendule en 2 secondes, success rate = 100%.
>
> On copie le checkpoint sur le pendule physique de l'atelier. Mesure du robot reel : `mass = 1.2 kg` (le moteur pese plus que dans le CAD), `length = 0.97 m` (tolerance fabrication), friction articulaire reelle = 0.05 Nm/(rad/s). On lance la policy. **Le pendule oscille pendant 30 secondes puis tombe.** Success rate = 0%.
>
> Que faire ? Re-collecter des donnees reelles et re-entrainer ? Trop cher. Ajouter de la robustesse en sim ? **Oui, c'est la domain randomization.**

C'est le scenario typique "sim-to-real". Il s'enonce simplement : *une policy optimisee pour exactement un environnement (le simulateur tel que tu l'as configure) ne genere pas vers un environnement legerement different (le robot reel).*

---

## 2. Le **reality gap** : d'ou vient l'ecart ?

Le **reality gap** designe l'ensemble des differences entre le simulateur et le monde reel. Il a 4 sources principales [CS285 L13] :

| Source | Exemple | Pourquoi c'est grave |
|---|---|---|
| **Parametres physiques mal connus** | masse, longueur, inertie, frottements, raideurs | Le CAD ment, le moteur pese 200g de plus que prevu |
| **Modelisation incomplete** | flexibilite des liens, jeu mecanique, vibrations, deformation des cables | MuJoCo modelise par defaut des corps rigides parfaits |
| **Capteurs imparfaits** | bruit gaussien, biais, derive, latence variable, occultations | En sim tu lis `qpos` exact, en reel tu lis un encodeur a 1000 Hz avec ±0.1° d'erreur |
| **Actuateurs imparfaits** | retard, saturation, dead zone, dynamique du moteur (PWM → couple) | Tu commandes `tau = 5 Nm`, le moteur delivre 4.7 Nm 8ms plus tard |

**Pourquoi le RL est particulierement vulnerable.** Une policy entrainee sans randomization "exploite" la sim : elle apprend des trajectoires qui dependent de timings exacts, de couplages d'oscillations precis, ou de valeurs de friction nulles. C'est de l'**overfitting a l'environnement d'entrainement**. Des qu'un parametre bouge, le comportement appris devient hors-distribution.

> **Mnemo** — *le RL apprend a "tricher" avec ton simulateur. La randomization l'oblige a apprendre la physique.*

---

## 3. Domain randomization : l'idee centrale

[Tobin et al., 2017] pose la question retournee : **et si la simulation etait suffisamment variee pour que le monde reel ressemble juste a une nouvelle randomisation ?**

Formellement, plutot que d'echantillonner les transitions sur **un seul** MDP `M*`, on tire pour chaque episode un MDP `M_xi ~ p(xi)` ou `xi` est un vecteur de parametres physiques (masse, friction, latence...). On entraine la policy `pi` pour maximiser :

```
J(pi) = E_{xi ~ p(xi)} [ E_{tau ~ pi, M_xi} [ R(tau) ] ]
```

Le reel est alors traite comme un tirage parmi tant d'autres : *si la distribution `p(xi)` est assez large pour englober le reel, la policy a deja "vu" quelque chose de proche.*

**Hypothese forte** : on assume que la moyenne des physics realistes contient le vrai. Si la friction reelle est 100x plus grande que le max de ta plage, la policy ne marchera pas. **Choisir la plage est une etape de design** (et souvent ce qui rate quand sim-to-real echoue).

---

## 4. Les deux familles : visuel et dynamics

### 4.1 Domain randomization **visuelle** (Tobin 2017)

Cible : tout ce que **la vision** percoit. Le papier original [Tobin et al., 2017] entraine un detecteur d'objets sur des images de simulation rendues avec :

- **Textures aleatoires** sur tous les objets (plus de 1000 textures procedurales par episode)
- **Couleurs** des objets distrack et de la table
- **Position et type** des distrack (cubes, cylindres, formes random)
- **Position** de la camera autour de la table
- **Position et caracteristiques** des sources de lumiere (intensite, direction, couleur)
- **Bruit** ajoute aux images rendues

Resultat : un detecteur entraine **uniquement en simulation** (jamais d'image reelle) localise des objets sur le robot reel a ±1.5 cm. C'est la premiere demonstration sim-to-real visuelle qui marche en zero-shot.

### 4.2 **Dynamics** randomization (Peng 2017, OpenAI 2019)

Cible : la **physique** du systeme. On randomise a chaque episode (ou chaque rollout) :

- **Masses** des corps : `m ~ Uniform([0.7 m_nom, 1.3 m_nom])`
- **Frottements** Coulomb et visqueux : multipliers ~ U([0.5, 2.0])
- **Inertie** des liens : leger jitter (typiquement ±10%)
- **Latence** observation→action : `delay ~ U([0, 30 ms])` (la latence reelle d'un control loop)
- **Bruit capteur** : `qpos_obs = qpos_true + N(0, sigma)`, `sigma ~ U([0, 0.05 rad])`
- **Bruit actionneur** : `tau_applied = tau_command * (1 + eps)`, `eps ~ N(0, 0.1)`
- **Gravite** (rare mais utile pour humanoids transferes) : `g ~ U([9.5, 10.1])`
- **Dead zone** moteur : action plus petite qu'un seuil → 0

C'est cette famille qu'on utilise dans le code du jour, parce que **le dynamics gap est le tueur silencieux** : un pendule reel a toujours plus de friction qu'un pendule sim.

---

## 5. Comment ca marche concretement (la boucle d'entrainement)

```
Pour chaque episode:
  1. Tirer xi ~ p(xi)             # nouvelles masses, frictions, latences...
  2. Reset de l'env avec xi        # MuJoCo: modifier model.body_mass, model.dof_damping
  3. Rollout policy pi sur cet env # collecter (s, a, r, s')
  4. Stocker dans buffer
  5. PPO update sur buffer melange
```

Trois detail importants :

1. **Re-init du simulateur a chaque episode**. Sinon la policy voit toujours la meme physique pendant la trajectoire, et le gradient est biaise.
2. **Distribution suffisamment large**. Une regle empirique [CS285 L13] : ta plage doit deborder de ~30% de chaque cote du nominal estime, parce que ton estimation est probablement biaisee.
3. **PPO comme baseline standard** [Schulman 2017]. PPO supporte naturellement la non-stationnarite induite par le changement de xi a chaque episode (c'est on-policy, donc le ratio clipping reste valide). SAC marche aussi mais demande plus de tuning.

---

## 6. Variantes au-dela de la randomization fixe

### 6.1 System Identification (SysID)

Plutot que de randomiser uniformement, on **mesure le robot reel** sur quelques trajectoires (typiquement 2-5 minutes), on optimise les parametres physiques de la sim pour reproduire ces trajectoires (least-squares ou MLE), puis on randomise autour des valeurs identifiees. C'est moins robuste mais plus efficient en sample.

### 6.2 **Adaptive** policies (RMA, OpenAI Solving Rubik's Cube)

La policy prend en entree, en plus de l'observation, un **embedding** des parametres physiques courants (issu d'un encodeur entraine a regresser xi depuis l'historique observation/action recent). En reel, l'encodeur regresse implicitement xi. C'est ce que fait OpenAI sur la main shadow + Rubik's Cube (2019) : la policy reagit en quelques pas a une variation de friction.

### 6.3 Automatic Domain Randomization (ADR, OpenAI 2019)

Au lieu de fixer la plage `p(xi)` a la main, on l'**elargit automatiquement** : si la policy reussit consistamment sur la plage actuelle, on l'agrandit ; si elle echoue, on la retrecit. Le **curriculum** est appris.

---

## 7. Pourquoi simu seul ne suffit jamais (mais pour ce cours, on reste full sim)

Les approches modernes sim-to-real **mixent toujours** un peu de donnees reelles. Les VLAs (OpenVLA, π0, GR00T) utilisent des dataset de teleoperation reels parce que :

- La randomization couvre les parametres **qu'on a pense a randomiser**. Tout phenomene non modelise (cables qui s'enchevetrent, table qui glisse) reste un trou.
- Le temps de calcul augmente fortement avec la largeur de la plage. Une policy "robuste a tout" est plus lente a converger qu'une policy "specialiste".
- Le visuel sim-to-real a des limites : meme avec textures random, il y a des regularites de simulation (ombres parfaites, pas de motion blur) que la policy peut latch on.

**Pour ce cours, on reste full simu.** L'objectif du jour est de demontrer la **mecanique** : entrainer en sim "facile", tester en sim "differente" (qui simule le reel), et montrer que la randomization ferme le gap. Le passage au reel reel demanderait du materiel.

---

## 8. Conclusion en 4 phrases

1. Une policy RL entrainee sur **un** simulateur overfitte a sa physique exacte ; le reality gap (parametres, modelisation, capteurs, actuateurs) la fait crasher en deploiement.
2. La **domain randomization** [Tobin et al., 2017] traite le reel comme un echantillon parmi une distribution riche de simulations, et oblige la policy a apprendre une politique robuste plutot que specialiste.
3. Deux familles : **visuelle** (textures, lumiere, camera) pour la perception ; **dynamics** (masses, frictions, latence, bruit) pour le controle. Les deux se cumulent.
4. PPO [Schulman 2017] est la baseline standard ; SysID, policies adaptatives (RMA) et ADR sont des extensions qui reduisent le cout d'une plage trop large.

---

## Flash cards (spaced repetition)

**Q1.** Quelles sont les 4 sources principales du reality gap ?
**R1.** (i) parametres physiques mal connus (masse, friction, inertie), (ii) modelisation incomplete (flexibilite, jeu, vibrations), (iii) capteurs imparfaits (bruit, biais, latence), (iv) actuateurs imparfaits (retard, saturation, dead zone).

**Q2.** Citation : qui a introduit la domain randomization visuelle et en quelle annee ?
**R2.** Tobin, Fong, Ray, Schneider, Zaremba, Abbeel — 2017 (arxiv 1703.06907). Ils entrainent un detecteur d'objet **uniquement en sim** avec textures/lumieres/cameras randomisees, et il marche zero-shot sur le robot reel.

**Q3.** Pourquoi PPO est-il pratique pour la dynamics randomization ?
**R3.** PPO est on-policy avec clipping du ratio de probabilites ; il tolere la non-stationnarite introduite par le changement de physique a chaque episode sans casser la stabilite du gradient.

**Q4.** Citer 4 parametres typiques de dynamics randomization sur un pendule.
**R4.** masse, longueur, friction articulaire (damping), bruit observation. On peut ajouter latence, gain actionneur, gravite.

**Q5.** Pourquoi la randomization seule est-elle souvent insuffisante en pratique ?
**R5.** Elle ne couvre que les parametres qu'on a pense a randomiser ; tout phenomene non modelise (cables, glissement de la table, deformations) reste un trou, et la regression sur xi explose le temps d'entrainement.

---

## Sources

- [Tobin et al., 2017] — *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World* — arxiv 1703.06907.
- [Schulman et al., 2017] — *Proximal Policy Optimization Algorithms* — arxiv 1707.06347. Baseline standard du jour.
- [CS285 L13] — Berkeley CS285 Deep RL (Sergey Levine, 2023), Lecture 13 sur sim-to-real, dynamics randomization, RMA, ADR.
