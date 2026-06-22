# Solutions (medium) — Module 04 : Messages qui collent

> Corriges modeles. Rappel central : SUCCESs amplifie la forme, il ne valide pas le fond. On verifie le fond CTR avant d'optimiser la forme.

---

## Exercice 1 — Un message qui colle sur du faux

**Message** : "Mon grand-pere a fume jusqu'a 96 ans et n'a jamais ete malade... la preuve que les etudes c'est du business."

### 1. Diagnostic SUCCESs (pourquoi ça colle)

- **Simple** : "fumeur = vieux et en pleine forme", une idee nette.
- **Unexpected** : casse le schema attendu (tabac = maladie).
- **Concrete** : un grand-pere, un paquet par jour, 96 ans, un village — tout est visualisable.
- **Emotional** : affection pour le grand-pere, mefiance envers "le business".
- **Stories** : mini-recit complet.

-> Le message coche 5 dimensions. Il colle parfaitement.

### 2. Diagnostic du fond (pourquoi c'est trompeur)

- **Anecdote isolee vs donnees** : un cas ne dit rien d'une tendance statistique sur des millions de personnes.
- **Biais du survivant** : on entend le grand-pere qui a "echappe" ; on n'entend pas les fumeurs morts a 55 ans (ils ne racontent rien).
- **Faux raisonnement** : "il a vecu plus vieux que les non-fumeurs du village" est une correlation locale sans controle, pas une preuve causale.
- **"Etudes = business"** : disqualification gratuite, non etayee.

### 3. Le piege

> Le fait qu'un message colle ne dit **rien** de sa veracite. SUCCESs est un amplificateur neutre : une legende urbaine est simple, inattendue, concrete, emotionnelle et narrative — elle colle. C'est precisement l'avertissement du module : la forme ne valide jamais le fond.

### 4. Reecriture honnete ET collante

> "Mon oncle Daniel etait persuade qu'il s'en sortirait : 'mon pere a fume toute sa vie sans probleme'. A 58 ans, on lui a diagnostique un cancer du poumon. Son pere avait eu de la chance — lui non. Le tabac ne tue pas tout le monde : il joue a la roulette avec chacun. Selon les autorites de sante, fumer multiplie nettement le risque de cancer du poumon [chiffre precis a verifier sur source officielle]. La vraie question n'est pas 'est-ce que ça tue tout le monde ?' mais 'est-ce que je veux jouer ça avec ma vie ?'"

*(Collant : Story, Emotional merite, Concrete, Unexpected — "roulette" — et Credible via source officielle, sans surinterpreter.)*

---

## Exercice 2 — Story Spine adapte au canal

**Idee centrale** : documenter au fil de l'eau, pas apres coup.

### Version A — Slack/chat (<=40 mots)

> "Rappel utile : documenter une procedure prend 5 min sur le moment, et 2 jours a reconstituer 6 mois plus tard. Notez au fil de l'eau — votre futur vous dira merci."

*(Dimensions : **Simple** + **Concrete** — "5 min" vs "2 jours". Format court = peu de dimensions, bien choisies.)*

### Version B — oral 60 secondes

**Story Spine** :
- Il etait une fois une equipe qui codait vite et bien.
- Tous les jours, chacun gardait ses procedures "dans sa tete".
- Jusqu'au jour ou Marie a du reprendre un module ecrit 6 mois plus tot par quelqu'un parti depuis.
- A cause de ça, elle a passe deux jours entiers a deviner comment ça marchait.
- A cause de ça, une livraison a pris du retard.
- Jusqu'a ce que finalement on decide d'ecrire une note de 5 lignes a chaque procedure terminee.
- Et depuis lors, reprendre le code d'un collegue prend des minutes, pas des jours.

**Texte oral** :
> "La semaine derniere, Marie a passe deux jours a dechiffrer un module ecrit il y a six mois — du code que l'un de nous a ecrit. Personne ne se souvenait comment ça marchait. Ce n'est la faute de personne : on code vite, on documente 'plus tard', et 'plus tard' n'arrive jamais. Documenter une procedure au moment ou on la finit, c'est cinq minutes. La reconstituer six mois apres, c'est deux jours. Je propose une regle simple : cinq lignes de note a chaque procedure terminee. C'est une note a votre futur vous."

*(Dimensions : Simple, Concrete, Emotional — Marie —, Stories, Unexpected en creux. 5 dimensions.)*

---

## Exercice 3 — Concrete + Emotional sans manipuler

### Enonce 1 — Gaspillage cantine

- **Concrete** : "Chaque midi, la cantine jette l'equivalent de 80 repas complets a la poubelle — soit deux grands conteneurs pleins."
- **Emotional** : "Sofia, qui sert au self, dit qu'elle a du mal a vider ces bacs alors qu'elle sait que des familles du quartier comptent leurs courses."
- **Test** : si on retire l'emotion, le fait (80 repas jetes) tient seul. **OK, pas de manipulation.**

### Enonce 2 — Temps de reponse support

- **Concrete** : "Notre delai de premiere reponse est passe de 2h a 9h en trois mois."
- **Emotional** : "Un client, M. Royer, a ecrit : 'J'ai cru que vous aviez ferme.'"
- **Test** : sans l'emotion, le chiffre (2h -> 9h) tient. **OK.**

### Enonce 3 — Personnes agees isolees

- **Concrete** : "Dans notre rue, 12 personnes de plus de 75 ans vivent seules ; 4 d'entre elles n'ont eu aucune visite le mois dernier."
- **Emotional** : "Mme Lopez, 81 ans, dit que le facteur est parfois la seule personne a qui elle parle dans la semaine."
- **Test** : sans l'emotion, les chiffres tiennent. **OK** — l'emotion *illustre*, elle ne *remplace* pas.

> Si, pour l'un des enonces, on n'avait eu QUE "Mme Lopez est triste" sans aucun fait derriere, l'argument se serait effondre sans l'emotion -> signal de manipulation -> il aurait fallu rajouter le socle factuel (les 12 personnes, les 4 sans visite).

---

### Points de vigilance communs (medium)

- **Colle != vrai.** La premiere question n'est jamais "est-ce memorable ?" mais "est-ce vrai ?".
- **Le canal dicte la densite** : un chat porte 1-2 dimensions, un oral/une video en porte 4-6. Forcer 6 dimensions dans 40 mots les noie.
- **L'emotion doit etre meritee par les faits** : test systematique — "si je retire l'emotion, l'argument tient-il ?". Si non, on ajoute du fond, on ne garde pas l'emotion seule.
