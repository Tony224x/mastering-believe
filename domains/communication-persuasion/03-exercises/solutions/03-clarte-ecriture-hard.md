# Solutions (hard) — Module 03 : Clarte & ecriture qui se comprend

> Corriges modeles. L'opacite strategique est etudiee ici pour la detecter et la corriger — jamais pour la reproduire.

---

## Exercice 1 — Document long et opaque -> note actionnable

### 1. Diagnostic

- **Nominalisations** : "non-atteinte des objectifs", "sous-dimensionnement des effectifs", "recours a une solution externalisee", "prise de decision", "mise en oeuvre" (5+).
- **Passifs sans acteur** : "il est apparu que", "il a ete evoque", "il a ete convenu que", "ait ete menee une analyse" (4).
- **Mots-coussins** : "en date du", "au niveau de" (x2), "notamment", "etant entendu que", "a caractere provisoire", "par ailleurs" (6+).
- **Phrase la plus longue** : la deuxieme (~70 mots) — illisible d'un trait.

### 2. Note reecrite (pyramide inversee, <120 mots)

> **Objet : Delais commandes non tenus — 2 causes, plan d'action**
>
> Les delais de traitement des commandes ne sont pas tenus. Deux causes identifiees a la reunion du 12 :
> - **Effectifs insuffisants** sur la periode.
> - **Bugs de l'outil de gestion des stocks.**
>
> Decisions :
> - **[Resp. ops]** met en place des mesures correctives provisoires des cette semaine.
> - **[Resp. IT]** ouvre un ticket sur l'outil de stocks (echeance : vendredi).
> - **[Directeur]** evalue l'option d'externalisation et tranche au prochain comite (date : __).
>
> Aucune decision definitive sur l'externalisation avant cette evaluation.

*(~95 mots, chaque action a un acteur et une echeance.)*

### 3. Objet d'email

> "Delais commandes : 2 causes + plan d'action"

---

## Exercice 2 — Opacite strategique (information noyee)

**Texte** : "plaisir de vous informer de l'enrichissement... actualisation de la grille tarifaire... prendra effet automatiquement..."

### 1. Information enterree

La vraie nouvelle : **le prix de l'abonnement augmente.** Elle est noyee au *milieu* du texte, sans chiffre, derriere "actualisation de la grille tarifaire". Le debut ("plaisir", "enrichissement") et la fin ("merci de votre fidelite") sont du rembourrage positif destine a adoucir et a detourner l'attention de la seule info qui compte.

### 2. Techniques d'enfouissement

- **Enrobage positif** : "plaisir de vous informer", "enrichissement continu de votre experience".
- **Euphemisme** : "actualisation de la grille tarifaire", "evolution" (= hausse).
- **Automaticite/passif** : "prendra effet automatiquement", "sera repercutee" — pas d'acteur, l'abonne n'a "rien a faire" (donc ne reagit pas).
- **Absence de chiffre** : ni le nouveau prix, ni le pourcentage, ni la date exacte.
- **Dilution** : "demarche d'amelioration globale", "environnement en constante mutation" — abstractions qui remplissent.

### 3. Evaluation CTR

- **C — Consentement** : **viole de fait.** "Prendra effet automatiquement" suggere qu'il n'y a rien a faire — l'abonne n'est pas invite a reagir ni informe de son droit de resilier.
- **T — Transparence** : **violee.** Le fait materiel central (le montant, la date) est absent. On ne peut pas consentir a ce qu'on ne connait pas.
- **R — Reciprocite** : la hausse profite a l'entreprise ; aucune contrepartie claire pour l'abonne ("amelioration globale" reste vide).

### 4. Version transparente

> **Objet : Hausse de votre abonnement a partir du 1er aout**
>
> Bonjour,
>
> Le prix de votre abonnement passera de 12 EUR a 15 EUR par mois (+3 EUR) a compter du prelevement du 1er aout. Cette hausse couvre [raison reelle].
>
> Vous avez le choix : continuer au nouveau tarif (rien a faire), ou resilier sans frais avant le 25 juillet depuis votre espace client / en repondant a cet email. Pour toute question, on est joignables au [contact]."

**Conclusion** : la clarte est la *condition du consentement reel*. On ne peut pas consentir librement a une hausse qu'on n'a pas vue. L'entreprise a le droit d'augmenter ses prix — pas de le cacher.

---

## Exercice 3 — Malediction du savoir, grandeur reelle

*(Solution-type : l'exercice porte sur le domaine du lecteur. Exemple calibre sur la cuisine pour illustrer la methode.)*

### 1. Paragraphe "entre experts"

> "Pour une emulsion stable, monte la mayo a temperature ambiante, ajoute l'huile en filet une fois l'appareil pris, et si ça tranche, redemarre sur un nouveau jaune. Le ratio acide-matiere grasse fait tout, et un coup de mixeur plongeant rattrape la plupart des ratages."

### 2. Test de paraphrase — termes opaques pour un novice

- "emulsion" : un debutant ne sait pas ce que c'est.
- "monter / l'appareil pris" : jargon de cuisine.
- "ça tranche" : incomprehensible hors contexte.
- "ratio acide-matiere grasse" : abstraction.
- "filet" : implicite (= mince filet continu).

### 3. Version pour un lecteur qui ne connait rien

> "Une mayonnaise, c'est de l'huile melangee a un jaune d'oeuf jusqu'a ce que ça devienne epais et cremeux. Le secret : verse l'huile tres lentement, presque goutte a goutte au debut, en fouettant sans arret — sinon le melange reste liquide et 'se separe'. Sors les ingredients du frigo une heure avant : le froid empeche la prise. Si ça rate, recommence avec un nouveau jaune d'oeuf et incorpore ton melange rate dedans, petit a petit."

### 4. Boucle de controle — implicites "evidents" qui ne l'etaient pas

- Je pensais "evident" que monter signifie *fouetter pour epaissir* — un novice ne le sait pas.
- Je pensais "evident" qu'il faut verser l'huile *progressivement* — c'est precisement l'erreur n1 des debutants.
- "Temperature ambiante" me paraissait trivial, mais c'est une etape *causale* (le froid fait rater) qu'il fallait expliciter.

> Ces trois implicites sont la malediction du savoir en action : ce qui est automatique pour moi est invisible — donc absent — pour le lecteur qui en a le plus besoin.

---

### Points de vigilance communs (hard)

- **L'opacite strategique se reconnait a sa structure** : info defavorable au milieu, enrobage positif aux extremites, jamais de chiffre. La detecter, c'est la moitie de s'en proteger.
- **Pas de consentement sans clarte** : cacher le fait materiel (prix, date, consequence) rend le "oui" sans valeur ethique.
- **La malediction du savoir ne se corrige pas par la volonte mais par la methode** : test de paraphrase, remplacement des abstractions par du concret, relecture "premier paragraphe comprehensible ?".
