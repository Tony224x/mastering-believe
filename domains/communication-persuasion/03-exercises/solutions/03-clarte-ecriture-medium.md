# Solutions (medium) — Module 03 : Clarte & ecriture qui se comprend

> Corriges modeles. Toute reecriture qui respecte acteur/action, concision, cohesion et honnetete est valide.

---

## Exercice 1 — Cohesion (information connue -> information nouvelle)

**Texte original** (phrases qui repartent de zero) :
> "Notre nouvelle politique de remboursement entre en vigueur le 1er septembre. Les clients devront fournir une preuve d'achat. Un delai de 30 jours est la regle desormais. Le service apres-vente traitera les demandes. Une reponse sera donnee sous 48 heures."

**Version avec cohesion** :
> "**Notre nouvelle politique de remboursement** entre en vigueur le 1er septembre. **Elle** demande aux clients de fournir une preuve d'achat dans un delai de 30 jours. **Ces demandes** seront traitees par le service apres-vente, **qui** repondra sous 48 heures."

**Sujets-connus en ouverture** :
- Phrase 1 : "Notre nouvelle politique" (introduit le theme).
- Phrase 2 : "Elle" (= la politique, info connue) -> info nouvelle : preuve d'achat + 30 jours.
- Phrase 3 : "Ces demandes" (= les demandes de remboursement, deja en tete du lecteur) -> info nouvelle : service apres-vente.
- Phrase 4 (relative) : "qui" (= le service) -> info nouvelle : delai de reponse 48h.

> Chaque phrase reprend le fil de la precedente : le lecteur n'a jamais a relire. On est passe de 5 phrases sautillantes a 3 phrases liees, sans perte d'info.

---

## Exercice 2 — Message ambigu -> clair ET honnete

**Email original** : "reevaluation des priorites strategiques... repositionnement... ajustements organisationnels... accompagner les parties prenantes..."

### 1. Decodage en une phrase

> "Deux projets sont arretes et deux postes vont etre supprimes ce mois-ci ; un accompagnement est propose."

### 2. Diagnostic de l'opacite ethique

- **Euphemismes** : "repositionnement" (= arret), "ajustements organisationnels" (= suppressions de postes), "periode de transition" (= licenciements).
- **Passif sans acteur** : "feront l'objet d'un repositionnement", "il est possible que des ajustements interviennent" — *personne* n'assume la decision.
- **Abstractions** : "rationalisation des ressources", "priorites strategiques" — aucun contenu concret.
- **Manque de respect** : le flou ici protege l'emetteur (il n'assume pas), pas le lecteur (qui ne sait pas si SON poste est concerne et reste dans l'angoisse). Le lecteur perd son temps et son autonomie : il ne peut pas agir sur une information qu'on lui cache.

### 3. Version claire ET honnete

> "Bonjour a toutes et tous,
>
> Je dois vous annoncer une decision difficile. La direction a decide d'arreter deux projets (X et Y) et, dans ce cadre, deux postes de l'equipe seront supprimes d'ici la fin du mois.
>
> Les personnes directement concernees seront recues individuellement cette semaine — elles ne l'apprendront pas par cet email. Un accompagnement (reclassement, soutien RH, ...) leur sera propose.
>
> Je sais que cette nouvelle est preoccupante. Je suis disponible des aujourd'hui pour repondre a vos questions, en groupe jeudi a 11h ou en entretien individuel."

> Direct, l'acteur est assume ("la direction a decide", "je suis disponible"), aucune info enterree — mais humain (entretiens individuels, disponibilite). Clair n'est pas brutal.

---

## Exercice 3 — Adapter a deux lecteurs

**Contenu de depart** : migration, script ETL, fenetre de maintenance 4h, SSO indisponible, rollback si seuil d'erreur depasse.

### Version A — equipe technique

> "Migration vers le nouveau systeme : bascule des donnees par script ETL le samedi 12, de 2h a 6h (fenetre de maintenance de 4h). Pendant ce creneau, le SSO sera indisponible — aucune connexion possible. [Nom] pilote la bascule ; [Nom] surveille le taux d'erreur. Plan B : rollback automatique declenche si le taux d'erreur depasse 2 %. Point de validation a 6h avant reouverture."

*(Garde le vocabulaire technique mais ajoute quand / qui / plan B chiffres.)*

### Version B — utilisateurs finaux

> "**Le systeme sera inaccessible samedi 12 de 2h a 6h du matin.** Pendant ce creneau, vous ne pourrez pas vous connecter — c'est une mise a jour planifiee. Tout sera revenu a la normale a 6h. Aucune action de votre part n'est necessaire ; pensez juste a sauvegarder votre travail avant samedi soir."

*(Pyramide inversee : l'info actionnable d'abord. Zero jargon.)*

### Ce qui a ete retire pour la version B et pourquoi

J'ai retire "ETL", "SSO", "rollback", "seuil d'erreur" et qui pilote : c'est du **savoir d'emetteur** (le mecanisme interne) sans valeur d'action pour un utilisateur. Lui n'a besoin que de : quand c'est coupe, quoi faire. Le reste relevait de la malediction du savoir — evident pour l'expert, inutile et opaque pour le lecteur.

---

### Points de vigilance communs (medium)

- **La cohesion est invisible quand elle marche** : on ne remarque un texte fluide que par le fait qu'on n'a jamais a le relire.
- **Le flou n'est pas neutre** : un message vague pour ne pas assumer une mauvaise nouvelle protege l'emetteur et abandonne le lecteur. Clair et humain ne sont pas opposes.
- **Adapter au lecteur, pas a soi** : la question n'est jamais "qu'est-ce que je sais ?" mais "de quoi CE lecteur a besoin pour agir ?".
