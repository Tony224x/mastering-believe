## Verdict

Fondation solide et responsable, mais je ne publierais pas ce domaine tel quel. Le contenu est bien séquencé, Alex est un bon persona et les garde-fous sont clairs. Les bloqueurs publics sont : moitié des modules sans visuel d’ouverture, jargon trop précoce, accessibilité éditoriale insuffisante et exercices sans feedback immédiat.

## Top 8 findings

- **[P1] · Le contrat “visual-first” n’est pas tenu ·** [`README.md`](/Users/anthonyvb/GitRepo/mastering-believe/domains/vie/ia-quotidien/README.md:81), [`assets/README.md`](/Users/anthonyvb/GitRepo/mastering-believe/domains/vie/ia-quotidien/assets/README.md:7), [`01-theory/04-partenaire-reflexion.md`](/Users/anthonyvb/GitRepo/mastering-believe/domains/vie/ia-quotidien/01-theory/04-partenaire-reflexion.md:1) à [`13-capstone-brouillon.md`](/Users/anthonyvb/GitRepo/mastering-believe/domains/vie/ia-quotidien/01-theory/13-capstone-brouillon.md:1) · **Pourquoi :** seulement 7 modules sur 14 commencent par un visuel. J4, J5, J7, J8, J11, J12 et J13 sont text-first. **Fix :** un visuel pédagogique obligatoire par module, placé avant la scène et accompagné d’un résumé textuel.

- **[P1] · Le français et l’accessibilité sont sous le seuil public ·** [`README.md`](/Users/anthonyvb/GitRepo/mastering-believe/domains/vie/ia-quotidien/README.md:10), [`01-ia-sans-panique.md`](/Users/anthonyvb/GitRepo/mastering-believe/domains/vie/ia-quotidien/01-theory/01-ia-sans-panique.md:26), [`02-prompts-qui-marchent.md`](/Users/anthonyvb/GitRepo/mastering-believe/domains/vie/ia-quotidien/01-theory/02-prompts-qui-marchent.md:34) · **Pourquoi :** accents absents presque partout, typo `confance`, anglicismes (`Key takeaway`, `few-shot`, `outline`), et les SVG ne contiennent ni `<title>` ni `<desc>`. **Fix :** passe éditoriale complète, remplacer par exemple :
  
  > **À retenir :** une réponse qui paraît sûre n’est pas forcément vraie.
  
  Ajouter une description textuelle après chaque visuel : “Le modèle peut reformuler et structurer, mais les chiffres et les sources doivent être vérifiés.”

- **[P1] · J1 et J2 chargent la mémoire avant de créer de la confiance ·** [`PLAN.md`](/Users/anthonyvb/GitRepo/mastering-believe/domains/vie/ia-quotidien/PLAN.md:7), [`01-ia-sans-panique.md`](/Users/anthonyvb/GitRepo/mastering-believe/domains/vie/ia-quotidien/01-theory/01-ia-sans-panique.md:30), [`02-prompts-qui-marchent.md`](/Users/anthonyvb/GitRepo/mastering-believe/domains/vie/ia-quotidien/01-theory/02-prompts-qui-marchent.md:38) · **Pourquoi :** LLM, token, “stochastic parrot”, System Card, RCCFC, few-shot, JSON et FR-CA arrivent très tôt pour un public débutant. **Fix :** commencer par trois phrases :
  
  > ChatGPT propose du texte à partir de ta demande.  
  > Il peut écrire quelque chose de faux avec assurance.  
  > Ton réflexe : demander, vérifier, décider.
  
  Garder LLM et RCCFC dans une section “Pour aller plus loin”.

- **[P1] · Les exercices vérifient une production, pas une compétence ·** [`03-exercises/01-easy/01-ia-sans-panique.md`](/Users/anthonyvb/GitRepo/mastering-believe/domains/vie/ia-quotidien/03-exercises/01-easy/01-ia-sans-panique.md:6), [`03-exercises/01-easy/06-excel-bases-ia.md`](/Users/anthonyvb/GitRepo/mastering-believe/domains/vie/ia-quotidien/03-exercises/01-easy/06-excel-bases-ia.md:6) · **Pourquoi :** pas de données de départ standardisées, pas d’indice, pas de réponse attendue ni de boucle “si ton résultat diffère”. **Fix :** fournir un petit jeu de données, une réponse modèle et un indice progressif. Exemple Excel :
  
  > `=SOMME.SI(D2:D7;"entrée";C2:C7)`  
  > Résultat attendu : entrées `600`, sorties `150`, solde `450`.

- **[P1] · Le premier exercice dépend d’une réponse web réelle et potentiellement instable ·** [`03-exercises/01-easy/01-ia-sans-panique.md`](/Users/anthonyvb/GitRepo/mastering-believe/domains/vie/ia-quotidien/03-exercises/01-easy/01-ia-sans-panique.md:7) · **Pourquoi :** l’apprenant doit demander trois statistiques 2025 et juger des sources qu’il ne sait pas encore évaluer. Une source ouvrable n’est pas forcément une source fiable. **Fix :** fournir d’abord une réponse ChatGPT fictive contenant trois affirmations, une source inventée et une source réelle. Demander : auteur, date, méthode, lien direct, décision “utiliser / vérifier / jeter”. Garder la version live en bonus.

- **[P2] · Le capstone crée une forte chute de charge ·** [`README.md`](/Users/anthonyvb/GitRepo/mastering-believe/domains/vie/ia-quotidien/README.md:46), [`PLAN.md`](/Users/anthonyvb/GitRepo/mastering-believe/domains/vie/ia-quotidien/PLAN.md:84), [`14-capstone-deck-hec.md`](/Users/anthonyvb/GitRepo/mastering-believe/domains/vie/ia-quotidien/domains/vie/ia-quotidien/01-theory/14-capstone-deck-hec.md:21) · **Pourquoi :** en 60–90 minutes, il faut finaliser 8–12 slides, vérifier les faits, produire les notes, le journal IA et répéter 6–8 minutes. Le nombre de slides annotées varie aussi entre 3 et 4. **Fix :** contrat unique :
  
  - Minimum : 8 slides, notes sur 3 slides, journal de 5 puces.
  - Bonus : 10–12 slides, notes sur 4 slides, Excel en annexe.

- **[P2] · La sécurité est une interdiction, pas encore un geste appris ·** [`README.md`](/Users/anthonyvb/GitRepo/mastering-believe/domains/vie/ia-quotidien/README.md:66), [`06-excel-bases-ia.md`](/Users/anthonyvb/GitRepo/mastering-believe/domains/vie/ia-quotidien/01-theory/06-excel-bases-ia.md:78), [`meta.toml`](/Users/anthonyvb/GitRepo/mastering-believe/domains/vie/ia-quotidien/meta.toml:14) · **Pourquoi :** “ne colle pas de données réelles” est clair, mais l’apprenant ne voit pas comment anonymiser. **Fix :** ajouter une carte vert/orange/rouge :
  
  > Vert : exemple fictif.  
  > Orange : anonymiser noms, dates et montants.  
  > Rouge : santé, RH, salaires, comptes, données client — ne pas envoyer.
  
  Montrer un avant/après : `Marie Dupont, salaire 2 450 €` → `Employée A, montant supprimé`.

- **[P2] · RCCFC est trop rigide et la locale Excel reste ambiguë ·** [`02-prompts-qui-marchent.md`](/Users/anthonyvb/GitRepo/mastering-believe/domains/vie/ia-quotidien/01-theory/02-prompts-qui-marchent.md:42), [`06-excel-bases-ia.md`](/Users/anthonyvb/GitRepo/mastering-believe/domains/vie/ia-quotidien/01-theory/06-excel-bases-ia.md:43) · **Pourquoi :** demander de mémoriser cinq lettres et choisir entre FR-CA, FR et EN ajoute une décision inutile. **Fix :** commencer par trois blocs :
  
  > **Contexte :** mon tableau contient Date, Montant et Type.  
  > **Demande :** calcule le total des entrées.  
  > **Résultat attendu :** donne la formule et un test numérique.
  
  Présenter RCCFC ensuite comme une version avancée.

## Visual-first redesign kit

### Assets à conserver et compléter

Sous `assets/` :

- `01-llm-vs-knowledge.svg` — garder, ajouter un résumé textuel.
- `02-rccfc-prompt.svg` — simplifier en “Contexte → Demande → Résultat”.
- `03-vair-checklist.svg` — ajouter une décision finale en trois couleurs.
- `06-excel-flow.svg` — garder.
- `10-pitch-story.svg` — garder.
- `04-socratique.svg` — une question → une réponse → synthèse.
- `05-avant-apres-texte.svg` — notes brutes → plan → réécriture humaine.
- `07-formule-expliquee.svg` — formule colorée morceau par morceau.
- `08-nettoyage-donnees.svg` — données sales → données propres → graphique.
- `09-classeur-onglets.svg` — Transactions → Résumé → Scénarios.
- `11-slide-avant-apres.svg` — mur de texte → slide lisible.
- `12-chrono-oral.svg` — ouverture, détails, transition.
- `13-deck-8-slides.svg` — carte minimale du capstone.
- `14-check-final.svg` — vérifier, répéter, livrer.

Chaque SVG devrait contenir `<title>`, `<desc>` et un équivalent texte dans le module.

### Template de module

```md
# Jx — Titre concret

> 20–45 min · Mission du jour

![Description complète du visuel](../assets/xx-visuel.svg)

> En une phrase : ce que montre le visuel.

## La mission

Une action faisable en 10–15 minutes.

## Les 3 gestes

1. Regarder
2. Essayer
3. Vérifier

## Si ça bloque

Un indice, puis un exemple corrigé.

## Preuve de réussite

Une capture, une phrase ou un résultat numérique.

## À retenir

Une seule phrase mémorisable.
```

### Carte de mission

```md
### Mission du jour — 12 min

**But :** obtenir une formule qui donne le bon total.

**À faire :**
1. Crée le tableau fictif.
2. Demande la formule.
3. Compare avec le calcul manuel.

**Indice :** cherche `SOMME.SI`.

**Réussite :** les deux totaux sont identiques.

**Bonus :** explique la formule avec tes propres mots.
```

Ajouter encouragements et feedback, sans classement, streak obligatoire, vies limitées ni message culpabilisant.

## Quick wins (<2h)

- Ajouter accents, corriger les typos et remplacer `Key takeaway` par `À retenir`.
- Ajouter un résumé accessible sous les 5 SVG existants.
- Ajouter `<title>` et `<desc>` aux SVG.
- Insérer un bloc “Mission / Indice / Réussite” dans J1, J2, J6 et J14.
- Fournir le tableau Excel fixe et le résultat attendu.
- Ajouter une réponse ChatGPT fictive pour l’exercice J1.
- Unifier le capstone en “Minimum / Bonus”.

## What NOT to change

- Garder Alex comme persona fictif.
- Garder l’interdiction des données réelles, la vérification et le journal IA.
- Garder le pitch PowerPoint comme objectif principal et Excel comme projet secondaire.
- Garder les SVG éditables et les exemples concrets Excel/PowerPoint.
- Ne pas ajouter de classement, de compétition ou de streak culpabilisant.
