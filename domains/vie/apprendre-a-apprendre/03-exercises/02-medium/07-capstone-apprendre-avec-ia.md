# Exercices (medium) — Module 07 (Capstone) : Apprendre avec l'IA

> **Niveau** : Intermediaire | **Temps estime** : 50-60 min
> Extensions du plan d'apprentissage personnel : on muscle le gabarit du capstone (`04-projects/README.md`) sur des points precis avant la version complete (hard).

---

## Exercice 1 — Auditer un plan d'apprentissage genere par IA

### Objectif
Evaluer de facon critique un plan produit par un LLM au regard des principes du domaine, plutot que de l'accepter tel quel.

### Consigne
Demande a un LLM : *"Fais-moi un plan pour apprendre [un sujet de ton choix] en 4 semaines."* (volontairement vague, comme le ferait un debutant).

Puis audite le plan obtenu avec une grille fondee sur le domaine :
1. **Retrieval** : le plan prevoit-il de la recuperation active, ou surtout de la lecture/visionnage passif ?
2. **Espacement** : les revisions sont-elles espacees a intervalles croissants, ou tout est-il vu une seule fois ?
3. **Interleaving** : melange-t-il les sous-themes confondables, ou traite-t-il un theme par bloc ?
4. **Indicateurs objectifs** : propose-t-il une mesure de maitrise objective, ou reste-t-il sur "comprendre" ?
5. **Neuromythes** : le plan glisse-t-il un conseil douteux (styles d'apprentissage, brain-training, "10 000 h") ?

Pour chaque point manquant, ecris la reformulation de prompt qui corrige.

### Criteres de reussite
- [ ] Les 5 points sont audites explicitement (present/absent)
- [ ] Chaque manque donne lieu a une reformulation de prompt concrete
- [ ] Le point 5 verifie activement l'absence de neuromythe — et si le plan en contient un, tu le signales et le corriges
- [ ] L'audit conclut par les 2 corrections les plus importantes a apporter au plan

---

## Exercice 2 — Definir des indicateurs de maitrise objectifs

### Objectif
Renforcer la section "indicateurs de maitrise" du gabarit capstone : remplacer les objectifs subjectifs par des tests externes verifiables.

### Consigne
Pour un sujet que tu veux apprendre, transforme 3 objectifs vagues en indicateurs objectifs.

Pour chacun :
1. Pars d'un objectif subjectif typique ("comprendre X", "etre a l'aise avec Y").
2. Reecris-le en **indicateur observable et mesurable** : un test, une production, une evaluation externe, avec un seuil et idealement un delai.
3. Classe-le en niveau : minimum viable / cible / stretch.
4. Indique *comment* tu obtiendras la mesure (qui evalue, avec quoi).

Exemple de transformation :
- Vague : "comprendre les fonctions en Python."
- Objectif : "Resoudre 8/10 exercices 'fonctions' d'un jeu donne, en autonomie, en moins de 30 min — cible."

### Criteres de reussite
- [ ] 3 objectifs vagues transformes en indicateurs observables
- [ ] Chaque indicateur a un seuil (et si pertinent un delai)
- [ ] Les 3 niveaux (minimum viable / cible / stretch) sont distingues
- [ ] Le mode d'obtention de la mesure est precise (pas "je verrai bien")
- [ ] Aucun indicateur ne repose sur un ressenti subjectif (JOL)

---

## Exercice 3 — Cadrer l'usage de l'IA pour eviter la dependance passive

### Objectif
Definir des regles d'usage du LLM qui maximisent l'apprentissage (effort de recuperation) et evitent le piege de la consommation passive d'explications.

### Consigne
Le risque #1 d'apprendre avec un LLM : se faire expliquer, hocher la tete, et ne rien retenir (illusion de fluidite, version augmentee).

1. Pour chacun des 3 roles du LLM (tuteur socratique, generateur de retrieval, partenaire Feynman), ecris **un prompt type** et **le moment** ou tu l'utilises.
2. Ecris 3 **garde-fous** personnels (ex. "je ne lis pas d'explication passive sans faire un blank-page juste apres", "je verifie les faits critiques dans une source primaire car le LLM peut confabuler").
3. Explique pourquoi le mode passif (demander une explication et lire) reproduit l'illusion de competence du Module 01, mais en pire (le texte est encore plus fluide).
4. Rappelle la nuance du 2-sigma de Bloom : pourquoi un LLM utilise passivement ne reproduit PAS l'effet du tutorat avec mastery learning.

### Criteres de reussite
- [ ] Les 3 roles ont chacun un prompt type et un moment d'usage
- [ ] 3 garde-fous personnels concrets, dont la verification des faits (confabulation possible du LLM)
- [ ] Le lien mode passif / illusion de competence (Module 01) est explicite
- [ ] La nuance de Bloom (1984) est correctement restituee : l'effet tient au mastery learning + feedback, pas a l'ecoute passive
- [ ] Aucune affirmation ne presente le LLM comme un raccourci magique

---

*Solutions disponibles dans `03-exercises/solutions/07-capstone-apprendre-avec-ia-medium.md`*
