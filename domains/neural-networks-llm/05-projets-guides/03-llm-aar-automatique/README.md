# Projet 03 — LLM pour AAR automatique

## Contexte metier

Apres un exercice, le formateur passe 2 a 4 heures a ecrire son **AAR** : un document qui resume ce qu'il s'est passe, pointe les moments cles, explique les lecons apprises. C'est crucial pedagogiquement mais lourd a produire.

Objectif : un **assistant LLM** qui prend en entree les traces structurees d'un exercice (les events enregistres par le pipeline AAR) et produit une **premiere version d'AAR** que le formateur n'a plus qu'a reviser. Reduit 4h a 30 min.

Contraintes defense :
- **Air-gap** : pas d'API OpenAI/Anthropic/Google. Modele local (Llama, Mistral, ou distillation d'un gros vers petit).
- **Explicabilite** : chaque affirmation du rapport doit pointer vers les events concrets qui la supportent. RAG obligatoire.
- **Pas d'hallucination** : un AAR avec des faits inventes est pire qu'un AAR lent. Garde-fous a construire.

## Objectif technique

Construire un pipeline :
1. Prend en entree une liste d'events JSON (format du projet system-design 02)
2. Extrait les **moments cles** (detection contact, engagement majeur, neutralisation, etc.)
3. Pour chaque moment, recupere le contexte (events 30s avant/apres)
4. Appelle un LLM avec un prompt structure pour generer le paragraphe d'AAR correspondant
5. Assemble le rapport final avec citations (chaque affirmation pointe `event_id`s)

Pour le dev, utilise l'**API Anthropic** (claude-haiku-4-5) en mode "local equivalent" — dans le vrai projet on utiliserait un Llama on-prem, mais pedagogiquement le format de prompt est identique.

## Consigne

Livrables :
- `solution/extract_key_moments.py` : heuristique qui extrait les moments cles des events
- `solution/aar_prompt.py` : templates de prompt (system + user)
- `solution/generate_aar.py` : pipeline complet events -> rapport markdown
- `solution/eval_aar.py` : checks automatiques sur la sortie (citations presentes, pas de contradiction factuelle)

## Prompt template (extrait)

```
System:
Tu es un assistant d'analyse tactique pour un formateur militaire. Tu dois
produire un paragraphe d'AAR factuel, neutre, et base **exclusivement** sur
les events fournis. Chaque affirmation doit etre suivie d'une citation
au format [ev:<event_id>]. Si tu n'as pas l'information, ecris "non
documente", ne fabrique jamais.

Ton style est : phrases courtes, vocabulaire militaire standard, pas de
jugement de valeur.

User:
Moment cle : contact ennemi detecte a t=0h47m12s
Unite concernee : peloton Alpha-2 (BLUFOR)

Events du contexte (30s avant / 60s apres) :
[ev:42871] t=0h46m50s MOVE Alpha-2 to grid 304-512
[ev:42903] t=0h47m12s DETECT Alpha-2 observes OPFOR squad at 304-518 (dist 600m)
[ev:42905] t=0h47m13s ORDER Bravo-HQ to Alpha-2: take cover and report
[ev:42931] t=0h47m45s FIRE Alpha-2 engages OPFOR (preemptive)
[ev:42942] t=0h47m58s DAMAGE OPFOR suffers 2 neutralized
[ev:42951] t=0h48m05s DAMAGE Alpha-2 suffers 1 wounded
...

Redige le paragraphe d'AAR pour ce moment.
```

## Criteres de reussite

- Rapport genere en < 60 s pour un exercice de 4h
- Chaque phrase a une citation vers des events
- Eval automatique : pour 10 exercices de test, 100% des affirmations citees existent vraiment dans les events
- Zero hallucination detectee (eval manuelle sur 5 exemples)
- Le formateur le note "utile" ou plus sur 5 exercices de test

## Garde-fous anti-hallucination

1. **Constrained prompting** — le prompt interdit explicitement d'inventer, exige les citations
2. **Post-check** — parser les citations, verifier que chaque `event_id` existe
3. **Fact extraction retour** — relire le rapport, extraire les affirmations, les confronter aux events
4. **Low temperature** — T=0.2 max pour le factuel
5. **Few-shot** — 2-3 exemples d'AAR bien formates en contexte

## Solution

Voir `solution/` pour le pipeline complet.

## Pour aller plus loin

- **Fine-tuning leger** — LoRA sur quelques centaines d'AAR humains pour ajuster le style
- **Style par client** — chaque armee a son format d'AAR, prompt parametre
- **Interactive** — mode chat, le formateur questionne ("pourquoi Alpha-2 a pris contact avant le signal ?"), le modele repond avec citations
- **Distillation offline** — partir d'un gros modele (Claude Opus), generer 10k AAR de qualite, fine-tuner un petit Mistral 7B local pour deployer on-prem
