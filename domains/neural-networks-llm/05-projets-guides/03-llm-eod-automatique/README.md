# Projet 03 — LLM pour EOD Review automatique

## Contexte metier

Apres un shift, l'OCC ou un planner passe 2 a 4 heures a ecrire son **EOD Review** : un document qui resume ce qu'il s'est passe, pointe les moments cles (incidents, surcharge, panne, deviation SOP), explique les ajustements a faire au prochain shift. C'est crucial pour l'amelioration continue mais lourd a produire.

Objectif : un **assistant LLM** qui prend en entree les traces structurees d'un shift (les events enregistres par le pipeline EOD) et produit une **premiere version de rapport** que l'OCC n'a plus qu'a reviser. Reduit 4h a 30 min.

Contraintes deploiement on-premise :
- **Air-gap / quasi air-gap** : pas d'API cloud autorisee chez la plupart des clients. Modele local (Llama, Mistral, ou distillation d'un gros vers petit).
- **Explicabilite** : chaque affirmation du rapport doit pointer vers les events concrets qui la supportent. RAG obligatoire.
- **Pas d'hallucination** : un EOD avec des faits inventes est pire qu'un EOD lent. Garde-fous a construire.

## Objectif technique

Construire un pipeline :
1. Prend en entree une liste d'events JSON (format du projet system-design 02)
2. Extrait les **moments cles** (collision, FAULT critique, surcharge dock, deviation de Routing Plan, etc.)
3. Pour chaque moment, recupere le contexte (events 30s avant/apres)
4. Appelle un LLM avec un prompt structure pour generer le paragraphe d'EOD correspondant
5. Assemble le rapport final avec citations (chaque affirmation pointe `event_id`s)

Pour le dev, utilise l'**API Anthropic** (claude-haiku-4-5) en mode "local equivalent" — dans le vrai projet on utiliserait un Llama on-prem, mais pedagogiquement le format de prompt est identique.

## Consigne

Livrables :
- `solution/extract_key_moments.py` : heuristique qui extrait les moments cles des events
- `solution/eod_prompt.py` : templates de prompt (system + user)
- `solution/generate_eod.py` : pipeline complet events -> rapport markdown
- `solution/eval_eod.py` : checks automatiques sur la sortie (citations presentes, pas de contradiction factuelle)

## Prompt template (extrait)

```
System:
Tu es un assistant d'analyse EOD pour un superviseur OCC d'entrepot. Tu dois
produire un paragraphe d'EOD factuel, neutre, et base **exclusivement** sur
les events fournis. Chaque affirmation doit etre suivie d'une citation
au format [ev:<event_id>]. Si tu n'as pas l'information, ecris "non
documente", ne fabrique jamais.

Ton style est : phrases courtes, vocabulaire operationnel logistique, pas de
jugement de valeur.

User:
Moment cle : COLLISION inter-flotte detectee a t=0h47m12s
Unite concernee : AGV-Alpha-2 (own_fleet)

Events du contexte (30s avant / 60s apres) :
[ev:42871] t=0h46m50s MOVE Alpha-2 to zone B-12-N
[ev:42903] t=0h47m12s DETECT Alpha-2 observes Sorter-7 at B-12-NE (dist 4m)
[ev:42905] t=0h47m13s ORDER OCC to Alpha-2: hold position
[ev:42931] t=0h47m45s PICKUP Alpha-2 (preemptive) parcel PCL-3
[ev:42942] t=0h47m58s COLLISION Alpha-2 vs Sorter-7, severity 0.3
[ev:42951] t=0h48m05s FAULT Alpha-2 BATTERY_LOW minor
...

Redige le paragraphe d'EOD pour ce moment.
```

## Criteres de reussite

- Rapport genere en < 60 s pour un shift de 4h
- Chaque phrase a une citation vers des events
- Eval automatique : pour 10 shifts de test, 100% des affirmations citees existent vraiment dans les events
- Zero hallucination detectee (eval manuelle sur 5 exemples)
- L'OCC le note "utile" ou plus sur 5 shifts de test

## Garde-fous anti-hallucination

1. **Constrained prompting** — le prompt interdit explicitement d'inventer, exige les citations
2. **Post-check** — parser les citations, verifier que chaque `event_id` existe
3. **Fact extraction retour** — relire le rapport, extraire les affirmations, les confronter aux events
4. **Low temperature** — T=0.2 max pour le factuel
5. **Few-shot** — 2-3 exemples d'EOD bien formates en contexte

## Solution

Voir `solution/` pour le pipeline complet.

## Pour aller plus loin

- **Fine-tuning leger** — LoRA sur quelques centaines d'EOD humains pour ajuster le style
- **Style par client** — chaque client a son format d'EOD attendu, prompt parametre
- **Interactive** — mode chat, l'OCC questionne ("pourquoi Alpha-2 a tente le pickup avant le feu vert ?"), le modele repond avec citations
- **Distillation offline** — partir d'un gros modele (Claude Opus), generer 10k EOD de qualite, fine-tuner un petit Mistral 7B local pour deployer on-prem
