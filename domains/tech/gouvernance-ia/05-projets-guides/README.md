# 05 — Projets guides (Gouvernance de l'IA)

> Voir `shared/logistics-context.md` pour le contexte metier de LogiSim / FleetSim.

Trois missions qui appliquent la gouvernance de l'IA agentique a un cas reel :
gouverner la **flotte d'agents LLM "fleet brain"** deployes chez un client LogiSim
pour piloter FleetSim. Les trois projets rejouent, sur un scenario metier, les
**trois temps du pipeline de gouvernance** vu au capstone
(`ingest -> enforce -> log -> score -> map -> report`) :

1. **Inventorier & auditer** la flotte (qui tourne, qui est responsable) ;
2. **Decider & bloquer** les actions sensibles en runtime, de maniere prouvable ;
3. **Scorer le risque, mapper la conformite et rendre un verdict** au comite.

Le projet phare est le **02 — Policy Gate runtime** : le coeur operationnel de la
gouvernance agentique (PDP/PEP + budgets + kill-switch + audit tamper-evident).

## Projets

| # | Projet | Brique de gouvernance | Difficulte |
|---|---|---|---|
| 01 | **Registry & audit de flotte** | Inventaire, 4 piliers, agents orphelins (J2/J3) | medium |
| 02 | **Policy Gate runtime OCC** ⭐ | PDP/PEP, deny>oblige>allow, kill-switch, audit chaine (J8/J9/J10/J14) | hard |
| 03 | **Rapport de gouvernance board** | Score de risque NIST, crosswalk conformite, verdict (J4/J7/J12) | medium |

## Methodologie

Pour chaque projet :
1. Lire le contexte metier (et `shared/logistics-context.md`)
2. Lire les specs (entrees/sorties, mecanismes attendus)
3. Coder la v0 a partir de la consigne
4. Confronter a la correction commentee (`solution/`)
5. Faire tourner la demo et **observer la trace** (chaque chiffre est derive d'un
   mecanisme, jamais saisi a la main)

## Pourquoi ce contexte FleetSim ?

LogiSim deploie des agents LLM "fleet brain" (cf. Agentic AI, projet 01) pour
piloter des flottes robotisees. Ces agents **agissent sur le monde reel** :
engager une flotte tierce (cout, contrat), rouvrir une zone aux pietons (securite),
exporter de la telemetrie client (confidentialite contractuelle, site souvent
air-gap). C'est exactement le profil ou la gouvernance cesse d'etre theorique :
identite, owner, permissions et audit deviennent la difference entre un incident
trace/borne et un incident subi. Les contraintes du domaine logistique —
certification ISO 9001 / SOC 2, determinisme pour reconstitution d'incident,
confidentialite des flux — donnent des accroches directes vers l'EU AI Act, le
NIST AI RMF et l'ISO/IEC 42001.

## Stack technique

- **Python 3.11+ stdlib uniquement** — aucune dependance externe, aucune cle API.
  La gouvernance se code avec des `dataclass`, un peu de `hashlib` et de la rigueur,
  pas avec un framework lourd. Chaque solution tourne hors-ligne et de maniere
  deterministe (essentiel pour l'auditabilite).
- **Observabilite** : les scripts impriment une trace structuree des decisions ;
  en prod on brancherait un logger / un SIEM, mais le mecanisme reste celui-ci.

## Requirements

```bash
python domains/tech/gouvernance-ia/05-projets-guides/<projet>/solution/<fichier>.py
```

Rien a installer. Chaque solution est un fichier autonome qui s'execute et
imprime sa demo (exit 0).
