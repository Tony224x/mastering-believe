"""
Solutions -- Jour 11 : LLM Infrastructure
"""


def solution_exercice_1() -> None:
    """
    Exercice 1 -- Routing policy pour un assistant email.

    +---------------+---------+-------------------------------------------+
    | Tache         | Primary | Condition de basculement                  |
    +---------------+---------+-------------------------------------------+
    | Classify      | nano    | Si confidence < 0.8 -> mini                |
    |               |         | Si spam + litigieux -> std (faux positifs  |
    |               |         | couteux)                                   |
    +---------------+---------+-------------------------------------------+
    | Summarize     | mini    | Si thread > 10 emails ou > 5K tokens ->    |
    |               |         | std                                        |
    +---------------+---------+-------------------------------------------+
    | Draft reply   | mini    | Si email pro avec acteur senior ou ton     |
    |               |         | delicat (detecte par un classifier nano)   |
    |               |         | -> std. Si negociation complexe -> frontier|
    +---------------+---------+-------------------------------------------+
    | Extract       | nano    | Si le classifier nano n'est pas sur ou si  |
    |               |         | schema JSON strict non valide -> mini ->   |
    |               |         | std                                        |
    +---------------+---------+-------------------------------------------+
    | Answer        | std     | Si la question exige du reasoning profond  |
    | question      |         | (multi-step, calculs) -> frontier           |
    +---------------+---------+-------------------------------------------+

    Q2 -- Cout comparatif (unites relatives par appel : nano=1, mini=4, std=15, frontier=80) :

    Volume repartition :
      classify : 50% * 2M = 1M
      extract  : 25% * 2M = 500K
      summarize: 15% * 2M = 300K
      draft    :  8% * 2M = 160K
      answer   :  2% * 2M =  40K

    Politique proposee (avec ~15% de basculement) :
      classify  : 1M * (0.85*1 + 0.15*4)  = 1M * 1.45 = 1.45M units
      extract   : 500K * (0.9*1 + 0.1*4)  = 500K * 1.3 = 0.65M
      summarize : 300K * (0.8*4 + 0.2*15) = 300K * 6.2 = 1.86M
      draft     : 160K * (0.8*4 + 0.18*15 + 0.02*80)
                = 160K * (3.2+2.7+1.6) = 160K * 7.5 = 1.20M
      answer    : 40K * (0.7*15 + 0.3*80)
                = 40K * 34.5 = 1.38M
      TOTAL     ~ 6.54M units

    Politique naive "frontier pour tout" :
      2M * 80 = 160M units.

    => Facteur d'economie ~= **24x**. On est bien au-dessus de la cible.

    Q3 -- Quand accepter frontier :
    - Draft reply pour des emails sensibles detectes (negociation, RH,
      conflit client). Faux sans valeur = risque business.
    - Answer question quand la query necessite du chaining (plusieurs
      sources, calcul, reasoning sur un contrat).
    - Jamais pour classify ou extract : un classifier doit etre rapide
      et pas cher.
    """


def solution_exercice_2() -> None:
    """
    Exercice 2 -- Guardrails par produit.

    1) CHATBOT MEDICAL
       Input :
         - PII detection (nom, SSN) + masking avant envoi au LLM
         - Prompt injection detection
         - Rate limit par user
         - Contenu manifestement suicide / urgence -> escalade immediate
       Output :
         - Disclaimer automatique "je ne remplace pas un avis medical"
         - Validation que l'output ne contient pas de diagnostic ferme
         - Toxicity check (Llama Guard)
         - Escalade humaine si la question contient des mots-cles critiques
       Echec : ESCALADE HUMAINE. Jamais "reponse degradee".

    2) GENERATEUR DE CV
       Input :
         - PII collectee volontairement (autorisee) mais chiffree
         - Prompt injection detection
         - Pas de contenu diffamatoire
       Output :
         - Verification que chaque experience citee est dans l'input
           user (pas d'hallucination d'experience)
         - Validation JSON sur le schema (sections, dates, etc.)
         - Retry with constraints si format invalide
       Echec : retry + fallback template par defaut.

    3) SUPPORT E-COMMERCE AVEC ACCES COMMANDES
       Input :
         - Authentification forte (session, non-LLM)
         - Enrichissement du prompt avec les donnees user COTE BACKEND,
           jamais "trusted" depuis le prompt user
         - Prompt injection detection
       Output :
         - Validation que les numeros de commande cites dans la reponse
           appartiennent bien a l'utilisateur connecte (authz check
           post-generation)
         - Masking des donnees d'autres clients
         - JSON validation si API retournant une action (ex: "refund")
       Echec : bloquer + message d'erreur sans leak. Log l'incident.

    4) ASSISTANT JURIDIQUE (extraction de clauses)
       Input :
         - Doc PDF parse + chunking controle
         - Prompt injection detection dans le document lui-meme
           (documents adversariaux)
       Output :
         - Groundedness check : chaque clause citee doit exister dans
           le doc source (string match ou LLM-as-a-judge)
         - JSON validation (clause_id, text, commentary)
         - Disclaimer "n'est pas un conseil juridique"
       Echec : retry avec prompt plus strict, puis bypass avec output
       "je n'ai pas trouve d'information fiable dans le document".

    Note transverse : tous ces systemes doivent avoir un trace ID
    visible dans l'output pour permettre au support humain de retrouver
    la conversation en cas de reclamation.
    """


def solution_exercice_3() -> None:
    """
    Exercice 3 -- Semantic cache tuning.

    Etat : threshold=0.90, hit rate=45%, 3 faux positifs sur des
    questions proches mais semantiquement differentes.

    Q1 -- Cause racine :
      Le cosine sur BoW / embeddings generiques ne capture pas bien les
      petits mots qui changent tout le sens ("deployer" vs "mettre a jour
      un deploy"). Le modele voit un overlap massif (k8s, deploy, service)
      et score haut. DILEMME : si on monte le threshold, on perd du hit
      rate. Si on le baisse, on augmente les faux positifs.

    Q2 -- Techniques pour reduire les faux positifs sans sacrifier le
    hit rate :

      1. **Threshold adaptatif par type de query** : les questions
         "how to" tolerent moins d'approximation que les questions
         "what is". On apprend le bon threshold par categorie sur un
         dataset de validation.

      2. **Boost des verbes et mots-cles d'action** : donner un poids
         plus grand aux verbes principaux (deploy, update, delete,
         rollback). Un match imparfait sur le verbe = mismatch.

      3. **LLM-as-a-judge pre-serve** : avant de renvoyer un hit, un
         petit modele nano verifie que la question en cache et la
         question courante demandent vraiment la meme chose. Cout : un
         appel nano ajoute, mais cache hit toujours moins cher qu'un
         appel std.

      4. **Segmentation du cache par topic** : un classifier nano assigne
         un topic ("k8s", "python-debug", "security"), et le cache
         cherche seulement dans le meme topic. Reduit les collisions
         inter-sujets.

      5. **Cross-encoder leger** comme verificateur final (plus precis
         que le cosine BoW).

      6. **TTL raccourci** pour les sujets qui evoluent vite (secu,
         depreciations, versions de lib).

    Q3 -- Plan de mesure :

      a) Dataset de validation : 500 paires (query_a, query_b) dont la
         moitie sont "equivalentes" et l'autre "differentes". Label humain.
      b) Metriques avant / apres changement :
         - Taux de faux positifs (cache hit incorrect) sur le dataset
         - Taux de hit rate en production
         - User feedback (thumbs down sur les reponses cachees vs fresh)
         - Latence moyenne
      c) Deployer les changements en shadow mode : le cache hit est
         calcule mais la vraie reponse est quand meme generee et
         comparee. Si ecart > seuil, log.

    Q4 -- Quand desactiver le cache :
      Si le taux de faux positifs > 1% ET que les faux positifs sont
      visibles pour l'utilisateur (pas juste des phrasings differents).
      Pour ce use case (devs, confiance technique critique), mieux vaut
      payer le cout LLM que risquer la perte de confiance. Desactiver
      totalement le cache OU le restreindre aux queries les plus simples
      (definitions, glossaire).
    """


if __name__ == "__main__":
    for fn in (solution_exercice_1, solution_exercice_2, solution_exercice_3):
        print(f"\n--- {fn.__name__} ---")
        print(fn.__doc__)
