# Exercice (medium) — Chaine de hash tamper-evident

## Objectif
Construire un **journal append-only chaine par hash** et un **verifier d'integrite** qui detecte toute alteration d'une entree passee et indique **a quelle position** elle a eu lieu.

## Consigne
1. Reprends (ou recree) une fonction qui produit une entree d'audit (le quintuple de l'exercice easy suffit ; tu peux la simplifier).
2. Ecris une classe `HashChainLog` avec :
   - un attribut interne `chain` (liste),
   - une propriete/methode `head_hash` qui renvoie le hash de la derniere entree (ou la chaine `"GENESIS"` si vide),
   - une methode `append(entry: dict)` qui calcule `entry_hash = sha256(prev_hash + canonical(entry))` et stocke un enregistrement `{index, entry, prev_hash, entry_hash}`.
3. La serialisation `canonical(entry)` DOIT etre **deterministe** : utilise `json.dumps(..., sort_keys=True)` (sinon deux processus calculent des hash differents pour le meme contenu).
4. Ecris `verify()` qui re-parcourt la chaine, recalcule chaque hash, et renvoie `(True, None)` si tout est coherent, sinon `(False, index)` ou `index` est la **premiere** position incoherente.
5. Dans le `if __name__ == "__main__":` : ajoute 3 entrees, verifie que `verify()` renvoie `(True, None)`, puis **modifie silencieusement** un champ d'une entree deja stockee (ex. un montant) et montre que `verify()` renvoie `(False, <index modifie>)`.

## Criteres de reussite
- [ ] `append` chaine correctement chaque entree au `head_hash` precedent (premiere entree chainee a `"GENESIS"`).
- [ ] La serialisation est deterministe (`sort_keys=True`) — re-hasher la meme entree donne le meme hash.
- [ ] `verify()` renvoie `(True, None)` sur une chaine intacte.
- [ ] Apres une modification silencieuse de l'entree d'index *k*, `verify()` renvoie `(False, k)`.
- [ ] Le script tourne en stdlib seule (`hashlib`, `json`), sans erreur.
