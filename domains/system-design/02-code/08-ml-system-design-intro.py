"""
Jour 8 -- ML System Design : Feature Store + Model Registry + Batch Pipeline
Mini systeme ML end-to-end en memoire (no external dependencies).

Usage:
    python 08-ml-system-design-intro.py

Le script simule un cycle de vie ML :
1. Un feature store (offline + online)
2. Un model registry avec versions et stages
3. Un mini modele (rule-based pour rester sans dependances)
4. Un pipeline de batch prediction
5. Un shadow deployment compare les predictions v1 et v2
"""

import time
import json
import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Optional
from datetime import datetime, timedelta

SEPARATOR = "=" * 70


# =============================================================================
# SECTION 1 : Feature Store minimaliste
# =============================================================================


@dataclass
class FeatureDefinition:
    """Definition d'une feature : comment la calculer a partir d'un event dict."""

    name: str
    compute_fn: Callable[[dict], Any]  # same fn used offline and online -> no skew
    description: str = ""


class FeatureStore:
    """Feature store avec un offline store (historique) et un online store (latest).

    Le offline store supporte le point-in-time lookup : etant donne un timestamp,
    il retourne la valeur de la feature telle qu'elle etait a cet instant.
    L'online store ne garde que la derniere valeur, pour l'inference rapide.
    """

    def __init__(self):
        # offline: {entity_id: [(timestamp, feature_name, value)]}
        self.offline: dict[str, list[tuple[float, str, Any]]] = defaultdict(list)
        # online: {entity_id: {feature_name: value}} - last value wins
        self.online: dict[str, dict[str, Any]] = defaultdict(dict)
        self.definitions: dict[str, FeatureDefinition] = {}

    def register(self, fdef: FeatureDefinition) -> None:
        """Enregistre une feature definition."""
        self.definitions[fdef.name] = fdef

    def ingest(self, entity_id: str, event: dict, ts: Optional[float] = None) -> None:
        """Ingere un event et calcule toutes les features enregistrees.

        Offline: append une ligne historique (timestamp + feature + valeur).
        Online: override la derniere valeur.
        """
        if ts is None:
            ts = time.time()
        for fname, fdef in self.definitions.items():
            try:
                value = fdef.compute_fn(event)
            except Exception:
                value = None  # in real life, log + alert on compute failure
            self.offline[entity_id].append((ts, fname, value))
            self.online[entity_id][fname] = value

    def get_online(self, entity_id: str) -> dict[str, Any]:
        """Lookup online pour le serving (latence ms)."""
        return dict(self.online.get(entity_id, {}))

    def get_historical(self, entity_id: str, feature: str, as_of: float) -> Any:
        """Lookup point-in-time : valeur de la feature a l'instant as_of.

        Necessaire pour generer des datasets d'entrainement sans data leakage.
        """
        # Scan timeline, keep values strictly before as_of
        candidates = [
            (ts, v) for (ts, f, v) in self.offline.get(entity_id, []) if f == feature and ts <= as_of
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda x: x[0])[1]  # most recent before as_of


# =============================================================================
# SECTION 2 : Model Registry avec versions et stages
# =============================================================================


@dataclass
class ModelVersion:
    """Une version de modele dans le registry."""

    name: str
    version: str
    artifact: Any  # in real life: path to weights, tokenizer, etc.
    metadata: dict = field(default_factory=dict)
    stage: str = "dev"  # dev | staging | production | archived
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def fingerprint(self) -> str:
        """Hash stable pour detecter les changements."""
        payload = json.dumps(
            {"name": self.name, "version": self.version, "metadata": self.metadata},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


class ModelRegistry:
    """Registry qui indexe les modeles par (name, version).

    Permet de promote/rollback entre stages et de retrouver le modele
    production d'un nom donne.
    """

    def __init__(self):
        self.versions: dict[tuple[str, str], ModelVersion] = {}

    def register(self, mv: ModelVersion) -> None:
        self.versions[(mv.name, mv.version)] = mv

    def promote(self, name: str, version: str, stage: str) -> None:
        """Promote un modele. Si stage=production, archive l'ancien production."""
        key = (name, version)
        if key not in self.versions:
            raise KeyError(f"Unknown model {name}:{version}")
        if stage == "production":
            for (n, v), mv in self.versions.items():
                if n == name and mv.stage == "production":
                    mv.stage = "archived"  # rollback-friendly
        self.versions[key].stage = stage

    def get_production(self, name: str) -> Optional[ModelVersion]:
        """Retourne le modele actuellement en prod pour ce nom."""
        for (n, _), mv in self.versions.items():
            if n == name and mv.stage == "production":
                return mv
        return None


# =============================================================================
# SECTION 3 : Un "modele" rule-based (aucune dependance ML requise)
# =============================================================================


def model_v1_predict(features: dict) -> dict:
    """Credit scoring simpliste v1 : seuils sur revenu et dettes.

    Retourne {'score': 0..1, 'decision': 'approve'|'reject'}.
    """
    income = features.get("monthly_income", 0) or 0
    debt = features.get("monthly_debt", 0) or 0
    ratio = debt / income if income > 0 else 1.0
    # Simple linear combo then squash into [0,1]
    raw = 0.6 * (income / 5000) - 0.8 * ratio
    score = max(0.0, min(1.0, 0.5 + raw / 2))
    return {"score": round(score, 3), "decision": "approve" if score >= 0.5 else "reject"}


def model_v2_predict(features: dict) -> dict:
    """V2 : ajoute la feature 'num_late_payments' (meilleure capture du risque)."""
    income = features.get("monthly_income", 0) or 0
    debt = features.get("monthly_debt", 0) or 0
    late = features.get("num_late_payments", 0) or 0
    ratio = debt / income if income > 0 else 1.0
    raw = 0.6 * (income / 5000) - 0.8 * ratio - 0.2 * late
    score = max(0.0, min(1.0, 0.5 + raw / 2))
    return {"score": round(score, 3), "decision": "approve" if score >= 0.5 else "reject"}


# =============================================================================
# SECTION 4 : Pipeline de batch prediction
# =============================================================================


def batch_predict(fs: FeatureStore, registry: ModelRegistry, entity_ids: list[str]) -> list[dict]:
    """Pipeline batch : pour chaque entity, lit online features + applique le modele prod.

    En vrai, on lirait depuis l'offline store (Parquet/BigQuery) pour des millions
    d'entities avec Spark. Ici on simule sur du petit volume.
    """
    prod = registry.get_production("credit-scoring")
    if prod is None:
        raise RuntimeError("No production model for credit-scoring")
    results = []
    for eid in entity_ids:
        features = fs.get_online(eid)
        prediction = prod.artifact(features)  # artifact = fn in this demo
        results.append(
            {
                "entity_id": eid,
                "model_version": prod.version,
                "prediction": prediction,
                "features_fingerprint": hashlib.md5(
                    json.dumps(features, sort_keys=True).encode()
                ).hexdigest()[:8],
            }
        )
    return results


# =============================================================================
# SECTION 5 : Shadow deployment
# =============================================================================


def shadow_compare(
    fs: FeatureStore,
    entity_ids: list[str],
    prod_fn: Callable,
    shadow_fn: Callable,
) -> dict:
    """Execute prod + shadow sur chaque entity et mesure le disagreement rate.

    Un disagreement > 10-20% est un signal d'alarme : le nouveau modele prend
    des decisions significativement differentes. A inspecter avant promotion.
    """
    disagreements = 0
    score_diffs = []
    for eid in entity_ids:
        feats = fs.get_online(eid)
        p = prod_fn(feats)
        s = shadow_fn(feats)
        if p["decision"] != s["decision"]:
            disagreements += 1
        score_diffs.append(abs(p["score"] - s["score"]))
    n = len(entity_ids)
    return {
        "n": n,
        "disagreement_rate": disagreements / n if n else 0,
        "avg_score_diff": sum(score_diffs) / n if n else 0,
        "max_score_diff": max(score_diffs) if score_diffs else 0,
    }


# =============================================================================
# SECTION 6 : Demo end-to-end
# =============================================================================


def demo() -> None:
    random.seed(42)
    print(SEPARATOR)
    print("DEMO ML SYSTEM DESIGN -- Feature Store + Registry + Batch + Shadow")
    print(SEPARATOR)

    # 1) Setup feature store
    fs = FeatureStore()
    fs.register(FeatureDefinition("monthly_income", lambda e: e["income"]))
    fs.register(FeatureDefinition("monthly_debt", lambda e: e["debt"]))
    fs.register(FeatureDefinition("num_late_payments", lambda e: e.get("late", 0)))

    # 2) Ingest events (simulate 5 users with different profiles)
    users = [
        ("u1", {"income": 3000, "debt": 2500, "late": 3}),
        ("u2", {"income": 6000, "debt": 1000, "late": 0}),
        ("u3", {"income": 4500, "debt": 3800, "late": 2}),
        ("u4", {"income": 8000, "debt": 500, "late": 0}),
        ("u5", {"income": 2500, "debt": 1500, "late": 5}),
    ]
    for uid, event in users:
        fs.ingest(uid, event)

    print(f"\n[FeatureStore] online snapshot for u3 = {fs.get_online('u3')}")

    # 3) Register v1 and v2 in model registry
    reg = ModelRegistry()
    reg.register(
        ModelVersion(
            name="credit-scoring",
            version="1.0.0",
            artifact=model_v1_predict,
            metadata={"auc_offline": 0.87, "dataset": "credit_2025_Q1"},
        )
    )
    reg.register(
        ModelVersion(
            name="credit-scoring",
            version="2.0.0",
            artifact=model_v2_predict,
            metadata={"auc_offline": 0.91, "dataset": "credit_2025_Q2", "uses": ["late_payments"]},
        )
    )
    reg.promote("credit-scoring", "1.0.0", "production")
    reg.promote("credit-scoring", "2.0.0", "staging")

    prod = reg.get_production("credit-scoring")
    print(f"\n[Registry] production model = {prod.name}:{prod.version} fp={prod.fingerprint()}")

    # 4) Batch prediction with prod model
    preds = batch_predict(fs, reg, [uid for uid, _ in users])
    print("\n[BatchPipeline] predictions from production model:")
    for p in preds:
        print(f"  {p['entity_id']}: {p['prediction']} (model={p['model_version']})")

    # 5) Shadow comparison v1 vs v2
    stats = shadow_compare(
        fs,
        [uid for uid, _ in users],
        prod_fn=model_v1_predict,
        shadow_fn=model_v2_predict,
    )
    print("\n[Shadow] v1 (prod) vs v2 (candidate):")
    print(f"  disagreement_rate = {stats['disagreement_rate']:.0%}")
    print(f"  avg score diff    = {stats['avg_score_diff']:.3f}")
    print(f"  max score diff    = {stats['max_score_diff']:.3f}")
    if stats["disagreement_rate"] > 0.2:
        print("  -> high disagreement : inspect before promoting !")
    else:
        print("  -> safe-ish to canary deploy")

    # 6) Point-in-time historical lookup example
    # simulate an update to u1's debt later in time
    fs.ingest("u1", {"income": 3000, "debt": 1000, "late": 0}, ts=time.time() + 10)
    early = fs.get_historical("u1", "monthly_debt", as_of=time.time() + 1)
    late_val = fs.get_historical("u1", "monthly_debt", as_of=time.time() + 100)
    print(f"\n[PointInTime] u1 monthly_debt early={early} late={late_val}")
    print("  (training datasets must use point-in-time to avoid data leakage)")


if __name__ == "__main__":
    demo()
