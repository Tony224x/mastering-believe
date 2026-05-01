"""
Training du detecteur de quasi-collisions LogiSim — correction.

Pipeline :
1. Baseline majoritaire (predict "pas collision" tout le temps)
2. Regression logistique avec class_weight='balanced'
3. MLP avec BCEWithLogitsLoss(pos_weight)
4. Threshold tuning sur val pour maximiser F1
5. Eval finale sur test : accuracy, precision, recall, F1, AUC-PR, confusion matrix

Le point pedagogique majeur : comparer accuracy vs F1. La baseline majoritaire
a 97% accuracy et F1=0 — les bons eleves voient immediatement pourquoi accuracy
est inutile ici.

Dependances : numpy, torch, scikit-learn (pour logistique + metriques).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# sklearn et torch sont optionnels. Le script tourne avec juste numpy et
# calcule les metriques a la main. Si sklearn est present, on ajoute la
# regression logistique officielle. Si torch est present, le MLP.
try:
    from sklearn.linear_model import LogisticRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

sys.path.insert(0, str(Path(__file__).parent))
from generate_dataset import FEATURE_NAMES, load_dataset, stratified_split

SEED = 42


def _set_seeds() -> None:
    np.random.seed(SEED)
    if HAS_TORCH:
        torch.manual_seed(SEED)


def _standardize(X_train: np.ndarray, *rest: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    return tuple((X - mean) / std for X in (X_train, *rest))


# ---------- Models -------------------------------------------------------


if HAS_TORCH:
    class CollisionMLP(nn.Module):
        def __init__(self, in_dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(16, 1),  # logit, pas de sigmoid (BCEWithLogitsLoss gere)
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)


def train_mlp(X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 100, lr: float = 1e-3):
    """MLP avec loss ponderee pour compenser le desequilibre."""
    _set_seeds()
    model = CollisionMLP(in_dim=X_train.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # pos_weight = n_neg / n_pos : force le modele a payer plus cher les
    # faux negatifs. Sans ca, il converge vers "tout negatif" (loss mini).
    n_pos = max(1, int(y_train.sum()))
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    X_tr_t = torch.from_numpy(X_train).float()
    y_tr_t = torch.from_numpy(y_train).float()
    X_val_t = torch.from_numpy(X_val).float()
    y_val_t = torch.from_numpy(y_val).float()

    best_val_f1 = 0.0
    best_state = None
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(X_tr_t)
        loss = loss_fn(logits, y_tr_t)
        loss.backward()
        opt.step()

        # Val F1 pour early stopping
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_proba = torch.sigmoid(val_logits).numpy()
            val_pred = (val_proba > 0.5).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(
                y_val, val_pred, average="binary", zero_division=0,
            )
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ---------- Metriques (implementees a la main, numpy only) --------------


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """Precision, recall, F1 sur la classe positive."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def _auc_pr(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Aire sous la courbe precision-recall (approche average-precision).

    Trie par proba decroissante, construit la courbe PR, integre avec la
    methode du "step function" (AP = somme des (R_k - R_{k-1}) * P_k).
    """
    order = np.argsort(-y_proba)
    y_sorted = y_true[order]
    tp_cum = np.cumsum(y_sorted)
    fp_cum = np.cumsum(1 - y_sorted)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return 0.0
    precisions = tp_cum / (tp_cum + fp_cum + 1e-12)
    recalls = tp_cum / n_pos
    # Ajoute le point (0, 1) au debut
    precisions = np.concatenate([[1.0], precisions])
    recalls = np.concatenate([[0.0], recalls])
    return float(np.sum((recalls[1:] - recalls[:-1]) * precisions[1:]))


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def evaluate(y_true: np.ndarray, y_proba: np.ndarray,
             threshold: float = 0.5, label: str = "") -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    accuracy = float((y_pred == y_true).mean())
    prec, rec, f1 = _binary_metrics(y_true, y_pred)
    auc_pr = _auc_pr(y_true, y_proba)
    cm = _confusion(y_true, y_pred)

    print(f"\n--- {label} (threshold={threshold:.2f}) ---")
    print(f"  accuracy  : {accuracy:.3f}   (attention : trompeuse avec desequilibre)")
    print(f"  precision : {prec:.3f}")
    print(f"  recall    : {rec:.3f}")
    print(f"  F1        : {f1:.3f}")
    print(f"  AUC-PR    : {auc_pr:.3f}")
    print(f"  confusion : TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")

    return {"accuracy": accuracy, "precision": prec, "recall": rec,
            "f1": f1, "auc_pr": auc_pr, "confusion_matrix": cm}


def tune_threshold(y_val: np.ndarray, y_val_proba: np.ndarray) -> float:
    """Trouve le threshold qui maximise F1 sur le val set."""
    best_f1 = 0.0
    best_t = 0.5
    for t in np.arange(0.05, 0.95, 0.05):
        y_pred = (y_val_proba >= t).astype(int)
        _, _, f1 = _binary_metrics(y_val, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    print(f"\nThreshold optimal sur val : {best_t:.2f} (F1={best_f1:.3f})")
    return float(best_t)


# ---------- Regression logistique fallback (numpy) ---------------------


def _train_logistic_numpy(X: np.ndarray, y: np.ndarray, epochs: int = 500,
                           lr: float = 0.1) -> np.ndarray:
    """Regression logistique avec gradient descent, class_weight balanced.

    Implem minimale pour que le projet tourne sans sklearn. Le poids de
    classe compense le desequilibre en multipliant le gradient des exemples
    positifs par n_neg / n_pos.
    """
    n, d = X.shape
    X_aug = np.concatenate([X, np.ones((n, 1))], axis=1)  # biais
    w = np.zeros(d + 1)
    n_pos = max(1, int(y.sum()))
    n_neg = n - n_pos
    weights = np.where(y == 1, n_neg / n_pos, 1.0)
    for _ in range(epochs):
        z = X_aug @ w
        p = 1.0 / (1.0 + np.exp(-z))
        grad = (X_aug * (weights * (p - y))[:, None]).mean(axis=0)
        w -= lr * grad
    return w


def _predict_logistic_numpy(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    X_aug = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    return 1.0 / (1.0 + np.exp(-(X_aug @ w)))


# ---------- Pipeline -----------------------------------------------------


def main() -> None:
    _set_seeds()

    print("=" * 60)
    print("DETECTION DE QUASI-COLLISIONS — correction")
    print("=" * 60)

    X, y = load_dataset(n_samples=5000, collision_rate=0.03, seed=SEED)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = stratified_split(X, y, seed=SEED)
    X_tr_s, X_va_s, X_te_s = _standardize(X_tr, X_va, X_te)

    print(f"Train : {len(y_tr)} ({y_tr.sum()} positifs)")
    print(f"Val   : {len(y_va)} ({y_va.sum()} positifs)")
    print(f"Test  : {len(y_te)} ({y_te.sum()} positifs)")

    # -- Baseline majoritaire : predit toujours 0
    print("\n" + "=" * 60)
    print("1. BASELINE MAJORITAIRE")
    print("=" * 60)
    y_proba_zero = np.zeros_like(y_te, dtype=float)
    evaluate(y_te, y_proba_zero, threshold=0.5, label="Baseline zero")
    # Observation : accuracy ~97% mais F1=0 -> toutes les quasi-collisions sont manquees

    # -- Regression logistique (sklearn si dispo, sinon fallback numpy)
    print("\n" + "=" * 60)
    print("2. REGRESSION LOGISTIQUE (class_weight='balanced')")
    print("=" * 60)
    if HAS_SKLEARN:
        logreg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
        logreg.fit(X_tr_s, y_tr)
        y_val_proba_lr = logreg.predict_proba(X_va_s)[:, 1]
        y_te_proba_lr = logreg.predict_proba(X_te_s)[:, 1]
    else:
        print("  [INFO] sklearn absent, fallback sur implementation numpy manuelle.")
        w = _train_logistic_numpy(X_tr_s, y_tr)
        y_val_proba_lr = _predict_logistic_numpy(w, X_va_s)
        y_te_proba_lr = _predict_logistic_numpy(w, X_te_s)
    t_lr = tune_threshold(y_va, y_val_proba_lr)
    evaluate(y_te, y_te_proba_lr, threshold=t_lr, label="Logistic regression (test)")

    # -- MLP (optionnel : necessite torch)
    print("\n" + "=" * 60)
    print("3. MLP (BCEWithLogitsLoss + pos_weight)")
    print("=" * 60)
    if not HAS_TORCH:
        print("  [SKIP] torch non installe. Pip install torch pour activer cette partie.")
        print("  La baseline logistique suffit a valider les concepts du projet.")
    else:
        mlp = train_mlp(X_tr_s, y_tr, X_va_s, y_va)
        mlp.eval()
        with torch.no_grad():
            y_val_proba_mlp = torch.sigmoid(mlp(torch.from_numpy(X_va_s).float())).numpy()
            y_te_proba_mlp = torch.sigmoid(mlp(torch.from_numpy(X_te_s).float())).numpy()
        t_mlp = tune_threshold(y_va, y_val_proba_mlp)
        evaluate(y_te, y_te_proba_mlp, threshold=t_mlp, label="MLP (test)")

    print("\n" + "=" * 60)
    print("LECON")
    print("=" * 60)
    print("La baseline majoritaire a l'accuracy la plus haute mais F1=0.")
    print("Accuracy n'est PAS une metrique valide sur dataset desequilibre.")
    print("Regarde precision/recall/F1 sur la classe positive, et l'AUC-PR.")


if __name__ == "__main__":
    main()
