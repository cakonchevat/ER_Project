import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Iterable

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


@dataclass
class TrainedMatcher:
    # Defining a wrapper object that stores the model, scaler, list of feature columns names, optimal decision threshold,
    # training metrics and out-of-fold probabilities
    model_name: str
    model: Any
    scaler: StandardScaler
    feature_cols: List[str]
    best_threshold: float
    metrics: Dict[str, Any]
    oof_prob: np.ndarray

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_cols]
        X = X.astype(float).fillna(0.0)  # forcing columns to be numeric floats (for scaling)
        X_scaled = self.scaler.transform(X.values)
        proba = self.model.predict_proba(X_scaled)
        return proba[:, 1]  # selecting the probability of being a match

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        p = self.predict_proba(df)
        return (p >= self.best_threshold).astype(int)


def select_threshold_by_fbeta(y_true: np.ndarray, y_prob: np.ndarray, beta: float = 2.0) -> Tuple[float, Dict[str, float]]:
    precision, recall, thresh = precision_recall_curve(y_true, y_prob)
    if len(thresh) == 0:
        return 0.5, {"f_beta": 0.0}

    fbeta_vals = [
        ((1 + beta**2) * p * r) / ((beta**2) * p + r) if (p + r) > 0 else 0.0
        for p, r in zip(precision[1:], recall[1:])
    ]

    ind_of_best_thresh = int(np.argmax(fbeta_vals))
    best_thresh = float(thresh[ind_of_best_thresh])
    return best_thresh, {
        "f_beta": float(fbeta_vals[ind_of_best_thresh])
    }

def model_constructor(model_name: str, class_weight: Dict[int, float], scale_pos_weight: float):
    # The models and its parameters are based on the hyperparameter_tuning results made from the same named ipynb file in src/classifier
    if model_name == "logreg":
        return LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            class_weight=class_weight,
            C=0.01,
            penalty="l1"
        )

    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=2,
            n_jobs=-1,
            class_weight=class_weight,
            random_state=42
        )

    if model_name == "xgb":
        return XGBClassifier(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            tree_method="hist",
            n_jobs=-1,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=42
        )

    raise ValueError(f"Unknown model: {model_name}.")

def train_pairwise_matcher(
    df: pd.DataFrame,
    feature_cols: List[str],
    model_name: str,
    label_col: str = "label",
    n_folds: int = 5,
    random_state: int = 42,
    beta: float = 2.0,
) -> TrainedMatcher:
    X = df[feature_cols].astype(float).fillna(0.0).values
    y = df[label_col].astype(int).values

    # Handling imbalance
    classes = np.array([0, 1])
    cw_vals = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    class_weight = {int(k): float(v) for k, v in zip(classes, cw_vals)}
    scale_pos_weight = float(cw_vals[0] / cw_vals[1])

    clf = model_constructor(model_name, class_weight, scale_pos_weight)
    scaler = StandardScaler()

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    oof_prob = np.zeros(len(df), dtype=float)

    for train_ind, validation_ind in skf.split(X, y):
        X_train, X_validation = X[train_ind], X[validation_ind]
        y_tr = y[train_ind]

        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_validation_scaled = scaler.transform(X_validation)

        clf.fit(X_train_scaled, y_tr)
        oof_prob[validation_ind] = clf.predict_proba(X_validation_scaled)[:, 1]

    best_thr, _ = select_threshold_by_fbeta(y, oof_prob, beta=beta)

    oof_pred = (oof_prob >= best_thr).astype(int)
    oof_prec = float(precision_score(y, oof_pred, zero_division=0))
    oof_rec  = float(recall_score(y, oof_pred, zero_division=0))
    oof_fbeta = (
        ((1 + beta**2) * oof_prec * oof_rec) / ((beta**2) * oof_prec + oof_rec)
        if (oof_prec + oof_rec) > 0 else 0.0
    )

    oof_roc = float(roc_auc_score(y, oof_prob))
    oof_pr  = float(average_precision_score(y, oof_prob))

    metrics = {
        "oof_best_thr": float(best_thr),
        "oof_fbeta": float(oof_fbeta),
        "oof_precision": oof_prec,
        "oof_recall": oof_rec,
        "oof_pr_auc": oof_pr,
        "oof_roc_auc": oof_roc,
        "beta": float(beta),
        "pos_frac": float(y.mean()),
        "model": model_name,
        "features": list(feature_cols),
    }

    scaler.fit(X)
    X_full = scaler.transform(X)
    clf.fit(X_full, y)

    return TrainedMatcher(
        model_name=model_name,
        model=clf,
        scaler=scaler,
        feature_cols=list(feature_cols),
        best_threshold=float(best_thr),
        metrics=metrics,
        oof_prob=oof_prob,
    )

def _save_predictions_csv(
    tm: TrainedMatcher,
    df: pd.DataFrame,
    src_col: str,
    cand_col: str,
    out_path: str
) -> pd.DataFrame:
    probs = tm.predict_proba(df)
    preds = (probs >= tm.best_threshold).astype(int)

    out = df[[src_col, cand_col]].copy()
    out["prob_match"] = probs
    out["pred_match"] = preds
    out.to_csv(out_path, index=False)
    return out


def train_and_save_all_models(
    df: pd.DataFrame,
    feature_cols: List[str],
    src_col: str = "src_id",
    cand_col: str = "cand_id",
    label_col: str = "label",
    models: Iterable[str] = ("logreg", "rf", "xgb"),
    out_dir: str = "../data",
    file_stem: str = "classifier_predictions",
    file_suffix: str = ""
) -> Dict[str, TrainedMatcher]:
    short = {"logreg": "lr", "rf": "rf", "xgb": "xgb"}
    tms = {}

    for m in models:
        print(f"\nTraining model: {m.upper()}")
        tm = train_pairwise_matcher(
            df=df,
            feature_cols=feature_cols,
            label_col=label_col,
            model_name=m,
            n_folds=5,
            random_state=42
        )
        tms[m] = tm

        out_path = f"{out_dir}/{file_stem}_{short[m]}{file_suffix}.csv"
        _save_predictions_csv(tm, df, src_col, cand_col, out_path)
        print(f"Saved predictions to {out_path}")

    return tms
