import pandas as pd
from models.pairwise_classifier import train_pairwise_matcher

feature_cols = [
    "edit_ratio", "jaro_winkler", "lcs_ratio",
    "token_jaccard", "token_cosine",
    "tfidf_word_cosine", "tfidf_char_cosine",
    "dmetaphone_match",
]

models = ["logreg", "rf", "xgb"]
results = {}

if __name__ == "main":
    df = pd.read_csv("../data/feature_extraction/er_blocking_candidates_k40_features_labeled.csv")

    for model_name in models:
        print(f"\nTraining {model_name.upper()} model")

        matcher = train_pairwise_matcher(
            df=df,
            feature_cols=feature_cols,
            model_name=model_name,
            label_col="label",
            n_folds=5,
            random_state=42,
            beta=2.0,  # prioritizing recall with Fβ metric
        )

        results[model_name] = matcher

        # Save per-model predictions
        probs = matcher.predict_proba(df)
        preds = matcher.predict(df)

        out = df[["src_id", "cand_id"]].copy()
        out["prob_match"] = probs
        out["pred_match"] = preds

        out_path = f"../data/classifier_predictions_{model_name}.csv"
        out.to_csv(out_path, index=False)

        print(f"Saved predictions to {out_path}")
        print(
            f"OOF Fβ={matcher.metrics['oof_fbeta']:.4f} | "
            f"Precision={matcher.metrics['oof_precision']:.4f} | "
            f"Recall={matcher.metrics['oof_recall']:.4f} | "
            f"PR-AUC={matcher.metrics['oof_pr_auc']:.4f} | "
            f"ROC-AUC={matcher.metrics['oof_roc_auc']:.4f}"
        )

    # Picking the best model based on OOF F-beta
    best_model, best_tm = max(results.items(), key=lambda kv: kv[1].metrics["oof_fbeta"])
    print(f"\nBest model by Fβ: {best_model.upper()} ({best_tm.metrics['oof_fbeta']:.4f})")
