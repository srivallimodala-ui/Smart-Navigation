import argparse
import json
from pathlib import Path
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit, cross_val_score
from sklearn.pipeline import make_pipeline

NUMERIC_FEATURES = [
    "current_speed_mps",
    "avg_speed_5s_mps",
    "avg_speed_10s_mps",
    "speed_std_10s_mps",
    "max_speed_10s_mps",
    "speed_trend_10s_mps",
    "speed_samples_5s",
    "speed_samples_10s",
    "speed_history_coverage_s",
    "distance_to_turn_m",
    "stop_duration_s",
]
CATEGORICAL_FEATURES = ["turn_type"]
FEEDBACK_TARGETS = {"too_early": -1, "just_right": 0, "too_late": 1}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="artifacts/lead_time_model.joblib")
    ap.add_argument("--adjustment-seconds", type=float, default=2.0)
    ap.add_argument("--min-labels", type=int, default=8)
    ap.add_argument("--min-sessions", type=int, default=2)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    # Accept both browser/backend feedback exports and the richer maneuver dataset.
    aliases = {
        "lead_time_s": ["lead_time_seconds", "lead_time_s"],
        "current_speed_mps": ["walking_speed_mps", "current_speed_mps", "speed_mps"],
        "stop_duration_s": ["stop_duration_seconds", "stop_duration_s"],
        "distance_to_turn_m": ["distance_to_turn_m"],
        "turn_type": ["turn_type"],
    }
    for canonical, options in aliases.items():
        if canonical not in df.columns:
            for candidate in options:
                if candidate in df.columns:
                    df[canonical] = df[candidate]
                    break
    if "session_id" not in df.columns:
        df["session_id"] = "session_unknown"
    for c in NUMERIC_FEATURES + ["lead_time_s"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "feedback" not in df.columns:
        raise SystemExit("Missing required column: feedback")

    df["feedback"] = df["feedback"].fillna("").astype(str).str.strip().str.lower()
    # Browser-downloaded CSVs use the button keys early/right/late; API exports
    # use the canonical labels. Accept both so the two collection paths agree.
    feedback_aliases = {
        "early": "too_early", "right": "just_right", "late": "too_late",
        "too early": "too_early", "just right": "just_right", "too late": "too_late",
    }
    df["feedback"] = df["feedback"].replace(feedback_aliases)
    df = df[df["feedback"].isin(FEEDBACK_TARGETS)].copy()
    df = df[df["lead_time_s"].notna()].copy()
    if "turn_outcome" in df.columns:
        df = df[df["turn_outcome"].fillna("reached").astype(str).str.lower() != "pending"].copy()
    if df.empty:
        raise SystemExit(
            "Training stopped safely: there are no rows with valid feedback and lead_time_s. "
            "Collect feedback values: too_early, just_right, or too_late."
        )

    # Weakly-supervised target, explicitly documented in README.
    df["preferred_lead_time_s"] = df.apply(
        lambda r: r["lead_time_s"] + FEEDBACK_TARGETS[r["feedback"]] * args.adjustment_seconds,
        axis=1,
    ).clip(3, 20)

    for c in NUMERIC_FEATURES:
        if c not in df.columns:
            df[c] = float("nan")
    if "turn_type" not in df.columns:
        df["turn_type"] = "unknown"
    df["turn_type"] = df["turn_type"].fillna("unknown").astype(str)
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].replace([float("inf"), -float("inf")], float("nan"))

    if len(df) < args.min_labels:
        raise SystemExit(f"Training stopped safely: {len(df)} labelled rows; need at least {args.min_labels}.")
    groups = df["session_id"].fillna("unknown").astype(str) if "session_id" in df.columns else pd.Series(["unknown"] * len(df))
    if groups.nunique() < args.min_sessions:
        raise SystemExit(f"Training stopped safely: only {groups.nunique()} distinct sessions; need at least {args.min_sessions}.")

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
            ]), NUMERIC_FEATURES),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", Ridge(alpha=1.0)),
    ])
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["preferred_lead_time_s"]
    model.fit(X, y)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "min_lead_time_s": 3.0,
        "max_lead_time_s": 20.0,
        "default_lead_time_s": 10.0,
        "adjustment_seconds": args.adjustment_seconds,
        "training_rows": int(len(df)),
        "training_sessions": int(groups.nunique()),
        "target_method": "weak_feedback_rule",
    }
    joblib.dump(artifact, out)
    report = {
        "status": "trained_prototype",
        "training_rows": int(len(df)),
        "training_sessions": int(groups.nunique()),
        "feedback_counts": df["feedback"].value_counts().to_dict(),
        "target_method": "weak_feedback_rule",
        "warning": "Metrics on the training data are not generalization metrics. Evaluate on held-out sessions/users before deployment.",
        "training_mae_s_in_sample": round(float(mean_absolute_error(y, model.predict(X))), 3),
        "artifact": str(out),
    }
    out.with_suffix(".report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
