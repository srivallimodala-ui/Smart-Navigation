"""
Train + compare models for instruction lead-time prediction.

  python train_compare.py --data data/simulated_maneuvers.csv --target preferred_lead_time_s

Evaluation = GroupKFold by user_id, i.e. every test fold contains users the model has
NEVER seen.  This is the honest question: "does it work for a new person?"

Saves the chosen model in the SAME artifact format backend/main.py already loads:
  artifacts/lead_time_model.joblib  ->  {"model": sklearn Pipeline, ...}
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.inspection import permutation_importance

NUMERIC = ["current_speed_mps", "avg_speed_5s_mps", "avg_speed_10s_mps", "speed_std_10s_mps",
           "max_speed_10s_mps", "speed_trend_10s_mps", "speed_samples_5s", "speed_samples_10s",
           "speed_history_coverage_s", "distance_to_turn_m", "stop_duration_s",
           "user_profile_lead_s"]
CATEGORICAL = ["turn_type"]


def prep(scale):
    num = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        num.append(("scaler", StandardScaler()))
    return ColumnTransformer([
        ("numeric", Pipeline(num), NUMERIC),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ], remainder="drop")


def build_models():
    return {
        "Baseline (always predict mean)": Pipeline([("p", prep(False)), ("r", DummyRegressor())]),
        "Ridge": Pipeline([("p", prep(True)), ("r", Ridge(alpha=1.0))]),
        "Random Forest": Pipeline([("p", prep(False)), ("r", RandomForestRegressor(
            n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=0))]),
        "Gradient Boosting": Pipeline([("p", prep(False)), ("r", HistGradientBoostingRegressor(
            max_iter=250, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, random_state=0))]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/simulated_maneuvers.csv")
    ap.add_argument("--target", default="preferred_lead_time_s")
    ap.add_argument("--out", default="artifacts/lead_time_model.joblib")
    ap.add_argument("--save", default="Ridge", help="which model to ship as the artifact")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--no-profile", action="store_true", help="drop user_profile_lead_s (ablation)")
    a = ap.parse_args()

    global NUMERIC
    if a.no_profile:
        NUMERIC = [c for c in NUMERIC if c != "user_profile_lead_s"]
    df = pd.read_csv(a.data)
    df = df[df[a.target].notna()].copy()
    X, y, groups = df[NUMERIC + CATEGORICAL], df[a.target].clip(3, 20), df["user_id"]
    print(f"{len(df)} rows | {groups.nunique()} users | target={a.target}")

    gkf = GroupKFold(n_splits=a.folds)
    results, preds_store = {}, {}
    for name, model in build_models().items():
        oof = np.zeros(len(df))
        for tr, te in gkf.split(X, y, groups):
            model.fit(X.iloc[tr], y.iloc[tr])
            oof[te] = model.predict(X.iloc[te])
        oof = np.clip(oof, 3, 20)
        results[name] = dict(
            MAE_s=round(mean_absolute_error(y, oof), 3),
            RMSE_s=round(mean_squared_error(y, oof) ** 0.5, 3),
            R2=round(r2_score(y, oof), 3),
            within_1_5s_pct=round(float((np.abs(y - oof) <= 1.5).mean() * 100), 1),
        )
        preds_store[name] = oof
        print(f"{name:32s} {results[name]}")

    # ---- plots ----
    Path("reports").mkdir(exist_ok=True)
    names = list(results)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].bar(range(len(names)), [results[n]["MAE_s"] for n in names], color="#4C72B0")
    ax[0].set_xticks(range(len(names))); ax[0].set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=8)
    ax[0].set_ylabel("MAE on unseen users (seconds)  — lower is better"); ax[0].set_title("Model comparison")
    best_plot = a.save
    ax[1].scatter(y, preds_store[best_plot], s=6, alpha=0.35)
    ax[1].plot([3, 20], [3, 20], "r--", lw=1)
    ax[1].set_xlabel("True ideal lead time (s)"); ax[1].set_ylabel("Predicted (s)")
    ax[1].set_title(f"{best_plot}: predicted vs true (held-out users)")
    plt.tight_layout(); plt.savefig("reports/model_comparison.png", dpi=140); plt.close()

    # ---- tune + fit final model on ALL data and save ----
    final = build_models()[a.save]
    final.fit(X, y)

    # permutation importance on a held-out-user split (honest importance)
    tr, te = next(GroupKFold(n_splits=a.folds).split(X, y, groups))
    probe = build_models()[a.save].fit(X.iloc[tr], y.iloc[tr])
    pi = permutation_importance(probe, X.iloc[te], y.iloc[te], n_repeats=10, random_state=0,
                                scoring="neg_mean_absolute_error")
    imp = pd.Series(pi.importances_mean, index=NUMERIC + CATEGORICAL).sort_values()
    plt.figure(figsize=(7, 4.5)); imp.plot.barh(color="#55A868")
    plt.xlabel("Increase in MAE (s) when feature is shuffled"); plt.title(f"What {a.save} relies on")
    plt.tight_layout(); plt.savefig("reports/feature_importance.png", dpi=140); plt.close()

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": final, "numeric_features": NUMERIC, "categorical_features": CATEGORICAL,
        "min_lead_time_s": 3.0, "max_lead_time_s": 20.0, "default_lead_time_s": 10.0,
        "training_rows": int(len(df)), "training_sessions": int(df["session_id"].nunique()),
        "target_method": f"simulated:{a.target}", "model_name": a.save,
    }, out)
    report = dict(data=a.data, target=a.target, rows=len(df), users=int(groups.nunique()),
                  evaluation="GroupKFold by user_id (unseen users)", results=results,
                  shipped_model=a.save, artifact=str(out),
                  top_features=imp.sort_values(ascending=False).round(3).head(6).to_dict(),
                  warning="Trained on SIMULATED data: shows the pipeline works, not real-world accuracy.")
    Path("reports/training_report.json").write_text(json.dumps(report, indent=2))
    print("\nTop features:", report["top_features"])
    print("Saved", out)


if __name__ == "__main__":
    main()
