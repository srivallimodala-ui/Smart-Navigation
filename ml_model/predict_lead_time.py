import argparse
from pathlib import Path
import json
import joblib
import pandas as pd

FEATURES = {
    "current_speed_mps": 0.0,
    "avg_speed_5s_mps": 0.0,
    "avg_speed_10s_mps": 0.0,
    "speed_std_10s_mps": 0.0,
    "max_speed_10s_mps": 0.0,
    "speed_trend_10s_mps": 0.0,
    "speed_samples_5s": 0,
    "speed_samples_10s": 0,
    "speed_history_coverage_s": 0.0,
    "distance_to_turn_m": 0.0,
    "stop_duration_s": 0.0,
    "turn_type": "unknown",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/lead_time_model.joblib")
    ap.add_argument("--user-id", required=True)
    ap.add_argument("--turn-type", default="unknown")
    ap.add_argument("--current-speed-mps", type=float, default=0.0)
    ap.add_argument("--avg-speed-5s-mps", type=float, default=0.0)
    ap.add_argument("--avg-speed-10s-mps", type=float, default=0.0)
    ap.add_argument("--speed-std-10s-mps", type=float, default=0.0)
    ap.add_argument("--max-speed-10s-mps", type=float, default=0.0)
    ap.add_argument("--speed-trend-10s-mps", type=float, default=0.0)
    ap.add_argument("--speed-samples-5s", type=int, default=0)
    ap.add_argument("--speed-samples-10s", type=int, default=0)
    ap.add_argument("--speed-history-coverage-s", type=float, default=0.0)
    ap.add_argument("--distance-to-turn-m", type=float, default=0.0)
    ap.add_argument("--stop-duration-s", type=float, default=0.0)
    ap.add_argument("--feedback-history", help="Optional CSV with user_id,feedback columns")
    args = ap.parse_args()

    x = dict(FEATURES)
    x.update({
        "current_speed_mps": args.current_speed_mps,
        "avg_speed_5s_mps": args.avg_speed_5s_mps,
        "avg_speed_10s_mps": args.avg_speed_10s_mps,
        "speed_std_10s_mps": args.speed_std_10s_mps,
        "max_speed_10s_mps": args.max_speed_10s_mps,
        "speed_trend_10s_mps": args.speed_trend_10s_mps,
        "speed_samples_5s": args.speed_samples_5s,
        "speed_samples_10s": args.speed_samples_10s,
        "speed_history_coverage_s": args.speed_history_coverage_s,
        "distance_to_turn_m": args.distance_to_turn_m,
        "stop_duration_s": args.stop_duration_s,
        "turn_type": args.turn_type or "unknown",
    })

    default_s = 10.0
    model_path = Path(args.model)
    prediction_source = "safe_default"
    predicted = default_s
    if model_path.exists():
        artifact = joblib.load(model_path)
        model = artifact["model"]
        predicted = float(model.predict(pd.DataFrame([x]))[0])
        default_s = float(artifact.get("default_lead_time_s", 10.0))
        prediction_source = "ridge_model"

    # Transparent feedback-rule personalization. This does not claim that
    # elapsed time alone measures satisfaction.
    user_adjustment = 0.0
    feedback_counts = {"too_early": 0, "just_right": 0, "too_late": 0}
    if args.feedback_history and Path(args.feedback_history).exists():
        hist = pd.read_csv(args.feedback_history)
        if {"user_id", "feedback"}.issubset(hist.columns):
            mine = hist[hist["user_id"].astype(str) == str(args.user_id)]
            fb = mine["feedback"].fillna("").astype(str).str.lower().str.strip()
            feedback_counts = {k: int((fb == k).sum()) for k in feedback_counts}
            # One-second conservative adjustment per net feedback, capped at +/- 3s.
            user_adjustment = max(-3.0, min(3.0, feedback_counts["too_late"] - feedback_counts["too_early"]))
            if prediction_source == "safe_default" and mine.shape[0] > 0:
                prediction_source = "feedback_rule"

    final_s = max(3.0, min(20.0, predicted + user_adjustment))
    print(json.dumps({
        "user_id": args.user_id,
        "predicted_lead_time_s": round(final_s, 2),
        "model_prediction_s": round(predicted, 2),
        "user_feedback_adjustment_s": round(user_adjustment, 2),
        "source": prediction_source,
        "feedback_counts": feedback_counts,
        "note": "Prototype output; validate in live walking tests before relying on it."
    }, indent=2))

if __name__ == "__main__":
    main()
