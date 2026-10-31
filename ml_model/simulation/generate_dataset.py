"""
Simulated dataset generator for Smart Turn-by-Turn Navigation (VNIT prototype).

WHAT THIS IS
  A *synthetic* dataset whose columns match ml_model/data/current_maneuver_dataset.csv.
  Every row = one turn instruction (maneuver) given to one simulated walker.

HOW THE GROUND TRUTH IS BUILT (the "hidden rules" the model must rediscover)
  Each simulated user has hidden traits (base speed, reaction time, how early they like
  to be told, how erratic their pace is).  The ideal lead time (seconds before the turn
  that the instruction should be spoken) is:

    ideal = reaction_s                      # user's own processing time
          + turn_complexity[turn_type]      # u-turns / sharp turns need more warning
          + resume_penalty(stop_duration)   # just started walking again -> needs more time
          + k * speed_std                   # erratic walkers -> earlier warning
          + poor_gps_penalty(accuracy)      # noisy GPS -> be earlier, position is uncertain
          + speed term + slow-u-turn interaction
          + user_pref_offset                # personal taste (+/- seconds)
          + noise

  IMPORTANT: these coefficients are ASSUMPTIONS made by us, not measured human data.
  A model trained on this data proves the PIPELINE works; it does not prove real-world accuracy.

OUTPUT
  data/simulated_maneuvers.csv
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Same labels the backend accepts (LeadTimePredictionIn.turn_type)
TURN_TYPES = ["left", "right", "straight", "uturn", "roundabout"]
TURN_P = [0.36, 0.36, 0.14, 0.06, 0.08]
TURN_COMPLEXITY_S = {"straight": -1.0, "left": 0.0, "right": 0.0, "roundabout": 1.5, "uturn": 2.8}
GIVEN_LEAD_CHOICES = [6, 8, 10, 12, 14]      # lead times the app tried (exploration)
ADJUST_SECONDS = 2.0                          # backend/main.py ADJUST_SECONDS
DEFAULT_LEAD_S = 10.0
FEEDBACK_DEADZONE_S = 1.5                     # "just right" if within +/-1.5 s of ideal


def make_users(rng, n_users):
    users = []
    for i in range(n_users):
        users.append(dict(
            user_id=f"user_sim_{i:03d}",
            base_speed=float(np.clip(rng.normal(1.30, 0.18), 0.8, 1.8)),
            reaction_s=float(np.clip(rng.normal(7.8, 1.1), 5.0, 11.0)),
            pref_offset=float(rng.normal(0.0, 1.8)),
            erratic=float(np.clip(rng.normal(1.0, 0.35), 0.4, 2.0)),
            phone_checker=float(rng.uniform(0.05, 0.35)),
            label_noise=float(rng.uniform(0.5, 1.2)),
        ))
    return users


def simulate_maneuver(rng, u, profile_lead):
    turn = str(rng.choice(TURN_TYPES, p=TURN_P))

    stopped = rng.random() < u["phone_checker"]
    stop_dur = float(np.clip(rng.gamma(2.0, 2.5), 1, 25)) if stopped else 0.0
    acc = float(np.clip(rng.lognormal(np.log(4.5), 0.45), 2.0, 25.0))

    mood = rng.normal(0, 0.12)
    speed = float(np.clip(u["base_speed"] + mood, 0.3, 2.2))
    if stopped:
        speed *= rng.uniform(0.45, 0.85)
    std10 = float(np.clip(rng.gamma(2.0, 0.045) * u["erratic"] * (1 + acc / 40), 0.01, 0.9))
    trend = float(rng.normal(0, 0.18) + (0.35 if stopped else 0))
    avg10 = float(np.clip(speed - 0.3 * trend + rng.normal(0, 0.05), 0.2, 2.2))
    avg5 = float(np.clip(0.5 * (speed + avg10) + rng.normal(0, 0.05), 0.2, 2.2))
    max10 = float(max(speed, avg10) + abs(rng.normal(0.15, 0.08)) + std10)
    samples10 = int(np.clip(rng.normal(10.5, 1.5), 3, 14))
    samples5 = int(np.clip(samples10 // 2 + rng.integers(-1, 2), 1, 8))
    coverage = float(np.clip(rng.normal(9.6, 0.8), 3, 10.5))
    dist = float(np.clip(rng.uniform(4, 45), 3, 60))

    resume = 2.0 * (1 - np.exp(-stop_dur / 6.0))
    gps_pen = 0.12 * max(0.0, acc - 4.0)
    speed_term = 2.0 * (avg10 - 1.3)
    slow_u_turn = 1.2 if (turn == "uturn" and avg10 < 1.0) else 0.0
    ideal = (u["reaction_s"] + TURN_COMPLEXITY_S[turn] + resume + 5.4 * std10
             + gps_pen + speed_term + slow_u_turn + u["pref_offset"]
             + rng.normal(0, u["label_noise"]))
    ideal = float(np.clip(ideal, 3, 20))

    given = int(rng.choice(GIVEN_LEAD_CHOICES))
    diff = ideal - given
    if diff > FEEDBACK_DEADZONE_S:
        fb = "too_late"
    elif diff < -FEEDBACK_DEADZONE_S:
        fb = "too_early"
    else:
        fb = "just_right"

    return dict(
        user_id=u["user_id"], turn_type=turn, accuracy_m=round(acc, 2),
        current_speed_mps=round(speed, 3), avg_speed_5s_mps=round(avg5, 3),
        avg_speed_10s_mps=round(avg10, 3), speed_std_10s_mps=round(std10, 3),
        max_speed_10s_mps=round(max10, 3), speed_trend_10s_mps=round(trend, 3),
        speed_samples_5s=samples5, speed_samples_10s=samples10,
        speed_history_coverage_s=round(coverage, 2), distance_to_turn_m=round(dist, 2),
        stop_duration_s=round(stop_dur, 2), user_profile_lead_s=round(profile_lead, 2),
        lead_time_s=given, feedback=fb,
        preferred_lead_time_s=round(ideal, 2), turn_outcome="reached",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/simulated_maneuvers.csv")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    rows = []
    for u in make_users(rng, a.users):
        profile = DEFAULT_LEAD_S            # starts at 10 s, like the real app
        for s in range(int(rng.integers(3, 7))):
            sid = f"S_sim_{u['user_id'][-3:]}_{s:02d}"
            for m in range(int(rng.integers(6, 16))):
                r = simulate_maneuver(rng, u, profile)
                delta = {"too_early": -ADJUST_SECONDS, "just_right": 0.0, "too_late": ADJUST_SECONDS}[r["feedback"]]
                profile = float(np.clip(profile + delta, 3, 30))   # same rule as the backend
                r.update(session_id=sid, maneuver_id=f"M{m+1:03d}")
                rows.append(r)
    df = pd.DataFrame(rows)
    first = ["user_id", "session_id", "maneuver_id"]
    df = df[first + [c for c in df.columns if c not in first]]
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows, {df.user_id.nunique()} users, "
          f"{df.session_id.nunique()} sessions -> {out}")
    print(df.feedback.value_counts().to_dict())
    print(df.preferred_lead_time_s.describe().round(2).to_dict())


if __name__ == "__main__":
    main()
