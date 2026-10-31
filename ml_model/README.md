# Smart Navigation — First ML Model (Prototype)

This package sets up the first personalized instruction-timing model for the VNIT smart-walking navigation prototype.

## Important current-data status

The supplied dataset currently has 4 maneuver rows. The feedback field is blank in all 4 rows, and one maneuver is still pending. Therefore **there are currently zero usable feedback-labelled examples**. A supervised model cannot be honestly trained or evaluated from this dataset yet.

This package includes:
- a training script that builds a `StandardScaler + OneHotEncoder + Ridge` regression pipeline when enough feedback-labelled rows exist;
- a prediction script that uses the trained model when available and otherwise uses a transparent feedback-rule fallback;
- a copy of the current maneuver dataset and a CSV template for collecting feedback.

No synthetic training records or fabricated accuracy scores are used.

## How the weak target is constructed

The initial target is a *bootstrapping heuristic*, not ground truth:
- `too_early`: preferred lead time = current `lead_time_s - 2` seconds
- `just_right`: preferred lead time = current `lead_time_s`
- `too_late`: preferred lead time = current `lead_time_s + 2` seconds

Values are clamped to 3–20 seconds. Change `--adjustment-seconds` in the training command if you choose a different feedback adjustment step. After enough repeated feedback trials, replace this heuristic with a more defensible target based on repeated user-specific trials.

`actual_time_to_turn_s` is deliberately **not** used as the target. Actual time-to-turn and ETA error describe movement/estimation; they do not by themselves tell us whether the instruction timing felt right.

## Install

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

Put your collected CSV at `data/maneuver_records.csv`, then run:

```bash
python train_model.py --data data/maneuver_records.csv --out artifacts/lead_time_model.joblib
```

By default the script requires at least 8 labelled rows and 2 distinct sessions. These are only minimum safety checks, not a claim that 8 rows are enough for reliable real-world performance. For a real evaluation, collect substantially more varied data and split by session/user to avoid leakage.

## Predict

```bash
python predict_lead_time.py --model artifacts/lead_time_model.joblib \
  --user-id user_demo \
  --turn-type left \
  --current-speed-mps 1.2 \
  --avg-speed-5s-mps 1.1 \
  --avg-speed-10s-mps 1.0 \
  --speed-std-10s-mps 0.12 \
  --speed-trend-10s-mps 0.05 \
  --distance-to-turn-m 12 \
  --stop-duration-s 0
```

If no trained model exists, prediction falls back to the default lead time and available user feedback history. The result is clamped to 3–20 seconds. Integrate this prediction into the navigation app at the point where it decides when to speak an upcoming maneuver.

## Feedback collection

Record one row per maneuver. After the user experiences the instruction, update that same row with:
- `feedback`: exactly `too_early`, `just_right`, or `too_late`
- `feedback_timestamp`
- `turn_outcome`: `reached`, `missed`, `rerouted`, or `pending`
- `actual_time_to_turn_s` and `turn_reached_timestamp` only when the turn was actually reached

Do not count pending/unreached maneuvers as negative feedback. Do not create a second maneuver row just to store feedback; update the original row by `maneuver_id`/session.
