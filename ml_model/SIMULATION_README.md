# Simulated dataset + model comparison

**All numbers here come from SIMULATED data. They prove the pipeline works, not real-world accuracy.**

## Run order (from `ml_model/`)
```bash
pip install -r requirements.txt matplotlib
python simulation/generate_dataset.py          # -> data/simulated_maneuvers.csv (60 users, ~2,700 maneuvers)
python train_compare.py                        # -> artifacts/lead_time_model.joblib + reports/*.png|json
python train_compare.py --no-profile --out artifacts/_tmp.joblib   # ablation: drop personalization feature
```
Restart the FastAPI server afterwards; `/api/predict-lead-time` picks the artifact up automatically.

## Results (5-fold GroupKFold by user = evaluated on people the model never saw)
| Model | MAE (s) | R2 | within ±1.5 s |
|---|---|---|---|
| Baseline (mean) | 2.04 | -0.06 | 44% |
| **Ridge** | **1.29** | 0.57 | 66% |
| Random Forest | 1.35 | 0.52 | 64% |
| Gradient Boosting | 1.33 | 0.54 | 65% |
Without the `user_profile_lead_s` feature, every model falls to MAE ≈ 1.9 s.

## Real-data caveat
Replace `data/simulated_maneuvers.csv` with real exports (same columns) as soon as you have them; the target column
`preferred_lead_time_s` does not exist in real data, so use the existing weak target in `train_model.py`
or collect repeated trials.
