Place the latest collected maneuver-level CSV here as `maneuver_records.csv`.

Expected fields include:
`user_id`, `session_id`, `maneuver_id`, `lead_time_s`, `feedback`, `turn_type`,
`current_speed_mps`, `avg_speed_5s_mps`, `avg_speed_10s_mps`,
`speed_std_10s_mps`, `max_speed_10s_mps`, `speed_trend_10s_mps`,
`speed_samples_5s`, `speed_samples_10s`, `speed_history_coverage_s`,
`distance_to_turn_m`, and `stop_duration_s`.

`feedback_history_template.csv` is for the prediction script's lightweight per-user feedback adjustment. Fill one row per feedback event. Keep the detailed maneuver data in the main dataset.
