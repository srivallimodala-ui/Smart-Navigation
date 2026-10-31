# VNIT Smart Walk — latest KML + reliable Build Route

This version uses the newly supplied `Untitled map (1)(1).kml` as the routing source.

## Routing
- Uses the supplied KML LineStrings as the pedestrian network.
- Exact geometric line intersections become graph junctions.
- Only very small endpoint/path gaps (3 m) are connected; arbitrary 10–25 m gaps are NOT bridged.
- Start/destination points can attach to existing mapped segments without connecting unrelated segments to each other.
- Destination POIs can finish at the nearest mapped pedestrian access point.
- A* is used for path selection.

## Build Route reliability
- The visible Start/Destination text fields are authoritative when **Build route** is pressed.
- Older route requests are cancelled/superseded and cannot overwrite the latest route.
- Route results are accepted only if they belong to the current route generation.
- Manual Build Route requests force a fresh route calculation, avoiding stale cached results after repeated destination changes.
- A destination change while GPS navigation is active can still trigger a new route from the latest GPS position.
- The Build route button is protected against accidental duplicate requests.
- Browser fallback uses the same latest KML data if the backend route endpoint is unavailable.

## Important behavior
If two KML paths are visually close but are not actually connected, the router will not invent a junction. If the supplied KML does not contain a walkable connection to a destination, the app reports that the network is disconnected instead of drawing a false route.

## Updating the map data
1. Export your My Maps as KML and save it as `data/Untitled_map.kml`.
2. Run `python tools/kml_to_json.py data/Untitled_map.kml`. It rewrites `data/vnit_pedestrian_map.json` (used by the server) and the `LOCAL_VNIT_POINTS/PATHS` constants in `website/index.html` (used by the browser fallback), keeping each Y pin's stem direction.
3. Restart the server and hard-refresh the page (Ctrl+F5).
Start and destination must be within 60 m of a mapped line; draw a path to any place (for example LAD Chowk Metro Station) that is farther away.


## ML personalization integration
- `ml_model/` contains the Ridge Regression trainer and inference scripts.
- The website periodically calls `POST /api/predict-lead-time` while navigating, using rolling walking-speed and turn-context features.
- If `ml_model/artifacts/lead_time_model.joblib` exists, the backend blends the model estimate with the MongoDB user's feedback-updated profile. If no trained artifact exists, the existing per-user feedback rule remains the fallback.
- The included dataset currently has no usable feedback labels, so **no trained model artifact is shipped**. Train only after collecting valid feedback-labelled records.
- To train from the backend, export feedback using `GET /api/export/feedback`, save the returned `records` as a CSV, then run `python ml_model/train_model.py --data path/to/feedback.csv --out ml_model/artifacts/lead_time_model.joblib`. Training stops if labels/session counts are insufficient.
- Install `ml_model/requirements.txt` (or project requirements) before training/inference. Restart the API after adding a model artifact.


## Integration validation
- The training script accepts both canonical API feedback labels (`too_early`, `just_right`, `too_late`) and browser CSV labels (`early`, `right`, `late`).
- Repeated feedback for a maneuver with the same `session_id` and `maneuver_id` updates the existing feedback record rather than adding a duplicate row.
- Automated syntax and smoke tests are documented in `INTEGRATION_VALIDATION.md`. A real MongoDB connection, browser permission flow, and live walking/GPS session still require testing in your environment.
