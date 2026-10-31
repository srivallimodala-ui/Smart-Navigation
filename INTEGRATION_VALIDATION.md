# Integration Validation Report

## Checks run on this build

- Python files compiled with `compileall`.
- Inline JavaScript blocks extracted from `website/index.html` passed `node --check`.
- `frontend/personalization.js` passed `node --check`.
- ZIP archive integrity check passed.
- API logic smoke test passed with a mocked MongoDB layer: health endpoint logic, bundled map-data loading, default prediction fallback, feedback creation, and updating feedback for an existing maneuver without creating a second row.
- Model inference smoke test passed using a temporary test-only Ridge artifact. The test artifact was removed and is **not** shipped.
- Trainer correctly refused to train the included dataset because it has no valid feedback labels.
- Trainer successfully generated a temporary test artifact from a synthetic smoke-test dataset. Those synthetic rows and that artifact are **not** shipped.

## Important limits

These checks are not proof of perfect operation. They did not connect to your real MongoDB instance, run the website in a real browser with GPS permission, walk a route outdoors, or verify spoken directions/rerouting on a phone. Please test those flows before relying on the application.

## Feedback and training compatibility

- API-export feedback uses `too_early`, `just_right`, `too_late`.
- Browser CSV download uses `early`, `right`, `late`; the trainer now normalizes these labels.
- When `session_id` and `maneuver_id` are both supplied, repeated feedback updates that maneuver's existing feedback record. Feedback without both identifiers is stored as a separate event because it cannot be safely associated with a unique maneuver.
- The training target is still an initial feedback-based heuristic (current lead time ± 2 seconds), not a measured ground-truth optimum. Do not interpret in-sample MAE as proof of real-world accuracy.
