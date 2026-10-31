// Add this script to your existing navigation website.
// It stores a random anonymous ID in this browser and fetches that user's lead-time preference.
// Do not use a name, phone number, or email as the ID.

const API_BASE = window.SMART_NAV_API_BASE || window.location.origin;
const USER_ID_KEY = "smartNavAnonymousUserId";

function getOrCreateUserId() {
  let id = localStorage.getItem(USER_ID_KEY);
  if (!id) {
    id = "user_" + crypto.randomUUID();
    localStorage.setItem(USER_ID_KEY, id);
  }
  return id;
}

async function getPersonalizedProfile() {
  const userId = getOrCreateUserId();
  const response = await fetch(`${API_BASE}/api/profile/${encodeURIComponent(userId)}`);
  if (!response.ok) throw new Error("Could not load personal profile");
  return response.json();
}

// Call this when the user taps a feedback button.
// feedback must be: "too_early", "just_right", or "too_late".
// leadTimeSeconds is the lead time used for the instruction the user is rating.
async function sendNavigationFeedback({
  feedback,
  leadTimeSeconds,
  walkingSpeedMps = 0,
  distanceToTurnM = 0,
  stopDurationSeconds = 0,
  turnType = "straight",
  sessionId = "",
  maneuverId = "",
  note = "",
}) {
  const userId = getOrCreateUserId();
  const response = await fetch(`${API_BASE}/api/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      user_id: userId,
      feedback,
      lead_time_seconds: Number(leadTimeSeconds),
      walking_speed_mps: Number(walkingSpeedMps),
      distance_to_turn_m: Number(distanceToTurnM),
      stop_duration_seconds: Number(stopDurationSeconds),
      turn_type: turnType,
      session_id: sessionId,
      maneuver_id: maneuverId,
      note: String(note || "").slice(0, 1000),
    }),
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(`Feedback could not be saved: ${message}`);
  }
  return response.json();
}

// Example integration:
// const profile = await getPersonalizedProfile();
// let leadTimeSeconds = profile.preferred_lead_time_seconds; // Starts at 10 for a new user
//
// document.querySelector("#tooEarly").addEventListener("click", async () => {
//   const result = await sendNavigationFeedback({
//     feedback: "too_early",
//     leadTimeSeconds,
//     walkingSpeedMps: currentSpeedMps,
//     distanceToTurnM: currentDistanceToTurnM,
//     stopDurationSeconds: currentStopDurationSeconds,
//     turnType: currentTurnType,
//   });
//   leadTimeSeconds = result.preferred_lead_time_seconds;
// });


// Ask the backend for a feature-aware lead-time prediction. If the model has not
// been trained yet, the API returns the existing feedback-updated profile value.
async function getPredictedLeadTime(features = {}) {
  const userId = getOrCreateUserId();
  const response = await fetch(`${API_BASE}/api/predict-lead-time`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId, ...features }),
  });
  if (!response.ok) throw new Error("Could not predict instruction timing");
  return response.json();
}
