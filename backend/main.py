import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pymongo import MongoClient, ASCENDING
from .route_engine import route as build_custom_route
import math
import json

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
load_dotenv(BASE_DIR / ".env")

MONGODB_URI = os.getenv("MONGODB_URI", "")
DB_NAME = os.getenv("DB_NAME", "smart_navigation")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")

app = FastAPI(title="Personalized Navigation API", version="2.0.0")

allowed_origins = [x.strip() for x in FRONTEND_ORIGIN.split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins or ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=8000) if MONGODB_URI else None
db = client[DB_NAME] if client is not None else None
profiles = db["user_profiles"] if db is not None else None
feedbacks = db["feedback"] if db is not None else None
sessions = db["sessions"] if db is not None else None
trajectories = db["trajectory_points"] if db is not None else None

DEFAULT_LEAD_SECONDS = 10
MIN_LEAD_SECONDS = 3
MAX_LEAD_SECONDS = 30
ADJUST_SECONDS = 2
ML_ARTIFACT = PROJECT_DIR / "ml_model" / "artifacts" / "lead_time_model.joblib"


class LeadTimePredictionIn(BaseModel):
    user_id: str = Field(min_length=8, max_length=100)
    turn_type: Literal["left", "right", "straight", "uturn", "roundabout"] = "straight"
    current_speed_mps: float = Field(default=0, ge=0, le=15)
    avg_speed_5s_mps: float = Field(default=0, ge=0, le=15)
    avg_speed_10s_mps: float = Field(default=0, ge=0, le=15)
    speed_std_10s_mps: float = Field(default=0, ge=0, le=15)
    max_speed_10s_mps: float = Field(default=0, ge=0, le=15)
    speed_trend_10s_mps: float = Field(default=0, ge=-15, le=15)
    speed_samples_5s: int = Field(default=0, ge=0, le=1000)
    speed_samples_10s: int = Field(default=0, ge=0, le=2000)
    speed_history_coverage_s: float = Field(default=0, ge=0, le=60)
    distance_to_turn_m: float = Field(default=0, ge=0, le=100000)
    stop_duration_s: float = Field(default=0, ge=0, le=86400)
    # Demo only: lets /demo simulate "what if this user had given feedback" without a database.
    profile_lead_override_s: Optional[float] = Field(default=None, ge=3, le=30)


class FeedbackIn(BaseModel):
    user_id: str = Field(min_length=8, max_length=100)
    feedback: Literal["too_early", "just_right", "too_late"]
    lead_time_seconds: float = Field(ge=0, le=120)
    walking_speed_mps: float = Field(default=0, ge=0, le=15)
    distance_to_turn_m: float = Field(default=0, ge=0, le=100000)
    stop_duration_seconds: float = Field(default=0, ge=0, le=86400)
    turn_type: Literal["left", "right", "straight", "uturn", "roundabout"] = "straight"
    session_id: str = Field(default="", max_length=100)
    maneuver_id: str = Field(default="", max_length=100)
    note: str = Field(default="", max_length=1000)


class SessionIn(BaseModel):
    user_id: str = Field(min_length=8, max_length=100)
    route_segment_id: int = Field(default=0, ge=0)
    destination: str = Field(default="", max_length=500)


class TrajectoryPoint(BaseModel):
    timestamp: str
    participant: str = Field(min_length=8, max_length=100)
    session_id: str = Field(min_length=1, max_length=100)
    route_segment_id: int = Field(default=0, ge=0)
    maneuver_id: str = Field(default="", max_length=100)
    event: str = Field(default="gps_sample", max_length=50)
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    accuracy_m: float = Field(default=0, ge=0, le=10000)
    speed_mps: float = Field(default=0, ge=0, le=15)
    distance_to_turn_m: float = Field(default=0, ge=0, le=100000)
    stop_duration_s: float = Field(default=0, ge=0, le=86400)
    lead_time_s: float = Field(default=10, ge=0, le=120)
    turn_type: Literal["left", "right", "straight", "uturn", "roundabout"] = "straight"
    turn_text: str = Field(default="", max_length=200)


class TrajectoryBatch(BaseModel):
    user_id: str = Field(min_length=8, max_length=100)
    session_id: str = Field(min_length=1, max_length=100)
    points: list[TrajectoryPoint] = Field(min_length=1, max_length=100)


def require_db():
    if any(x is None for x in (profiles, feedbacks, sessions, trajectories)):
        raise HTTPException(status_code=503, detail="Database is not configured. Set MONGODB_URI.")
    try:
        client.admin.command("ping")
    except Exception:
        raise HTTPException(status_code=503, detail="Could not connect to database.")


@app.on_event("startup")
def create_indexes():
    if profiles is not None:
        profiles.create_index([("user_id", ASCENDING)], unique=True)
        feedbacks.create_index([("user_id", ASCENDING), ("created_at", ASCENDING)])
        sessions.create_index([("session_id", ASCENDING)], unique=True)
        sessions.create_index([("user_id", ASCENDING), ("started_at", ASCENDING)])
        trajectories.create_index([("session_id", ASCENDING), ("timestamp", ASCENDING)])
        trajectories.create_index([("user_id", ASCENDING), ("maneuver_id", ASCENDING)])


@app.get("/", include_in_schema=False)
def serve_website():
    index_file = PROJECT_DIR / "website" / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=500, detail="Website file was not found.")
    return FileResponse(index_file)


@app.get("/demo", include_in_schema=False)
def serve_demo():
    f = PROJECT_DIR / "website" / "demo.html"
    if not f.exists():
        raise HTTPException(status_code=404, detail="demo.html not found")
    return FileResponse(f)


@app.get("/health")
def health():
    return {"status": "ok", "database_configured": bool(MONGODB_URI)}




ROUTE_DATA_FILE = PROJECT_DIR / "data" / "vnit_pedestrian_map.json"
try:
    with open(ROUTE_DATA_FILE, "r", encoding="utf-8") as _f:
        VNIT_ROUTE_DATA = json.load(_f)
except Exception:
    VNIT_ROUTE_DATA = {"points": [], "paths": []}

@app.post("/api/route")
def custom_route(payload: dict):
    try:
        start = payload.get("start") or {}
        destination = payload.get("destination") or {}
        slat, slon = float(start["lat"]), float(start["lng"])
        dlat, dlon = float(destination["lat"]), float(destination["lng"])
    except Exception:
        raise HTTPException(status_code=400, detail="start and destination must contain numeric lat/lng.")
    route, error = build_custom_route(VNIT_ROUTE_DATA, (slat, slon), (dlat, dlon))
    if error:
        raise HTTPException(status_code=404, detail=error)
    route["engine"] = "custom_kml_vnit_astar"
    return route

@app.get("/api/map-data")
def map_data():
    return VNIT_ROUTE_DATA

@app.post("/api/predict-lead-time")
def predict_lead_time(item: LeadTimePredictionIn):
    """Predict instruction lead time with a trained artifact when available.

    Until a trained model exists, use the existing feedback-updated user profile.
    If a model exists, blend its prediction with the user's profile preference so
    online user feedback remains useful without treating user_id as a numeric feature.
    """
    profile_lead = DEFAULT_LEAD_SECONDS
    if profiles is not None:
        try:
            profile = profiles.find_one({"user_id": item.user_id}, {"_id": 0})
            if profile:
                profile_lead = float(profile.get("preferred_lead_time_seconds", DEFAULT_LEAD_SECONDS))
        except Exception:
            pass
    if item.profile_lead_override_s is not None:
        profile_lead = float(item.profile_lead_override_s)
    model_lead = None
    source = "feedback_profile_fallback"
    if ML_ARTIFACT.exists():
        try:
            import joblib
            import pandas as pd
            artifact = joblib.load(ML_ARTIFACT)
            features = {
                "current_speed_mps": item.current_speed_mps,
                "avg_speed_5s_mps": item.avg_speed_5s_mps,
                "avg_speed_10s_mps": item.avg_speed_10s_mps,
                "speed_std_10s_mps": item.speed_std_10s_mps,
                "max_speed_10s_mps": item.max_speed_10s_mps,
                "speed_trend_10s_mps": item.speed_trend_10s_mps,
                "speed_samples_5s": item.speed_samples_5s,
                "speed_samples_10s": item.speed_samples_10s,
                "speed_history_coverage_s": item.speed_history_coverage_s,
                "distance_to_turn_m": item.distance_to_turn_m,
                "stop_duration_s": item.stop_duration_s,
                "user_profile_lead_s": profile_lead,
                "turn_type": item.turn_type,
            }
            model_lead = float(artifact["model"].predict(pd.DataFrame([features]))[0])
            # Blend the global feature model with the explicit per-user preference.
            if "user_profile_lead_s" in artifact.get("numeric_features", []):
                # The model already takes the user's profile as an input feature,
                # so blending again would double-count it.
                lead = model_lead
                source = "ml_model_with_user_profile_feature"
            else:
                lead = 0.5 * model_lead + 0.5 * profile_lead
                source = "ridge_model_plus_user_profile"
        except Exception:
            lead = profile_lead
            source = "feedback_profile_fallback_model_error"
    else:
        lead = profile_lead
    lead = max(MIN_LEAD_SECONDS, min(MAX_LEAD_SECONDS, float(lead)))
    return {
        "user_id": item.user_id,
        "lead_time_seconds": round(lead, 2),
        "model_prediction_seconds": round(model_lead, 2) if model_lead is not None else None,
        "profile_preference_seconds": round(profile_lead, 2),
        "source": source,
        "model_available": ML_ARTIFACT.exists(),
    }


@app.get("/api/profile/{user_id}")
def get_profile(user_id: str):
    require_db()
    profile = profiles.find_one({"user_id": user_id}, {"_id": 0})
    if not profile:
        profile = {
            "user_id": user_id,
            "preferred_lead_time_seconds": DEFAULT_LEAD_SECONDS,
            "feedback_count": 0,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        profiles.update_one({"user_id": user_id}, {"$setOnInsert": profile}, upsert=True)
    return profile


@app.post("/api/sessions")
def create_session(item: SessionIn):
    require_db()
    now = datetime.now(timezone.utc).isoformat()
    session_id = "S_" + uuid4().hex
    record = {
        "session_id": session_id,
        "user_id": item.user_id,
        "route_segment_id": item.route_segment_id,
        "destination": item.destination,
        "started_at": now,
        "ended_at": None,
        "status": "active",
    }
    sessions.insert_one(record)
    return {"session_id": session_id, "started_at": now}


@app.post("/api/sessions/{session_id}/end")
def end_session(session_id: str):
    require_db()
    now = datetime.now(timezone.utc).isoformat()
    result = sessions.update_one(
        {"session_id": session_id},
        {"$set": {"ended_at": now, "status": "ended"}},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"session_id": session_id, "ended_at": now, "status": "ended"}


@app.post("/api/trajectory")
def save_trajectory(batch: TrajectoryBatch):
    require_db()
    session = sessions.find_one({"session_id": batch.session_id, "user_id": batch.user_id})
    if not session:
        raise HTTPException(status_code=404, detail="Navigation session not found.")

    now = datetime.now(timezone.utc).isoformat()
    records = []
    for point in batch.points:
        record = point.model_dump()
        record["user_id"] = batch.user_id
        record["session_id"] = batch.session_id
        record["received_at"] = now
        records.append(record)
    if records:
        trajectories.insert_many(records)
    return {"saved": len(records), "session_id": batch.session_id}


@app.post("/api/feedback")
def submit_feedback(item: FeedbackIn):
    require_db()
    now = datetime.now(timezone.utc).isoformat()
    profiles.update_one(
        {"user_id": item.user_id},
        {"$setOnInsert": {
            "user_id": item.user_id,
            "preferred_lead_time_seconds": DEFAULT_LEAD_SECONDS,
            "feedback_count": 0,
            "created_at": now,
        }},
        upsert=True,
    )
    if item.session_id:
        session = sessions.find_one({"session_id": item.session_id, "user_id": item.user_id})
        if not session:
            raise HTTPException(status_code=404, detail="Navigation session not found.")

    profile = profiles.find_one({"user_id": item.user_id})
    current = float(profile.get("preferred_lead_time_seconds", DEFAULT_LEAD_SECONDS))

    # Keep one feedback record per maneuver. Re-submitting feedback for the same
    # maneuver updates that record instead of creating duplicate training rows.
    has_maneuver_key = bool(item.session_id and item.maneuver_id)
    feedback_key = {"user_id": item.user_id}
    if has_maneuver_key:
        feedback_key.update({"session_id": item.session_id, "maneuver_id": item.maneuver_id})
    existing = feedbacks.find_one(feedback_key) if has_maneuver_key else None

    feedback_delta = {"too_early": -ADJUST_SECONDS, "just_right": 0, "too_late": ADJUST_SECONDS}
    old_delta = feedback_delta.get(existing.get("feedback"), 0) if existing else 0
    new_delta = feedback_delta[item.feedback]
    updated = max(MIN_LEAD_SECONDS, min(MAX_LEAD_SECONDS, current + new_delta - old_delta))

    record = item.model_dump()
    record["created_at"] = existing.get("created_at", now) if existing else now
    record["updated_at"] = now
    if has_maneuver_key:
        feedbacks.update_one(feedback_key, {"$set": record}, upsert=True)
    else:
        feedbacks.insert_one(record)

    is_new_feedback = existing is None
    profile_update = {"$set": {
        "preferred_lead_time_seconds": updated,
        "updated_at": now,
    }}
    if is_new_feedback:
        profile_update["$inc"] = {"feedback_count": 1}
    profiles.update_one({"user_id": item.user_id}, profile_update)
    return {
        "user_id": item.user_id,
        "previous_lead_time_seconds": current,
        "preferred_lead_time_seconds": updated,
        "feedback_count": int(profile.get("feedback_count", 0)) + (1 if is_new_feedback else 0),
        "feedback_record_updated": not is_new_feedback,
        "message": "Maneuver feedback saved.",
    }


@app.get("/api/export/feedback")
def export_feedback():
    require_db()
    rows=list(feedbacks.find({}, {"_id":0}).sort("created_at", ASCENDING))
    return {"count":len(rows),"records":rows}

FRONTEND_DIR = PROJECT_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")
