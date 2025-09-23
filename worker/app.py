#!/usr/bin/env python3
"""Cloud Run worker service for processing video analysis jobs.

Receives Pub/Sub push messages containing `{ "jobId": "..." }` in the data payload.
Looks up the job in Firestore, downloads the video from Firebase Storage, runs
analysis via `analysis.shooting_drill_feedback.analyze_drill`, then generates summaries using
`agents.parent_feedback_agent.generate_summary_with_gemini` and
`agents.improvement_coach_agent.generate_sections`.

Environment variables:
- PB_STORAGE_BUCKET: GCS bucket name for Firebase Storage (e.g., your-project.appspot.com)
- GOOGLE_API_KEY: API key for Gemini (via Secret Manager)

Notes:
- Uses /tmp for local working directory and sets PB_WORK_DIR for analysis.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import pathlib
import tempfile
from typing import Any, Dict

from fastapi import FastAPI, Request, Response

from google.cloud import firestore  # type: ignore
from google.cloud import storage  # type: ignore

# Import local analysis modules from repository
from analysis.shooting_drill_feedback import analyze_drill  # type: ignore
from agents.parent_feedback_agent import generate_summary_with_gemini  # type: ignore
from agents.improvement_coach_agent import generate_sections  # type: ignore


app = FastAPI()


def _update_job(doc_ref: Any, data: Dict[str, Any]) -> None:
    """Safely update a Firestore document with server timestamp."""
    data["updatedAt"] = firestore.SERVER_TIMESTAMP
    doc_ref.set(data, merge=True)


def _download_video_to_tmp(storage_path: str, bucket_name: str) -> str:
    """Download a GCS object to a local temporary file and return its path."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(storage_path)
    if not blob.exists():
        raise FileNotFoundError(f"GCS object not found: gs://{bucket_name}/{storage_path}")
    fd, local_path = tempfile.mkstemp(prefix="pb_video_", suffix=pathlib.Path(storage_path).suffix)
    os.close(fd)
    blob.download_to_filename(local_path)
    return local_path


def _process_job(job_id: str) -> None:
    """Execute the full pipeline for a jobId."""
    bucket_name = os.environ.get("PB_STORAGE_BUCKET")
    if not bucket_name:
        raise RuntimeError("PB_STORAGE_BUCKET env var not set")

    fs = firestore.Client()
    doc_ref = fs.collection("jobs").document(job_id)
    snap = doc_ref.get()
    if not snap.exists:
        raise KeyError(f"Job not found: {job_id}")
    job = snap.to_dict() or {}

    storage_path = job.get("storagePath")
    user_id = job.get("userId")
    if not storage_path or not user_id:
        raise ValueError("Job missing required fields: storagePath or userId")

    logging.info("Processing job %s for user %s from %s", job_id, user_id, storage_path)
    _update_job(doc_ref, {"status": "processing", "progress": 10})

    # Prepare work dir
    work_dir = os.path.join(tempfile.gettempdir(), "pb_work")
    os.makedirs(work_dir, exist_ok=True)
    os.environ["PB_WORK_DIR"] = work_dir

    # Download video
    local_video = _download_video_to_tmp(storage_path, bucket_name)
    logging.info("Downloaded to %s", local_video)

    try:
        # Analysis
        _update_job(doc_ref, {"progress": 40})
        drill_result = analyze_drill(local_video)
        _update_job(doc_ref, {"progress": 70, "drill": drill_result})

        # Summaries
        _update_job(doc_ref, {"status": "summarizing", "progress": 80})
        parent_summary = generate_summary_with_gemini(drill_result)
        coach_summary = generate_sections(drill_result)

        _update_job(
            doc_ref,
            {
                "parent_summary": parent_summary,
                "coach_summary": coach_summary,
                "status": "completed",
                "progress": 100,
            },
        )
        logging.info("Job %s completed", job_id)

    except Exception as exc:  # noqa: BLE001
        logging.exception("Job %s failed: %s", job_id, exc)
        _update_job(doc_ref, {"status": "failed", "error": str(exc)})
        raise


@app.post("/")
async def pubsub_push(request: Request) -> Response:
    """Pub/Sub push endpoint. Accepts JSON envelope.

    Expected body:
    {
      "message": {
         "data": base64("{\"jobId\": \"...\"}")
      }
    }
    """
    envelope = await request.json()
    message = envelope.get("message", {}) if isinstance(envelope, dict) else {}
    data_b64 = message.get("data")
    if not data_b64:
        return Response(status_code=204)
    payload_raw = base64.b64decode(data_b64)
    try:
        payload = json.loads(payload_raw.decode("utf-8"))
    except Exception:
        logging.error("Invalid payload: %s", payload_raw)
        return Response(status_code=204)

    job_id = payload.get("jobId")
    if not job_id:
        logging.error("Missing jobId in payload")
        return Response(status_code=204)

    _process_job(job_id)
    return Response(status_code=204)


if __name__ == "__main__":
    import uvicorn  # type: ignore

    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)


