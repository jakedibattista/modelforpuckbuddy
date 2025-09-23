## PuckBuddy Processing Architecture (Option A)

This document describes the end-to-end design to let the iOS app send videos for analysis and receive feedback using the current Python modules: `analysis/shooting_drill_feedback.py`, `agents/parent_feedback_agent.py`, and `agents/improvement_coach_agent.py`.

### Goals
- **Reliability**: Avoid client timeouts; handle retries; isolate heavy compute.
- **Simplicity**: Minimal changes to current Python code; containerized worker.
- **Real-time UX**: iOS listens to Firestore job document for status/results.
- **Ephemeral**: Results are returned once; no long-term retention required.

## High-level Flow
1. iOS uploads video to Firebase Storage at `users/{uid}/{uuid}.mov`.
2. iOS creates a Firestore job doc: `jobs/{jobId}` with status `queued`.
3. A small Cloud Function (2nd gen) listens for `jobs.onCreate` and publishes a message to Pub/Sub topic `process-video` with `jobId`.
4. Pub/Sub pushes the message to a Cloud Run service (Python worker container).
5. Worker downloads the video from Storage, runs analysis and summaries, and writes updates back to `jobs/{jobId}` (`status`, `progress`, `results`).
6. iOS listens to `jobs/{jobId}` in real-time and renders updates; on completion, shows text outputs. An optional cleanup removes the job after a short TTL.


## TL;DR of high-level flow (technical)
Client (iOS)
  Uploads video to Storage at users/{uid}/{uuid}.mov.
  Creates Firestore doc jobs/{jobId} with:
    userId, storagePath, status='queued', progress=0, createdAt/updatedAt.
Cloud Function (enqueue)
  Trigger: on create of jobs/{jobId}, if status=='queued'.
  Publishes { jobId } to Pub/Sub topic process-video.
Cloud Run worker
  Receives Pub/Sub push with { jobId }.
  Loads Firestore doc, sets status='processing', progress≈10.
  Downloads video from Storage, sets PB_WORK_DIR, runs `analysis.shooting_drill_feedback.analyze_drill()`.
  Writes drill results into jobs/{jobId}.drill and progress≈70.
  Generates summaries with agents (status='summarizing', progress≈80).
  Writes parent_summary, coach_summary, then status='completed', progress=100.
Client (iOS)
  Listens to jobs/{jobId}.
  Renders progress and, when completed, reads parent_summary, coach_summary, and drill.

## Data Model (Firestore)
Collection: `jobs`
- `userId: string` — Firebase Auth UID of the requester
- `storagePath: string` — Firebase Storage path, e.g., `videos/{uid}/{uuid}.mov`
- `status: string` — one of `queued|processing|summarizing|completed|failed`
- `progress: number` — 0..100
- `options: map` — optional runtime knobs, e.g., `{ stride: 2, width: 960 }`
- `drill: map` — result of `analyze_drill` (subset)
- `parent_summary: string` — text from `parent_feedback_agent`
- `coach_summary: string` — text from `improvement_coach_agent`
- `error: string` — present only on `failed`
- `createdAt: timestamp`, `updatedAt: timestamp`

Example document snapshot:
```json
{
  "userId": "abc123",
  "storagePath": "videos/abc123/9c9e...-clip.mov",
  "status": "processing",
  "progress": 70,
  "options": { "stride": 2, "width": 960 },
  "drill": {
    "fps": 30.0,
    "duration_est_sec": 5.2,
    "shots": [ { "shot_time_sec": 1.2, "knee_bend_min_deg": 125.0, "hip_drive": 0.41, "control_smoothness": 0.55 } ]
  },
  "parent_summary": "2 shots detected: ...",
  "coach_summary": "What went well:\n- ...\n\nWhat to work on:\n- ...",
  "createdAt": { ".sv": "timestamp" },
  "updatedAt": { ".sv": "timestamp" }
}
```

## Components

### iOS app
- Upload video to Storage under `videos/{uid}/{uuid}.mov`.
- Create `jobs/{jobId}` with `status=queued`.
- Set up a Firestore snapshot listener to render status and final results.

Minimal Swift outline:
```swift
import FirebaseAuth
import FirebaseFirestore
import FirebaseStorage

struct Job: Codable {
  let userId: String
  let storagePath: String
  let status: String
  let createdAt: Timestamp
}

func startProcessing(localUrl: URL, uid: String, completion: @escaping (String) -> Void) {
  let bucketPath = "videos/\(uid)/\(UUID().uuidString).mov"
  let ref = Storage.storage().reference(withPath: bucketPath)
  ref.putFile(from: localUrl, metadata: nil) { _, err in
    if let err = err { print("upload error: \(err)"); return }
    let jobRef = Firestore.firestore().collection("jobs").document()
    let job: [String: Any] = [
      "userId": uid,
      "storagePath": bucketPath,
      "status": "queued",
      "progress": 0,
      "createdAt": FieldValue.serverTimestamp(),
      "updatedAt": FieldValue.serverTimestamp()
    ]
    jobRef.setData(job) { err in
      if let err = err { print("job create error: \(err)"); return }
      completion(jobRef.documentID)
    }
  }
}

func observeJob(jobId: String, onUpdate: @escaping ([String: Any]) -> Void) -> ListenerRegistration {
  let docRef = Firestore.firestore().collection("jobs").document(jobId)
  return docRef.addSnapshotListener { snap, err in
    guard let data = snap?.data(), err == nil else { return }
    onUpdate(data)
  }
}
```

### Cloud Function (enqueue)
- 2nd gen Firestore trigger on `jobs/{jobId}` `onCreate`.
- Validates `status == queued` and publishes `{ jobId }` to Pub/Sub topic `process-video`.
- Keeps this function lightweight; worker does the heavy lifting.

### Pub/Sub → Cloud Run (worker)
- Pub/Sub push subscription targets Cloud Run HTTPS endpoint.
- Cloud Run service verifies Pub/Sub auth header and processes the job.

Worker responsibilities:
- Resolve `jobs/{jobId}`; fetch `storagePath` and `userId`.
- Download video to `/tmp` from Firebase Storage (GCS) using service account.
- Normalize via ffmpeg (same as `run_ffmpeg_normalize`).
- Call `analyze_drill(local_path)` to get metrics.
- Generate `parent_summary` with `generate_summary_with_gemini`.
- Generate `coach_summary` with `generate_sections`.
- Update Firestore during phases: `processing` (10→80%), `summarizing` (80→95%), `completed` (100%).
- On failure: set `status=failed` with an `error` message.
- Optional: schedule deletion of the job after N minutes.

## Security & Access Control

### Firestore Rules (sketch)
```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /jobs/{jobId} {
      allow create: if request.auth != null
        && request.resource.data.userId == request.auth.uid
        && request.resource.data.status == 'queued';
      allow read, update, delete: if request.auth != null
        && resource.data.userId == request.auth.uid;
    }
  }
}
```

### Firebase Storage Rules (sketch)
```javascript
rules_version = '2';
service firebase.storage {
  match /b/{bucket}/o {
    match /users/{uid}/{file} {
      allow write: if request.auth != null && request.auth.uid == uid;
      allow read: if false; // end-users do not read raw videos
    }
  }
}
```

### Service accounts & secrets
- Cloud Run service account needs roles: `Storage Object Viewer`, `Datastore User` (Firestore), `Secret Manager Secret Accessor` (for `GOOGLE_API_KEY`).
- Pub/Sub Invoker for calling Cloud Run.
- Store `GOOGLE_API_KEY` in Secret Manager, injected as env var.

## Deployment Steps (summary)
1. Enable APIs: Cloud Run, Cloud Functions v2, Pub/Sub, Secret Manager, Firestore, Storage.
2. Create Pub/Sub topic `process-video` and push subscription to Cloud Run URL (after deploy).
3. Build and deploy Cloud Run worker:
   - Dockerfile: Python base, `apt-get install -y ffmpeg`, `pip install mediapipe opencv-python-headless numpy google-genai google-cloud-storage google-cloud-firestore`.
   - Set memory (e.g., 2–4 GB), CPU (2 vCPU), concurrency 1–2, timeout 15m.
   - Mount `GOOGLE_API_KEY` from Secret Manager.
4. Deploy Firestore-triggered Cloud Function (2nd gen) to publish to Pub/Sub on `jobs.onCreate`.
5. Set IAM bindings and verify Pub/Sub → Cloud Run authenticated invocation.
6. Test end-to-end with a small sample video.

## Operational Considerations
- Logging: include `jobId` and `userId` fields in all logs.
- Retries: enable at-least-once processing; use idempotent updates.
- Performance: keep `width=960`, `stride=2` as defaults; consider `fast` option for larger stride.
- Cleanup: daily job deletes completed/failed `jobs` and uploaded user videos older than 24 hours (ephemeral policy).

## Mapping to Current Code
- Use `analyze_drill(video_path)` from `analysis.shooting_drill_feedback`.
- Use `generate_summary_with_gemini(raw)` from `agents.parent_feedback_agent`.
- Use `generate_sections(raw)` from `agents.improvement_coach_agent`.
- Avoid writing local files in worker; write results to Firestore instead.

## iOS Listener Notes
- Subscribe to `jobs/{jobId}` after creation and render:
  - `status` and `progress` for a progress bar.
  - When `completed`, display `parent_summary` and `coach_summary`.
- Optionally call a Cloud Function to mark job for deletion once shown.

---
This architecture prioritizes reliability and a smooth user experience while reusing your existing Python analysis logic.


