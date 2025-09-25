## PuckBuddy Processing Architecture

This document describes the end-to-end design to let the iOS app send videos for analysis and receive feedback using the current Python modules: `analysis/shooting_drill_feedback.py`, `agents/data_summary_agent.py`, `agents/seth_shooting_agent.py`, and the new `agents/openice_agent.py` for conversational AI coaching.

### Goals
- **Reliability**: Avoid client timeouts; handle retries; isolate heavy compute.
- **Simplicity**: Minimal changes to current Python code; containerized worker.
- **Real-time UX**: iOS listens to Firestore job document for status/results.
- **Security**: Use signed URLs for secure file access without exposing credentials.
- **Scalability**: Support both Firestore-only and signed URL result delivery options.
- **Ephemeral**: Results are returned once; no long-term retention required.

## Production Architecture ✅ DEPLOYED

**Backend API**: `https://puck-buddy-model-22317830094.us-central1.run.app`
**Cloud Functions**: ✅ **ACTIVE** - `enqueueJob`, `notifyOnCompletion`, `cleanupOldJobs`
**Pub/Sub Topic**: ✅ **ACTIVE** - `process-video`

### Simple Workflow (Recommended)
1. **App requests upload URL**: `POST /api/upload-url`
2. **Backend generates signed URL** for `users/{uid}/{timestamp}_{filename}` (1 hour expiration)
3. **App uploads video** directly to Firebase Storage using signed URL
4. **App requests analysis**: `POST /api/analyze-video` (waits for completion, 5 min timeout)
5. **Backend processes video**: Downloads, runs pose analysis, generates AI summaries
6. **Backend returns results**: Data analysis + coach summary as plain text in JSON response
7. **Optional**: **App starts OpenIce chat**: `POST /api/start-chat` with analysis data for conversational coaching

### Advanced Workflow (For Progress Updates)
1-3. Same as simple workflow (upload)
4. **App submits job**: `POST /api/submit-video` (returns immediately with job_id)
5. **App polls for results**: `GET /api/results/{user_id}` (includes progress updates)
6. **Backend uploads results** to Firebase Storage with signed download URLs
7. **App downloads results** using signed URLs when job completes


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
- `storagePath: string` — Firebase Storage path, e.g., `users/{uid}/{timestamp}_{filename}.mov`
- `status: string` — one of `queued|processing|summarizing|completed|failed`
- `progress: number` — 0..100
- `options: map` — optional runtime knobs, e.g., `{ stride: 2, width: 960 }`
- `delivery_method: string` — `firestore` (default) or `signed_urls`
- `drill: map` — result of `analyze_drill` (subset) - only if delivery_method is `firestore`
- `data_analysis: string` — text from `data_summary_agent` (structured shot data) - only if delivery_method is `firestore`
- `coach_summary: string` — text from `seth_shooting_agent` (coaching feedback) - only if delivery_method is `firestore`
- `result_urls: map` — signed download URLs for results - only if delivery_method is `signed_urls`
  - `analysis_url: string` — signed URL for analysis JSON (24h expiration)
  - `parent_summary_url: string` — signed URL for parent summary text (24h expiration)  
  - `coach_summary_url: string` — signed URL for coach analysis text (24h expiration)
- `error: string` — present only on `failed`
- `createdAt: timestamp`, `updatedAt: timestamp`

Example document snapshots:

**Option A: Firestore Results (Current)**
```json
{
  "userId": "abc123",
  "storagePath": "users/abc123/20241201_143022_hockey_drill.mov",
  "status": "completed",
  "progress": 100,
  "delivery_method": "firestore",
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

**Option B: Signed URL Results (Recommended)**
```json
{
  "userId": "abc123",
  "storagePath": "users/abc123/20241201_143022_hockey_drill.mov",
  "status": "completed", 
  "progress": 100,
  "delivery_method": "signed_urls",
  "options": { "stride": 2, "width": 960 },
  "result_urls": {
    "analysis_url": "https://storage.googleapis.com/puck-buddy.appspot.com/users/abc123/results/20241201_143022/analysis.json?X-Goog-Algorithm=...",
    "parent_summary_url": "https://storage.googleapis.com/puck-buddy.appspot.com/users/abc123/results/20241201_143022/parent_summary.txt?X-Goog-Algorithm=...",
    "coach_summary_url": "https://storage.googleapis.com/puck-buddy.appspot.com/users/abc123/results/20241201_143022/coach_analysis.txt?X-Goog-Algorithm=..."
  },
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

## Signed URL Integration (Option B)

### Backend API Endpoints
For signed URL workflow, add these endpoints to your backend:

```python
# GET /api/upload-url
# Generate signed URL for video upload
@app.route('/api/upload-url', methods=['POST'])
def generate_upload_url():
    data = request.get_json()
    user_id = data['user_id']
    filename = data['filename']
    
    from utils.firebase_storage import FirebaseStorageManager
    storage_manager = FirebaseStorageManager()
    
    upload_url, storage_path = storage_manager.generate_upload_url(
        user_id=user_id, filename=filename, expiration_minutes=60
    )
    
    return {
        "upload_url": upload_url,
        "storage_path": storage_path,
        "expires_in": 3600
    }

# GET /api/results/{user_id}
# Get analysis results with signed URLs
@app.route('/api/results/<user_id>')
def get_user_results(user_id):
    storage_manager = FirebaseStorageManager()
    results = storage_manager.list_user_results(user_id)
    
    # Generate fresh download URLs
    for result_group in results:
        if result_group['files']:
            storage_paths = {
                filename.split('.')[0]: file_info['storage_path']
                for filename, file_info in result_group['files'].items()
            }
            download_urls = storage_manager.generate_results_download_urls(
                storage_paths, expiration_minutes=60
            )
            result_group['download_urls'] = download_urls
    
    return {"results": results}
```

### Worker Integration
Update Cloud Run worker to support signed URLs:

```python
def process_job_with_signed_urls(job_data):
    from utils.firebase_storage import FirebaseStorageManager
    storage_manager = FirebaseStorageManager()
    
    # Generate signed download URL for video
    video_download_url = storage_manager.generate_download_url(
        job_data['storagePath'], expiration_minutes=30
    )
    
    # Download and process video
    with tempfile.NamedTemporaryFile(suffix=".mov") as tmp_file:
        urllib.request.urlretrieve(video_download_url, tmp_file.name)
        analysis_result = analyze_drill(tmp_file.name)
        parent_summary = generate_summary_with_gemini(analysis_result)
        coach_analysis = generate_sections(analysis_result)
    
    # Upload results to Firebase Storage
    results_paths = storage_manager.upload_analysis_results(
        user_id=job_data['userId'],
        video_filename=os.path.basename(job_data['storagePath']),
        analysis_data=analysis_result,
        parent_summary=parent_summary,
        coach_analysis=coach_analysis
    )
    
    # Generate signed URLs for results
    result_urls = storage_manager.generate_results_download_urls(
        results_paths, expiration_minutes=1440  # 24 hours
    )
    
    # Update job with result URLs
    job_ref.update({
        'status': 'completed',
        'progress': 100,
        'delivery_method': 'signed_urls',
        'result_urls': result_urls
    })
```

## Security & Access Control

### Firebase Storage Rules (Updated for Signed URLs)
```javascript
rules_version = '2';
service firebase.storage {
  match /b/{bucket}/o {
    match /users/{uid}/{allPaths=**} {
      // Authenticated users can write their own files
      allow write: if request.auth != null && request.auth.uid == uid;
      
      // Authenticated users can read their own files  
      allow read: if request.auth != null && request.auth.uid == uid;
      
      // Allow signed URL access (no authentication required for signed URLs)
      // This enables secure temporary access for video processing and result delivery
      allow read, write: if request.auth == null;
    }
  }
}
```

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

## AI Agents Architecture

### Core Analysis Agents (Used in Video Processing)
- **`data_summary_agent.py`**: Structured data analysis with timestamps and metrics
- **`seth_shooting_agent.py`**: Coaching feedback with "What went well" and "What to work on" sections

### OpenIce Conversational Agent (Optional Enhancement)
- **`agents/openice_agent.py`**: Intelligent conversational coach using Gemini with Google Search
- **Purpose**: Answer follow-up questions about technique, provide player comparisons, suggest drills
- **Integration**: Completely additive - existing workflows unchanged
- **Session Management**: In-memory chat sessions with cleanup (24h expiration)

### OpenIce API Endpoints
```
POST /api/start-chat       - Create chat session with analysis data
POST /api/ask-question     - Ask questions in existing session  
GET  /api/chat-info/<id>   - Get session information
```

### OpenIce Usage Flow
1. **After video analysis**: Client optionally creates OpenIce chat session with `data_analysis` result
2. **Conversational Q&A**: Client asks questions like "How can I shoot like McDavid?"
3. **Intelligent responses**: OpenIce provides personalized advice using web search + technical data
4. **Session persistence**: Chat memory maintained for follow-up questions

### OpenIce Features
- 🧠 **Contextual AI**: References specific shots and timestamps from analysis
- 🌐 **Real-time research**: Google Search integration for current hockey knowledge
- 🏒 **Player comparisons**: Compare technique to NHL players (McDavid, Crosby, Ovechkin)
- 📚 **Practice recommendations**: Specific drills and training methods
- 💬 **Conversation memory**: Maintains context across multiple questions

## Mapping to Current Code
- Use `analyze_drill(video_path)` from `analysis.shooting_drill_feedback`.
- Use `generate_summary_with_gemini(raw)` from `agents.data_summary_agent`.
- Use `generate_sections(raw)` from `agents.seth_shooting_agent`.
- Use `OpenIceAgent()` from `agents.openice_agent` for conversational coaching.
- Avoid writing local files in worker; write results to Firestore instead.

## iOS Listener Notes
- Subscribe to `jobs/{jobId}` after creation and render:
  - `status` and `progress` for a progress bar.
  - When `completed`, display `data_analysis` and `coach_summary`.
- **Optional OpenIce Integration**: After displaying results, offer "Ask OpenIce" feature to start conversational coaching.
- Optionally call a Cloud Function to mark job for deletion once shown.

---
This architecture prioritizes reliability and a smooth user experience while reusing your existing Python analysis logic.


