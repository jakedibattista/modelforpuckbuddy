# Hockey Shooting Drill Feedback

Runtime: MediaPipe Pose + OpenCV + FFmpeg.
Enhanced form analysis: head position, upper body square, lower body triangle.
Agents: data analysis, coaching feedback, and conversational AI coach (Gemini + Google Search).

## Project layout
```text
modelforpuckbuddy/
  analysis/                # Pose analysis
    __init__.py
    pose_extraction_shooting_drills.py  # analyze_drill(video_path) → dict
  agents/                  # AI agents for analysis and coaching
    __init__.py
    data_summary_agent.py    # structured per-shot data analysis
    seth_shooting_agent.py   # 2-section coaching feedback
    openice_agent.py         # conversational AI coach with Google Search
  worker/                  # Cloud Run worker services
    app.py                 # Original worker for Pub/Sub processing
    app_signed_urls.py     # Signed URL worker for secure processing
  functions/               # Firebase Functions (enqueue + cleanup)
    index.js
    package.json
  firebase/                # Firebase rules
    firestore.rules
    storage.rules
  utils/                   # Shared helpers
    config.py              # load_env(), get_required_env()
    io.py                  # load_json_file()
    firebase_storage.py    # Firebase Storage Manager with signed URLs
  tests/                   # Testing utilities
    test_complete_signed_url_workflow.py # End-to-end signed URL test
  helpfuldocs/             # Guides and architecture notes
    API_GUIDE.md           # Complete API integration guide
    IOS_INTEGRATION.md     # iOS/React Native implementation guide
    ARCHITECTURE.md        # System design and workflow details
    HOCKEY_VIDEO_TESTING_GUIDE.md  # Local testing and development
  signed_url_api.py        # Flask backend API for signed URL endpoints
  app.py                   # Cloud Run entrypoint for backend API
  requirements.txt         # Dependencies for Cloud Run deployment
  firebase.json            # Firebase config
  README.md, LICENSE
  videos/                  # Local inputs (gitignored)
  results/                 # Local outputs (gitignored)
```

## Quick start (local)
1) Create a virtual environment and install deps you need for running agents locally:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install google-genai python-dotenv
```

2) Provide your Gemini key (one of):
```bash
echo "GOOGLE_API_KEY=YOUR_KEY" > .env     # preferred
# or
export GOOGLE_API_KEY=YOUR_KEY
```

3) Run agents against an existing analysis JSON:
```bash
# Core analysis agents
python -m agents.data_summary_agent results/drill/kidshoot4_drill_feedback.json
python -m agents.seth_shooting_agent results/drill/kidshoot4_drill_feedback.json

# OpenIce conversational coach
python -m agents.openice_agent --analysis-file results/drill/kidshoot4_drill_feedback.json --question "How can I shoot like Connor McDavid?"
```

To generate an analysis JSON from a local video, install analysis dependencies (see `worker/requirements.txt` for reference: mediapipe, opencv-python-headless, numpy, etc.) and run:
```bash
python -c "from analysis.pose_extraction_shooting_drills import analyze_drill; import json; print(json.dumps(analyze_drill('videos/input/your_clip.mov'), indent=2))"
```

## Production Deployment
The backend API is deployed to Cloud Run and ready for production use:

**Backend API URL**: `https://puck-buddy-model-22317830094.us-central1.run.app`

### Available Endpoints:

**Core Video Analysis:**
- `GET /health` - Service health check
- `POST /api/upload-url` - Generate signed URL for video upload
- `POST /api/analyze-video` - ⭐ **Simple**: MediaPipe pose analysis + structured data summary
- `POST /api/submit-video` - Create analysis job (advanced workflow)
- `GET /api/results/{user_id}` - List user's analysis results (advanced workflow)

**Coaching Feedback:**
- `GET /api/coaches` - List available coaching personalities
- `POST /api/coach/seth` - Get Seth's technical coaching feedback

**OpenIce AI Coach (Optional):**
- `POST /api/start-chat` - Create conversational coaching session with analysis data
- `POST /api/ask-question` - Ask questions about technique, get personalized advice
- `GET /api/chat-info/<session_id>` - Get chat session information
- `POST /api/openice/init` - Client-compatible: Initialize OpenIce session with immediate response
- `POST /api/openice/chat` - Client-compatible: Send chat messages to OpenIce

**Job Management & Cleanup:**
- `POST /api/job/complete` - Mark job as completed and clean up from queue
- `POST /api/jobs/cleanup` - Clean up old jobs (completed, failed, or stale)

### Testing the deployment:
```bash
# Health check
curl https://puck-buddy-model-22317830094.us-central1.run.app/health

# Test signed URL generation (requires Firebase service account setup)
curl -X POST https://puck-buddy-model-22317830094.us-central1.run.app/api/upload-url \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test-user","filename":"test.mp4"}'
```

## Quick Integration Guide

### Simple Approach (Recommended for most apps):
```bash
# Step 1: Get upload URL
curl -X POST https://puck-buddy-model-22317830094.us-central1.run.app/api/upload-url \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test123","content_type":"video/mov"}'

# Step 2: Upload video using returned URL, then analyze
curl -X POST https://puck-buddy-model-22317830094.us-central1.run.app/api/analyze-video \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test123","storage_path":"users/test123/..."}'

# Step 3 (Optional): Get coaching feedback  
curl -X POST https://puck-buddy-model-22317830094.us-central1.run.app/api/coach/seth \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test123","raw_analysis":"[analysis_data_from_step_2]"}'
```

### OpenIce AI Coach Integration (Optional):
```bash
# Method 1: Traditional endpoints
curl -X POST https://puck-buddy-model-22317830094.us-central1.run.app/api/start-chat \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test123","analysis_data":"**Shots detected at timestamps:** 00:08, 00:15..."}'

curl -X POST https://puck-buddy-model-22317830094.us-central1.run.app/api/ask-question \
  -H "Content-Type: application/json" \
  -d '{"session_id":"your-session-id","question":"How can I shoot like Connor McDavid?"}'

# Method 2: Client-compatible endpoints (with CORS support)
curl -X POST https://puck-buddy-model-22317830094.us-central1.run.app/api/openice/init \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test123","analysis_data":"**Shots detected at timestamps:** 00:08, 00:15..."}'

curl -X POST https://puck-buddy-model-22317830094.us-central1.run.app/api/openice/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"session-id","question":"What drill should I practice?"}'
```

### Coaching Feedback Integration:
```bash
# List available coaches
curl -X GET https://puck-buddy-model-22317830094.us-central1.run.app/api/coaches

# Get Seth's coaching feedback (requires raw_analysis from analyze-video)
curl -X POST https://puck-buddy-model-22317830094.us-central1.run.app/api/coach/seth \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test123","raw_analysis":{"shots":[{"shot_time_sec":8.2,"knee_bend_min_deg":95}]}}'
```

### Job Management & Cleanup:
```bash
# Clean up completed jobs for a user (prevents queue accumulation)
curl -X POST https://puck-buddy-model-22317830094.us-central1.run.app/api/job/complete \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test123"}'

# Clean up old jobs (completed, failed, or stale after 24 hours)
curl -X POST https://puck-buddy-model-22317830094.us-central1.run.app/api/jobs/cleanup \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test123","max_age_hours":24}'
```

### Advanced Integration:
```bash
# Test complete signed URL workflow  
cd tests
python3 test_complete_signed_url_workflow.py "test_user_123" "../videos/input/kidshoot2.MOV"
```

**📖 Documentation**: See `helpfuldocs/API_GUIDE.md` for complete integration examples including OpenIce.

## Cloud components
- **Backend API** (`signed_url_api.py`) deployed to Cloud Run provides signed URL endpoints + OpenIce chat endpoints
- **Worker services** (`worker/app_signed_urls.py`) handle video processing with signed URLs
- **AI Agents**: data analysis, coaching feedback, and OpenIce conversational coach (Gemini + Google Search)
- Firebase Functions (2nd gen) in `functions/` can enqueue jobs and clean old data
- Cloud Run handles video analysis via `analysis.analyze_drill` and agent summaries

## Docs
- See `helpfuldocs/HOCKEY_VIDEO_TESTING_GUIDE.md` for local testing tips.
- See `helpfuldocs/ARCHITECTURE.md` for end-to-end design.
