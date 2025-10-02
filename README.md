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

### Security & Rate Limits

**🔒 Security:**
- Firebase Authentication required for all video operations
- Signed URLs with 1-hour expiration for secure uploads
- Private storage (users can only access their own files)
- Automatic cleanup of old files (30 days)

**⏱️ Rate Limits (Per User):**
- **Video Analysis**: 10 videos/hour (main cost driver)
- **Upload URLs**: 20 requests/hour
- **All Endpoints**: 200 requests/day, 50 requests/hour
- **Response**: HTTP 429 when limit exceeded

### Available Endpoints:

**Core Video Analysis:**
- `GET /health` - Service health check
- `POST /api/upload-url` - Generate signed URL for video upload
- `POST /api/analyze-video` - Analyze video with MediaPipe pose detection (~2 minutes)

**Coaching Feedback:**
- `GET /api/coaches` - List available coaching personalities
- `POST /api/coach/seth` - Get Seth's technical coaching feedback

**OpenIce AI Coach:**
- `POST /api/openice/init` - Initialize OpenIce coaching session
- `POST /api/openice/chat` - Chat with AI coach about technique
- `GET /api/chat-info/<session_id>` - Get chat session information


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

**Three simple steps to analyze a hockey video:**

```bash
# Step 1: Get upload URL
curl -X POST https://puck-buddy-model-22317830094.us-central1.run.app/api/upload-url \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test123","content_type":"video/mov"}'

# Step 2: Upload video using returned URL (use PUT with video file)

# Step 3: Analyze video (waits ~2 min, returns complete results)
curl -X POST https://puck-buddy-model-22317830094.us-central1.run.app/api/analyze-video \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test123","storage_path":"users/test123/..."}'

# Optional: Get coaching feedback  
curl -X POST https://puck-buddy-model-22317830094.us-central1.run.app/api/coach/seth \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test123","raw_analysis":"[analysis_data_from_step_3]"}'
```


### OpenIce AI Coach (Optional):
```bash
# Initialize AI coaching session
curl -X POST https://puck-buddy-model-22317830094.us-central1.run.app/api/openice/init \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test123","analysis_data":"**Shots detected at timestamps:** 00:08, 00:15..."}'

# Ask follow-up questions
curl -X POST https://puck-buddy-model-22317830094.us-central1.run.app/api/openice/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"session-id","question":"What drill should I practice?"}'
```

**📖 Complete Documentation**: See `helpfuldocs/API_GUIDE.md` for detailed integration guide and examples.

## Cloud components
- **Backend API** (`signed_url_api.py`) deployed to Cloud Run - handles video uploads, analysis, and AI coaching
- **Video Analysis Engine** (`analysis/`) - MediaPipe pose detection for hockey technique analysis
- **AI Agents** (`agents/`) - Data summary, Seth coaching, and OpenIce conversational coach (Gemini)
- **Firebase Storage** - Secure video storage with automatic 30-day cleanup
- **Rate Limiting** - Per-user limits to ensure fair usage (10 videos/hour)

## Docs
- See `helpfuldocs/HOCKEY_VIDEO_TESTING_GUIDE.md` for local testing tips.
- See `helpfuldocs/ARCHITECTURE.md` for end-to-end design.
