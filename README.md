# Hockey Shooting Drill Feedback

Runtime: MediaPipe Pose + OpenCV + FFmpeg.
Enhanced form analysis: head position, upper body square, lower body triangle.
Agents: parent per-shot report, improvement coach sections (Gemini).

## Project layout
```text
modelforpuckbuddy/
  analysis/                # Pose analysis
    __init__.py
    shooting_drill_feedback.py  # analyze_drill(video_path) → dict
  agents/                  # Text-generation agents
    __init__.py
    parent_feedback_agent.py        # per-shot parent report
    improvement_coach_agent.py      # 2-section coach summary
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
python -m agents.improvement_coach_agent results/drill/kidshoot4_drill_feedback.json
python -m agents.parent_feedback_agent results/drill/kidshoot4_drill_feedback.json
```

To generate an analysis JSON from a local video, install analysis dependencies (see `worker/requirements.txt` for reference: mediapipe, opencv-python-headless, numpy, etc.) and run:
```bash
python -c "from analysis.shooting_drill_feedback import analyze_drill; import json; print(json.dumps(analyze_drill('videos/input/your_clip.mov'), indent=2))"
```

## Production Deployment
The backend API is deployed to Cloud Run and ready for production use:

**Backend API URL**: `https://puck-buddy-model-22317830094.us-central1.run.app`

### Available Endpoints:
- `GET /health` - Service health check
- `POST /api/upload-url` - Generate signed URL for video upload
- `POST /api/analyze-video` - ⭐ **Simple**: Upload video and get analysis results with MediaPipe pose detection
- `POST /api/submit-video` - Create analysis job (advanced workflow)
- `GET /api/results/{user_id}` - List user's analysis results (advanced workflow)

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
# Test the simple workflow
curl -X POST https://puck-buddy-model-22317830094.us-central1.run.app/api/upload-url \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test123","content_type":"video/mov"}'

# Upload video using returned URL, then:
curl -X POST https://puck-buddy-model-22317830094.us-central1.run.app/api/analyze-video \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test123","storage_path":"users/test123/..."}'
```

### Advanced Integration:
```bash
# Test complete signed URL workflow  
cd tests
python3 test_complete_signed_url_workflow.py "test_user_123" "../videos/input/kidshoot2.MOV"
```

**📖 Documentation**: See `helpfuldocs/API_GUIDE.md` for complete integration examples.

## Cloud components
- **Backend API** (`signed_url_api.py`) deployed to Cloud Run provides signed URL endpoints
- **Worker services** (`worker/app_signed_urls.py`) handle video processing with signed URLs
- Firebase Functions (2nd gen) in `functions/` can enqueue jobs and clean old data
- Cloud Run handles video analysis via `analysis.analyze_drill` and agent summaries

## Docs
- See `helpfuldocs/HOCKEY_VIDEO_TESTING_GUIDE.md` for local testing tips.
- See `helpfuldocs/ARCHITECTURE.md` for end-to-end design.
