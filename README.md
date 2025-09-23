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
  worker/                  # Cloud Run HTTP worker (Pub/Sub push)
    app.py
    Dockerfile
    requirements.txt
  functions/               # Firebase Functions (enqueue + cleanup)
    index.js
    package.json
  firebase/                # Firebase rules
    firestore.rules
    storage.rules
  utils/                   # Shared helpers
    config.py              # load_env(), get_required_env()
    io.py                  # load_json_file()
  tests/                   # Testing utilities
    test_firebase_video.py # Full pipeline test with Firebase videos
    test_signed_url_workflow.py # Complete signed URL workflow test
    get_firebase_url.py    # Firebase Storage URL generator
  helpfuldocs/             # Guides and architecture notes
    HOCKEY_VIDEO_TESTING_GUIDE.md
    ARCHITECTURE.md        # Supports both direct Firebase and signed URL workflows
    IOS_INTEGRATION.md     # iOS implementation for both approaches
    FIREBASE_SIGNED_URL_INTEGRATION.md  # Production integration guide
    MIGRATION_TO_SIGNED_URLS.md        # Migration from current to signed URL approach
    QUESTIONS_FOR_CONSIDERATION.md
  setup_firebase_admin.py # Firebase Admin SDK setup helper
  firebase.json            # Firebase config (kept at repo root)
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

## Firebase Signed URL Integration
For production app integration with secure video uploads and result delivery:

```bash
# Setup Firebase Admin SDK
python3 setup_firebase_admin.py

# Test complete signed URL workflow
cd tests
python3 test_signed_url_workflow.py "test_user_123" "../videos/input/kidshoot2.MOV"
```

## Testing Firebase videos
To test the complete pipeline with a video from Firebase Storage:
```bash
cd tests
python3 get_firebase_url.py "users/your-uid/video.mov"
export GOOGLE_API_KEY=your_key
python3 test_firebase_video.py "https://firebasestorage.googleapis.com/v0/b/project.appspot.com/o/video.mov?alt=media"
```
See `helpfuldocs/FIREBASE_SIGNED_URL_INTEGRATION.md` for complete integration guide.

## Cloud components
- Firebase Functions (2nd gen) in `functions/` enqueue jobs and clean old data.
- Cloud Run `worker/` handles Pub/Sub push, downloads video, runs `analysis.analyze_drill`, then calls agents for summaries, and writes results to Firestore.

## Docs
- See `helpfuldocs/HOCKEY_VIDEO_TESTING_GUIDE.md` for local testing tips.
- See `helpfuldocs/ARCHITECTURE.md` for end-to-end design.
