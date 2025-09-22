# Hockey Drill Feedback

Runtime: MediaPipe Pose + OpenCV + FFmpeg.
Agents: parent per-shot report, improvement coach sections (Gemini).

## Project layout
```text
modelforpuckbuddy/
  analysis/                # Pose analysis
    __init__.py
    drill_feedback.py      # analyze_drill(video_path) → dict
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
  helpfuldocs/             # Guides and architecture notes
    HOCKEY_VIDEO_TESTING_GUIDE.md
    ARCHITECTURE.md
    IOS_INTEGRATION.md
    QUESTIONS_FOR_CONSIDERATION.md
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
python -c "from analysis.drill_feedback import analyze_drill; import json; print(json.dumps(analyze_drill('videos/input/your_clip.mov'), indent=2))"
```

## Cloud components
- Firebase Functions (2nd gen) in `functions/` enqueue jobs and clean old data.
- Cloud Run `worker/` handles Pub/Sub push, downloads video, runs `analysis.analyze_drill`, then calls agents for summaries, and writes results to Firestore.

## Docs
- See `helpfuldocs/HOCKEY_VIDEO_TESTING_GUIDE.md` for local testing tips.
- See `helpfuldocs/ARCHITECTURE.md` for end-to-end design.
