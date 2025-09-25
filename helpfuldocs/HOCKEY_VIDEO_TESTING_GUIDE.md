# 🏒 Hockey Shooting Drill Feedback - Testing Guide (MediaPipe Pose)

## Overview

This guide helps you run the MediaPipe Pose-based analyzer to measure hockey shooting drills: shot detection, knee bend, hip drive, and control smoothness.

## Dependencies

- OpenCV (`opencv-python`) for video I/O (reading frames, color conversion)
- MediaPipe (`mediapipe`) for pose landmarks
- NumPy (`numpy`) for numeric processing
- FFmpeg (system binary) for video normalization

Note: The runtime stack is MediaPipe Pose + OpenCV + FFmpeg.

## Metrics Measured

### Core Body Position Metrics at Shot Release:
- **Shots detected** with timestamps
- **Front knee bend**: angle during shot (degrees), score (0..1) - should be 90-110° for power
- **Back leg extension**: should be 160-180° for drive through shot
- **Hip drive**: normalized (0..1) with good/bad flag - measures forward drive
- **Wrist steadiness**: label from pull-back stability (smooth / mixed / jerky)

### Enhanced Form Analysis:
- **Head position**: forward lean, eye level consistency, target facing
- **Upper body square**: shoulder level, arm extension, target alignment  
- **Lower body triangle**: front knee bend, back leg extension (no stance width tracking)

## Explanation of Outputs (Plain English)

This section explains how we create the JSON and what each number means, without technical jargon.

1) We prepare the video
- We first pass the video through FFmpeg so every clip has similar frame rate and size. This does not change the content, it just makes it easier to analyze.

2) We find body points in every frame
- We use MediaPipe Pose to locate key points like shoulders, hips, knees, ankles, and wrists. Think of it as dots on the body that move frame to frame.

3) We find shots by wrist speed “bursts”
- We measure how fast the wrist is moving. A shot shows up as a sharp burst of wrist speed.
- We only count a burst as a shot if it’s: (a) clearly above normal movement, (b) at least ~2 seconds apart from the last shot, and (c) not tiny (it must be at least 20% of the fastest wrist speed in the clip). This avoids counting small wiggles as shots.

4) We look at two short time windows around each shot
- Control window (before the shot): When the wrist is moving back. We allow brief pauses so it’s not too strict.
- Shot window (the release): The moment right around the wrist speed burst.

5) What each metric means

### Core Position Metrics:
- **Front knee bend** (shooting leg)
  - We measure the front knee angle during shot release. Smaller angle = deeper bend. Example: 100° is "deep", 168° is "shallow".
  - Good shooting form: 90-110°. Score from 0 to 1: 1.0 = very deep (≤100°), 0.0 = very shallow (≥140°).
- **Back leg extension** (support leg)  
  - We measure how straight the back leg is during shot release. Should be 160-180° for proper drive.
  - Angles <150° indicate "too bent" - player not driving through properly.
- **Hip drive**
  - We measure forward hip movement during shot release. Scale 0 to 1 based on movement in the clip.
  - "hip_drive_good: true" if ≥0.3. Higher = more power through the shot.
- **Wrist steadiness**
  - How steady the wrist moves during pull-back. Smooth setup = better shot consistency.
  - We present this to users as a label: "smooth" (≥0.6), "mixed" (0.31–0.59), or "jerky" (≤0.3).

### Enhanced Form Analysis:
- **Head position**: Measures forward lean, eye level consistency, and target-facing direction
- **Upper body square**: Shoulder level, arm extension, and target alignment for proper shooting form
- **Lower body triangle**: Front knee bend and back leg extension for optimal power transfer

6) How to read the JSON at a glance
- "shots" is a list where each item is one shot with its time and metrics.
- **Core metrics**: `shot_time_sec`, `hip_drive` (≥0.3 = good), `control_smoothness` (generates wrist steadiness label)
- **Form analysis**: `head_position.*`, `upper_body_square.*`, `lower_body_triangle.front_knee_bend_deg` & `back_leg_extension_deg`
- **Great rep indicators**: front knee ≤110°, back leg 160-180°, hip drive ≥0.3, head/upper body metrics ≥0.8

## Step-by-Step Testing Instructions

### Step 1: Environment Setup

```bash
pip install mediapipe opencv-python numpy
# Ensure ffmpeg is installed on your system (brew install ffmpeg on macOS)
```

### Step 2: Prepare Your Hockey Video

```bash
# Place your hockey video in the input folder
cp /path/to/your/hockey_video.mp4 videos/input/
```

**Supported formats**: MP4, AVI, MOV, MKV, WMV
**Recommended**: 
- Video length: 30 seconds to 5 minutes (for meaningful analysis)
- Resolution: 720p or higher
- Clear view of the action (not too zoomed out)

### Step 3: Run the Analysis (Drill Feedback)

```bash
# Run analyze_drill from the packaged module
python -c "from analysis.shooting_drill_feedback import analyze_drill; import json; print(json.dumps(analyze_drill('videos/input/your_clip.mov'), indent=2))"
```

The script will automatically:
1. 🎥 Normalize video with FFmpeg (fps/scale/codec)
2. 🧍 Extract pose landmarks with MediaPipe Pose
3. 🕒 Detect shots via wrist-speed peaks (with robust thresholds)
4. 🦵 Compute knee bend during shot (+ score and validity)
5. 🧠 Measure hip drive (forward-only, normalized, with good/bad flag)
6. 📈 Compute control smoothness from wrist-speed stability
7. 💾 Save JSON to `results/drill/`

### Step 4: Generate Reports (Agents)

We separate the end-user report into two parts using lightweight LLM agents:

- Parent per-shot report (times and metrics only)

```bash
python -m agents.data_summary_agent results/drill/<your_video>_drill_feedback.json
```

Output: First line with shot times, then one bullet per shot:
"time — front knee bend XXX°, hip drive H.HHH (good/not good), wrist steadiness: LABEL, head position: excellent/good/needs work, back leg: XXX°".

- Improvement coaching sections (What went well / What to work on)

```bash
python -m agents.seth_shooting_agent results/drill/<your_video>_drill_feedback.json
```

Output: Two sections with 2–3 concise, coach-like bullets each.

Note: Set `GOOGLE_API_KEY` in your environment or `.env` for the agents. We auto-load `.env` if present.

### Step 5: Test OpenIce AI Coach (Optional)

OpenIce is an intelligent conversational coach that combines your technical analysis with real-time hockey knowledge from web search.

#### Start a conversation with your analysis data:

```bash
python -m agents.openice_agent --analysis-file results/drill/<your_video>_drill_feedback.json --question "How can my shot look more like Connor McDavid?"
```

#### Example OpenIce Questions:

**Player Comparisons:**
```bash
python -m agents.openice_agent --analysis-file results/drill/kidshoot3_drill_feedback.json --question "How can I shoot like Sidney Crosby?"
```

**Practice Planning:**
```bash
python -m agents.openice_agent --analysis-file results/drill/kidshoot3_drill_feedback.json --question "What's the most important thing to work on this week?"
```

**Technique Analysis:**
```bash
python -m agents.openice_agent --analysis-file results/drill/kidshoot3_drill_feedback.json --question "Why is my wrist movement jerky in shot 3?"
```

**Follow-up Questions (using session ID):**
```bash
# After your first question, use the session ID for follow-ups:
python -m agents.openice_agent --session-id [SESSION_ID_FROM_FIRST_RESPONSE] --question "What drill should I focus on for that?"
```

#### OpenIce Output Example:
```
🏒 OpenIce Response:
Based on your data and McDavid's documented technique:

Your shots vs McDavid:
• Hip drive: Your 0.456 at 00:15 is good - McDavid emphasizes explosive rotation
• Knee bend: Your 142° needs work - McDavid gets down to 95-105° 
• Release time: Your 1.8s vs his lightning 0.9s

Focus on shot 2 (00:15) - you almost had McDavid's hip drive there!

🔍 Searched for: Connor McDavid shooting technique, McDavid knee bend mechanics
📚 Sources: NHL.com, Hockey Training Pro, Elite Prospects
```

### Step 6: Review Results

Results will be saved in `results/drill/`:

#### **Drill Feedback JSON** (example snippet)
```json
{
  "video": "hockeyshoot (1).mov",
  "fps": 30.0,
  "shots": [
    {
      "shot_time_sec": 10.133,
      "knee_bend_min_deg": 109.4,
      "knee_bend_score": 0.765,
      "hip_drive": 0.408,
      "hip_drive_good": true,
      "control_smoothness": 0.573,
      "head_position": {"forward_lean": 0.943, "eye_level": 0.991, "target_facing": 0.899},
      "upper_body_square": {"shoulder_level": 0.93, "arm_extension": 0.917, "target_alignment": 0.928},
      "lower_body_triangle": {"front_knee_bend_deg": 168.9, "back_leg_extension_deg": 151.9}
    }
  ]
}
