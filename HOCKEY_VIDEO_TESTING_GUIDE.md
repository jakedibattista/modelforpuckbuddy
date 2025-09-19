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

- Shots detected with timestamps
- Knee bend: min angle during shot (degrees), score (0..1), validity
- Hip drive: normalized (0..1) with good/bad flag
- Wrist steadiness (formerly "control"): label derived from wrist-speed stability (smooth / mixed / jerky)
- Follow-through (formerly "stick lift"): present, first_time_sec, peak_norm

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
- Knee bend
  - We look during the shot window. We estimate the knee angle in degrees (smaller angle = deeper bend). For example, 100° is “deep”, 150° is “shallow”.
  - We also give a score from 0 to 1: 1.0 means very deep bend (≤100°), 0.0 means very shallow (≥140°), with a smooth scale in between. “valid: true” means we saw enough knee points to trust the number.
- Hip drive
  - We measure how strongly the hips move forward at the shot moment. We convert it to a 0 to 1 scale by comparing to fast movements elsewhere in the clip.
  - We also mark “hip_drive_good: true” if the forward drive is strong (≥0.3). Higher is better.
- Wrist steadiness
  - We check how steady the wrist speed is during the control (pull-back) window.
  - We present this to users as a label: “smooth” (≥0.6), “mixed” (0.31–0.59), or “jerky” (≤0.3).
- Follow-through
  - We see if the wrist rises above the shoulder during the shot (a simple proxy for stick follow-through) by more than 15% of torso height.
  - We report whether it was present, when it first happened, and the “peak_norm” height (how high relative to your torso). Rough guide: ≥0.3 shows a clear follow-through.

6) How to read the JSON at a glance
- “shots” is a list where each item is one shot with its time and metrics.
- Look for: knee_bend_min_deg (lower = deeper), hip_drive (closer to 1 is stronger drive), control_smoothness (used internally to label “wrist steadiness”), and stick_lift.present (true/false).
- A great rep usually has: knee_bend_min_deg ≤ 110°, hip_drive ≥ 0.3, wrist steadiness = smooth, and a sensible follow-through.

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
# Run the drill feedback analyzer
python drill_feedback.py
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
python parent_feedback_agent.py results/drill/<your_video>_drill_feedback.json
```

Output: First line with shot times, then one bullet per shot:
“time — knee bend XXX°, hip drive H.HHH (good/not good), wrist steadiness: LABEL, follow-through: yes/no”.

- Improvement coaching sections (What went well / What to work on)

```bash
python improvement_coach_agent.py results/drill/<your_video>_drill_feedback.json
```

Output: Two sections with 2–3 concise, coach-like bullets each.

Note: Set `GOOGLE_API_KEY` in your environment or `.env` for the agents. We auto-load `.env` if present.

### Step 4: Review Results

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
      "knee_bend_valid": true,
      "hip_drive": 0.408,
      "hip_drive_good": true,
      "control_smoothness": 0.573,
      "stick_lift": {"present": true, "first_time_sec": 10.2, "peak_norm": 0.22}
    }
  ]
}
