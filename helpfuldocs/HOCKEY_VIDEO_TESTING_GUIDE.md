# 🏒 Hockey Video Testing Guide

Quick guide to test the PuckBuddy video analysis system locally.

## Setup

### 1. Install Dependencies
```bash
pip install google-genai python-dotenv
```

### 2. Set API Key
```bash
echo "GOOGLE_API_KEY=your_key_here" > .env
```

### 3. Test Sample Videos
Sample videos are in `videos/input/` - try with `kidshoot4.mov` for best results.

---

## Testing APIs

### Test Video Analysis
```bash
# Basic analysis (MediaPipe + data summary)
curl -X POST https://puck-buddy-model-22317830094.us-central1.run.app/api/analyze-video \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test123","storage_path":"users/test123/videos/test.mov"}'
```

### Test Seth Coaching
```bash
# Get coaching feedback (requires analysis data)
curl -X POST https://puck-buddy-model-22317830094.us-central1.run.app/api/coach/seth \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test123","raw_analysis":{"shots":[{"shot_time_sec":8.2,"knee_bend_min_deg":95}]}}'
```

### Test Available Coaches
```bash
# List coaching options
curl -X GET https://puck-buddy-model-22317830094.us-central1.run.app/api/coaches
```

---

## Local Testing

### Run Analysis Locally
```bash
# Test data summary agent
python -m agents.data_summary_agent results/drill/kidshoot4_drill_feedback.json

# Test Seth coaching agent  
python -m agents.seth_shooting_agent results/drill/kidshoot4_drill_feedback.json

# Test OpenIce conversational agent
python -m agents.openice_agent --analysis-file results/drill/kidshoot4_drill_feedback.json --question "How can I improve my shot?"
```

### Expected Outputs

**Data Summary:**
```
**Shots detected at timestamps:** 00:08, 00:15, 00:23

**Shot 1: 00:08:**
**head position:** good (0.78)
**wrist steadiness:** smooth (0.85)
**hip drive:** 0.650 (good)
**front knee angle:** 95°
**back leg angle:** 165°
```

**Seth Coaching:**
```
**What went well:**
- Strong hip drive on shot at 00:08 — driving power through the puck!
- Excellent front knee bend at 00:08 (95°) — getting low for power!

**What to work on:**
- Get lower on your front knee — aim for 100-110° bend
- Drive hips forward more aggressively — push through the puck!
```

---

## Good Test Videos

### What Works Well
- **Side angle**: Player filmed from the side
- **Full body visible**: Head to skates in frame
- **Multiple shots**: 3-5 shots in 20-30 seconds
- **Clear movement**: Player clearly winds up and shoots
- **Good lighting**: Indoor rink lighting is fine

### What Doesn't Work
- **Front/back angles**: Can't see leg bend properly
- **Partial body**: Missing arms, legs, or stick
- **Too few shots**: Less than 2 clear shots
- **Too fast**: Player rushing through motions
- **Dark/blurry**: Poor video quality

### Sample Test Videos
In the `videos/input/` folder:
- `kidshoot4.mov` - **Best example** (4 shots, good angle)
- `kidshoot2.mov` - Good (3 shots, clear motion)
- `sethshoot.MOV` - OK (fewer shots but clear)

---

## Common Issues

### "No shots detected"
- **Fix**: Ensure player makes clear, deliberate shooting motions
- **Check**: Is the stick and puck visible?
- **Try**: Record 5+ shots with clear wind-up

### "Analysis failed"
- **Fix**: Check video format (.mov or .mp4)
- **Check**: Video under 100MB
- **Try**: Re-record with better lighting

### "Coaching seems wrong"
- **Fix**: This is normal - AI coaching is based on detected poses
- **Check**: Was the player's form actually good/bad in the video?
- **Try**: Record clearer examples of good technique

---

## Quick Debugging

### Check API Health
```bash
curl https://puck-buddy-model-22317830094.us-central1.run.app/health
```

### Check Processing Time
- Upload URL: instant
- Video analysis: 2-3 minutes
- Seth coaching: 15 seconds
- OpenIce chat: 5-15 seconds

### Check Response Format
All successful responses have `"success": true` and relevant data. Errors return helpful messages with specific guidance.

---

## What Gets Measured

The system tracks these key hockey shooting metrics:

- **Shot timing**: When each shot happens (timestamps)
- **Knee bend**: How low the player gets (90-110° is ideal)
- **Hip drive**: Power generation through hips (0-1 scale)
- **Wrist control**: Smoothness of stick handling
- **Head position**: Eyes on target
- **Body alignment**: Square to target

All measurements are relative to good hockey shooting technique and provided in simple, coaching-friendly language.