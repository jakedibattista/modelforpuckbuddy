# 🏒 Hockey Video Testing Guide

Quick guide to test the PuckBuddy video analysis system locally and via API.

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

## Testing Complete API Flow

### Full 3-Step Workflow Test

**Step 1: Get Upload URL**
```bash
curl -X POST https://puck-buddy-model-22317830094.us-central1.run.app/api/upload-url \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test123","content_type":"video/mov"}'
```

**Step 2: Upload Video**
```bash
# Use the upload_url from Step 1
curl -X PUT "[upload_url_from_step_1]" \
  -H "Content-Type: video/mov" \
  --data-binary "@videos/input/kidshoot4.mov"
```

**Step 3: Analyze Video**
```bash
# Use the storage_path from Step 1
curl -X POST https://puck-buddy-model-22317830094.us-central1.run.app/api/analyze-video \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test123","storage_path":"users/test123/videos/20240315_143022_shooting_drill.mov","age_group":"10-12"}'
```

**Expected wait time:** ~2 minutes for analysis to complete

### Test Seth Coaching
```bash
# Get coaching feedback (requires analysis data)
curl -X POST https://puck-buddy-model-22317830094.us-central1.run.app/api/coach/seth \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test123","data_analysis":{"shots":[{"shot_time_sec":8.2,"front_knee_bend_deg":95,"hip_rotation_power":{"max_rotation_speed":25.0},"wrist_extension":{"follow_through_score":80.0}}]}}'
```

### Test Available Coaches
```bash
# List coaching options
curl -X GET https://puck-buddy-model-22317830094.us-central1.run.app/api/coaches
```

### Test OpenIce AI Chat
```bash
# Initialize OpenIce session
curl -X POST https://puck-buddy-model-22317830094.us-central1.run.app/api/openice/init \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test123","analysis_data":"**Shots detected:** 3 shots at 00:08, 00:15, 00:23"}'

# Ask follow-up questions (use session_id from previous response)
curl -X POST https://puck-buddy-model-22317830094.us-central1.run.app/api/openice/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"session-abc123","question":"What drill should I practice?"}'
```

---

## Rate Limits & Security

### Rate Limits (Per User)
When testing, be aware of these limits:

| Action | Limit | Reset |
|--------|-------|-------|
| Upload URLs | 20/hour | Per hour |
| Video Analysis | **10/hour** | Per hour |
| All requests | 200/day | Per day |

**If you hit the limit:**
- Wait 1 hour for hourly limits to reset
- Response: `429 Too Many Requests`

### Security Requirements
- **Firebase Auth**: Production requires authenticated users
- **Signed URLs**: Upload URLs expire after 1 hour
- **Auto-cleanup**: Files deleted after 30 days

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
**head position:** head excellent (100), eyes focused (78)
**wrist performance:** excellent extension (85/100)
**hip drive:** excellent (78/100, 65.2 speed)
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
- Upload URL generation: instant
- Video upload: 10-30 seconds (depends on video size)
- Video analysis: ~2 minutes (MediaPipe pose detection)
- Seth coaching: ~15 seconds
- OpenIce chat: 5-10 seconds per message

### Check Response Format
All successful responses have `"success": true` and relevant data. Errors return helpful messages with specific guidance.

### Common Error Responses

**Rate Limit Exceeded:**
```json
{
  "error": "429 Too Many Requests: 10 per 1 hour"
}
```
**Solution:** Wait 1 hour before trying again

**Video Not Found:**
```json
{
  "error": "Video not found: users/test123/videos/test.mov"
}
```
**Solution:** Verify the storage_path matches the upload_url response

**Invalid Request:**
```json
{
  "error": "user_id and storage_path are required"
}
```
**Solution:** Include all required fields in request

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

---

## Testing Best Practices

### For Local Development
1. Start with existing analysis JSON files in `results/drill/`
2. Test agents individually before testing full pipeline
3. Use small test videos (10-30 seconds) to iterate quickly

### For API Testing
1. **Check health endpoint** before running tests
2. **Use test user IDs** (e.g., "test123") to avoid mixing with production data
3. **Wait for completion** - video analysis takes ~2 minutes, don't retry immediately
4. **Track rate limits** - max 10 videos per hour during testing
5. **Save responses** - helpful for debugging and comparing results

### Automated Testing
```bash
# Quick health check
curl -s https://puck-buddy-model-22317830094.us-central1.run.app/health | jq .

# Test with timeout (analysis takes ~2 min)
curl -X POST https://puck-buddy-model-22317830094.us-central1.run.app/api/analyze-video \
  --max-time 180 \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test123","storage_path":"users/test123/videos/test.mov"}'
```

---

## Need Help?

- **API Documentation**: See `API_GUIDE.md` for complete endpoint reference
- **Architecture**: See `ARCHITECTURE.md` for system design details  
- **iOS Integration**: See `IOS_INTEGRATION.md` for mobile app examples