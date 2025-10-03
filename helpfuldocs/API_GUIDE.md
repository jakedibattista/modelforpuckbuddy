# 🏒 PuckBuddy API Guide

**Base URL**: `https://puck-buddy-model-22317830094.us-central1.run.app`

## Quick Start

**🎯 Three Simple Steps:**
1. **Get upload URL** → 2. **Upload video** → 3. **Analyze video** (waits ~2 min, returns results)

**Optional**: Get coaching feedback or chat with OpenIce AI.

---

## Core Video Analysis

### Step 1: Get Upload URL
```bash
POST /api/upload-url

{
  "user_id": "user123",
  "content_type": "video/mov"
}
```

**Response:**
```json
{
  "success": true,
  "upload_info": {
    "upload_url": "https://storage.googleapis.com/...",
    "storage_path": "users/user123/20240315_143022_video.mov"
  }
}
```

### Step 2: Upload Video
```bash
PUT [upload_url from step 1]
Content-Type: video/mov

[video file data]
```

---

## Step 3: Analyze Video

**Processes immediately and returns complete results in ~2 minutes**
```bash
POST /api/analyze-video

{
  "user_id": "user123", 
  "storage_path": "users/user123/20240315_143022_video.mov",
  "age_group": "10-12"  // Optional: "7-9", "10-12", "13-16", "17+" (default: "17+")
}
```

**Response (~2 minutes):**
```json
{
  "success": true,
  "analysis": {
    "data_analysis": "**Shots detected at timestamps:** 00:05, 00:11\n\n**Shot 1: 00:05:**\n**Age Group:** High School/College (17+)\n\n**RAW PERFORMANCE:**\nPower: 3.8/10, Form: 2.8/10\n\n**COACHING SCORE (High School/College (17+) adjusted):**\nPower: 3.8/10, Form: 2.8/10\nOverall: 3.5/10",
    "shots_detected": 2,
    "video_duration": 14.9,
    "video_size_mb": 7.6,
    "raw_analysis": {
      "video": "video.mov",
      "fps": 30.0,
      "duration_est_sec": 14.9,
      "shots": [
        {
          "shot_time_sec": 4.533,
          "age_group": "17+",
          "front_knee_bend_deg": 151.9,
          "hip_rotation_power": {
            "max_rotation_speed": 3.5,
            "rotation_angle_change": 1.9,
            "rotation_consistency": 0.225
          },
          "wrist_extension": {
            "left_wrist_extension_score": 89.4,
            "right_wrist_extension_score": 94.0,
            "follow_through_score": 91.9
          },
          "head_position": {
            "head_up_score": 100.0,
            "eyes_forward_score": 65.4
          },
          "body_stability": {
            "stability_score": 0.867,
            "max_movement": 3.0,
            "movement_consistency": 0.867
          }
        }
      ]
    }
  }
}
```

**That's it!** Your app gets:
- ✅ **Human-readable summary** in `data_analysis` (show this to users)
- ✅ **Complete technical data** in `raw_analysis` (use for coaching endpoints or advanced features)

---

## Optional: Coaching Feedback

### Get Available Coaches
```bash
GET /api/coaches
```

**Response:**
```json
{
  "coaches": [
    {
      "id": "seth",
      "name": "Seth", 
      "description": "Technical feedback with specific improvements"
    }
  ]
}
```

### Get Seth's Coaching
```bash
POST /api/coach/seth

{
  "user_id": "user123",
  "raw_analysis": "analysis data from step 3"
}
```

**Response (~15 seconds):**
```json
{
  "success": true,
  "coach_id": "seth",
  "coaching_feedback": "**What went well:**\n- Strong hip drive at 00:08\n\n**What to work on:**\n- Get lower on front knee bend"
}
```

---

## Optional: OpenIce AI Chat

**⭐ Recommended Endpoints** (with CORS support and enhanced features):

### Initialize Session
```bash
POST /api/openice/init

{
  "user_id": "user123",
  "analysis_data": "data analysis text from step 3"
}
```

**OR with raw analysis data:**
```bash
POST /api/openice/init

{
  "user_id": "user123",
  "raw_analysis": {
    "video": "video.mov",
    "shots": [...],
    "duration_est_sec": 25.4
  }
}
```

**Response:** Includes `session_id` and immediate AI coaching response.

### Continue Conversation
```bash
POST /api/openice/chat

{
  "session_id": "session-abc123",
  "question": "How can I shoot like Connor McDavid?"
}
```

**Note:** The `/api/openice/*` endpoints are recommended over the legacy `/api/start-chat` and `/api/ask-question` endpoints as they provide immediate responses and better mobile app support.

---

## Complete Example

```javascript
// 1. Get upload URL
const uploadResponse = await fetch('YOUR_API_BASE/api/upload-url', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ user_id: 'user123', content_type: 'video/mov' })
});

// 2. Upload video
await fetch(uploadResponse.upload_info.upload_url, {
  method: 'PUT', 
  body: videoFile
});

// 3. Analyze video
const analysis = await fetch('YOUR_API_BASE/api/analyze-video', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ 
    user_id: 'user123',
    storage_path: uploadResponse.upload_info.storage_path 
  })
});

// Done! Display analysis.data_analysis to user
```

---

## All Endpoints Reference

### Core Video Analysis
| Endpoint | Purpose | Time |
|----------|---------|------|
| `GET /health` | Health check | instant |
| `POST /api/upload-url` | Get upload URL | instant |
| `POST /api/analyze-video` | Analyze video with MediaPipe | ~2 min |

### Coaching & AI
| Endpoint | Purpose | Time |
|----------|---------|------|
| `GET /api/coaches` | List available coaches | instant |
| `POST /api/coach/seth` | Get Seth's technical coaching | ~15 sec |
| `POST /api/openice/init` | Start AI coaching chat | ~5 sec |
| `POST /api/openice/chat` | Ask AI coaching questions | ~10 sec |
| `GET /api/chat-info/{session_id}` | Get chat session info | instant |

## Rate Limits & Security

### Rate Limits (Per User)
To ensure fair usage and control costs, the following limits apply per `user_id`:

| Endpoint | Limit | Reset Period |
|----------|-------|--------------|
| `/api/upload-url` | 20 requests | Per hour |
| `/api/analyze-video` | **10 videos** | **Per hour** |
| All endpoints | 200 requests | Per day |

**When limit exceeded:**
- Status: `429 Too Many Requests`
- Response: `{"error": "429 Too Many Requests: 10 per 1 hour"}`
- Action: Wait until the hour resets, or contact support for higher limits

**iOS Error Handling Example:**
```swift
if httpResponse.statusCode == 429 {
    error = "You've reached the hourly video analysis limit. Please try again in a bit!"
}
```

### Storage & Cleanup
- **Automatic cleanup**: Videos and results older than 30 days are deleted automatically
- **Job cleanup**: Only last 10 completed jobs per user are retained
- **No action needed**: Cleanup happens after each analysis

### Security
- **Authentication required**: Firebase Auth is required for all video operations
- **Signed URLs**: Videos use time-limited signed URLs (1 hour expiration)
- **Private storage**: Each user can only access their own files

---

## Tips

- **.mov files** work best
- **Keep videos under 100MB** 
- **Set 10 min timeout** for analysis
- **Always check** `success: true` in responses
- **Monitor rate limits**: Track your usage to avoid hitting limits