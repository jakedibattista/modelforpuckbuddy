# 🏒 PuckBuddy API Guide

**Base URL**: `https://puck-buddy-model-22317830094.us-central1.run.app`

## Quick Start

**🎯 Choose Your Workflow:**

### ⚡ **Simple/Direct** (Recommended for Mobile Apps)
Most apps need just 3 steps:
1. **Get upload URL** → 2. **Upload video** → 3. **Analyze video** (waits ~2 min, returns results)

**Optional**: Get coaching feedback from Seth or chat with OpenIce AI.

---

### 🔄 **Advanced/Queue** (⚠️ DEPRECATED)
This workflow requires a separate worker system and is no longer recommended for most use cases. Use the Simple/Direct workflow above instead.

---

## Core Video Analysis

### Step 1: Get Upload URL
*(Same for both workflows)*
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
*(Same for both workflows)*
```bash
PUT [upload_url from step 1]
Content-Type: video/mov

[video file data]
```

---

## 🎯 Step 3: Choose Your Analysis Method

### ⚡ Option A: Direct Analysis (Recommended)
**Use this for mobile apps - processes immediately and returns results**
```bash
POST /api/analyze-video

{
  "user_id": "user123", 
  "storage_path": "users/user123/20240315_143022_video.mov"
}
```

**Response (~2 minutes):**
```json
{
  "success": true,
  "analysis": {
    "data_analysis": "**Shots detected at timestamps:** 00:08, 00:15\n\n**Shot 1: 00:08:**\n**head position:** head excellent (100), eyes focused (68)\n**wrist performance:** excellent extension (82/100)\n**hip drive:** excellent (78/100, 75.3 speed)",
    "shots_detected": 3,
    "video_duration": 25.4,
    "video_size_mb": 15.2,
    "raw_analysis": {
      "video": "video.mov",
      "fps": 30,
      "duration_est_sec": 25.4,
      "shots": [
        {
          "shot_time_sec": 8.2,
          "head_position": {
            "head_up_score": 85.2,
            "eyes_forward_score": 72.1
          },
          "wrist_control": {
            "setup_control_score": 78.5,
            "setup_control_category": "controlled"
          },
          "hip_drive_analysis": {
            "hip_drive_score": 82.3,
            "hip_drive_category": "excellent"
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

## All Endpoints

### Core Endpoints
| Endpoint | Purpose | Time | Workflow |
|----------|---------|------|----------|
| `/health` | Health check | instant | Both |
| `/api/upload-url` | Get upload URL | instant | Both |

### Analysis Methods
| Endpoint | Purpose | Time | Workflow |
|----------|---------|------|----------|
| `/api/analyze-video` | ⚡ **Direct analysis** (includes raw_analysis) | ~2 min | **Simple/Direct** |
| `/api/submit-video` | 🔄 Submit to job queue (⚠️ DEPRECATED) | instant | **Advanced/Queue** |
| `/api/results/{user_id}` | List completed results (⚠️ DEPRECATED) | instant | **Advanced/Queue** |

### Optional Features
| Endpoint | Purpose | Time | Notes |
|----------|---------|------|-------|
| `/api/coaches` | List coaches | instant | |
| `/api/coach/seth` | Seth's coaching | 15 sec | |
| `/api/openice/init` | ⭐ Start AI chat | 5 sec | **Recommended** |
| `/api/openice/chat` | ⭐ Ask AI questions | 10 sec | **Recommended** |
| `/api/start-chat` | Start OpenIce chat (legacy) | 5 sec | Use `/api/openice/init` |
| `/api/ask-question` | Ask question (legacy) | 10 sec | Use `/api/openice/chat` |
| `/api/chat-info/{session_id}` | Get chat info | instant | |

## Tips

- **.mov files** work best
- **Keep videos under 100MB** 
- **Set 10 min timeout** for analysis
- **Always check** `success: true` in responses