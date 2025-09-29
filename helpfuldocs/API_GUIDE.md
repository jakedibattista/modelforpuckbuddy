# 🏒 PuckBuddy API Guide

**Base URL**: `https://puck-buddy-model-22317830094.us-central1.run.app`

## Quick Start

Most apps need just 3 steps:
1. **Get upload URL** → 2. **Upload video** → 3. **Get analysis**

Optional: Get coaching feedback from Seth or chat with OpenIce AI.

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

### Step 3: Analyze Video
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
    "raw_analysis": "full analysis data for coaching"
  }
}
```

**That's it!** Your app now has pose analysis and structured shot data.

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

### Start Conversation
```bash
POST /api/openice/init

{
  "user_id": "user123",
  "analysis_data": "data analysis text from step 3"
}
```

### Ask Questions
```bash
POST /api/openice/chat

{
  "session_id": "session-abc123",
  "question": "How can I shoot like Connor McDavid?"
}
```

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

| Endpoint | Purpose | Time |
|----------|---------|------|
| `/health` | Health check | instant |
| `/api/upload-url` | Get upload URL | instant |
| `/api/analyze-video` | Analyze video | 2 min |
| `/api/submit-video` | Submit job (advanced) | instant |
| `/api/download-url` | Get download URL | instant |
| `/api/results/{user_id}` | List user results | instant |
| `/api/coaches` | List coaches | instant |
| `/api/coach/seth` | Seth's coaching | 15 sec |
| `/api/start-chat` | Start OpenIce chat | 5 sec |
| `/api/ask-question` | Ask OpenIce question | 10 sec |
| `/api/chat-info/{session_id}` | Get chat info | instant |
| `/api/openice/init` | Start AI chat (client) | 5 sec |
| `/api/openice/chat` | Ask AI questions (client) | 10 sec |
| `/api/job/complete` | Mark job complete | instant |
| `/api/jobs/cleanup` | Clean up old jobs | instant |

## Tips

- **.mov files** work best
- **Keep videos under 100MB** 
- **Set 10 min timeout** for analysis
- **Always check** `success: true` in responses