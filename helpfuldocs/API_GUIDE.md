# 🏒 Puck Buddy API Integration Guide

**Base URL**: `https://puck-buddy-model-22317830094.us-central1.run.app`

## 🎯 Two Integration Options

### Option 1: Simple Integration (Recommended)
Upload video → Get analysis results directly. Perfect for most apps.

### Option 2: Advanced Integration  
More control over the workflow with real-time progress updates.

---

## 🚀 Simple Integration (2 Steps)

**Best for**: Most mobile apps that just want to send a video and get feedback text.

### Step 1: Get Upload URL
```bash
POST /api/upload-url
Content-Type: application/json

{
  "user_id": "your-user-123",
  "content_type": "video/mov"
}
```

**Response:**
```json
{
  "success": true,
  "upload_info": {
    "upload_url": "https://storage.googleapis.com/...",
    "storage_path": "users/your-user-123/1642781234_video.mov",
    "expires_in": "1 hour"
  }
}
```

### Step 2: Upload Video  
```bash
PUT [upload_url from step 1]
Content-Type: video/mov

[Your video file data]
```

### Step 3: Get Analysis Results
```bash
POST /api/analyze-video
Content-Type: application/json

{
  "user_id": "your-user-123",
  "storage_path": "users/your-user-123/1642781234_video.mov"
}
```

**Response (after ~2-5 minutes):**
```json
{
  "success": true,
  "analysis": {
    "data_analysis": "Shots detected at timestamp: 8.2s, 15.7s, 23.1s...",
    "coach_summary": "## What Went Well\n\n- Good knee bend on most shots...",
    "shots_detected": 7,
    "video_duration": 54.3,
    "video_size_mb": 23.1
  }
}
```

---

## ⚡ Advanced Integration (More Control)

**Best for**: Apps that need real-time progress updates or want to manage results differently.

### Steps 1-2: Same as Simple Integration
(Get upload URL and upload video)

### Step 3: Submit for Processing
```bash
POST /api/submit-video
Content-Type: application/json

{
  "user_id": "your-user-123",
  "storage_path": "users/your-user-123/1642781234_video.mov"
}
```

**Response:**
```json
{
  "success": true,
  "job_id": "abc123def456"
}
```

### Step 4: Check Results (Poll or use Firestore)
```bash
GET /api/results/your-user-123?limit=10
```

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "job_id": "abc123def456",
      "status": "completed",
      "created_at": "2024-01-20T10:30:00Z",
      "result_urls": {
        "parent_summary_url": "https://storage.googleapis.com/...",
        "coach_summary_url": "https://storage.googleapis.com/...",
        "drill_analysis_url": "https://storage.googleapis.com/..."
      }
    }
  ]
}
```

### Step 5: Download Results
```bash
GET [result_url from step 4]
```
**Response**: Plain text content

---

## 📱 JavaScript Examples

### Simple Integration
```javascript
const API_BASE = 'https://puck-buddy-model-22317830094.us-central1.run.app';

async function analyzeHockeyVideo(videoFile, userId) {
  try {
    // Step 1: Get upload URL
    const uploadResponse = await fetch(`${API_BASE}/api/upload-url`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_id: userId,
        content_type: 'video/mov'
      })
    });
    
    const uploadData = await uploadResponse.json();
    const { upload_url, storage_path } = uploadData.upload_info;
    
    // Step 2: Upload video
    await fetch(upload_url, {
      method: 'PUT',
      headers: { 'Content-Type': 'video/mov' },
      body: videoFile
    });
    
    // Step 3: Get analysis (waits for completion)
    const analysisResponse = await fetch(`${API_BASE}/api/analyze-video`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_id: userId,
        storage_path: storage_path
      })
    });
    
    const result = await analysisResponse.json();
    
    if (result.success) {
      return {
        parentFeedback: result.analysis.data_analysis,
        coachFeedback: result.analysis.coach_summary
      };
    } else {
      throw new Error(result.error);
    }
    
  } catch (error) {
    console.error('Analysis failed:', error);
    throw error;
  }
}

// Usage:
// const results = await analyzeHockeyVideo(myVideoFile, 'user123');
// showFeedback(results.parentFeedback, results.coachFeedback);
```

### Advanced Integration with Progress
```javascript
async function analyzeWithProgress(videoFile, userId, onProgress) {
  // Steps 1-2: Upload (same as simple)
  const { storage_path, job_id } = await uploadVideo(videoFile, userId);
  
  // Step 3: Submit for processing
  await fetch(`${API_BASE}/api/submit-video`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: userId, storage_path })
  });
  
  // Step 4: Poll for results with progress
  while (true) {
    const results = await fetch(`${API_BASE}/api/results/${userId}`);
    const data = await results.json();
    
    const job = data.results.find(j => j.job_id === job_id);
    if (job) {
      onProgress(job.status, job.progress || 0);
      
      if (job.status === 'completed') {
        // Download and return results
        const parentSummary = await downloadText(job.result_urls.parent_summary_url);
        const coachSummary = await downloadText(job.result_urls.coach_summary_url);
        return { parentSummary, coachSummary };
      }
      
      if (job.status === 'failed') {
        throw new Error('Analysis failed');
      }
    }
    
    await new Promise(resolve => setTimeout(resolve, 5000)); // Wait 5 seconds
  }
}
```

---

## 🛠️ All Available Endpoints

| Endpoint | Method | Purpose | Usage |
|----------|--------|---------|-------|
| `/health` | GET | Health check | Always available |
| `/api/upload-url` | POST | Get signed upload URL | Both workflows |
| `/api/analyze-video` | POST | ⭐ Simple: Get results directly | Simple workflow |
| `/api/submit-video` | POST | Submit for processing | Advanced workflow |
| `/api/results/{user_id}` | GET | List completed analyses | Advanced workflow |

---

## ⚠️ Important Notes

### Processing Time
- **Simple**: API waits up to 10 minutes, then returns results or timeout error
- **Advanced**: Usually 2-5 minutes, poll every 5-10 seconds for updates

### File Requirements  
- **Format**: .mov, .mp4 recommended
- **Size**: Keep under 100MB for best performance
- **Content**: Hockey shooting drills work best

### Error Handling
The API now returns concise, generic error messages (no internal stack traces):

```json
// 400 examples
{ "error": "user_id and storage_path are required" }
{ "error": "Video not found: users/uid/foo.mov" }

// 500 examples
{ "error": "Failed to create upload URL" }
{ "error": "Failed to analyze video" }
```

### Rate Limits
- No formal rate limits currently
- Processing is resource-intensive, so avoid concurrent requests per user

---

## 🎯 Quick Decision Guide

**Use Simple Integration if:**
- ✅ You just want text feedback  
- ✅ You don't need progress updates
- ✅ Your app can wait 2-5 minutes for results
- ✅ You want the easiest implementation

**Use Advanced Integration if:**
- ✅ You need real-time progress updates
- ✅ You want to store/manage result files yourself  
- ✅ You're building complex workflow features
- ✅ You want to poll results later

---

## 🤖 OpenIce AI Coach (Optional Enhancement)

**What is OpenIce?** An intelligent conversational AI coach that provides personalized hockey advice based on your video analysis data. Players can ask questions like "How can I shoot like Connor McDavid?" or "What drill should I focus on this week?"

### Key Features:
- 💬 **Conversational AI** - Remembers previous questions in chat sessions
- 🌐 **Real-time Research** - Uses Google Search for current hockey knowledge  
- 🎯 **Personalized Advice** - References your specific shot data and timestamps
- 🏒 **Player Comparisons** - Compare technique to NHL stars like McDavid, Crosby, Ovechkin
- 📚 **Practice Recommendations** - Suggests specific drills and training methods

### OpenIce API Endpoints

#### 1. Start Chat Session
Create a new conversation with your analysis data:

```bash
POST /api/start-chat
Content-Type: application/json

{
  "analysis_data": "**Shots detected at timestamps:** 00:08, 00:15...",
  "user_id": "your-user-123"
}
```

**Response:**
```json
{
  "success": true,
  "session_id": "abc123-session-id",
  "user_id": "your-user-123",
  "message": "OpenIce chat session created successfully"
}
```

#### 2. Ask Questions
Continue the conversation with coaching questions:

```bash
POST /api/ask-question  
Content-Type: application/json

{
  "session_id": "abc123-session-id",
  "question": "How can my shot look more like Connor McDavid?"
}
```

**Response:**
```json
{
  "success": true,
  "openice_response": "Based on your data and McDavid's technique: Your hip drive (0.456) at 00:15 is good - McDavid emphasizes explosive rotation. However, your knee bend needs work - McDavid gets down to 95-105° vs your 142°...",
  "search_queries": ["Connor McDavid shooting technique", "McDavid knee bend mechanics"],
  "sources": ["NHL.com", "Hockey Training Pro", "Elite Prospects"],
  "session_id": "abc123-session-id",
  "message_count": 1
}
```

#### 3. Get Session Info
Check conversation details:

```bash
GET /api/chat-info/abc123-session-id
```

**Response:**
```json
{
  "success": true,
  "session_info": {
    "session_id": "abc123-session-id",
    "user_id": "your-user-123",
    "created_at": "2025-09-25T18:30:00Z",
    "last_activity": "2025-09-25T18:35:00Z", 
    "message_count": 3
  }
}
```

### Complete OpenIce Integration Example

```javascript
// After getting video analysis results:
async function startCoachingSession(analysisData, userId) {
  // 1. Create OpenIce chat session
  const chatResponse = await fetch(`${baseUrl}/api/start-chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      analysis_data: analysisData,
      user_id: userId
    })
  });
  
  const { session_id } = await chatResponse.json();
  
  // 2. Ask coaching questions
  const questionResponse = await fetch(`${baseUrl}/api/ask-question`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: session_id,
      question: "How can I shoot like Connor McDavid?"
    })
  });
  
  const coaching = await questionResponse.json();
  
  // Display personalized coaching advice
  console.log('OpenIce Coach:', coaching.openice_response);
  console.log('Research Sources:', coaching.sources);
  
  return session_id; // Save for follow-up questions
}

// Follow-up questions in the same session
async function askFollowUp(sessionId, question) {
  const response = await fetch(`${baseUrl}/api/ask-question`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      question: question
    })
  });
  
  return await response.json();
}
```

### Example OpenIce Questions

**Technique Comparisons:**
- "How can my shot look more like Sidney Crosby?"
- "What does Connor McDavid do differently in his release?"
- "Compare my hip drive to Alex Ovechkin's technique"

**Practice Planning:**
- "What's the most important thing to work on this week?"
- "What drill should I focus on for my knee bend?"
- "How can I improve my wrist steadiness?"

**Specific Analysis:**
- "Why was my shot at 00:15 better than the others?"
- "What's causing my jerky wrist movement?"
- "How can I improve my front knee angle?"

### Error Handling

```javascript
try {
  const result = await askQuestion(sessionId, question);
  if (result.success) {
    displayCoaching(result.openice_response);
  } else {
    console.error('OpenIce error:', result.error);
  }
} catch (error) {
  if (error.status === 404) {
    console.log('Session expired, create new session');
  } else {
    console.error('Network error:', error);
  }
}
```

**Note:** OpenIce is completely optional and additive. Your existing video analysis workflow continues to work identically whether you use OpenIce features or not.

---

## 🚀 Ready to Integrate!

1. **Copy the JavaScript example** for your preferred workflow
2. **Replace `userId`** with your actual user identification  
3. **Test with the health endpoint** to verify connectivity
4. **Start with a short test video** to verify the complete flow

Need help? Check the other guides in `/helpfuldocs/` for iOS-specific examples and system architecture details.
