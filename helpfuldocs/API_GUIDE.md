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
    "parent_summary": "Great job! Your shooting form improved in several areas...",
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
        parentFeedback: result.analysis.parent_summary,
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
- **Simple**: API waits up to 5 minutes, then returns results or timeout error
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

## 🚀 Ready to Integrate!

1. **Copy the JavaScript example** for your preferred workflow
2. **Replace `userId`** with your actual user identification  
3. **Test with the health endpoint** to verify connectivity
4. **Start with a short test video** to verify the complete flow

Need help? Check the other guides in `/helpfuldocs/` for iOS-specific examples and system architecture details.
