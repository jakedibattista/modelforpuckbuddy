# 🏒 Puck Buddy API Reference - Simple Guide for Developers

**Base URL**: `https://puck-buddy-model-22317830094.us-central1.run.app`

## 📋 Quick Start Workflow

1. **Get Upload URL** → Upload video → **Submit for Analysis** → **Check Results**

---

## 🔌 API Endpoints

### 1. Health Check
**Purpose**: Check if the API is working

**Request:**
```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "puck-buddy-signed-url-api"
}
```

---

### 2. Get Upload URL
**Purpose**: Get a secure URL to upload a video file

**Request:**
```bash
POST /api/upload-url
Content-Type: application/json

{
  "user_id": "your-user-123",
  "content_type": "video/mov"
}
```

**Response (Success):**
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

**Response (Error):**
```json
{
  "error": "user_id is required"
}
```

---

### 3. Upload Video File
**Purpose**: Upload your video using the URL from step 2

**Request:**
```bash
PUT [upload_url from step 2]
Content-Type: video/mov

[Your video file data]
```

**Response**: HTTP 200 (no body content)

---

### 4. Submit Video for Analysis
**Purpose**: Tell the system to analyze your uploaded video

**Request:**
```bash
POST /api/submit-video
Content-Type: application/json

{
  "user_id": "your-user-123",
  "storage_path": "users/your-user-123/1642781234_video.mov"
}
```

**Response (Success):**
```json
{
  "success": true,
  "job_id": "abc123def456"
}
```

**Response (Error):**
```json
{
  "error": "user_id and storage_path are required"
}
```

---

### 5. Get Analysis Results
**Purpose**: Get your completed analysis results

**Request:**
```bash
GET /api/results/your-user-123?limit=10
```

**Response (Success):**
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

**Response (No Results):**
```json
{
  "success": true,
  "results": []
}
```

---

### 6. Download Results
**Purpose**: Download your analysis text files

**Request:**
```bash
GET [result_url from step 5]
```

**Response**: Plain text content (the actual feedback/analysis)

---

## 💡 Simple Example (JavaScript/React Native)

```javascript
const API_BASE = 'https://puck-buddy-model-22317830094.us-central1.run.app';
const USER_ID = 'user123';

// Step 1: Get upload URL
async function getUploadUrl() {
  const response = await fetch(`${API_BASE}/api/upload-url`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      user_id: USER_ID,
      content_type: 'video/mov'
    })
  });
  
  const data = await response.json();
  return data.upload_info;
}

// Step 2: Upload video
async function uploadVideo(uploadUrl, videoFile) {
  const response = await fetch(uploadUrl, {
    method: 'PUT',
    headers: { 'Content-Type': 'video/mov' },
    body: videoFile
  });
  
  return response.ok;
}

// Step 3: Submit for analysis
async function submitForAnalysis(storagePath) {
  const response = await fetch(`${API_BASE}/api/submit-video`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      user_id: USER_ID,
      storage_path: storagePath
    })
  });
  
  const data = await response.json();
  return data.job_id;
}

// Step 4: Check results
async function getResults() {
  const response = await fetch(`${API_BASE}/api/results/${USER_ID}`);
  const data = await response.json();
  return data.results;
}

// Step 5: Download analysis text
async function downloadAnalysis(resultUrl) {
  const response = await fetch(resultUrl);
  const text = await response.text();
  return text;
}
```

---

## ❌ Common Error Codes

| Status Code | Meaning |
|-------------|---------|
| 200 | ✅ Success |
| 400 | ❌ Bad Request - Check your JSON |
| 500 | ❌ Server Error - Try again later |

---

## 🎯 Complete Workflow Example

```bash
# 1. Get upload URL
curl -X POST https://puck-buddy-model-22317830094.us-central1.run.app/api/upload-url \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test123","content_type":"video/mov"}'

# 2. Upload video (use URL from step 1)
curl -X PUT "https://storage.googleapis.com/..." \
  -H "Content-Type: video/mov" \
  --data-binary @your-video.mov

# 3. Submit for analysis (use storage_path from step 1)
curl -X POST https://puck-buddy-model-22317830094.us-central1.run.app/api/submit-video \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test123","storage_path":"users/test123/..."}'

# 4. Check results (wait a few minutes for processing)
curl https://puck-buddy-model-22317830094.us-central1.run.app/api/results/test123

# 5. Download analysis (use URLs from step 4)
curl "https://storage.googleapis.com/..."
```

---

## 🚀 Ready to Integrate!

Your backend API is live and ready to use. Just follow the 5-step workflow above and you'll have hockey analysis in your app! 🏒

**Questions?** Check the detailed documentation in `/helpfuldocs/` for more technical details.
