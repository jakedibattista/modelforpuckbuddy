# 🏒 PuckBuddy System Architecture

Simple overview of how the hockey video analysis system works.

## What It Does

1. **Accepts videos** from mobile apps
2. **Analyzes hockey shots** using AI pose detection  
3. **Returns structured data** about technique
4. **Provides coaching feedback** when requested
5. **Enables AI chat** for questions about improvement

---

## System Components

### Backend API
- **Location**: `https://puck-buddy-model-22317830094.us-central1.run.app`
- **Purpose**: Handles video uploads and analysis requests
- **Technology**: Python Flask on Google Cloud Run

### Video Analysis Engine
- **MediaPipe Pose**: Detects body positions in hockey videos
- **Shot Detection**: Finds when shots happen in the video
- **Metrics**: Measures knee bend, hip drive, wrist movement, etc.

### AI Agents
- **Data Summary Agent**: Converts pose data into readable summaries
- **Seth Coaching Agent**: Provides technical coaching feedback
- **OpenIce Agent**: Conversational AI for questions and tips

### Storage
- **Firebase Storage**: Stores uploaded videos securely
- **Temporary Processing**: Videos processed then deleted
- **No Long-term Storage**: System is ephemeral by design

---

## How It Works

### Video Analysis Flow
```
Mobile App → Upload Video → Analyze Video → Display Results
     ↓              ↓             ↓              ↓
Get Upload URL → PUT Video → POST /analyze-video → Show Data
```

**Timeline:**
- Upload URL: instant
- Video upload: 10-30 seconds  
- Analysis: ~2 minutes (processes immediately)
- Results: Complete data + human-readable summary

### Optional: Coaching Feedback
```
Get Analysis → Get Coaching → Show Feedback
     ↓              ↓              ↓  
  Raw Data → POST /coach/seth → Display Tips
```

**Additional time:** +15 seconds

### Optional: AI Chat
```
Get Analysis → Start Chat → Ask Questions → Get Answers
     ↓              ↓              ↓              ↓
  Summary → OpenIce Init → Chat Message → AI Response  
```

**Response time:** ~5-10 seconds per message

---

## Data Flow

### Video Analysis Pipeline
```
Raw Video → MediaPipe → Shot Detection → Metrics → Summary
   (MP4)      (Pose)       (AI)         (Data)    (Text)
```

### What Gets Measured
- **Shot timing**: When each shot happens
- **Knee bend**: How low the player gets
- **Hip drive**: Power generation through hips
- **Wrist control**: Smoothness of stick handling
- **Head position**: Eyes on target
- **Body alignment**: Square to target

### Output Formats
- **Structured Data**: Timestamps, angles, scores
- **Human Text**: "Good knee bend at 00:08 (95°)"
- **Coaching Tips**: "Work on getting lower for more power"

---

## Security & Performance

### Security
- **Signed URLs**: Videos uploaded directly to secure cloud storage with 1-hour expiration
- **No Credentials**: Apps never handle storage keys
- **Firebase Auth Required**: All storage operations require authenticated users
- **Private Storage**: Users can only access their own files (enforced by Storage Rules)
- **Automatic Cleanup**: Videos and results older than 30 days deleted automatically
- **Rate Limiting**: Per-user limits prevent abuse (10 videos/hour, 200 requests/day)

### Performance  
- **Auto-scaling**: Handles multiple videos simultaneously (up to 10 concurrent)
- **Efficient Processing**: Optimized for 2-3 minute analysis
- **Smart Cleanup**: Old files removed to maintain performance
- **Cost Control**: Rate limits and cleanup prevent runaway costs

### Reliability
- **Error Handling**: Graceful failures with helpful messages
- **Timeout Protection**: Won't hang indefinitely (10 min max)
- **Health Monitoring**: System status always available at `/health`
- **Rate Limit Protection**: Prevents server overload with 429 responses

---

## For Developers

### Getting Started
1. Read the **API_GUIDE.md** for integration steps
2. Test with sample videos
3. Implement the 3-step flow: upload → analyze → display

### iOS Integration
- See **IOS_INTEGRATION.md** for SwiftUI examples
- Copy-paste ready code for video analysis
- Includes error handling and UI patterns

### Testing
- Use **HOCKEY_VIDEO_TESTING_GUIDE.md** for local testing
- Sample videos and expected outputs provided
- Command-line tools for debugging

---

## Technical Details

### Cloud Infrastructure
- **Google Cloud Run**: Serverless container hosting
- **Firebase Storage**: Video file storage
- **Cloud Functions**: Job management and cleanup
- **Auto-scaling**: 0 to 10 instances based on demand

### AI Models
- **MediaPipe**: Google's pose detection (local processing)
- **Gemini**: Google's AI for text generation
- **Custom Logic**: Hockey-specific shot detection algorithms

### Processing Power
- **2 CPU cores** per analysis
- **2GB RAM** for video processing  
- **~10 concurrent** video analyses supported

This system is designed to be simple for developers while providing powerful hockey analysis capabilities.
