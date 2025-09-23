# Firebase Signed URL Integration Guide

This guide shows you how to integrate Firebase signed URLs into your hockey drill analysis app for secure video uploads and result delivery.

## Overview

**Problem**: Your app needs to upload videos and receive analysis results without exposing Firebase credentials or making files publicly accessible.

**Solution**: Use Firebase Admin SDK to generate time-limited signed URLs that allow secure upload/download without authentication.

## Architecture Flow

```
Mobile App → Backend (Signed URL) → Firebase Storage → Video Processing → Results Upload → Download URLs → Mobile App
```

### Detailed Workflow:

1. **Upload Request**: Mobile app requests upload URL from your backend
2. **Generate Upload URL**: Backend creates signed URL with PUT permissions
3. **Direct Upload**: App uploads video directly to Firebase Storage using signed URL
4. **Processing Trigger**: Firebase Function or Pub/Sub triggers video processing
5. **Analysis**: Backend downloads video, runs pose detection, generates AI summaries
6. **Store Results**: Upload analysis results back to Firebase Storage
7. **Delivery URLs**: Generate signed download URLs for results
8. **App Download**: Mobile app downloads results using signed URLs

## Production Deployment

✅ **The backend API is already deployed and ready for production use:**

**Backend API URL**: `https://puck-buddy-model-22317830094.us-central1.run.app`

### Available Endpoints:
- `GET /health` - Service health check  
- `POST /api/upload-url` - Generate signed URL for video upload
- `POST /api/submit-video` - Create analysis job
- `POST /api/download-url` - Generate signed URL for file download  
- `GET /api/results/{user_id}` - List user's analysis results

### Quick Test:
```bash
# Health check
curl https://puck-buddy-model-22317830094.us-central1.run.app/health

# Test upload URL generation (requires Firebase service account setup)
curl -X POST https://puck-buddy-model-22317830094.us-central1.run.app/api/upload-url \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test-user","filename":"test.mp4"}'
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install firebase-admin google-cloud-storage python-dotenv
```

### 2. Get Firebase Service Account Key

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Select your project: `puck-buddy`
3. Go to **Project Settings → Service Accounts**
4. Click **"Generate new private key"**
5. Save JSON file as: `firebase-service-account-key.json`

### 3. Configure Environment Variables

Add to your `.env` file:

```bash
# Firebase Admin SDK
GOOGLE_APPLICATION_CREDENTIALS=./firebase-service-account-key.json
FIREBASE_STORAGE_BUCKET=puck-buddy.appspot.com
GOOGLE_API_KEY=your_gemini_api_key

# Optional
GOOGLE_CLOUD_PROJECT=puck-buddy
```

### 4. Update Firebase Storage Rules

In Firebase Console → Storage → Rules:

```javascript
rules_version = '2';
service firebase.storage {
  match /b/{bucket}/o {
    // Allow signed URL access for user-specific paths
    match /users/{userId}/{allPaths=**} {
      allow read, write: if request.auth != null && request.auth.uid == userId;
      allow read, write: if request.auth == null; // For signed URLs
    }
  }
}
```

## Code Integration

### Backend Implementation

#### 1. Initialize Firebase Storage Manager

```python
from utils.firebase_storage import FirebaseStorageManager

# Initialize (automatically uses environment variables)
storage_manager = FirebaseStorageManager()
```

#### 2. Generate Upload URL (API Endpoint)

```python
@app.route('/api/upload-url', methods=['POST'])
def generate_upload_url():
    data = request.get_json()
    user_id = data['user_id']
    filename = data['filename']
    
    try:
        upload_url, storage_path = storage_manager.generate_upload_url(
            user_id=user_id, 
            filename=filename,
            expiration_minutes=60  # 1 hour
        )
        
        return {
            "upload_url": upload_url,
            "storage_path": storage_path,
            "expires_in": 3600  # seconds
        }
    except Exception as e:
        return {"error": str(e)}, 500
```

#### 3. Process Video (Triggered by Upload)

```python
def process_uploaded_video(user_id: str, storage_path: str):
    """Process video after upload completes."""
    
    # Generate download URL for internal processing
    download_url = storage_manager.generate_download_url(
        storage_path, expiration_minutes=30
    )
    
    # Download video temporarily
    with tempfile.NamedTemporaryFile(suffix=".mov") as tmp_file:
        urllib.request.urlretrieve(download_url, tmp_file.name)
        
        # Run analysis
        from analysis.shooting_drill_feedback import analyze_drill
        from agents.parent_feedback_agent import generate_summary_with_gemini
        from agents.improvement_coach_agent import generate_sections
        
        analysis_result = analyze_drill(tmp_file.name)
        parent_summary = generate_summary_with_gemini(analysis_result)
        coach_analysis = generate_sections(analysis_result)
    
    # Upload results to Firebase Storage
    results_paths = storage_manager.upload_analysis_results(
        user_id=user_id,
        video_filename=os.path.basename(storage_path),
        analysis_data=analysis_result,
        parent_summary=parent_summary,
        coach_analysis=coach_analysis
    )
    
    # Generate download URLs for results (24 hour expiration)
    result_urls = storage_manager.generate_results_download_urls(
        results_paths, expiration_minutes=1440
    )
    
    # Notify app that results are ready (push notification, webhook, etc.)
    notify_app_results_ready(user_id, result_urls)
```

#### 4. Get Results API Endpoint

```python
@app.route('/api/results/<user_id>')
def get_user_results(user_id):
    """Get all analysis results for a user."""
    try:
        # List user's results
        results = storage_manager.list_user_results(user_id)
        
        # Generate fresh download URLs for recent results
        for result_group in results:
            if result_group['files']:
                # Extract storage paths
                storage_paths = {
                    filename.split('.')[0]: file_info['storage_path']
                    for filename, file_info in result_group['files'].items()
                }
                
                # Generate download URLs
                download_urls = storage_manager.generate_results_download_urls(
                    storage_paths, expiration_minutes=60
                )
                
                result_group['download_urls'] = download_urls
        
        return {"results": results}
    except Exception as e:
        return {"error": str(e)}, 500
```

### Mobile App Implementation

#### 1. Request Upload URL

```swift
// iOS Example
func requestUploadURL(for filename: String, userId: String, completion: @escaping (UploadInfo?) -> Void) {
    let url = URL(string: "https://your-backend.com/api/upload-url")!
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    
    let body = ["user_id": userId, "filename": filename]
    request.httpBody = try! JSONSerialization.data(withJSONObject: body)
    
    URLSession.shared.dataTask(with: request) { data, response, error in
        guard let data = data,
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let uploadURL = json["upload_url"] as? String,
              let storagePath = json["storage_path"] as? String else {
            completion(nil)
            return
        }
        
        completion(UploadInfo(uploadURL: uploadURL, storagePath: storagePath))
    }.resume()
}
```

#### 2. Upload Video Using Signed URL

```swift
func uploadVideo(fileURL: URL, to uploadURL: String, completion: @escaping (Bool) -> Void) {
    guard let url = URL(string: uploadURL) else {
        completion(false)
        return
    }
    
    var request = URLRequest(url: url)
    request.httpMethod = "PUT"
    request.setValue("video/quicktime", forHTTPHeaderField: "Content-Type")
    
    URLSession.shared.uploadTask(with: request, fromFile: fileURL) { data, response, error in
        DispatchQueue.main.async {
            if let httpResponse = response as? HTTPURLResponse {
                completion(httpResponse.statusCode == 200)
            } else {
                completion(false)
            }
        }
    }.resume()
}
```

#### 3. Download Results

```swift
func downloadResults(from downloadURL: String, completion: @escaping (Data?) -> Void) {
    guard let url = URL(string: downloadURL) else {
        completion(nil)
        return
    }
    
    URLSession.shared.dataTask(with: url) { data, response, error in
        DispatchQueue.main.async {
            completion(data)
        }
    }.resume()
}
```

## Testing Your Implementation

### 1. Run Setup Helper

```bash
python setup_firebase_admin.py
```

### 2. Test Complete Workflow

```bash
# Test with local video
python tests/test_signed_url_workflow.py "test_user_123" "videos/input/kidshoot2.MOV"

# Test URL generation only
python tests/test_signed_url_workflow.py "test_user_123"
```

### 3. Verify Firebase Storage

1. Check Firebase Console → Storage
2. Look for files under `users/test_user_123/`
3. Verify signed URLs work by opening in browser

## Security Considerations

### ✅ Benefits

- **No credentials in mobile app**: Signed URLs provide temporary access without exposing Firebase credentials
- **Time-limited access**: URLs expire automatically (1 hour for uploads, 24 hours for downloads)
- **User isolation**: Each user can only access their own files through the path structure
- **Direct uploads**: Videos go directly to Firebase Storage, not through your backend

### 🔒 Best Practices

1. **Short expiration times**: Use 1 hour for uploads, 24 hours for downloads
2. **User validation**: Always validate user permissions before generating URLs
3. **Path restrictions**: Use consistent path patterns (`users/{userId}/...`)
4. **Monitor usage**: Track Firebase Storage usage and costs
5. **Error handling**: Implement proper retry logic for network failures

## File Organization

Firebase Storage structure:

```
your-bucket/
├── users/
│   └── {user_id}/
│       ├── videos/
│       │   └── 20241201_143022_hockey_drill.mov
│       └── results/
│           └── 20241201_143022/
│               ├── analysis.json
│               ├── parent_summary.txt
│               └── coach_analysis.txt
```

## Monitoring and Analytics

### Firebase Console Metrics

- Storage usage and costs
- Request counts and patterns
- Error rates and types

### Custom Analytics

```python
# Track video processing metrics
def track_video_processing(user_id: str, video_size: int, processing_time: float, shots_detected: int):
    """Track video processing analytics."""
    analytics_data = {
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "video_size_mb": video_size / (1024 * 1024),
        "processing_time_seconds": processing_time,
        "shots_detected": shots_detected
    }
    
    # Send to your analytics service
    # e.g., Firebase Analytics, Google Analytics, etc.
```

## Troubleshooting

### Common Issues

1. **403 Forbidden**: Check storage rules and service account permissions
2. **URL expired**: Generate fresh URLs, check expiration times
3. **File not found**: Verify storage path and file existence
4. **Upload fails**: Check content-type headers and file size limits

### Debug Commands

```bash
# Test Firebase connection
python -c "from utils.firebase_storage import FirebaseStorageManager; FirebaseStorageManager()"

# List user files
python -c "
from utils.firebase_storage import FirebaseStorageManager
sm = FirebaseStorageManager()
print(sm.list_user_videos('test_user_123'))
"

# Generate test URL
python -c "
from utils.firebase_storage import FirebaseStorageManager
sm = FirebaseStorageManager()
url, path = sm.generate_upload_url('test', 'test.mov')
print(f'URL: {url[:100]}...')
"
```

## Production Deployment

### Environment Variables

Set these in your production environment:

```bash
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
FIREBASE_STORAGE_BUCKET=puck-buddy.appspot.com
GOOGLE_API_KEY=your_production_gemini_key
```

### Scaling Considerations

- **Firebase Storage quotas**: Monitor usage and upgrade plan if needed
- **Processing time**: Consider async processing for large videos
- **Cost optimization**: Set lifecycle rules to delete old files
- **CDN integration**: Use Firebase hosting or CDN for result delivery

## Next Steps

1. **Implement backend endpoints** for upload URL generation and result retrieval
2. **Integrate mobile app** upload and download functionality
3. **Set up monitoring** for storage usage and processing metrics
4. **Test with real users** to validate the complete workflow
5. **Optimize for scale** based on usage patterns

This integration provides a secure, scalable foundation for your hockey drill analysis app! 🏒
