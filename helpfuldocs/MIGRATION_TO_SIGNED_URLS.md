# Migration Guide: From Direct Firebase Access to Signed URLs

This guide helps you migrate from the current direct Firebase Storage and Firestore approach to the recommended signed URL approach for better security and scalability.

## Overview

**Current Architecture (Option A):**
- iOS app uploads directly to Firebase Storage
- Results stored in Firestore documents
- iOS app reads results directly from Firestore

**New Architecture (Option B):**
- Backend generates signed URLs for uploads
- iOS app uploads using signed URLs
- Results stored in Firebase Storage with signed URLs for access
- Better security, no Firebase credentials in app

## Current Status

✅ **Phase 1-3 Complete: Backend is deployed and operational**

**Backend API URL**: `https://puck-buddy-model-22317830094.us-central1.run.app`

The signed URL infrastructure is ready for production use. Your app can now integrate with the deployed backend for secure video processing.

## Migration Strategy

### Phase 1: Backend Setup (No Breaking Changes) ✅ COMPLETE

1. **Add Firebase Admin SDK to your backend**
   ```bash
   pip install firebase-admin google-cloud-storage
   ```

2. **Add the Firebase Storage Manager**
   - Copy `utils/firebase_storage.py` to your project
   - Set up service account credentials
   - Add environment variables

3. **Add API endpoints for signed URLs**
   ```python
   # Add to your backend (Flask/FastAPI)
   from utils.firebase_storage import FirebaseStorageManager
   
   @app.route('/api/upload-url', methods=['POST'])
   def generate_upload_url():
       # Implementation from ARCHITECTURE.md
       pass
   
   @app.route('/api/results/<user_id>')
   def get_user_results(user_id):
       # Implementation from ARCHITECTURE.md
       pass
   ```

4. **Update Firebase Storage Rules**
   ```javascript
   // Add to firebase/storage.rules
   rules_version = '2';
   service firebase.storage {
     match /b/{bucket}/o {
       match /users/{uid}/{allPaths=**} {
         allow write: if request.auth != null && request.auth.uid == uid;
         allow read: if request.auth != null && request.auth.uid == uid;
         // NEW: Allow signed URL access
         allow read, write: if request.auth == null;
       }
     }
   }
   ```

### Phase 2: Worker Updates (Backward Compatible)

1. **Update your Cloud Run worker to support both modes**
   ```python
   def process_job(job_data):
       delivery_method = job_data.get('delivery_method', 'firestore')
       
       if delivery_method == 'signed_urls':
           return process_job_with_signed_urls(job_data)
       else:
           return process_job_with_firestore(job_data)  # Current implementation
   ```

2. **Add signed URL processing function**
   ```python
   def process_job_with_signed_urls(job_data):
       # Implementation from ARCHITECTURE.md
       # Upload results to Firebase Storage instead of Firestore
       # Generate signed URLs for result access
       pass
   ```

3. **Update Firestore job schema**
   ```python
   # Support both delivery methods
   job_data = {
       "userId": user_id,
       "storagePath": storage_path,
       "status": "queued",
       "delivery_method": delivery_method,  # NEW FIELD
       # Conditional fields based on delivery_method
   }
   ```

### Phase 3: iOS App Updates (Gradual Migration)

1. **Add new signed URL service alongside existing service**
   ```swift
   // Keep existing VideoAnalysisService
   // Add new SignedURLVideoAnalysisService
   ```

2. **Add feature flag for testing**
   ```swift
   class VideoAnalysisManager {
       let useSignedURLs = UserDefaults.standard.bool(forKey: "useSignedURLs")
       
       func submitVideo(_ url: URL) async throws {
           if useSignedURLs {
               return try await signedURLService.submitVideoWithSignedURL(url)
           } else {
               return try await directService.submitVideo(url)
           }
       }
   }
   ```

3. **Test with small user group**
   - Enable signed URLs for beta users
   - Monitor performance and error rates
   - Gradually roll out to more users

### Phase 4: Full Migration

1. **Switch default to signed URLs**
   ```swift
   let useSignedURLs = UserDefaults.standard.bool(forKey: "useSignedURLs") // Default: true
   ```

2. **Remove old direct Firebase implementation**
   - After confirming signed URLs work well
   - Clean up old code paths
   - Update documentation

## Detailed Migration Steps

### Step 1: Backend API Setup

1. **Download Firebase service account key**
   ```bash
   # From Firebase Console → Project Settings → Service Accounts
   # Download JSON key file
   ```

2. **Set environment variables**
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
   export FIREBASE_STORAGE_BUCKET=puck-buddy.appspot.com
   export GOOGLE_API_KEY=your_gemini_key
   ```

3. **Test Firebase Admin SDK**
   ```bash
   python3 setup_firebase_admin.py
   ```

4. **Test signed URL generation**
   ```bash
   python3 tests/test_signed_url_workflow.py "test_user_123"
   ```

### Step 2: Worker Integration

1. **Update worker dependencies**
   ```dockerfile
   # Add to worker/Dockerfile
   RUN pip install firebase-admin google-cloud-storage
   ```

2. **Copy Firebase Storage Manager**
   ```dockerfile
   # Add to worker/Dockerfile
   COPY utils/ /app/utils/
   ```

3. **Update worker logic**
   ```python
   # In worker/app.py
   from utils.firebase_storage import FirebaseStorageManager
   
   def process_job_enhanced(job_data):
       storage_manager = FirebaseStorageManager()
       delivery_method = job_data.get('delivery_method', 'firestore')
       
       if delivery_method == 'signed_urls':
           # Generate signed download URL
           download_url = storage_manager.generate_download_url(
               job_data['storagePath'], expiration_minutes=30
           )
           
           # Process video from signed URL
           # ... existing processing logic ...
           
           # Upload results to Firebase Storage
           results_paths = storage_manager.upload_analysis_results(...)
           result_urls = storage_manager.generate_results_download_urls(
               results_paths, expiration_minutes=1440
           )
           
           # Update job with result URLs
           job_ref.update({
               'status': 'completed',
               'result_urls': result_urls
           })
       else:
           # Existing Firestore-based processing
           # ... current implementation ...
   ```

### Step 3: iOS App Integration

1. **Add backend API configuration**
   ```swift
   struct Config {
       static let backendURL = "https://your-backend.com/api"
       static let useSignedURLs = true  // Feature flag
   }
   ```

2. **Create combined service**
   ```swift
   class UnifiedVideoAnalysisService: ObservableObject {
       private let directService = VideoAnalysisService()
       private let signedURLService = SignedURLVideoAnalysisService()
       
       func submitVideo(_ url: URL) async throws -> (String, String) {
           if Config.useSignedURLs {
               return try await signedURLService.submitVideoWithSignedURL(url)
           } else {
               return try await directService.submitVideo(url)
           }
       }
   }
   ```

3. **Update UI to use unified service**
   ```swift
   struct VideoAnalysisView: View {
       @StateObject private var analysisService = UnifiedVideoAnalysisService()
       // ... rest of implementation ...
   }
   ```

### Step 4: Testing and Validation

1. **Backend API Testing**
   ```bash
   # Test upload URL generation
   curl -X POST https://your-backend.com/api/upload-url \
        -H "Content-Type: application/json" \
        -d '{"user_id": "test_user", "filename": "test.mov"}'
   ```

2. **Complete workflow testing**
   ```bash
   # Test with local video
   cd tests
   python3 test_signed_url_workflow.py "test_user_123" "../videos/input/kidshoot2.MOV"
   ```

3. **iOS app testing**
   - Test with small video files first
   - Verify upload progress tracking
   - Confirm result download works
   - Test error handling

## Rollback Strategy

If issues arise during migration:

1. **Quick rollback: Feature flag**
   ```swift
   // In iOS app
   Config.useSignedURLs = false  // Switch back to direct Firebase
   ```

2. **Backend rollback: Remove signed URL endpoints**
   - Comment out new API endpoints
   - Worker continues to support both modes

3. **Storage rules rollback**
   ```javascript
   // Remove signed URL access if needed
   allow read, write: if request.auth == null;  // Remove this line
   ```

## Benefits After Migration

### Security Improvements
- ✅ No Firebase credentials in mobile app
- ✅ Time-limited access to files
- ✅ Fine-grained permission control
- ✅ Audit trail of file access

### Scalability Improvements
- ✅ Direct uploads to storage (no backend bottleneck)
- ✅ Reduced Firestore read/write costs
- ✅ Better performance for large files
- ✅ Easier to implement CDN caching

### Operational Improvements
- ✅ Better monitoring and analytics
- ✅ Easier to implement rate limiting
- ✅ More flexible result delivery options
- ✅ Simplified client-side code

## Monitoring and Metrics

Track these metrics during migration:

1. **Upload Success Rate**
   - Direct Firebase vs Signed URL
   - Error types and frequencies

2. **Performance Metrics**
   - Upload time comparison
   - Result delivery time
   - App responsiveness

3. **Cost Analysis**
   - Firestore read/write costs
   - Storage operation costs
   - Bandwidth usage

4. **User Experience**
   - Crash rates
   - User satisfaction scores
   - Support ticket volume

## Troubleshooting Common Issues

### Upload Failures
```
Error: 403 Forbidden on signed URL upload
Solution: Check storage rules allow unsigned access
```

### Download Failures
```
Error: URL expired
Solution: Generate fresh signed URLs (24h expiration)
```

### Backend Errors
```
Error: Service account permissions
Solution: Verify IAM roles for Cloud Storage access
```

### iOS Integration Issues
```
Error: Network timeout
Solution: Implement proper retry logic with exponential backoff
```

This migration approach ensures zero downtime and allows gradual rollout with easy rollback options if needed.
