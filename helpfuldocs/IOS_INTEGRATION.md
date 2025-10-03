# 🏒 iOS Integration Guide

Quick guide to add hockey video analysis to your iOS app.

**🎯 For iOS/Mobile Apps - Use the Simple/Direct Workflow:**
This guide uses `/api/analyze-video` (not `/api/submit-video`) because it processes immediately and returns complete results (~2 minutes). Perfect for mobile apps!

## Setup

### 1. Add Firebase
```swift
import Firebase

@main
struct YourApp: App {
    init() {
        FirebaseApp.configure()
    }
}
```

### 2. Service Class
Create this service to handle video analysis:

```swift
import Foundation
import Firebase
import FirebaseAuth

@MainActor
class VideoAnalysisService: ObservableObject {
    @Published var isProcessing = false
    @Published var dataAnalysis: String?
    @Published var error: String?
    
    private let apiBase = "https://puck-buddy-model-22317830094.us-central1.run.app"
    
    func analyzeVideo(_ videoURL: URL) async {
        isProcessing = true
        error = nil
        
        do {
            // Step 1: Get upload URL
            let uploadInfo = try await getUploadURL()
            
            // Step 2: Upload video
            try await uploadVideo(videoURL, to: uploadInfo.uploadURL)
            
            // Step 3: Analyze video (direct processing - waits ~2 min, returns complete results)
            let analysis = try await requestAnalysis(storagePath: uploadInfo.storagePath)
            
            dataAnalysis = analysis.dataAnalysis
            
        } catch {
            self.error = error.localizedDescription
        }
        
        isProcessing = false
    }
    
    // MARK: - Private Methods
    
    private func getUploadURL() async throws -> UploadInfo {
        guard let user = Auth.auth().currentUser else {
            throw VideoError.notAuthenticated
        }
        
        let request = APIRequest(
            url: "\(apiBase)/api/upload-url",
            method: "POST",
            body: [
                "user_id": user.uid,
                "content_type": "video/mov"
            ]
        )
        
        let response: UploadResponse = try await request.send()
        return response.upload_info
    }
    
    private func uploadVideo(_ videoURL: URL, to uploadURL: String) async throws {
        let videoData = try Data(contentsOf: videoURL)
        
        var request = URLRequest(url: URL(string: uploadURL)!)
        request.httpMethod = "PUT"
        request.setValue("video/mov", forHTTPHeaderField: "Content-Type")
        request.httpBody = videoData
        
        let (_, response) = try await URLSession.shared.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            throw VideoError.uploadFailed
        }
    }
    
    private func requestAnalysis(storagePath: String, ageGroup: String = "17+") async throws -> AnalysisResult {
        guard let user = Auth.auth().currentUser else {
            throw VideoError.notAuthenticated
        }
        
        let request = APIRequest(
            url: "\(apiBase)/api/analyze-video",
            method: "POST",
            body: [
                "user_id": user.uid,
                "storage_path": storagePath,
                "age_group": ageGroup  // Optional: "7-9", "10-12", "13-16", "17+" (default: "17+")
            ]
        )
        
        let response: AnalysisResponse = try await request.send(timeout: 600) // 10 minutes (analyze-video takes ~2 min)
        return response.analysis
    }
}

// MARK: - Data Models

struct UploadResponse: Codable {
    let success: Bool
    let upload_info: UploadInfo
}

struct UploadInfo: Codable {
    let upload_url: String
    let storage_path: String
    
    var uploadURL: String { upload_url }
    var storagePath: String { storage_path }
}

struct AnalysisResponse: Codable {
    let success: Bool
    let analysis: AnalysisResult
}

struct AnalysisResult: Codable {
    let data_summary: String
    let shots_detected: Int
    let video_duration: Double
    let data_analysis: DataAnalysis?
    
    var dataAnalysis: String { data_summary }
}

struct DataAnalysis: Codable {
    // Include if you need coaching endpoints
}

enum VideoError: LocalizedError {
    case notAuthenticated
    case uploadFailed
    case analysisFailed
    case rateLimitExceeded
    
    var errorDescription: String? {
        switch self {
        case .notAuthenticated: return "Please sign in first"
        case .uploadFailed: return "Failed to upload video"
        case .analysisFailed: return "Analysis failed"
        case .rateLimitExceeded: return "You've reached the hourly analysis limit (10 videos/hour). Please try again later!"
        }
    }
}

// MARK: - API Helper

struct APIRequest {
    let url: String
    let method: String
    let body: [String: Any]?
    
    func send<T: Codable>(timeout: TimeInterval = 60) async throws -> T {
        var request = URLRequest(url: URL(string: url)!)
        request.httpMethod = method
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = timeout
        
        if let body = body {
            request.httpBody = try JSONSerialization.data(withJSONObject: body)
        }
        
        let (data, response) = try await URLSession.shared.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw VideoError.analysisFailed
        }
        
        // Handle rate limiting
        if httpResponse.statusCode == 429 {
            throw VideoError.rateLimitExceeded
        }
        
        guard 200...299 ~= httpResponse.statusCode else {
            throw VideoError.analysisFailed
        }
        
        return try JSONDecoder().decode(T.self, from: data)
    }
}
```

### 3. SwiftUI View
```swift
import SwiftUI
import PhotosUI

struct VideoAnalysisView: View {
    @StateObject private var analysisService = VideoAnalysisService()
    @State private var selectedVideo: PhotosPickerItem?
    @State private var showingResults = false
    
    var body: some View {
        VStack(spacing: 20) {
            PhotosPicker("Select Hockey Video", selection: $selectedVideo, matching: .videos)
                .buttonStyle(.borderedProminent)
                .disabled(analysisService.isProcessing)
            
            if analysisService.isProcessing {
                VStack {
                    ProgressView()
                    Text("Analyzing video...")
                        .font(.caption)
                }
            }
            
            if let error = analysisService.error {
                Text(error)
                    .foregroundColor(.red)
                    .font(.caption)
            }
        }
        .onChange(of: selectedVideo) { item in
            guard let item = item else { return }
            loadVideo(from: item)
        }
        .sheet(isPresented: $showingResults) {
            ResultsView(dataAnalysis: analysisService.dataAnalysis ?? "")
        }
        .onChange(of: analysisService.dataAnalysis) { analysis in
            if analysis != nil {
                showingResults = true
            }
        }
    }
    
    private func loadVideo(from item: PhotosPickerItem) {
        Task {
            guard let videoURL = try? await item.loadTransferable(type: VideoFile.self)?.url else {
                return
            }
            
            await analysisService.analyzeVideo(videoURL)
        }
    }
}

struct ResultsView: View {
    let dataAnalysis: String
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            ScrollView {
                Text(dataAnalysis)
                    .padding()
            }
            .navigationTitle("Analysis Results")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}

// Helper for loading videos
struct VideoFile: Transferable {
    let url: URL
    
    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { video in
            SentTransferredFile(video.url)
        } importing: { received in
            let copy = URL.documentsDirectory.appending(path: "video.mov")
            if FileManager.default.fileExists(atPath: copy.path()) {
                try FileManager.default.removeItem(at: copy)
            }
            try FileManager.default.copyItem(at: received.file, to: copy)
            return VideoFile(url: copy)
        }
    }
}
```

## Optional: Seth Coaching

To add Seth's coaching feedback:

```swift
// Add to VideoAnalysisService
@Published var coachingFeedback: String?

func getSethCoaching(from analysis: AnalysisResult) async {
    guard let dataAnalysis = analysis.data_analysis else { return }
    
    do {
        let request = APIRequest(
            url: "\(apiBase)/api/coach/seth",
            method: "POST", 
            body: [
                "user_id": Auth.auth().currentUser?.uid ?? "",
                "data_analysis": dataAnalysis
            ]
        )
        
        let response: CoachingResponse = try await request.send()
        coachingFeedback = response.coaching_feedback
    } catch {
        print("Coaching failed: \(error)")
    }
}

struct CoachingResponse: Codable {
    let success: Bool
    let coaching_feedback: String
}
```

## Rate Limits & Best Practices

### API Rate Limits (Per User)
The backend enforces the following limits per authenticated user:

| Endpoint | Limit | Purpose |
|----------|-------|---------|
| `/api/upload-url` | 20/hour | Upload URL generation |
| `/api/analyze-video` | **10/hour** | Video analysis (main limit) |
| All endpoints | 200/day | Total daily usage |

**What happens when limit exceeded:**
- API returns `429 Too Many Requests`
- User sees: "You've reached the hourly analysis limit (10 videos/hour). Please try again later!"
- Automatically handled by the `VideoError.rateLimitExceeded` case

### Best Practices

**1. Show User Feedback:**
```swift
// Add to VideoAnalysisService
@Published var videosAnalyzedToday = 0
@Published var videosRemainingThisHour = 10

// Update after each analysis
videosAnalyzedToday += 1
videosRemainingThisHour = max(0, 10 - videosAnalyzedToday % 10)
```

**2. Validate Before Upload:**
```swift
func canAnalyzeVideo() -> Bool {
    // Check file size
    guard videoSize < 100_000_000 else { // 100MB
        error = "Video too large. Please use a video under 100MB"
        return false
    }
    
    // Add your own tracking if needed
    return true
}
```

**3. Handle Long Processing Times:**
```swift
// Already configured with 10-minute timeout
let response: AnalysisResponse = try await request.send(timeout: 600)
```

### Security Notes
- **Firebase Auth Required**: All API calls require authenticated Firebase users
- **Private Storage**: Users can only access their own videos and results
- **Automatic Cleanup**: Videos and results are deleted after 30 days
- **Signed URLs**: All uploads use secure, time-limited signed URLs

---

## Tips

- **Video format**: .mov works best
- **Processing time**: ~2 minutes for analysis
- **Error handling**: Always check for errors and show user-friendly messages
- **Timeouts**: Use 10 minutes for video analysis
- **Testing**: Test with short (10-30 second) hockey videos first

That's it! Your app can now analyze hockey videos with proper error handling, rate limiting, and security.