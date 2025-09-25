# 🏒 iOS Integration Guide - Puck Buddy Video Analysis

Quick guide to integrate hockey video analysis into your iOS app using the simple API workflow.

## Prerequisites

1. **Add Firebase to your iOS project**
   - Download `GoogleService-Info.plist` from Firebase Console
   - Add to Xcode project and target
   - Install Firebase SDK via SPM: `https://github.com/firebase/firebase-ios-sdk`

2. **Configure Firebase in your app**
```swift
import Firebase

@main
struct PuckBuddyApp: App {
    init() {
        FirebaseApp.configure()
    }
    var body: some Scene { /* your app */ }
}
```

3. **Project settings**
   - **Backend API**: `https://puck-buddy-model-22317830094.us-central1.run.app`
   - **Firebase Storage bucket**: `puck-buddy.firebasestorage.app`

## Simple Integration (Recommended)

### VideoAnalysisService
```swift
import Foundation
import Firebase
import FirebaseAuth

@MainActor
class VideoAnalysisService: ObservableObject {
    @Published var isProcessing = false
    @Published var progress = 0.0
    @Published var dataAnalysis: String?
    @Published var coachSummary: String?
    @Published var error: String?
    
    private let apiBase = "https://puck-buddy-model-22317830094.us-central1.run.app"
    
    func analyzeVideo(_ videoURL: URL) async {
        guard let user = Auth.auth().currentUser else {
            error = "User not authenticated"
            return
        }
        
        isProcessing = true
        error = nil
        
        do {
            // Step 1: Get upload URL
            let uploadInfo = try await getUploadURL(userId: user.uid)
            
            // Step 2: Upload video
            progress = 0.3
            try await uploadVideo(videoURL, to: uploadInfo.uploadURL)
            
            // Step 3: Analyze video (waits for completion)
            progress = 0.6
            let results = try await analyzeVideo(userId: user.uid, storagePath: uploadInfo.storagePath)
            
            progress = 1.0
            dataAnalysis = results.dataAnalysis
            coachSummary = results.coachSummary
            
        } catch {
            self.error = error.localizedDescription
        }
        
        isProcessing = false
    }
    
    // MARK: - Private API calls
    
    private struct UploadInfo {
        let uploadURL: String
        let storagePath: String
    }
    
    private struct AnalysisResult {
        let dataAnalysis: String
        let coachSummary: String
    }
    
    private func getUploadURL(userId: String) async throws -> UploadInfo {
        let url = URL(string: "\(apiBase)/api/upload-url")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let body = ["user_id": userId, "content_type": "video/mov"]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        
        let (data, _) = try await URLSession.shared.data(for: request)
        let response = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let uploadInfo = response["upload_info"] as! [String: Any]
        
        return UploadInfo(
            uploadURL: uploadInfo["upload_url"] as! String,
            storagePath: uploadInfo["storage_path"] as! String
        )
    }
    
    private func uploadVideo(_ videoURL: URL, to uploadURL: String) async throws {
        let url = URL(string: uploadURL)!
        var request = URLRequest(url: url)
        request.httpMethod = "PUT"
        request.setValue("video/mov", forHTTPHeaderField: "Content-Type")
        
        let (_, response) = try await URLSession.shared.upload(for: request, fromFile: videoURL)
        
        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            throw URLError(.badServerResponse)
        }
    }
    
    private func analyzeVideo(userId: String, storagePath: String) async throws -> AnalysisResult {
        let url = URL(string: "\(apiBase)/api/analyze-video")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 600 // 10 minutes
        
        let body = ["user_id": userId, "storage_path": storagePath]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        
        let (data, _) = try await URLSession.shared.data(for: request)
        let response = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        
        guard response["success"] as? Bool == true,
              let analysis = response["analysis"] as? [String: Any],
              let dataAnalysis = analysis["data_analysis"] as? String,
              let coachSummary = analysis["coach_summary"] as? String else {
            let errorMsg = response["error"] as? String ?? "Analysis failed"
            throw NSError(domain: "Analysis", code: 0, userInfo: [NSLocalizedDescriptionKey: errorMsg])
        }
        
        return AnalysisResult(dataAnalysis: dataAnalysis, coachSummary: coachSummary)
    }
}
```

### SwiftUI View
```swift
import SwiftUI
import PhotosUI

struct VideoAnalysisView: View {
    @StateObject private var analysisService = VideoAnalysisService()
    @State private var selectedItem: PhotosPickerItem?
    @State private var showingResults = false
    
    var body: some View {
        VStack(spacing: 20) {
            // Video picker
            PhotosPicker(
                selection: $selectedItem,
                matching: .videos,
                photoLibrary: .shared()
            ) {
                Label("Select Hockey Video", systemImage: "video.circle")
                    .font(.headline)
                    .foregroundColor(.white)
                    .padding()
                    .background(Color.blue)
                    .cornerRadius(12)
            }
            .disabled(analysisService.isProcessing)
            
            // Progress
            if analysisService.isProcessing {
                VStack {
                    Text("Analyzing your hockey video...")
                        .font(.headline)
                    
                    ProgressView(value: analysisService.progress)
                        .progressViewStyle(LinearProgressViewStyle())
                        .frame(height: 8)
                    
                    Text("\(Int(analysisService.progress * 100))%")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding()
            }
            
            // Results
            if let dataAnalysis = analysisService.dataAnalysis,
               let coachSummary = analysisService.coachSummary {
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        Text("🎯 Performance Summary")
                            .font(.headline)
                        Text(dataAnalysis)
                            .padding()
                            .background(Color.gray.opacity(0.1))
                            .cornerRadius(8)
                        
                        Text("🏒 Coaching Feedback")
                            .font(.headline)
                        Text(coachSummary)
                            .padding()
                            .background(Color.blue.opacity(0.1))
                            .cornerRadius(8)
                        
                        Button("Analyze Another Video") {
                            reset()
                        }
                        .buttonStyle(.bordered)
                    }
                }
            }
            
            // Error
            if let error = analysisService.error {
                Text("Error: \(error)")
                    .foregroundColor(.red)
                    .multilineTextAlignment(.center)
                
                Button("Try Again") {
                    reset()
                }
                .buttonStyle(.bordered)
            }
            
            Spacer()
        }
        .padding()
        .onChange(of: selectedItem) { newItem in
            Task {
                await loadAndAnalyzeVideo(newItem)
            }
        }
    }
    
    private func loadAndAnalyzeVideo(_ item: PhotosPickerItem?) async {
        guard let item = item else { return }
        
        do {
            guard let videoURL = try await item.loadTransferable(type: VideoFile.self)?.url else {
                analysisService.error = "Failed to load video"
                return
            }
            
            await analysisService.analyzeVideo(videoURL)
        } catch {
            analysisService.error = error.localizedDescription
        }
    }
    
    private func reset() {
        analysisService.dataAnalysis = nil
        analysisService.coachSummary = nil
        analysisService.error = nil
        selectedItem = nil
    }
}

// Helper for video file transfer
struct VideoFile: Transferable {
    let url: URL
    
    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { video in
            SentTransferredFile(video.url)
        } importing: { received in
            let copy = URL.documentsDirectory.appending(path: "imported_video.mov")
            if FileManager.default.fileExists(atPath: copy.path()) {
                try FileManager.default.removeItem(at: copy)
            }
            try FileManager.default.copyItem(at: received.file, to: copy)
            return VideoFile(url: copy)
        }
    }
}
```

## Error Handling

Common errors and solutions:

```swift
// Handle specific API errors
switch errorMessage {
case "Video analysis system is temporarily unavailable":
    // System maintenance - try again later
    break
case "user_id and storage_path are required":
    // Authentication issue
    break
case "Failed to analyze video":
    // Video processing error - check format/size
    break
default:
    // Generic error handling
    break
}
```

## Requirements

- **Video format**: .mov, .mp4 recommended
- **Video size**: Under 100MB for best performance  
- **Content**: Hockey shooting drills work best
- **Processing time**: 2-10 minutes depending on video length
- **Authentication**: Firebase Auth required

## Response Format

Your app will receive:

```json
{
  "success": true,
  "analysis": {
    "data_analysis": "Shots detected at timestamp: 8.2s, 15.7s...",
    "coach_summary": "## What Went Well\n\n- Good knee bend...",
    "shots_detected": 3,
    "video_duration": 45.2,
    "pose_analysis": true
  }
}
```

## Quick Start Checklist

- [ ] Add Firebase to iOS project
- [ ] Install Firebase SDK packages  
- [ ] Copy `VideoAnalysisService` class
- [ ] Copy `VideoAnalysisView` SwiftUI example
- [ ] Test with a hockey video file

That's it! Your app will upload videos and receive detailed hockey analysis feedback. 🏒