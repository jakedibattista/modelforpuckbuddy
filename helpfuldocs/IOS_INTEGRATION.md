# 🏒 iOS Integration Guide

Quick guide to add hockey video analysis to your iOS app.

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
            
            // Step 3: Analyze video
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
    
    private func requestAnalysis(storagePath: String) async throws -> AnalysisResult {
        guard let user = Auth.auth().currentUser else {
            throw VideoError.notAuthenticated
        }
        
        let request = APIRequest(
            url: "\(apiBase)/api/analyze-video",
            method: "POST",
            body: [
                "user_id": user.uid,
                "storage_path": storagePath
            ]
        )
        
        let response: AnalysisResponse = try await request.send(timeout: 600) // 10 minutes
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
    let data_analysis: String
    let shots_detected: Int
    let video_duration: Double
    let raw_analysis: RawAnalysis?
    
    var dataAnalysis: String { data_analysis }
}

struct RawAnalysis: Codable {
    // Include if you need coaching endpoints
}

enum VideoError: LocalizedError {
    case notAuthenticated
    case uploadFailed
    case analysisFailed
    
    var errorDescription: String? {
        switch self {
        case .notAuthenticated: return "Please sign in first"
        case .uploadFailed: return "Failed to upload video"
        case .analysisFailed: return "Analysis failed"
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
        
        guard let httpResponse = response as? HTTPURLResponse,
              200...299 ~= httpResponse.statusCode else {
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
    guard let rawAnalysis = analysis.raw_analysis else { return }
    
    do {
        let request = APIRequest(
            url: "\(apiBase)/api/coach/seth",
            method: "POST", 
            body: [
                "user_id": Auth.auth().currentUser?.uid ?? "",
                "raw_analysis": rawAnalysis
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

## Tips

- **Video format**: .mov works best
- **Processing time**: ~2 minutes for analysis
- **Error handling**: Always check for errors and show user-friendly messages
- **Timeouts**: Use 10 minutes for video analysis
- **Testing**: Test with short (10-30 second) hockey videos first

That's it! Your app can now analyze hockey videos and provide feedback to users.