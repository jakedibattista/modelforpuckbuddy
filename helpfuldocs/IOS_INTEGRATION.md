## iOS Integration Guide: Upload → Analyze → Receive Feedback

This comprehensive guide shows how your iOS app uploads a user video, creates a processing job, listens for real-time updates, handles background/push notifications, and recovers results if the app is closed.

**Two Integration Options:**
- **Option A**: Direct Firebase upload + Firestore results (Current implementation)
- **Option B**: Signed URL upload + Signed URL results (Recommended for production)

Both options are supported. Option B provides better security and scalability.

### Prerequisites & Setup

1. **Add Firebase to your iOS project:**
   - Download `GoogleService-Info.plist` from Firebase Console
   - Add to your Xcode project (remember to add to target)
   - Install Firebase via SPM: `https://github.com/firebase/firebase-ios-sdk`
   - Import required modules: `FirebaseAuth`, `FirebaseFirestore`, `FirebaseStorage`, `FirebaseMessaging`

2. **Configure in AppDelegate or App struct:**
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

3. **Project settings:**
   - Firebase Storage bucket: `puck-buddy.firebasestorage.app`
   - Firestore collection: `jobs`
   - Storage path convention: `users/{uid}/{uuid}.mov`
   - **Backend API URL**: `https://puck-buddy-model-22317830094.us-central1.run.app`

### Error Handling
```swift
enum VideoAnalysisError: Error, LocalizedError {
    case notAuthenticated
    case uploadFailed(Error)
    case jobCreationFailed(Error)
    case processingFailed(String)
    case networkError
    case invalidVideo
    
    var errorDescription: String? {
        switch self {
        case .notAuthenticated: return "User not signed in"
        case .uploadFailed(let error): return "Upload failed: \(error.localizedDescription)"
        case .jobCreationFailed(let error): return "Job creation failed: \(error.localizedDescription)"
        case .processingFailed(let message): return "Processing failed: \(message)"
        case .networkError: return "Network connection error"
        case .invalidVideo: return "Invalid video format or size"
        }
    }
}
```

### Complete Service Implementation (Option A: Direct Upload)
```swift
import Foundation
import Firebase
import FirebaseAuth
import FirebaseFirestore
import FirebaseStorage
import FirebaseMessaging

@MainActor
class VideoAnalysisService: ObservableObject {
    @Published var isUploading = false
    @Published var uploadProgress: Double = 0
    @Published var analysisProgress = 0
    @Published var currentStatus = ""
    @Published var parentSummary: String?
    @Published var coachSummary: String?
    @Published var error: VideoAnalysisError?
    
    private var uploadTask: StorageUploadTask?
    private var jobListener: ListenerRegistration?
    
    // MARK: - Public Interface
    
    func submitVideo(_ videoURL: URL) async throws -> (parentSummary: String, coachSummary: String) {
        reset()
        
        guard Auth.auth().currentUser != nil else {
            throw VideoAnalysisError.notAuthenticated
        }
        
        // Validate video (optional - add your constraints)
        try validateVideo(videoURL)
        
        // Upload with progress
        let storagePath = try await uploadVideoWithProgress(videoURL)
        
        // Create job and listen
        let jobId = try await createJob(storagePath: storagePath)
        
        // Wait for completion
        return try await waitForCompletion(jobId: jobId)
    }
    
    func cancelCurrentJob() {
        uploadTask?.cancel()
        jobListener?.remove()
        reset()
    }
    
    func fetchLatestResult() async -> (jobId: String?, parentSummary: String?, coachSummary: String?)? {
        guard let uid = Auth.auth().currentUser?.uid else { return nil }
        
        do {
            let snapshot = try await Firestore.firestore().collection("jobs")
                .whereField("userId", isEqualTo: uid)
                .order(by: "createdAt", descending: true)
                .limit(to: 1)
                .getDocuments()
            
            guard let doc = snapshot.documents.first else { return nil }
            let data = doc.data()
            return (
                jobId: doc.documentID,
                parentSummary: data["parent_summary"] as? String,
                coachSummary: data["coach_summary"] as? String
            )
        } catch {
            return nil
        }
    }
    
    // MARK: - Private Implementation
    
    private func reset() {
        isUploading = false
        uploadProgress = 0
        analysisProgress = 0
        currentStatus = ""
        parentSummary = nil
        coachSummary = nil
        error = nil
        jobListener?.remove()
    }
    
    private func validateVideo(_ url: URL) throws {
        // Add your video validation logic
        let fileSize = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
        if fileSize > 100_000_000 { // 100MB limit example
            throw VideoAnalysisError.invalidVideo
        }
    }
    
    private func uploadVideoWithProgress(_ videoURL: URL) async throws -> String {
        guard let uid = Auth.auth().currentUser?.uid else {
            throw VideoAnalysisError.notAuthenticated
        }
        
        isUploading = true
        currentStatus = "Uploading video..."
        
        let path = "users/\(uid)/\(UUID().uuidString).mov"
        let ref = Storage.storage().reference(withPath: path)
        
        return try await withCheckedThrowingContinuation { continuation in
            uploadTask = ref.putFile(from: videoURL, metadata: nil) { [weak self] _, error in
                DispatchQueue.main.async {
                    self?.isUploading = false
                }
                
                if let error = error {
                    continuation.resume(throwing: VideoAnalysisError.uploadFailed(error))
                } else {
                    continuation.resume(returning: path)
                }
            }
            
            uploadTask?.observe(.progress) { [weak self] snapshot in
                DispatchQueue.main.async {
                    let progress = Double(snapshot.progress?.fractionCompleted ?? 0)
                    self?.uploadProgress = progress
                }
            }
        }
    }
    
    private func createJob(storagePath: String) async throws -> String {
        guard let uid = Auth.auth().currentUser?.uid else {
            throw VideoAnalysisError.notAuthenticated
        }
        
        currentStatus = "Creating analysis job..."
        
        let jobRef = Firestore.firestore().collection("jobs").document()
        let jobData: [String: Any] = [
            "userId": uid,
            "storagePath": storagePath,
            "status": "queued",
            "progress": 0,
            "createdAt": FieldValue.serverTimestamp(),
            "updatedAt": FieldValue.serverTimestamp()
        ]
        
        do {
            try await jobRef.setData(jobData)
            return jobRef.documentID
        } catch {
            throw VideoAnalysisError.jobCreationFailed(error)
        }
    }
    
    private func waitForCompletion(jobId: String) async throws -> (parentSummary: String, coachSummary: String) {
        currentStatus = "Waiting for analysis..."
        
        return try await withCheckedThrowingContinuation { continuation in
            jobListener = Firestore.firestore().collection("jobs").document(jobId)
                .addSnapshotListener { [weak self] snapshot, error in
                    guard let self = self else { return }
                    
                    if let error = error {
                        continuation.resume(throwing: VideoAnalysisError.networkError)
                        return
                    }
                    
                    guard let data = snapshot?.data() else { return }
                    
                    DispatchQueue.main.async {
                        let status = data["status"] as? String ?? "unknown"
                        let progress = data["progress"] as? Int ?? 0
                        
                        self.currentStatus = status.capitalized
                        self.analysisProgress = progress
                        
                        switch status {
                        case "completed":
                            let parent = data["parent_summary"] as? String ?? ""
                            let coach = data["coach_summary"] as? String ?? ""
                            self.parentSummary = parent
                            self.coachSummary = coach
                            self.jobListener?.remove()
                            continuation.resume(returning: (parentSummary: parent, coachSummary: coach))
                            
                        case "failed":
                            let errorMsg = data["error"] as? String ?? "Unknown error"
                            self.jobListener?.remove()
                            continuation.resume(throwing: VideoAnalysisError.processingFailed(errorMsg))
                            
                        default:
                            // Still processing - continue listening
                            break
                        }
                    }
                }
        }
    }
}
```

### Signed URL Service Implementation (Option B: Recommended)
```swift
import Foundation
import Firebase
import FirebaseAuth
import FirebaseFirestore

@MainActor
class SignedURLVideoAnalysisService: ObservableObject {
    @Published var isUploading = false
    @Published var uploadProgress: Double = 0
    @Published var analysisProgress = 0
    @Published var currentStatus = ""
    @Published var parentSummary: String?
    @Published var coachSummary: String?
    @Published var error: VideoAnalysisError?
    
    private var jobListener: ListenerRegistration?
    private let backendBaseURL = "https://your-backend.com/api"  // Update with your backend URL
    
    // MARK: - Public Interface
    
    func submitVideoWithSignedURL(_ videoURL: URL) async throws -> (parentSummary: String, coachSummary: String) {
        reset()
        
        guard let user = Auth.auth().currentUser else {
            throw VideoAnalysisError.notAuthenticated
        }
        
        // Validate video
        try validateVideo(videoURL)
        
        // Request upload URL from backend
        let uploadInfo = try await requestUploadURL(filename: videoURL.lastPathComponent, userId: user.uid)
        
        // Upload video using signed URL
        try await uploadVideoWithSignedURL(videoURL, to: uploadInfo.uploadURL)
        
        // Create job and listen for completion
        let jobId = try await createJobForSignedURL(storagePath: uploadInfo.storagePath, userId: user.uid)
        
        // Wait for completion
        return try await waitForCompletionWithSignedURLs(jobId: jobId)
    }
    
    func fetchLatestResultsWithSignedURLs() async -> (parentSummary: String?, coachSummary: String?)? {
        guard let uid = Auth.auth().currentUser?.uid else { return nil }
        
        // Get results from backend API with signed URLs
        guard let url = URL(string: "\(backendBaseURL)/results/\(uid)") else { return nil }
        
        do {
            let (data, _) = try await URLSession.shared.data(from: url)
            let response = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            let results = response?["results"] as? [[String: Any]]
            
            if let latestResult = results?.first,
               let downloadURLs = latestResult["download_urls"] as? [String: String],
               let parentURL = downloadURLs["parent_summary"],
               let coachURL = downloadURLs["coach_summary"] {
                
                // Download results from signed URLs
                let parentSummary = try await downloadTextFromSignedURL(parentURL)
                let coachSummary = try await downloadTextFromSignedURL(coachURL)
                
                return (parentSummary, coachSummary)
            }
        } catch {
            print("Error fetching results: \(error)")
        }
        
        return nil
    }
    
    // MARK: - Private Implementation
    
    private struct UploadInfo {
        let uploadURL: String
        let storagePath: String
    }
    
    private func reset() {
        isUploading = false
        uploadProgress = 0
        analysisProgress = 0
        currentStatus = ""
        parentSummary = nil
        coachSummary = nil
        error = nil
        jobListener?.remove()
    }
    
    private func validateVideo(_ url: URL) throws {
        let fileSize = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
        if fileSize > 100_000_000 { // 100MB limit
            throw VideoAnalysisError.invalidVideo
        }
    }
    
    private func requestUploadURL(filename: String, userId: String) async throws -> UploadInfo {
        guard let url = URL(string: "\(backendBaseURL)/upload-url") else {
            throw VideoAnalysisError.networkError
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let body = ["user_id": userId, "filename": filename]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        
        let (data, response) = try await URLSession.shared.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            throw VideoAnalysisError.networkError
        }
        
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        guard let uploadURL = json?["upload_url"] as? String,
              let storagePath = json?["storage_path"] as? String else {
            throw VideoAnalysisError.networkError
        }
        
        return UploadInfo(uploadURL: uploadURL, storagePath: storagePath)
    }
    
    private func uploadVideoWithSignedURL(_ videoURL: URL, to uploadURL: String) async throws {
        guard let url = URL(string: uploadURL) else {
            throw VideoAnalysisError.networkError
        }
        
        isUploading = true
        currentStatus = "Uploading video..."
        
        var request = URLRequest(url: url)
        request.httpMethod = "PUT"
        request.setValue("video/quicktime", forHTTPHeaderField: "Content-Type")
        
        return try await withCheckedThrowingContinuation { continuation in
            let uploadTask = URLSession.shared.uploadTask(with: request, fromFile: videoURL) { [weak self] data, response, error in
                DispatchQueue.main.async {
                    self?.isUploading = false
                }
                
                if let error = error {
                    continuation.resume(throwing: VideoAnalysisError.uploadFailed(error))
                } else if let httpResponse = response as? HTTPURLResponse,
                          httpResponse.statusCode == 200 {
                    continuation.resume()
                } else {
                    continuation.resume(throwing: VideoAnalysisError.uploadFailed(NSError(domain: "Upload", code: 0)))
                }
            }
            
            // Mock upload progress (real implementation would track actual progress)
            Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { timer in
                DispatchQueue.main.async { [weak self] in
                    self?.uploadProgress = min((self?.uploadProgress ?? 0) + 0.05, 0.95)
                }
                if self?.uploadProgress ?? 0 >= 0.95 {
                    timer.invalidate()
                }
            }
            
            uploadTask.resume()
        }
    }
    
    private func createJobForSignedURL(storagePath: String, userId: String) async throws -> String {
        currentStatus = "Creating analysis job..."
        
        let jobRef = Firestore.firestore().collection("jobs").document()
        let jobData: [String: Any] = [
            "userId": userId,
            "storagePath": storagePath,
            "status": "queued",
            "progress": 0,
            "delivery_method": "signed_urls",
            "createdAt": FieldValue.serverTimestamp(),
            "updatedAt": FieldValue.serverTimestamp()
        ]
        
        try await jobRef.setData(jobData)
        return jobRef.documentID
    }
    
    private func waitForCompletionWithSignedURLs(jobId: String) async throws -> (parentSummary: String, coachSummary: String) {
        currentStatus = "Waiting for analysis..."
        
        return try await withCheckedThrowingContinuation { continuation in
            jobListener = Firestore.firestore().collection("jobs").document(jobId)
                .addSnapshotListener { [weak self] snapshot, error in
                    guard let self = self else { return }
                    
                    if let error = error {
                        continuation.resume(throwing: VideoAnalysisError.networkError)
                        return
                    }
                    
                    guard let data = snapshot?.data() else { return }
                    
                    DispatchQueue.main.async {
                        let status = data["status"] as? String ?? "unknown"
                        let progress = data["progress"] as? Int ?? 0
                        
                        self.currentStatus = status.capitalized
                        self.analysisProgress = progress
                        
                        switch status {
                        case "completed":
                            if let resultURLs = data["result_urls"] as? [String: String],
                               let parentURL = resultURLs["parent_summary_url"],
                               let coachURL = resultURLs["coach_summary_url"] {
                                
                                // Download results from signed URLs
                                Task {
                                    do {
                                        let parentSummary = try await self.downloadTextFromSignedURL(parentURL)
                                        let coachSummary = try await self.downloadTextFromSignedURL(coachURL)
                                        
                                        await MainActor.run {
                                            self.parentSummary = parentSummary
                                            self.coachSummary = coachSummary
                                            self.jobListener?.remove()
                                        }
                                        
                                        continuation.resume(returning: (parentSummary: parentSummary, coachSummary: coachSummary))
                                    } catch {
                                        continuation.resume(throwing: VideoAnalysisError.networkError)
                                    }
                                }
                            }
                            
                        case "failed":
                            let errorMsg = data["error"] as? String ?? "Unknown error"
                            self.jobListener?.remove()
                            continuation.resume(throwing: VideoAnalysisError.processingFailed(errorMsg))
                            
                        default:
                            // Still processing - continue listening
                            break
                        }
                    }
                }
        }
    }
    
    private func downloadTextFromSignedURL(_ urlString: String) async throws -> String {
        guard let url = URL(string: urlString) else {
            throw VideoAnalysisError.networkError
        }
        
        let (data, _) = try await URLSession.shared.data(from: url)
        guard let text = String(data: data, encoding: .utf8) else {
            throw VideoAnalysisError.networkError
        }
        
        return text
    }
}
```

### SwiftUI Integration Example
```swift
import SwiftUI

struct VideoAnalysisView: View {
    @StateObject private var analysisService = VideoAnalysisService()
    @State private var selectedVideoURL: URL?
    @State private var showingVideoPicker = false
    @State private var showingResults = false
    
    var body: some View {
        VStack(spacing: 20) {
            // Upload Section
            if !analysisService.isUploading && analysisService.currentStatus.isEmpty {
                Button("Select Video to Analyze") {
                    showingVideoPicker = true
                }
                .buttonStyle(.borderedProminent)
            }
            
            // Progress Section
            if analysisService.isUploading {
                VStack {
                    Text("Uploading Video...")
                    ProgressView(value: analysisService.uploadProgress)
                        .progressViewStyle(LinearProgressViewStyle())
                }
            }
            
            if !analysisService.currentStatus.isEmpty && !analysisService.isUploading {
                VStack {
                    Text(analysisService.currentStatus)
                    ProgressView(value: Double(analysisService.analysisProgress) / 100.0)
                        .progressViewStyle(LinearProgressViewStyle())
                    Text("\(analysisService.analysisProgress)%")
                        .font(.caption)
                }
            }
            
            // Results Section
            if let parentSummary = analysisService.parentSummary,
               let coachSummary = analysisService.coachSummary {
                VStack(alignment: .leading, spacing: 16) {
                    Text("Analysis Complete!")
                        .font(.headline)
                        .foregroundColor(.green)
                    
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Performance Summary")
                            .font(.subheadline)
                            .fontWeight(.semibold)
                        Text(parentSummary)
                            .padding()
                            .background(Color.gray.opacity(0.1))
                            .cornerRadius(8)
                    }
                    
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Coaching Feedback")
                            .font(.subheadline)
                            .fontWeight(.semibold)
                        Text(coachSummary)
                            .padding()
                            .background(Color.blue.opacity(0.1))
                            .cornerRadius(8)
                    }
                    
                    Button("Analyze Another Video") {
                        analysisService.cancelCurrentJob()
                        showingVideoPicker = true
                    }
                    .buttonStyle(.bordered)
                }
            }
            
            // Error Section
            if let error = analysisService.error {
                VStack {
                    Text("Error")
                        .font(.headline)
                        .foregroundColor(.red)
                    Text(error.localizedDescription)
                        .foregroundColor(.red)
                    Button("Try Again") {
                        if let url = selectedVideoURL {
                            analyzeVideo(url)
                        } else {
                            showingVideoPicker = true
                        }
                    }
                    .buttonStyle(.bordered)
                }
            }
            
            Spacer()
        }
        .padding()
        .sheet(isPresented: $showingVideoPicker) {
            VideoPickerView { url in
                selectedVideoURL = url
                analyzeVideo(url)
            }
        }
        .onAppear {
            checkForPreviousResults()
        }
    }
    
    private func analyzeVideo(_ url: URL) {
        Task {
            do {
                let results = try await analysisService.submitVideo(url)
                // Results automatically appear via @Published properties
            } catch {
                analysisService.error = error as? VideoAnalysisError ?? .networkError
            }
        }
    }
    
    private func checkForPreviousResults() {
        Task {
            if let result = await analysisService.fetchLatestResult(),
               let parent = result.parentSummary,
               let coach = result.coachSummary {
                await MainActor.run {
                    analysisService.parentSummary = parent
                    analysisService.coachSummary = coach
                }
            }
        }
    }
}

// Simple video picker placeholder - implement based on your needs
struct VideoPickerView: UIViewControllerRepresentable {
    let onVideoSelected: (URL) -> Void
    
    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.mediaTypes = ["public.movie"]
        picker.delegate = context.coordinator
        return picker
    }
    
    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}
    
    func makeCoordinator() -> Coordinator {
        Coordinator(onVideoSelected: onVideoSelected)
    }
    
    class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let onVideoSelected: (URL) -> Void
        
        init(onVideoSelected: @escaping (URL) -> Void) {
            self.onVideoSelected = onVideoSelected
        }
        
        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            if let url = info[.mediaURL] as? URL {
                onVideoSelected(url)
            }
            picker.dismiss(animated: true)
        }
    }
}
```

### Push Notifications Setup
```swift
import FirebaseMessaging

final class AppDelegate: UIResponder, UIApplicationDelegate, MessagingDelegate {
    func application(_ application: UIApplication,
                   didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {
        FirebaseApp.configure()
        Messaging.messaging().delegate = self
        
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .badge, .sound]) { granted, _ in
            if granted {
                DispatchQueue.main.async {
                    application.registerForRemoteNotifications()
                }
            }
        }
        
        return true
    }
    
    func messaging(_ messaging: Messaging, didReceiveRegistrationToken fcmToken: String?) {
        guard let token = fcmToken, let uid = Auth.auth().currentUser?.uid else { return }
        
        // Store device token for push notifications
        Firestore.firestore().collection("users").document(uid).collection("devices").document(token)
            .setData([
                "createdAt": FieldValue.serverTimestamp(),
                "platform": "ios"
            ], merge: true)
    }
    
    // Handle notification tap
    func userNotificationCenter(_ center: UNUserNotificationCenter, 
                              didReceive response: UNNotificationResponse,
                              withCompletionHandler completionHandler: @escaping () -> Void) {
        let userInfo = response.notification.request.content.userInfo
        
        if let jobId = userInfo["jobId"] as? String {
            // Navigate to results screen with this jobId
            NotificationCenter.default.post(name: .showAnalysisResults, object: jobId)
        }
        
        completionHandler()
    }
}

extension Notification.Name {
    static let showAnalysisResults = Notification.Name("showAnalysisResults")
}
```

### Testing & Troubleshooting

1. **Test with small video first** (< 10MB, < 30 seconds)
2. **Check Firebase Console:**
   - Storage: verify video uploaded to `users/{uid}/...`
   - Firestore: verify job document created in `jobs` collection
   - Functions: check logs for any errors

3. **Common issues:**
   - **Upload fails:** Check Firebase Auth, Storage rules, network connection
   - **Job never starts:** Verify Functions deployed, Pub/Sub topic exists
   - **Processing fails:** Check Cloud Run logs for worker errors
   - **No push notifications:** Verify FCM setup, device token registration

4. **Debug logging:**
```swift
// Add to your service for debugging
private func log(_ message: String) {
    print("[VideoAnalysis] \(message)")
}
```

### Data Model Reference
Firestore `jobs/{jobId}` document structure:
```json
{
  "userId": "firebase_auth_uid",
  "storagePath": "users/{uid}/{uuid}.mov",
  "status": "queued|processing|summarizing|completed|failed",
  "progress": 0-100,
  "parent_summary": "Performance summary text...",
  "coach_summary": "What went well:\n- ...\n\nWhat to work on:\n- ...",
  "error": "Error message if failed",
  "createdAt": "2024-01-01T12:00:00Z",
  "updatedAt": "2024-01-01T12:05:30Z"
}
```

### Quick Start Checklist
- [ ] Add Firebase to iOS project (GoogleService-Info.plist)
- [ ] Install Firebase SDK packages
- [ ] Configure Firebase in app startup
- [ ] Copy `VideoAnalysisError` enum
- [ ] Copy `VideoAnalysisService` class
- [ ] Copy `VideoAnalysisView` SwiftUI example
- [ ] Set up push notifications (optional)
- [ ] Test with a small video file

That's it! Your app will upload videos, track progress in real-time, and receive both parent-friendly summaries and detailed coaching feedback.


