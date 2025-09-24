#!/usr/bin/env python3
"""
Complete Signed URL Workflow Test for Puck Buddy

This script tests the entire signed URL workflow:
1. Backend API generates upload URL
2. Simulates React Native app uploading video
3. Backend creates analysis job
4. Worker processes job with signed URLs
5. Client retrieves results via signed URLs

Run this script to verify the complete system works end-to-end.
"""

import os
import sys
import json
import time
import requests
import tempfile
import shutil
from pathlib import Path

# Add repo root to path for imports
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from utils.firebase_storage import FirebaseStorageManager
from worker.app_signed_urls import SignedURLWorker


class SignedURLWorkflowTester:
    """Tests the complete signed URL workflow."""
    
    def __init__(self, backend_url: str = "http://localhost:5000"):
        """
        Initialize the tester.
        
        Args:
            backend_url: URL of the backend API
        """
        self.backend_url = backend_url
        self.firebase_manager = FirebaseStorageManager()
        self.worker = SignedURLWorker()
        
        print(f"🔧 Initialized tester with backend: {backend_url}")
    
    def test_backend_health(self) -> bool:
        """Test if the backend API is running."""
        try:
            response = requests.get(f"{self.backend_url}/health", timeout=5)
            if response.status_code == 200:
                print("✅ Backend API is healthy")
                return True
            else:
                print(f"❌ Backend API unhealthy: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Backend API not reachable: {e}")
            return False
    
    def request_upload_url(self, user_id: str) -> dict:
        """Request an upload URL from the backend API."""
        try:
            payload = {
                "user_id": user_id,
                "content_type": "video/mov"
            }
            
            response = requests.post(
                f"{self.backend_url}/api/upload-url",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Upload URL generated: {data['upload_info']['storage_path']}")
                return data['upload_info']
            else:
                print(f"❌ Failed to get upload URL: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error requesting upload URL: {e}")
            return None
    
    def simulate_video_upload(self, upload_info: dict, video_path: str) -> bool:
        """Simulate React Native app uploading a video."""
        try:
            if not os.path.exists(video_path):
                print(f"❌ Test video not found: {video_path}")
                return False
            
            print(f"📤 Uploading video: {video_path}")
            
            # Read video file
            with open(video_path, 'rb') as f:
                video_data = f.read()
            
            # Upload to signed URL
            headers = {
                'Content-Type': upload_info['content_type']
            }
            
            response = requests.put(
                upload_info['upload_url'],
                data=video_data,
                headers=headers,
                timeout=60  # Allow time for large video uploads
            )
            
            if response.status_code in [200, 204]:
                print(f"✅ Video uploaded successfully")
                return True
            else:
                print(f"❌ Video upload failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error uploading video: {e}")
            return False
    
    def submit_video_for_analysis(self, user_id: str, storage_path: str) -> str:
        """Submit video for analysis via backend API."""
        try:
            payload = {
                "user_id": user_id,
                "storage_path": storage_path
            }
            
            response = requests.post(
                f"{self.backend_url}/api/submit-video",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                job_id = data['job_id']
                print(f"✅ Analysis job created: {job_id}")
                return job_id
            else:
                print(f"❌ Failed to submit video: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error submitting video: {e}")
            return None
    
    def wait_for_job_completion(self, job_id: str, timeout_seconds: int = 300) -> dict:
        """Wait for analysis job to complete."""
        try:
            print(f"⏳ Waiting for job {job_id} to complete...")
            
            start_time = time.time()
            
            while time.time() - start_time < timeout_seconds:
                # Check job status
                job_ref = self.firebase_manager.db.collection('jobs').document(job_id)
                job_doc = job_ref.get()
                
                if job_doc.exists:
                    job_data = job_doc.to_dict()
                    status = job_data.get('status')
                    
                    if status == 'completed':
                        print(f"✅ Job completed successfully")
                        return job_data
                    elif status == 'failed':
                        error = job_data.get('error', 'Unknown error')
                        print(f"❌ Job failed: {error}")
                        return None
                    elif status in ['queued', 'processing']:
                        print(f"   Status: {status}")
                        time.sleep(5)  # Wait 5 seconds before checking again
                    else:
                        print(f"   Unknown status: {status}")
                        time.sleep(5)
                else:
                    print(f"❌ Job {job_id} not found")
                    return None
            
            print(f"❌ Job timed out after {timeout_seconds} seconds")
            return None
            
        except Exception as e:
            print(f"❌ Error waiting for job completion: {e}")
            return None
    
    def test_result_download(self, signed_urls: dict) -> bool:
        """Test downloading results using signed URLs."""
        try:
            print("📥 Testing result downloads...")
            
            success_count = 0
            total_count = len(signed_urls)
            
            for result_type, url in signed_urls.items():
                try:
                    response = requests.get(url, timeout=30)
                    
                    if response.status_code == 200:
                        content_length = len(response.content)
                        print(f"   ✅ {result_type}: {content_length} bytes")
                        success_count += 1
                    else:
                        print(f"   ❌ {result_type}: HTTP {response.status_code}")
                        
                except Exception as e:
                    print(f"   ❌ {result_type}: {e}")
            
            if success_count == total_count:
                print(f"✅ All {total_count} results downloaded successfully")
                return True
            else:
                print(f"⚠️  {success_count}/{total_count} results downloaded")
                return False
                
        except Exception as e:
            print(f"❌ Error testing result downloads: {e}")
            return False
    
    def run_complete_test(self, test_video_path: str, user_id: str = "test_user_123") -> bool:
        """Run the complete signed URL workflow test."""
        print("🏒 STARTING COMPLETE SIGNED URL WORKFLOW TEST")
        print("=" * 60)
        
        try:
            # Step 1: Test backend health
            print("\n1️⃣  Testing Backend API Health")
            if not self.test_backend_health():
                print("❌ Backend API test failed - cannot continue")
                return False
            
            # Step 2: Request upload URL
            print("\n2️⃣  Requesting Upload URL")
            upload_info = self.request_upload_url(user_id)
            if not upload_info:
                print("❌ Upload URL request failed")
                return False
            
            # Step 3: Upload video
            print("\n3️⃣  Uploading Video")
            if not self.simulate_video_upload(upload_info, test_video_path):
                print("❌ Video upload failed")
                return False
            
            # Step 4: Submit for analysis
            print("\n4️⃣  Submitting for Analysis")
            job_id = self.submit_video_for_analysis(user_id, upload_info['storage_path'])
            if not job_id:
                print("❌ Job submission failed")
                return False
            
            # Step 5: Process job (simulate worker)
            print("\n5️⃣  Processing Analysis Job")
            print("   Running worker to process the job...")
            
            # Get job data for worker
            job_ref = self.firebase_manager.db.collection('jobs').document(job_id)
            job_doc = job_ref.get()
            job_data = job_doc.to_dict()
            job_data['job_id'] = job_id
            
            # Process with worker
            if not self.worker.process_job(job_data):
                print("❌ Worker processing failed")
                return False
            
            # Step 6: Wait for completion and get results
            print("\n6️⃣  Retrieving Results")
            completed_job = self.wait_for_job_completion(job_id, timeout_seconds=60)
            if not completed_job:
                print("❌ Job completion check failed")
                return False
            
            # Step 7: Test result downloads
            print("\n7️⃣  Testing Result Downloads")
            signed_urls = completed_job.get('signedResultUrls', {})
            if not signed_urls:
                print("❌ No signed URLs found in completed job")
                return False
            
            if not self.test_result_download(signed_urls):
                print("❌ Result download test failed")
                return False
            
            # Success!
            print("\n" + "=" * 60)
            print("🎉 COMPLETE SIGNED URL WORKFLOW TEST PASSED!")
            print("=" * 60)
            print(f"✅ Job ID: {job_id}")
            print(f"✅ Video Path: {upload_info['storage_path']}")
            print(f"✅ Results Available: {list(signed_urls.keys())}")
            print("\n🏒 Your Puck Buddy signed URL system is working perfectly!")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Test failed with exception: {e}")
            return False


def main():
    """Main entry point for the test script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Puck Buddy Signed URL Workflow')
    parser.add_argument('--video', help='Path to test video file', 
                       default='videos/input/kidshoot2.MOV')
    parser.add_argument('--user-id', help='Test user ID', default='test_user_123')
    parser.add_argument('--backend-url', help='Backend API URL', 
                       default='http://localhost:5000')
    parser.add_argument('--backend-only', action='store_true',
                       help='Test only backend API (no worker processing)')
    
    args = parser.parse_args()
    
    # Check if test video exists
    if not os.path.exists(args.video):
        print(f"❌ Test video not found: {args.video}")
        print("\nAvailable test videos:")
        video_dir = Path("videos/input")
        if video_dir.exists():
            for video_file in video_dir.glob("*.mov") or video_dir.glob("*.MOV"):
                print(f"   {video_file}")
        else:
            print("   No videos/input directory found")
        return
    
    # Initialize tester
    tester = SignedURLWorkflowTester(args.backend_url)
    
    if args.backend_only:
        # Test only the backend API workflow
        print("🔧 Testing Backend API Only")
        print("=" * 40)
        
        if tester.test_backend_health():
            upload_info = tester.request_upload_url(args.user_id)
            if upload_info:
                if tester.simulate_video_upload(upload_info, args.video):
                    job_id = tester.submit_video_for_analysis(args.user_id, upload_info['storage_path'])
                    if job_id:
                        print(f"✅ Backend API test successful! Job ID: {job_id}")
                        print("   Start the worker to process the job:")
                        print(f"   python worker/app_signed_urls.py --mode job --job-id {job_id}")
                    else:
                        print("❌ Backend API test failed")
                else:
                    print("❌ Video upload test failed")
            else:
                print("❌ Upload URL test failed")
        else:
            print("❌ Backend health check failed")
    else:
        # Run complete workflow test
        success = tester.run_complete_test(args.video, args.user_id)
        
        if not success:
            print("\n💡 Troubleshooting Tips:")
            print("1. Make sure Firebase Admin SDK is configured")
            print("2. Check that the backend API is running: python app.py")
            print("3. Verify your .env file has all required variables")
            print("4. Ensure the test video file exists and is readable")
            sys.exit(1)


if __name__ == "__main__":
    main()
