#!/usr/bin/env python3
"""
Puck Buddy Signed URL System Startup Script

This script helps you start and test the complete signed URL system.
"""

import os
import sys
import subprocess
import time
import threading
from pathlib import Path


def check_requirements():
    """Check if all requirements are met."""
    print("🔍 Checking Requirements...")
    
    # Check environment file
    if not os.path.exists('.env'):
        print("❌ .env file not found")
        return False
    
    # Check Firebase service account key
    cred_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', './firebase-service-account-key.json')
    if not os.path.exists(cred_path):
        print(f"❌ Firebase service account key not found: {cred_path}")
        print("\n💡 Download from: https://console.firebase.google.com/project/puck-buddy/settings/serviceaccounts/adminsdk")
        return False
    
    # Check test video
    test_videos = [
        'videos/input/kidshoot2.MOV',
        'videos/input/kidshoot1.MOV',
        'videos/input/jake1.MOV'
    ]
    
    test_video = None
    for video in test_videos:
        if os.path.exists(video):
            test_video = video
            break
    
    if not test_video:
        print("❌ No test videos found in videos/input/")
        return False
    
    print("✅ All requirements met!")
    return True


def start_backend_server():
    """Start the backend API server in a separate thread."""
    def run_backend():
        try:
            print("🚀 Starting Backend API Server...")
            subprocess.run([sys.executable, 'backend_api.py'], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Backend server failed: {e}")
        except KeyboardInterrupt:
            print("🛑 Backend server stopped")
    
    backend_thread = threading.Thread(target=run_backend, daemon=True)
    backend_thread.start()
    
    # Wait for server to start
    print("⏳ Waiting for backend server to start...")
    time.sleep(3)
    
    return backend_thread


def test_system():
    """Test the complete signed URL system."""
    print("\n🧪 Testing Complete Signed URL System...")
    
    # Find a test video
    test_videos = [
        'videos/input/kidshoot2.MOV',
        'videos/input/kidshoot1.MOV', 
        'videos/input/jake1.MOV'
    ]
    
    test_video = None
    for video in test_videos:
        if os.path.exists(video):
            test_video = video
            break
    
    if not test_video:
        print("❌ No test video found")
        return False
    
    try:
        # Run the complete workflow test
        result = subprocess.run([
            sys.executable, str(Path('tests/test_complete_signed_url_workflow.py')),
            '--video', test_video,
            '--user-id', 'test_user_123'
        ], capture_output=True, text=True, timeout=300)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("❌ Test timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def interactive_menu():
    """Show interactive menu for system operations."""
    while True:
        print("\n" + "="*50)
        print("🏒 PUCK BUDDY SIGNED URL SYSTEM")
        print("="*50)
        print("1. Start Backend API Server")
        print("2. Test Backend API Only") 
        print("3. Run Complete System Test")
        print("4. Start Worker (Job Listener)")
        print("5. Process Specific Job")
        print("6. Check System Status")
        print("7. Exit")
        
        choice = input("\nSelect option (1-7): ").strip()
        
        if choice == '1':
            print("\n🚀 Starting Backend API Server...")
            try:
                subprocess.run([sys.executable, 'backend_api.py'])
            except KeyboardInterrupt:
                print("\n🛑 Backend server stopped")
        
        elif choice == '2':
            print("\n🧪 Testing Backend API Only...")
            # Find test video
            test_video = None
            for video in ['videos/input/kidshoot2.MOV', 'videos/input/kidshoot1.MOV']:
                if os.path.exists(video):
                    test_video = video
                    break
            
            if test_video:
                subprocess.run([
                    sys.executable, str(Path('tests/test_complete_signed_url_workflow.py')),
                    '--backend-only', '--video', test_video
                ])
            else:
                print("❌ No test video found")
        
        elif choice == '3':
            # Start backend server in background
            backend_thread = start_backend_server()
            
            # Run complete test
            success = test_system()
            
            if success:
                print("\n🎉 Complete system test PASSED!")
            else:
                print("\n❌ Complete system test FAILED!")
        
        elif choice == '4':
            print("\n👂 Starting Worker (Job Listener)...")
            try:
                subprocess.run([sys.executable, 'worker/app_signed_urls.py', '--mode', 'listen'])
            except KeyboardInterrupt:
                print("\n🛑 Worker stopped")
        
        elif choice == '5':
            job_id = input("Enter Job ID: ").strip()
            if job_id:
                print(f"\n⚙️  Processing Job: {job_id}")
                subprocess.run([
                    sys.executable, 'worker/app_signed_urls.py',
                    '--mode', 'job', '--job-id', job_id
                ])
            else:
                print("❌ Job ID required")
        
        elif choice == '6':
            print("\n📊 System Status Check...")
            
            # Check requirements
            if check_requirements():
                print("✅ System ready for testing")
            else:
                print("❌ System not ready - check requirements above")
        
        elif choice == '7':
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please select 1-7.")


def main():
    """Main entry point."""
    print("🏒 Puck Buddy Signed URL System Startup")
    print("=" * 40)
    
    # Check if running with arguments (automated mode)
    if len(sys.argv) > 1:
        if sys.argv[1] == '--test':
            # Automated testing mode
            if not check_requirements():
                sys.exit(1)
            
            # Start backend
            backend_thread = start_backend_server()
            
            # Run test
            success = test_system()
            
            if success:
                print("\n🎉 AUTOMATED TEST PASSED!")
                sys.exit(0)
            else:
                print("\n❌ AUTOMATED TEST FAILED!")
                sys.exit(1)
        
        elif sys.argv[1] == '--check':
            # Just check requirements
            if check_requirements():
                print("\n✅ System ready!")
                sys.exit(0)
            else:
                print("\n❌ System not ready!")
                sys.exit(1)
    
    # Interactive mode
    if not check_requirements():
        print("\n💡 Please fix the requirements above before continuing.")
        return
    
    interactive_menu()


if __name__ == "__main__":
    main()
