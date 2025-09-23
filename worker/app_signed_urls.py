#!/usr/bin/env python3
"""
Puck Buddy Worker with Signed URL Support

This worker processes video analysis jobs using signed URLs for:
1. Downloading videos from Firebase Storage
2. Uploading analysis results back to Firebase Storage
3. Updating Firestore with signed download URLs

Modified from the original worker/app.py to support the signed URL workflow.
"""

import os
import sys
import json
import logging
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.firebase_storage import FirebaseStorageManager
from analysis.shooting_drill_feedback import analyze_drill
from agents.parent_feedback_agent import generate_summary_with_gemini
from agents.improvement_coach_agent import generate_coaching_summary

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SignedURLWorker:
    """Worker that processes video analysis jobs using signed URLs."""
    
    def __init__(self):
        """Initialize the worker with Firebase manager."""
        self.firebase_manager = FirebaseStorageManager()
        logger.info("Initialized SignedURLWorker")
    
    def process_job(self, job_data: dict) -> bool:
        """
        Process a single analysis job using signed URLs.
        
        Args:
            job_data: Job document data from Firestore
            
        Returns:
            True if processing successful, False otherwise
        """
        job_id = job_data.get('job_id')
        user_id = job_data.get('userId')
        video_storage_path = job_data.get('videoStoragePath')
        
        logger.info(f"Processing job {job_id} for user {user_id}")
        
        temp_video_path = None
        
        try:
            # Step 1: Generate signed URL for video download
            logger.info("Generating signed download URL for video")
            signed_download_url = self.firebase_manager.generate_video_download_url(
                video_storage_path, expiration_hours=2
            )
            
            # Step 2: Download video to temporary location
            logger.info("Downloading video for processing")
            temp_video_path = tempfile.mktemp(suffix='.mov')
            self.firebase_manager.download_video_for_processing(
                signed_download_url, temp_video_path
            )
            
            # Step 3: Run pose analysis
            logger.info("Running pose analysis")
            analysis_results = analyze_drill(temp_video_path)
            
            if not analysis_results or 'shot_events' not in analysis_results:
                raise ValueError("Analysis returned no valid results")
            
            # Step 4: Generate agent summaries
            logger.info("Generating parent summary")
            parent_summary = generate_summary_with_gemini(analysis_results)
            
            logger.info("Generating coach analysis")
            coach_analysis = generate_coaching_summary(analysis_results)
            
            # Step 5: Upload results to Firebase Storage
            logger.info("Uploading analysis results")
            video_filename = os.path.basename(video_storage_path)
            
            storage_paths = self.firebase_manager.upload_analysis_results(
                user_id=user_id,
                video_filename=video_filename,
                analysis_data=analysis_results,
                parent_summary=parent_summary,
                coach_analysis=coach_analysis
            )
            
            # Step 6: Complete job with signed URLs
            logger.info("Completing job with signed URLs")
            self.firebase_manager.complete_analysis_job(job_id, storage_paths)
            
            logger.info(f"Successfully processed job {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process job {job_id}: {e}")
            
            # Update job status to failed
            try:
                job_ref = self.firebase_manager.db.collection('jobs').document(job_id)
                job_ref.update({
                    'status': 'failed',
                    'error': str(e),
                    'failedAt': self.firebase_manager.db.SERVER_TIMESTAMP,
                    'updatedAt': self.firebase_manager.db.SERVER_TIMESTAMP
                })
            except Exception as update_error:
                logger.error(f"Failed to update job status: {update_error}")
            
            return False
            
        finally:
            # Cleanup temporary video file
            if temp_video_path and os.path.exists(temp_video_path):
                try:
                    os.unlink(temp_video_path)
                    logger.info("Cleaned up temporary video file")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup temp file: {cleanup_error}")
    
    def listen_for_jobs(self):
        """
        Listen for new jobs in Firestore and process them.
        
        This is a simple polling implementation. In production, you might want
        to use Pub/Sub triggers or Cloud Functions.
        """
        logger.info("Starting job listener")
        
        try:
            # Query for queued jobs with signed URL delivery method
            jobs_ref = (self.firebase_manager.db.collection('jobs')
                       .where('status', '==', 'queued')
                       .where('deliveryMethod', '==', 'signed_urls')
                       .limit(1))  # Process one job at a time
            
            while True:
                try:
                    # Get next queued job
                    jobs = list(jobs_ref.stream())
                    
                    if jobs:
                        for job_doc in jobs:
                            # Mark job as processing
                            job_data = job_doc.to_dict()
                            job_data['job_id'] = job_doc.id
                            
                            logger.info(f"Found job {job_doc.id}, marking as processing")
                            
                            job_doc.reference.update({
                                'status': 'processing',
                                'startedAt': self.firebase_manager.db.SERVER_TIMESTAMP,
                                'updatedAt': self.firebase_manager.db.SERVER_TIMESTAMP
                            })
                            
                            # Process the job
                            self.process_job(job_data)
                    
                    else:
                        # No jobs found, wait before checking again
                        import time
                        time.sleep(10)  # Wait 10 seconds
                        
                except Exception as e:
                    logger.error(f"Error in job listener loop: {e}")
                    import time
                    time.sleep(30)  # Wait 30 seconds before retrying
                    
        except KeyboardInterrupt:
            logger.info("Job listener stopped by user")
        except Exception as e:
            logger.error(f"Job listener failed: {e}")
            raise


def process_single_job_by_id(job_id: str) -> bool:
    """
    Process a specific job by ID (for testing/debugging).
    
    Args:
        job_id: Firestore job document ID
        
    Returns:
        True if successful, False otherwise
    """
    try:
        worker = SignedURLWorker()
        
        # Get job document
        job_ref = worker.firebase_manager.db.collection('jobs').document(job_id)
        job_doc = job_ref.get()
        
        if not job_doc.exists:
            logger.error(f"Job {job_id} not found")
            return False
        
        job_data = job_doc.to_dict()
        job_data['job_id'] = job_id
        
        logger.info(f"Processing specific job: {job_id}")
        
        # Process the job
        return worker.process_job(job_data)
        
    except Exception as e:
        logger.error(f"Failed to process job {job_id}: {e}")
        return False


def process_video_file_directly(video_path: str, user_id: str) -> dict:
    """
    Process a local video file directly (for testing).
    
    Args:
        video_path: Path to local video file
        user_id: User ID for result organization
        
    Returns:
        Dictionary with analysis results and summaries
    """
    try:
        logger.info(f"Processing video file directly: {video_path}")
        
        # Run analysis
        analysis_results = analyze_drill(video_path)
        
        # Generate summaries
        parent_summary = generate_summary_with_gemini(analysis_results)
        coach_analysis = generate_coaching_summary(analysis_results)
        
        return {
            'analysis': analysis_results,
            'parent_summary': parent_summary,
            'coach_analysis': coach_analysis
        }
        
    except Exception as e:
        logger.error(f"Failed to process video file: {e}")
        raise


def main():
    """Main entry point for the worker."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Puck Buddy Signed URL Worker')
    parser.add_argument('--mode', choices=['listen', 'job', 'video'], 
                       default='listen',
                       help='Worker mode: listen for jobs, process specific job, or process video file')
    parser.add_argument('--job-id', help='Specific job ID to process (for job mode)')
    parser.add_argument('--video-path', help='Path to video file (for video mode)')
    parser.add_argument('--user-id', help='User ID (for video mode)', default='test_user')
    
    args = parser.parse_args()
    
    if args.mode == 'listen':
        # Listen for new jobs continuously
        worker = SignedURLWorker()
        worker.listen_for_jobs()
        
    elif args.mode == 'job':
        # Process a specific job
        if not args.job_id:
            logger.error("Job ID required for job mode")
            return
        
        success = process_single_job_by_id(args.job_id)
        if success:
            logger.info("Job processed successfully")
        else:
            logger.error("Job processing failed")
            sys.exit(1)
            
    elif args.mode == 'video':
        # Process a video file directly
        if not args.video_path:
            logger.error("Video path required for video mode")
            return
        
        if not os.path.exists(args.video_path):
            logger.error(f"Video file not found: {args.video_path}")
            return
        
        results = process_video_file_directly(args.video_path, args.user_id)
        
        # Print results
        print("\n" + "="*50)
        print("ANALYSIS RESULTS")
        print("="*50)
        print(json.dumps(results['analysis'], indent=2))
        
        print("\n" + "="*50)
        print("PARENT SUMMARY")
        print("="*50)
        print(results['parent_summary'])
        
        print("\n" + "="*50)
        print("COACH ANALYSIS")
        print("="*50)
        print(results['coach_analysis'])


if __name__ == "__main__":
    main()
