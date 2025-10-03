#!/usr/bin/env python3
"""Flask API that exposes the signed URL workflow for Puck Buddy."""

from __future__ import annotations

import logging
import os
from typing import Dict

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from utils.firebase_storage import FirebaseStorageManager
from firebase_admin import firestore
from agents.openice_agent import OpenIceAgent


def create_app() -> Flask:
    app = Flask(__name__)
    
    # Enhanced CORS configuration for browser/Expo web compatibility
    CORS(app, 
         origins=os.getenv("CORS_ORIGIN", "*"),
         methods=["GET", "POST", "OPTIONS"],
         allow_headers=["Content-Type", "Authorization"],
         supports_credentials=True)

    # Admin users who get unlimited API access
    ADMIN_USERS = {
        "jakedibattista",              # Your username
        "xzRxXo2C65TaVzh32FpWDEqDycA2", # Your Firebase user ID (from logs)
        "admin",                       # Generic admin user
        # Add more admin user IDs here as needed
    }
    
    # Rate limiting - per user limits to prevent abuse and control costs
    def get_user_id_from_request():
        """Extract user_id from request payload for rate limiting."""
        try:
            data = request.get_json(silent=True) or {}
            user_id = data.get('user_id') or get_remote_address()
            username = data.get('username')  # Check username field for admin access
            
            # Check if user is admin (check both user_id and username fields)
            if user_id in ADMIN_USERS or username in ADMIN_USERS:
                return f"admin_{user_id or username}"  # Special admin key
            return user_id
        except:
            return get_remote_address()
    
    limiter = Limiter(
        app=app,
        key_func=get_user_id_from_request,
        default_limits=["200 per day", "50 per hour"],  # Global limits
        storage_uri="memory://"  # Use in-memory storage (simple, no Redis needed)
    )
    
    # Override rate limits for admin users
    @limiter.request_filter
    def admin_bypass():
        """Bypass rate limits for admin users."""
        try:
            data = request.get_json(silent=True) or {}
            user_id = data.get('user_id')
            username = data.get('username')
            
            logger.info(f"🔐 Rate limit check - user_id: {user_id}, username: {username}")
            
            if user_id in ADMIN_USERS or username in ADMIN_USERS:
                logger.info(f"✅ ADMIN BYPASS GRANTED for {user_id or username}")
                return True  # Bypass rate limiting
        except Exception as e:
            logger.error(f"❌ Error in admin_bypass: {e}")
            pass
        return False  # Apply normal rate limiting

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("signed_url_api")

    manager: FirebaseStorageManager | None = None
    openice_agent: OpenIceAgent | None = None

    def get_manager() -> FirebaseStorageManager:
        nonlocal manager
        if manager is None:
            manager = FirebaseStorageManager()
            logger.info("FirebaseStorageManager initialised")
        return manager

    def get_openice_agent() -> OpenIceAgent:
        nonlocal openice_agent
        if openice_agent is None:
            try:
                openice_agent = OpenIceAgent()
                logger.info("OpenIce agent initialised")
            except Exception as exc:
                logger.error(f"Failed to initialize OpenIce agent: {exc}")
                raise
        return openice_agent

    @app.route("/health", methods=["GET"])
    def health_check() -> Dict[str, str]:
        return {
            "status": "healthy",
            "service": "puck-buddy-signed-url-api",
        }
    
    @app.route("/api/admin/status", methods=["POST"])
    def check_admin_status() -> tuple:
        """Check if a user has admin privileges."""
        payload = request.get_json() or {}
        user_id = payload.get("user_id")
        username = payload.get("username")
        
        if not user_id and not username:
            return jsonify({"error": "user_id or username is required"}), 400
        
        is_admin = user_id in ADMIN_USERS or username in ADMIN_USERS
        return jsonify({
            "success": True,
            "user_id": user_id,
            "username": username,
            "is_admin": is_admin,
            "unlimited_access": is_admin,
            "rate_limits_bypassed": is_admin
        })

    @app.route("/api/upload-url", methods=["POST"])
    @limiter.limit("20 per hour")  # Allow more upload URL requests
    def generate_upload_url() -> tuple:
        payload = request.get_json() or {}
        user_id = payload.get("user_id")
        if not user_id:
            return jsonify({"error": "user_id is required"}), 400

        content_type = payload.get("content_type", "video/mov")

        try:
            upload_info = get_manager().generate_video_upload_url(
                user_id=user_id, content_type=content_type
            )
            return jsonify({"success": True, "upload_info": upload_info})
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to create upload URL")
            return jsonify({"error": "Failed to create upload URL"}), 500

    @app.route("/api/submit-video", methods=["POST"])
    @limiter.limit("10 per hour")  # Limit expensive video processing
    def submit_video() -> tuple:
        payload = request.get_json() or {}
        user_id = payload.get("user_id")
        storage_path = payload.get("storage_path")

        if not user_id or not storage_path:
            return jsonify({"error": "user_id and storage_path are required"}), 400

        try:
            job_id = get_manager().create_analysis_job(user_id, storage_path)
            return jsonify({"success": True, "job_id": job_id})
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to create analysis job")
            return jsonify({"error": "Failed to create analysis job"}), 500


    @app.route("/api/analyze-video", methods=["POST"])
    @limiter.limit("10 per hour")  # Limit expensive video processing (main cost driver)
    def analyze_video_simple() -> tuple:
        """Simple endpoint: process video analysis with simplified AI feedback."""
        payload = request.get_json() or {}
        user_id = payload.get("user_id")
        storage_path = payload.get("storage_path")
        age_group = payload.get("age_group", "17+")  # NEW: Accept age group parameter
        
        if not user_id or not storage_path:
            return jsonify({"error": "user_id and storage_path are required"}), 400
            
        try:
            manager = get_manager()
            
            # Check if user is admin and log accordingly
            username = payload.get("username")
            is_admin = user_id in ADMIN_USERS or username in ADMIN_USERS
            if is_admin:
                logger.info(f"🔑 ADMIN REQUEST - Processing video analysis for admin user {user_id or username}, storage_path: {storage_path}, age_group: {age_group}")
            else:
                logger.info(f"Processing video analysis for user {user_id}, storage_path: {storage_path}, age_group: {age_group}")
            
            # Verify the video exists and get metadata
            blob = manager.bucket.blob(storage_path)
            if not blob.exists():
                return jsonify({"error": f"Video not found: {storage_path}"}), 404
            
            # Get video metadata
            blob.reload()  # Refresh to get latest metadata
            video_size_mb = blob.size / (1024 * 1024) if blob.size else 0
            
            # Generate AI analysis based on video metadata and user patterns
            logger.info("Generating AI analysis based on video characteristics")
            
            try:
                # Import the real video analysis modules
                import tempfile
                import os
                from analysis.pose_extraction_shooting_drills import analyze_drill
                # Using local formatting only (Gemini integration removed)
                
                # Download video to temporary location for processing
                logger.info("Downloading video for pose analysis")
                temp_video_path = tempfile.mktemp(suffix='.mov')
                
                try:
                    # Download the video file
                    blob.download_to_filename(temp_video_path)
                    logger.info(f"Downloaded video to: {temp_video_path}")
                    
                    # Run the real pose analysis
                    logger.info("Running MediaPipe pose analysis on video")
                    analysis_results = analyze_drill(temp_video_path)
                    
                    # Use correct keys from analyzer output
                    shots = analysis_results.get('shots', []) if analysis_results else []
                    
                    # Add age_group to each shot for age-based scoring
                    for shot in shots:
                        shot['age_group'] = age_group
                    video_duration = analysis_results.get('duration_est_sec', 0) if analysis_results else 0
                    logger.info(f"Pose analysis complete: shots_detected={len(shots)}, duration_sec={video_duration}")
                    
                    if not shots:
                        logger.warning("Pose analysis returned no shot events")
                        # Provide a helpful, successful response for no-shots cases
                        data_analysis = (
                            "**No shots detected in this video.**\n\n"
                            "For best results: ensure the full body and stick are visible, keep the camera steady, and film 10–15 reps."
                        )
                        return jsonify({
                            "success": True,
                            "analysis": {
                                "data_analysis": {"shots": [], "duration_est_sec": video_duration, "fps": 30},  # Send structured data
                                "data_summary": data_analysis,  # Keep formatted text for display
                                "shots_detected": 0,
                                "video_duration": video_duration,
                                "video_size_mb": round(video_size_mb, 1),
                                "pose_analysis": True,
                                "no_shots_detected": True,
                                "message": "No shooting events detected; provided filming tips."
                            }
                        })
                    
                    # Generate data summary using local formatting (better scoring system)
                    logger.info("Generating data summary with local formatting")
                    from agents.data_summary_agent import format_shot_summary_locally
                    data_analysis = format_shot_summary_locally(analysis_results)
                    
                    logger.info("Full video analysis completed successfully")
                    
                    # Clean up any old completed jobs for this user to prevent accumulation
                    try:
                        old_jobs_ref = (
                            manager.db.collection("jobs")
                            .where("userId", "==", user_id)
                            .where("status", "==", "completed")
                            .limit(10)  # Clean up a reasonable number
                        )
                        
                        batch = manager.db.batch()
                        cleanup_count = 0
                        for old_job in old_jobs_ref.stream():
                            batch.delete(old_job.reference)
                            cleanup_count += 1
                            
                        if cleanup_count > 0:
                            batch.commit()
                            logger.info(f"Auto-cleaned {cleanup_count} old completed Firestore jobs for user {user_id}")
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to auto-cleanup old jobs: {cleanup_error}")
                    
                    # Clean up old storage files (videos and results older than 30 days)
                    try:
                        deleted_count = manager.cleanup_old_files(user_id, days_old=30)
                        if deleted_count > 0:
                            logger.info(f"Auto-cleaned {deleted_count} old storage files for user {user_id}")
                    except Exception as storage_cleanup_error:
                        logger.warning(f"Failed to auto-cleanup old storage files: {storage_cleanup_error}")
                    
                    return jsonify({
                        "success": True,
                        "analysis": {
                            "data_analysis": analysis_results,  # Complete structured data with age groups
                            "data_summary": data_analysis,  # Formatted text for display
                            "shots_detected": len(shots),
                            "video_duration": video_duration,
                            "video_size_mb": round(video_size_mb, 1),
                            "pose_analysis": True,
                            "message": "MediaPipe pose analysis with structured data summary"
                        }
                    })
                    
                finally:
                    # Clean up temporary video file
                    if os.path.exists(temp_video_path):
                        try:
                            os.unlink(temp_video_path)
                            logger.info("Cleaned up temporary video file")
                        except Exception as cleanup_error:
                            logger.warning(f"Failed to cleanup temp file: {cleanup_error}")
                
            except ImportError as e:
                logger.error(f"Video analysis dependencies not available: {e}")
                return jsonify({
                    "error": "Video analysis system is temporarily unavailable"
                }), 503
            
        except Exception as exc:
            logger.exception("Failed to analyze video")
            return jsonify({"error": "Failed to analyze video"}), 500


    @app.route("/api/results/<user_id>", methods=["GET"])
    def list_results(user_id: str):
        limit = int(request.args.get("limit", 10))
        manager = get_manager()

        try:
            jobs_ref = (
                manager.db.collection("jobs")
                .where("userId", "==", user_id)
                .where("status", "==", "completed")
                .order_by("createdAt", direction=firestore.Query.DESCENDING)
                .limit(limit)
            )

            result_payload = []
            for job_doc in jobs_ref.stream():
                data = job_doc.to_dict()
                storage_paths = data.get("resultUrls", {})
                signed_urls = (
                    manager.generate_results_download_urls(storage_paths)
                    if storage_paths
                    else {}
                )
                result_payload.append({
                    "job_id": job_doc.id,
                    "created_at": data.get("createdAt"),
                    "video_path": data.get("videoStoragePath"),
                    "delivery_method": data.get("deliveryMethod"),
                    "signed_result_urls": signed_urls,
                })

            return jsonify({
                "success": True,
                "results": result_payload,
                "count": len(result_payload),
            })
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to list results")
            return jsonify({"error": "Failed to list results"}), 500

    # ------------------------------------------------------------------
    # OpenIce AI Coach Endpoints
    # ------------------------------------------------------------------
    
    # Primary endpoints (current naming)
    @app.route("/api/start-chat", methods=["POST"])
    def start_chat() -> tuple:
        """Create a new OpenIce chat session with analysis data."""
        payload = request.get_json() or {}
        analysis_data = payload.get("analysis_data")
        user_id = payload.get("user_id", "anonymous")
        
        if not analysis_data:
            return jsonify({"error": "analysis_data is required"}), 400
        
        try:
            agent = get_openice_agent()
            session_id = agent.create_chat_session(analysis_data, user_id)
            
            return jsonify({
                "success": True,
                "session_id": session_id,
                "user_id": user_id,
                "message": "OpenIce chat session created successfully"
            })
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to create OpenIce chat session")
            return jsonify({"error": "Failed to create chat session"}), 500
    
    @app.route("/api/ask-question", methods=["POST"])
    def ask_question() -> tuple:
        """Ask a question in an existing OpenIce chat session."""
        payload = request.get_json() or {}
        session_id = payload.get("session_id")
        question = payload.get("question")
        
        if not session_id or not question:
            return jsonify({"error": "session_id and question are required"}), 400
        
        try:
            agent = get_openice_agent()
            result = agent.ask_question(session_id, question)
            
            return jsonify({
                "success": True,
                "openice_response": result["answer"],
                "search_queries": result["search_queries"],
                "sources": result["sources"],
                "session_id": session_id,
                "message_count": result["message_count"]
            })
        except ValueError as exc:
            # Session not found
            return jsonify({"error": str(exc)}), 404
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to process OpenIce question")
            return jsonify({"error": "Failed to process question"}), 500
    
    @app.route("/api/chat-info/<session_id>", methods=["GET"])
    def get_chat_info(session_id: str) -> tuple:
        """Get information about a chat session."""
        try:
            agent = get_openice_agent()
            info = agent.get_session_info(session_id)
            
            return jsonify({
                "success": True,
                "session_info": info
            })
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 404
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to get chat session info")
            return jsonify({"error": "Failed to get session info"}), 500

    # ------------------------------------------------------------------
    # Client-Expected OpenIce Endpoints (aliases for compatibility)
    # ------------------------------------------------------------------
    
    @app.route("/api/openice/init", methods=["POST", "OPTIONS"])
    def openice_init() -> tuple:
        """Initialize OpenIce session - client-expected endpoint."""
        if request.method == "OPTIONS":
            # Handle preflight request
            return jsonify({"status": "OK"}), 200
            
        payload = request.get_json() or {}
        analysis_data = payload.get("analysis_data")
        data_analysis = payload.get("data_analysis")  # Accept structured data with age groups
        user_id = payload.get("user_id", "anonymous")
        
        if not analysis_data and not data_analysis:
            return jsonify({"error": "analysis_data or data_analysis is required"}), 400
        
        try:
            agent = get_openice_agent()
            
            # Convert data_analysis to readable format if provided
            if data_analysis and not analysis_data:
                from agents.openice_agent import parse_raw_pose_analysis
                analysis_data = parse_raw_pose_analysis(data_analysis)
                logger.info("Converted data_analysis to readable format for OpenIce")
            
            session_id = agent.create_chat_session(analysis_data, user_id)
            
            # Get an initial response to provide immediate value
            initial_question = "What should I work on first based on this analysis?"
            result = agent.ask_question(session_id, initial_question)
            
            return jsonify({
                "success": True,
                "session_id": session_id,
                "user_id": user_id,
                "openice_response": result["answer"],
                "sources": result.get("sources", []),
                "search_queries": result.get("search_queries", []),
                "message": "OpenIce session initialized with initial coaching advice"
            })
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to initialize OpenIce session")
            return jsonify({"error": "Failed to initialize session"}), 500
    
    @app.route("/api/openice/chat", methods=["POST", "OPTIONS"])
    def openice_chat() -> tuple:
        """Send chat message to OpenIce - client-expected endpoint."""
        if request.method == "OPTIONS":
            # Handle preflight request
            return jsonify({"status": "OK"}), 200
            
        payload = request.get_json() or {}
        session_id = payload.get("session_id")
        question = payload.get("question") or payload.get("message")
        
        if not session_id or not question:
            return jsonify({"error": "session_id and question are required"}), 400
        
        try:
            agent = get_openice_agent()
            result = agent.ask_question(session_id, question)
            
            return jsonify({
                "success": True,
                "openice_response": result["answer"],
                "search_queries": result.get("search_queries", []),
                "sources": result.get("sources", []),
                "session_id": session_id,
                "message_count": result.get("message_count", 0),
                "word_count": result.get("word_count", 0)
            })
        except ValueError as exc:
            # Session not found
            return jsonify({"error": str(exc)}), 404
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to process OpenIce chat")
            return jsonify({"error": "Failed to process chat message"}), 500
    
    @app.route("/api/openice/session/<session_id>", methods=["GET"])
    def get_openice_session_info(session_id: str) -> tuple:
        """Get information about an OpenIce chat session."""
        try:
            agent = get_openice_agent()
            session_info = agent.get_session_info(session_id)
            
            return jsonify({
                "success": True,
                "session_info": session_info
            })
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 404
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to get session info")
            return jsonify({"error": "Failed to get session info"}), 500
    
    @app.route("/api/openice/stats", methods=["GET"])
    def get_openice_stats() -> tuple:
        """Get OpenIce agent statistics."""
        try:
            agent = get_openice_agent()
            stats = agent.get_session_stats()
            
            return jsonify({
                "success": True,
                "stats": stats
            })
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to get OpenIce stats")
            return jsonify({"error": "Failed to get stats"}), 500
    
    @app.route("/api/openice/reset-context/<session_id>", methods=["POST"])
    def reset_openice_context(session_id: str) -> tuple:
        """Force a context reset for an OpenIce session."""
        try:
            agent = get_openice_agent()
            success = agent.force_context_reset(session_id)
            
            if success:
                return jsonify({
                    "success": True,
                    "message": "Context reset successfully"
                })
            else:
                return jsonify({"error": "Session not found"}), 404
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to reset context")
            return jsonify({"error": "Failed to reset context"}), 500


    # ------------------------------------------------------------------
    # Coach-Specific Endpoints
    # ------------------------------------------------------------------
    
    @app.route("/api/coach/seth", methods=["POST"])
    def get_seth_coaching() -> tuple:
        """Generate Seth's coaching feedback from analysis data."""
        payload = request.get_json() or {}
        
        # Accept data_analysis (structured data with age groups)
        data_analysis = payload.get("data_analysis")
        user_id = payload.get("user_id")
        analysis_id = payload.get("analysis_id")  # Future: reference to stored analysis
        
        if not data_analysis:
            return jsonify({"error": "data_analysis is required"}), 400
            
        try:
            # Import seth shooting agent
            from agents.seth_shooting_agent import generate_sections
            
            logger.info("Generating Seth coaching feedback")
            coach_feedback = generate_sections(data_analysis)
            
            return jsonify({
                "success": True,
                "coach_id": "seth",
                "coach_name": "Seth",
                "coaching_feedback": coach_feedback,
                "message": "Seth's coaching analysis completed"
            })
            
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to generate Seth coaching feedback")
            return jsonify({"error": "Failed to generate coaching feedback"}), 500
    
    @app.route("/api/coaches", methods=["GET"])
    def list_available_coaches() -> tuple:
        """List available coaching personalities."""
        coaches = [
            {
                "id": "seth",
                "name": "Seth",
                "description": "Focused technical feedback with specific timestamps and metrics",
                "specialties": ["technique", "fundamentals", "specific_improvements"],
                "endpoint": "/api/coach/seth"
            }
            # Future coaches can be added here
        ]
        
        return jsonify({
            "success": True,
            "coaches": coaches,
            "count": len(coaches)
        })

    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.getenv("BACKEND_PORT", 5000))
    app.run(host="0.0.0.0", port=port)

