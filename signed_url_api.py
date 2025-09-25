#!/usr/bin/env python3
"""Flask API that exposes the signed URL workflow for Puck Buddy."""

from __future__ import annotations

import logging
import os
from typing import Dict

from flask import Flask, jsonify, request
from flask_cors import CORS

from utils.firebase_storage import FirebaseStorageManager
from firebase_admin import firestore
from agents.openice_agent import OpenIceAgent


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app, origins=os.getenv("CORS_ORIGIN", "*"))

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

    @app.route("/api/upload-url", methods=["POST"])
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

    @app.route("/api/download-url", methods=["POST"])
    def generate_download_url() -> tuple:
        payload = request.get_json() or {}
        storage_path = payload.get("storage_path")
        if not storage_path:
            return jsonify({"error": "storage_path is required"}), 400

        expiration_hours = int(payload.get("expiration_hours", 24))

        try:
            url = get_manager().generate_video_download_url(
                storage_path, expiration_hours=expiration_hours
            )
            return jsonify({
                "success": True,
                "download_url": url,
                "expires_in_hours": expiration_hours,
            })
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to create download URL")
            return jsonify({"error": "Failed to create download URL"}), 500

    @app.route("/api/analyze-video", methods=["POST"])
    def analyze_video_simple() -> tuple:
        """Simple endpoint: process video analysis with simplified AI feedback."""
        payload = request.get_json() or {}
        user_id = payload.get("user_id")
        storage_path = payload.get("storage_path")
        
        if not user_id or not storage_path:
            return jsonify({"error": "user_id and storage_path are required"}), 400
            
        try:
            manager = get_manager()
            
            logger.info(f"Processing video analysis for user {user_id}, storage_path: {storage_path}")
            
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
                from analysis.shooting_drill_feedback import analyze_drill
                from agents.data_summary_agent import generate_summary_with_gemini
                from agents.seth_shooting_agent import generate_sections
                
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
                    video_duration = analysis_results.get('duration_est_sec', 0) if analysis_results else 0
                    logger.info(f"Pose analysis complete: shots_detected={len(shots)}, duration_sec={video_duration}")
                    
                    if not shots:
                        logger.warning("Pose analysis returned no shot events")
                        # Provide a helpful, successful response for no-shots cases
                        parent_summary = (
                            "I didn't detect any clear shooting events in this video. "
                            "For best results: ensure the full body and stick are visible, keep the camera steady, and film 10–15 reps."
                        )
                        coach_summary = (
                            "What to try next:\n"
                            "- Capture from the side at waist height so knees and stick are visible\n"
                            "- Stand ~10–15 feet from the player\n"
                            "- Record at least 20–30 seconds with multiple shot attempts"
                        )
                        return jsonify({
                            "success": True,
                            "analysis": {
                                "data_analysis": parent_summary,
                                "coach_summary": coach_summary,
                                "shots_detected": 0,
                                "video_duration": video_duration,
                                "video_size_mb": round(video_size_mb, 1),
                                "pose_analysis": True,
                                "no_shots_detected": True,
                                "message": "No shooting events detected; provided filming tips."
                            }
                        })
                    
                    # Generate AI summaries using the real analysis
                    logger.info("Generating parent summary with Gemini")
                    parent_summary = generate_summary_with_gemini(analysis_results)
                    
                    logger.info("Generating coach analysis with Gemini")
                    coach_analysis = generate_sections(analysis_results)
                    
                    logger.info("Full video analysis completed successfully")
                    
                    return jsonify({
                        "success": True,
                        "analysis": {
                            "data_analysis": parent_summary,
                            "coach_summary": coach_analysis,
                            "shots_detected": len(shots),
                            "video_duration": video_duration,
                            "video_size_mb": round(video_size_mb, 1),
                            "pose_analysis": True,
                            "message": "Complete video analysis with MediaPipe pose detection"
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

    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.getenv("BACKEND_PORT", 5000))
    app.run(host="0.0.0.0", port=port)

