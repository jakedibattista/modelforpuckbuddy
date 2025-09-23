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


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app, origins=os.getenv("CORS_ORIGIN", "*"))

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("signed_url_api")

    manager: FirebaseStorageManager | None = None

    def get_manager() -> FirebaseStorageManager:
        nonlocal manager
        if manager is None:
            manager = FirebaseStorageManager()
            logger.info("FirebaseStorageManager initialised")
        return manager

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
            return jsonify({"error": str(exc)}), 500

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
            return jsonify({"error": str(exc)}), 500

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
            return jsonify({"error": str(exc)}), 500

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
            return jsonify({"error": str(exc)}), 500

    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.getenv("BACKEND_PORT", 5000))
    app.run(host="0.0.0.0", port=port)

