#!/usr/bin/env python3
"""
Enhanced Firebase Storage utilities for signed URL workflows used by Puck Buddy.

This module centralises every Firebase Storage interaction required by the
signed URL pipeline:

1. Secure video uploads from the mobile app using signed PUT URLs
2. Worker-side signed downloads for video processing
3. Uploading analysis results and producing signed GET URLs for delivery
4. Firestore job management helpers for the signed URL delivery method

The public API remains backward compatible with the previous
FirebaseStorageManager class so that existing documentation and scripts retain
their imports.
"""

from __future__ import annotations

import os
import json
import ssl
import urllib.request
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List

try:
    import firebase_admin
    from firebase_admin import credentials, storage, firestore
    from google.cloud import storage as gcs
    from google.cloud.exceptions import NotFound
except ImportError as exc:  # pragma: no cover - surface clearer message
    raise ImportError(
        f"Required Firebase dependencies not installed: {exc}\n"
        "Install with: pip install firebase-admin google-cloud-storage"
    ) from exc


class FirebaseStorageManager:
    """Manages Firebase Storage operations for the signed URL workflow."""

    def __init__(self, service_account_path: Optional[str] = None,
                 storage_bucket: Optional[str] = None,
                 project_id: Optional[str] = None) -> None:
        self.bucket_name = storage_bucket or os.getenv('FIREBASE_STORAGE_BUCKET')
        if not self.bucket_name:
            raise ValueError(
                "Storage bucket not specified. Set FIREBASE_STORAGE_BUCKET environment "
                "variable or pass storage_bucket parameter"
            )

        self.project_id = project_id or os.getenv('FIREBASE_PROJECT_ID')

        # Initialise Firebase Admin SDK once
        if not firebase_admin._apps:
            if service_account_path and os.path.exists(service_account_path):
                cred = credentials.Certificate(service_account_path)
            else:
                cred_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
                if cred_path and os.path.exists(cred_path):
                    cred = credentials.Certificate(cred_path)
                else:
                    cred = credentials.ApplicationDefault()

            firebase_admin.initialize_app(cred, {
                'storageBucket': self.bucket_name,
                **({'projectId': self.project_id} if self.project_id else {})
            })

        # Storage + Firestore clients
        self.storage_client = gcs.Client()
        # bucket() expects bucket name without the gs:// prefix
        self.bucket = self.storage_client.bucket(self.bucket_name.replace('gs://', ''))
        self.db = firestore.client()

    # ------------------------------------------------------------------
    # Upload helpers
    # ------------------------------------------------------------------
    def generate_video_upload_url(self, user_id: str,
                                  content_type: str = 'video/mov',
                                  expiration_hours: int = 1) -> Dict[str, str]:
        """Generate a signed PUT URL for the mobile app to upload a video."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_shooting_drill.mov"
        storage_path = f"users/{user_id}/videos/{filename}"

        blob = self.bucket.blob(storage_path)
        upload_url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(hours=expiration_hours),
            method="PUT",
            content_type=content_type,
            headers={'x-goog-content-length-range': '0,104857600'},  # 100 MB
        )

        return {
            'upload_url': upload_url,
            'storage_path': storage_path,
            'filename': filename,
            'content_type': content_type,
            'expires_at': (datetime.now() + timedelta(hours=expiration_hours)).isoformat()
        }

    # Backwards compatible helper used in older scripts/docs
    def generate_upload_url(self, user_id: str, filename: str,
                            expiration_minutes: int = 60) -> Tuple[str, str]:
        info = self.generate_video_upload_url(
            user_id=user_id,
            content_type='video/*',
            expiration_hours=max(1, expiration_minutes // 60 or 1)
        )

        # Respect requested filename if supplied
        if filename:
            storage_path = info['storage_path'].rsplit('/', 1)[0] + f"/{filename}"
            blob = self.bucket.blob(storage_path)
            upload_url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(minutes=expiration_minutes),
                method="PUT",
                content_type='video/*'
            )
            return upload_url, storage_path

        return info['upload_url'], info['storage_path']

    def create_analysis_job(self, user_id: str, video_storage_path: str) -> str:
        """Create a Firestore job document for the signed URL workflow."""
        job_data = {
            'userId': user_id,
            'videoStoragePath': video_storage_path,
            'status': 'queued',
            'deliveryMethod': 'signed_urls',
            'createdAt': firestore.SERVER_TIMESTAMP,
            'updatedAt': firestore.SERVER_TIMESTAMP,
        }

        job_ref = self.db.collection('jobs').document()
        job_ref.set(job_data)
        return job_ref.id

    # ------------------------------------------------------------------
    # Download helpers
    # ------------------------------------------------------------------
    def generate_video_download_url(self, storage_path: str,
                                    expiration_hours: int = 2) -> str:
        blob = self.bucket.blob(storage_path)
        if not blob.exists():
            raise NotFound(f"Video not found: {storage_path}")

        return blob.generate_signed_url(
            version="v4",
            expiration=timedelta(hours=expiration_hours),
            method="GET",
        )

    def download_video_for_processing(self, signed_url: str, local_path: str) -> str:
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        with urllib.request.urlopen(signed_url, context=ssl_context) as response:
            with open(local_path, 'wb') as fh:
                fh.write(response.read())

        return local_path

    # Backwards compatible helper name
    def generate_download_url(self, storage_path: str,
                              expiration_minutes: int = 60) -> str:
        return self.generate_video_download_url(
            storage_path,
            expiration_hours=max(1, expiration_minutes // 60 or 1)
        )

    # ------------------------------------------------------------------
    # Result management
    # ------------------------------------------------------------------
    def upload_analysis_results(self, user_id: str, video_filename: str,
                               analysis_data: Dict, parent_summary: str,
                               coach_analysis: str) -> Dict[str, str]:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        video_base = video_filename.rsplit('.', 1)[0]
        result_prefix = f"users/{user_id}/results/{timestamp}_{video_base}"

        payloads = {
            'analysis.json': json.dumps(analysis_data, indent=2),
            'parent_summary.txt': parent_summary,
            'coach_analysis.txt': coach_analysis,
        }

        storage_paths: Dict[str, str] = {}
        for filename, content in payloads.items():
            storage_path = f"{result_prefix}/{filename}"
            blob = self.bucket.blob(storage_path)
            content_type = 'application/json' if filename.endswith('.json') else 'text/plain'
            blob.upload_from_string(content, content_type=content_type)
            result_key = filename.replace('.json', '_url').replace('.txt', '_url')
            storage_paths[result_key] = storage_path

        return storage_paths

    def generate_results_download_urls(self, storage_paths: Dict[str, str],
                                       expiration_hours: int = 24) -> Dict[str, str]:
        download_urls: Dict[str, str] = {}
        for result_type, storage_path in storage_paths.items():
            blob = self.bucket.blob(storage_path)
            download_urls[result_type] = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(hours=expiration_hours),
                method="GET",
            )
        return download_urls

    def complete_analysis_job(self, job_id: str, storage_paths: Dict[str, str]) -> None:
        signed_urls = self.generate_results_download_urls(storage_paths)
        job_ref = self.db.collection('jobs').document(job_id)
        job_ref.update({
            'status': 'completed',
            'completedAt': firestore.SERVER_TIMESTAMP,
            'updatedAt': firestore.SERVER_TIMESTAMP,
            'resultUrls': storage_paths,
            'signedResultUrls': signed_urls,
        })

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def list_user_videos(self, user_id: str) -> List[Dict[str, Optional[str]]]:
        prefix = f"users/{user_id}/videos/"
        videos: List[Dict[str, Optional[str]]] = []
        for blob in self.bucket.list_blobs(prefix=prefix):
            videos.append({
                'name': blob.name.split('/')[-1],
                'storage_path': blob.name,
                'size_bytes': blob.size,
                'created_at': blob.time_created.isoformat() if blob.time_created else None,
                'content_type': blob.content_type,
            })
        return videos

    def list_user_results(self, user_id: str) -> List[Dict[str, any]]:
        prefix = f"users/{user_id}/results/"
        result_sessions: Dict[str, Dict[str, any]] = {}

        for blob in self.bucket.list_blobs(prefix=prefix):
            parts = blob.name.split('/')
            if len(parts) < 4:
                continue
            session = parts[3]
            filename = parts[4] if len(parts) > 4 else 'unknown'

            result_sessions.setdefault(session, {
                'session': session,
                'files': {},
                'created_at': blob.time_created.isoformat() if blob.time_created else None,
            })

            result_sessions[session]['files'][filename] = {
                'storage_path': blob.name,
                'size_bytes': blob.size,
                'content_type': blob.content_type,
            }

        return list(result_sessions.values())

    def cleanup_old_files(self, user_id: str, days_old: int = 30) -> int:
        cutoff = datetime.now() - timedelta(days=days_old)
        prefix = f"users/{user_id}/"
        deleted = 0
        for blob in self.bucket.list_blobs(prefix=prefix):
            if blob.time_created and blob.time_created < cutoff:
                blob.delete()
                deleted += 1
        return deleted


def main() -> None:  # pragma: no cover - manual utility
    try:
        manager = FirebaseStorageManager()
        user_id = 'demo_user_123'
        upload_url, storage_path = manager.generate_upload_url(user_id, 'demo.mov')
        print('Signed upload URL:', upload_url[:80] + '...')
        print('Storage path:', storage_path)
    except Exception as exc:  # noqa: BLE001
        print('❌ Error initialising Firebase Storage Manager:', exc)


if __name__ == '__main__':
    main()
