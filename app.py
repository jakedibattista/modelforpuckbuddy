"""Cloud Run entrypoint for the signed URL backend API."""

from backend.signed_url_api import create_app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

