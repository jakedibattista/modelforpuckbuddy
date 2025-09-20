FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy repo root modules needed by worker
COPY drill_feedback.py /app/drill_feedback.py
COPY parent_feedback_agent.py /app/parent_feedback_agent.py
COPY improvement_coach_agent.py /app/improvement_coach_agent.py

# Copy worker service
COPY worker/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY worker/app.py /app/app.py

ENV PORT=8080

CMD ["python", "app.py"]


