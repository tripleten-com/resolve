FROM python:3.12-slim

WORKDIR /app
RUN pip install --no-cache-dir "fastapi==0.115.12" "uvicorn==0.34.0"

# Build context is ./helpdesk, so the package lands at /app/helpdesk.
COPY . /app/helpdesk

ENV HELPDESK_DB=/app/db/helpdesk.db \
    HELPDESK_AUDIT_LOG=/app/db/audit.log \
    HELPDESK_DATA_DIR=/data

EXPOSE 8100
CMD ["uvicorn", "helpdesk.app:app", "--host", "0.0.0.0", "--port", "8100"]
