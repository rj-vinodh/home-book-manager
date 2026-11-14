FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    BOOKS_PORT=7861 \
    DATA_DIR=/app/data \
    EMBED_URL=http://host.docker.internal:9931 \
    EMBED_MODEL=ggml-org/embeddinggemma-2-GGUF:Q8_0 \
    EMBED_API_KEY=no-key \
    EMBED_DIM=768

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py books_app.py start.sh ./

# Run as a non-root user; outputs/ is where captured frames are written,
# data/ holds the book database (books.db), covers and CSV exports
RUN chmod +x start.sh \
    && useradd -m appuser \
    && mkdir -p /app/outputs /app/data \
    && chown -R appuser /app
USER appuser

EXPOSE 7860 7861
CMD ["./start.sh"]