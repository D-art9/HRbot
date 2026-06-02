FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Set target directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy private packages from libs directory to container site-packages
COPY libs/cyvia/ /usr/local/lib/python3.11/site-packages/cyvia/
COPY libs/claude_agent_sdk/ /usr/local/lib/python3.11/site-packages/claude_agent_sdk/

# Copy application source folders
COPY agents/ ./agents/
COPY api/ ./api/
COPY core/ ./core/
COPY modules/ ./modules/
COPY schemas/ ./schemas/
COPY services/ ./services/
COPY static/ ./static/
COPY templates/ ./templates/
COPY tools/ ./tools/

# Copy initial data folder (default vector database and documents)
COPY data/ ./data/

# Pre-warm models during Docker Build so container starts instantly
RUN python -c "import os; from core.embedding_provider import get_embedding_function; get_embedding_function(); from modules.rag_module import get_ranker; get_ranker()"

# Expose backend API port
EXPOSE 8000

# Run FastAPI server using production Uvicorn settings
CMD ["python", "-m", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
