# 🤖 HRbot Backend Service

This repository serves as the **AI API Backend** for the SVYIA HR Assistant application. It leverages FastAPI, LangGraph agent workflows, ChromaDB vector indexing, and Groq LLM inference to deliver real-time HR query routing and policy answers.

---

## 🚀 Speeding Up Application Startup on Render

FastAPI lifespan startup loads machine learning models (sentence embeddings and FlashRank re-rankers), which can take 15–30 seconds. To speed this up, especially on cold starts:

1. **Pre-cache Models (Done):** Our `Dockerfile` runs the caching script *during the build step* instead of startup, saving up to 20 seconds of boot delay.
2. **Move to Hosted DB:** Switch ChromaDB from local disk-load to a remote vector database service (like Pinecone, Qdrant, or Chroma Cloud) so the container doesn't load the index databases into active memory on start.
3. **Use Smaller Model Checkpoints:** For CPU-based environments like Render, use lighter weights (e.g., `all-MiniLM-L6-v2` instead of larger BGE or SentenceTransformer versions).
4. **Utilize ONNX Runtime Execution:** Convert PyTorch model weights to ONNX format (onnxruntime is natively loaded in our container) to drastically decrease memory footprint and model initialization times.

---

## 🐳 Running Uptime Pings on Render

Render's free tier spins down containers after 15 minutes of inactivity. To prevent this:
*   We've added a `/ping` route returning `"pong"`.
*   When running inside Render (detected via `RENDER=true`), the app automatically starts a background keep-alive loop task that pings its own server endpoint every 10 minutes (detected via `RENDER_EXTERNAL_URL`).

---

## 🛠️ Commands

### Build Image
```bash
docker build -t hrbot:latest .
```

### Run Container
```bash
docker run -d -p 8000:8000 --env-file .env -v hrbot_chroma_data:/app/data/chroma_db --name hr_assistant hrbot:latest
```
