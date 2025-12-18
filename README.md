# UniAssistant - D.K. Bhave Scholarship Chatbot

A powerful, RAG-based AI assistant designed to help students with information regarding the D.K. Bhave Scholarship at SPPU. The bot features natural language understanding, real-time voice interaction, and source verification.

![UniAssistant Demo](https://via.placeholder.com/800x400?text=UniAssistant+Demo+Image)

## 🚀 Features

-   **RAG (Retrieval-Augmented Generation)**: Uses Gemini 2.5 Flash to answer queries based on official scholarship documents.
-   **Local TTS (Text-to-Speech)**: Integrated **Piper TTS** for high-quality, low-latency, offline voice output (`en_US-lessac-medium`).
-   **STT (Speech-to-Text)**: Supports voice input via microphone using Google Speech Recognition.
-   **Source Citations**: Answers include references to the specific documents used.
-   **Modern UI**: Clean, dark-themed React interface with responsiveness and custom styling.

## 🛠️ Tech Stack

### Backend
-   **Framework**: FastAPI (Python)
-   **LLM**: Google Gemini (via `langchain-google-genai`)
-   **Vector DB**: ChromaDB (Local persistence)
-   **TTS**: Piper (ONNX-based neural TTS run locally)
-   **Dependency Manager**: `uv`

### Frontend
-   **Framework**: React (Vite)
-   **Styling**: CSS Modules / Standard CSS with Dark Theme
-   **Audio**: Web Audio API & MediaRecorder

---

## ⚙️ Prerequisites

-   **Python**: 3.10 or higher
-   **Node.js**: 16+
-   **uv** (Recommended for Python package management):
    ```bash
    pip install uv
    ```

## 📥 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd avatar-chatbot
```

### 2. Backend Setup
Navigate to the `backend` folder and install dependencies:
```bash
cd backend
uv sync
```

**Configuration**:
Create a `.env` file in the `backend/` directory:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash  # Note: 20 req/day limit. Use gemini-1.5-flash for more volume.
```

### 3. Frontend Setup
Navigate to the `frontend` folder and install dependencies:
```bash
cd ../frontend
npm install
```

---

## ▶️ Running the Application

You need to run both the backend and frontend terminals simultaneously.

### Terminal 1: Backend
```bash
cd backend
uv run uvicorn app.main:app --reload
```
*Wait for "Application startup complete" and "Piper model found/loaded".*

### Terminal 2: Frontend
```bash
cd frontend
npm run dev
```
Open your browser at `http://localhost:5173`.

---

## 🧩 Key Components

-   **`app/main.py`**: Entry point. Handles API routes (`/api/ask`, `/api/speak`, `/api/transcribe`).
-   **`app/rag_setup.py`**: RAG chain configuration, vector store initialization.
-   **`app/tts_piper.py`**: Manages the local Piper TTS process and model downloading.
-   **`frontend/src/api/client.js`**: API client for backend communication.

## ⚠️ Notes

-   **First Run**: The backend will automatically download the Piper Voice Model (~60MB) and the Embedding Model on the very first launch. This may take a minute.
-   **API Quotas**: If you encounter errors, check your Google Gemini API quota. The default `gemini-2.5-flash` has a strict daily limit.

## 📜 License
Unlicensed / Private Project.
