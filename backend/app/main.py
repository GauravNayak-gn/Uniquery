# app/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import logging
import asyncio
from typing import List, Optional

# Import your RAG setup
from .rag_setup import create_rag_chain, get_sources_from_query
from .tts_piper import ensure_model_exists, generate_audio_stream
from fastapi.responses import StreamingResponse

# --- Load environment variables ---
load_dotenv()

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- FastAPI Initialization ---
app = FastAPI(
    title="D. K. Bhave Scholarship RAG API",
    description="An AI-powered assistant to provide verified information about the D. K. Bhave Scholarship program at Savitribai Phule Pune University.",
    version="1.1.0",
)

# --- CORS Configuration ---
origins = [
    "http://localhost:5173",  # Vite
    "http://localhost:3000",  # CRA
    "http://127.0.0.1:5173",  # Alternate local address
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    query: str

class Source(BaseModel):
    doc_id: int
    source: str
    page_type: Optional[str] = None
    title: Optional[str] = None
    content_preview: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    # Always return a list (empty when no sources); easier for front-end
    sources: List[Source] = Field(default_factory=list)


# --- Global RAG Chain ---
rag_chain = None  # Lazy initialization

@app.on_event("startup")
async def startup_event():
    """Load the RAG chain once when the API starts."""
    global rag_chain
    logger.info("🔄 Initializing RAG Chain at startup...")
    try:
        rag_chain = await asyncio.to_thread(create_rag_chain)
        logger.info("✅ RAG Chain loaded successfully.")
    except Exception as e:
        logger.exception("❌ Failed to initialize RAG chain: %s", e)
        raise RuntimeError("Failed to initialize RAG chain. Check server logs.")

    # Initialize Piper TTS (download model if needed)
    try:
        await asyncio.to_thread(ensure_model_exists)
    except Exception as e:
        logger.warning(f"⚠️ Piper TTS model download failed: {e}")


# --- API Endpoints ---
@app.get("/", summary="Health Check", tags=["System"])
def root_status():
    """Simple health check endpoint."""
    return {
        "status": "✅ D. K. Bhave Scholarship RAG API is running.",
        "version": app.version,
        "docs": "/docs",
    }


@app.post("/api/ask", response_model=QueryResponse, summary="Ask a question to the RAG assistant", tags=["RAG"])
async def ask_question(request: QueryRequest):
    """
    Receives a query, retrieves relevant context, and returns an LLM-generated factual answer.
    Now unwraps the chain's structured output: {"answer": str, "sources": [...]}
    """
    global rag_chain
    if rag_chain is None:
        raise HTTPException(status_code=500, detail="RAG system not initialized yet.")

    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    logger.info(f"📩 Received query: {query}")

    try:
        out = await asyncio.to_thread(rag_chain.invoke, query)
        # Unwrap structured output
        if isinstance(out, dict) and "answer" in out:
            answer = out["answer"]
            sources = out.get("sources") or []
        else:
            answer = str(out)
            sources = []

        logger.info("✅ Answer generated successfully.")
        return QueryResponse(answer=answer, sources=sources)
    except Exception as e:
        logger.exception("❌ Error during RAG invocation: %s", e)
        raise HTTPException(status_code=500, detail=f"RAG processing error: {str(e)}")


@app.post("/api/ask_with_sources", response_model=QueryResponse, summary="Ask question and include source snippets", tags=["RAG"])
async def ask_question_with_sources(request: QueryRequest):
    """
    Returns the answer plus the exact sources used by the prompt (from the chain).
    Falls back to a similarity search list if the chain returns only a string.
    """
    global rag_chain
    if rag_chain is None:
        raise HTTPException(status_code=500, detail="RAG system not initialized yet.")

    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        out = await asyncio.to_thread(rag_chain.invoke, query)

        if isinstance(out, dict) and "answer" in out:
            # Preferred: use structured sources from the same retrieved docs
            answer = out["answer"]
            sources = out.get("sources") or []
            return QueryResponse(answer=answer, sources=sources)

        # Fallback: chain returned a plain string; use debug retriever for sources
        answer = str(out)
        debug_sources = await asyncio.to_thread(get_sources_from_query, query)
        # Map debug results to Source shape with incremental doc_id
        mapped_sources = [
            {
                "doc_id": i,
                "source": s.get("source", "Unknown"),
                "page_type": s.get("page_type"),
                "title": s.get("title"),  # may not exist in debug results
                "content_preview": s.get("content_preview"),
            }
            for i, s in enumerate(debug_sources, start=1)
        ]
        return QueryResponse(answer=answer, sources=mapped_sources)

    except Exception as e:
        logger.exception("❌ Error during ask_with_sources: %s", e)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/api/debug/sources", summary="Retrieve top sources for a sample query (debug)", tags=["Debug"])
async def debug_sources(query: str = "What is D.K. Bhave scholarship?"):
    """
    Fetch top retrieved documents for a query — useful for testing retrieval quality.
    """
    try:
        sources = await asyncio.to_thread(get_sources_from_query, query)
        return {"query": query, "results": sources}
    except Exception as e:
        logger.exception("❌ Debug retrieval failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Debug retrieval failed: {str(e)}")

# --- TTS Endpoint (Piper) ---
class SpeakRequest(BaseModel):
    text: str

@app.post("/api/speak", summary="Generate speech using Piper TTS", tags=["TTS"])
async def speak_text(request: SpeakRequest):
    """
    Generates audio from text using local Piper TTS (en_US-lessac-medium).
    Returns a WAV audio stream.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    try:
        # Generate audio stream (in-memory WAV)
        audio_buffer = await asyncio.to_thread(generate_audio_stream, request.text)
        return StreamingResponse(audio_buffer, media_type="audio/wav")
    except Exception as e:
        logger.exception("❌ TTS Generation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")


# --- STT Endpoint ---
import speech_recognition as sr
import shutil
import tempfile
import os
from fastapi import UploadFile, File

@app.post("/api/transcribe", summary="Transcribe audio file using Google Speech Recognition", tags=["STT"])
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Receives an audio file (typically WAV or FLAC), saves it temporarily, and uses
    SpeechRecognition (Google Web Speech API) to transcribe it.
    """
    temp_filename = f"temp_{file.filename}"
    try:
        # Save uploaded file temporarily
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        logger.info(f"🎤 Transcribing audio: {temp_filename}")
        
        # Use SpeechRecognition
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_filename) as source:
            # record the audio from the file
            audio_data = recognizer.record(source)
            # transcribe
            text = await asyncio.to_thread(recognizer.recognize_google, audio_data)
            
        logger.info(f"✅ Transcription result: {text}")
        return {"text": text}

    except sr.UnknownValueError:
        logger.warning("STT: Could not understand audio")
        raise HTTPException(status_code=400, detail="Could not understand audio")
    except sr.RequestError as e:
        logger.error(f"STT: Service error: {e}")
        raise HTTPException(status_code=503, detail=f"Speech service unavailable: {str(e)}")
    except Exception as e:
        logger.exception("❌ Transcription failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        # Cleanup temp file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)