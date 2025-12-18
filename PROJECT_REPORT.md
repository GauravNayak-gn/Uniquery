# UniAssistant Project Report - Technical Documentation

## 1. Project Overview
**UniAssistant** is an AI-powered conversational agent developed to provide accurate, verified information about the **D.K. Bhave Scholarship** program at Savitribai Phule Pune University (SPPU).

Unlike standard chatbots that hallucinate information, UniAssistant uses a **RAG (Retrieval-Augmented Generation)** architecture to ground its answers strictly in official documentation. It features a multimodal interface with support for text, speech-to-text (STT), and text-to-speech (TTS).

---

## 2. System Architecture

The project follows a decoupled **Client-Server Architecture**:

-   **Frontend (Client)**: A React.js Single Page Application (SPA) that handles user interaction, audio recording, and playback.
-   **Backend (Server)**: A Python FastAPI application that serves the RAG pipeline, manages the vector database, and performs voice processing.

### High-Level Data Flow
1.  **User Input**: User types text or speaks into the microphone (converted to text via Google Speech Recognition).
2.  **Vector Search**: The system converts the query into a vector and searches **ChromaDB** for relevant document chunks.
3.  **Context Construction**: Retrieved chunks are assembled into a strict context block.
4.  **Generative Reasoning**: The context + user query are sent to **Google Gemini LLM**.
5.  **Multimodal Output**: The LLM generates a text answer, which is simultaneously displayed and converted to audio via **Piper TTS**.

---

## 3. Internal Logic & Algorithms (Deep Dive)

### A. The RAG Pipeline Logic
The Retrieval-Augmented Generation process is the core intelligence implementation.

#### Step 1: Vector Embeddings (The "Brain" Index)
*   **Model**: `sentence-transformers/all-MiniLM-L6-v2`
*   **Logic**:
    *   The model converts any text (document chunk or user query) into a **384-dimensional dense vector**.
    *   Mathematically, this maps semantic meaning to geometric space. Similar concepts appear close together in this hyperspace.
    *   We store these vectors in **ChromaDB**, which uses the **HNSW (Hierarchical Navigable Small World)** algorithm for efficient approximate nearest neighbor search.

#### Step 2: Retrieval with MMR (Maximal Marginal Relevance)
Instead of just finding the "closest" matches (Standard Similarity Search), we use **MMR** to improve answer quality.
*   **Problem**: Standard search often returns 5 practically identical chunks (e.g., 5 versions of the same "eligibility" paragraph), missing other important details.
*   **Solution (MMR Algorithm)**:
    1.  Fetch `fetch_k=40` candidates based on pure similarity (Cosine Similarity).
    2.  Select the 1st result (highest similarity).
    3.  Iteratively select the next results that match the query but are **dissimilar to already selected docs**.
    4.  **Result**: We get a diverse set of 6 documents covering different aspects of the query.

#### Step 3: Prompt Engineering (The Control Layer)
We bypass the LLM's internal training data (which might be outdated or hallucinated) by using a **Strict Context Injection Prompt**:

```text
"You are a helpful and strictly factual assistant...
Rules:
1) Use ONLY the provided context below.
2) If the answer is not in the context, say 'I don't have that information'.
3) Do NOT add outside information."
```
This forces the model to act as a **reasoning engine** over our data, rather than a knowledge base.

---

### B. Text-to-Speech (TTS) Internal Logic
We use **Piper TTS**, an ONNX-based neural synthesizer, integrated via a custom pipeline to solve Windows compatibility issues.

1.  **Request Handling**: The `/api/speak` endpoint receives text.
2.  **Process spawning**: Python spawns a subprocess: `piper.exe -m model.onnx -f temp_file.wav`.
3.  **Buffer Management**:
    *   *Challenge*: Direct stdout piping (`-f -`) causes crashes on Windows due to binary file handling differences in Python/C++.
    *   *Logic Fix*: We implemented a **Temporary File Buffer** strategy. The output is written to a unique temp file on disk, read back into Python's memory (`io.BytesIO`), and then the file is immediately deleted.
4.  **Streaming**: The audio data is returned to the frontend as a `blob`, allowing immediate playback without waiting for the full file download.

---

## 4. Technical Stack & Justification

| Component | Technology | Reasoning |
| :--- | :--- | :--- |
| **Backend Framework** | **FastAPI** | High performance, native async support for concurrent requests (essential for real-time chat). |
| **LLM** | **Google Gemini** | `gemini-2.5-flash` offers superior reasoning capability with a generous API quota compared to OpenAI. |
| **Vector Database** | **ChromaDB** | Serverless, local persistence, and built-in HNSW selection make it ideal for embedded RAG. |
| **Embeddings** | **HuggingFace** | `all-MiniLM-L6-v2` balances speed (inference on CPU) with high semantic accuracy. |
| **TTS** | **Piper** | Runs 100% offline on CPU. Zero latency overhead compared to cloud APIs. |

---

## 5. Challenges & Solutions

| Challenge | Technical Solution |
| :--- | :--- |
| **Hallucinations** | Implemented **MMR Retrieval** and **Temperature=0** settings to force deterministic, fact-based answers. |
| **Rate Limiting** | Gemini has strict daily limits. We implemented **Quota-Aware Error Handling** and fallback prompts. |
| **Windows Audio** | The `piper` CLI crashed on Windows. We built a **NamedTemporaryFile wrapper** in Python to handle binary I/O safely. |
