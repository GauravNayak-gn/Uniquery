import React, { useEffect, useMemo, useRef, useState } from 'react';
import { ask, healthCheck, transcribeAudio } from './api/client';
import { audioBufferToWav } from './utils/wav_utils';

const Icons = {
    Trash: () => (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M3 6h18"></path>
            <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"></path>
            <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"></path>
        </svg>
    ),
    FileText: () => (
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
            <polyline points="14 2 14 8 20 8"></polyline>
            <line x1="16" y1="13" x2="8" y2="13"></line>
            <line x1="16" y1="17" x2="8" y2="17"></line>
            <polyline points="10 9 9 9 8 9"></polyline>
        </svg>
    ),
    Mic: () => (
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
            <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
            <line x1="12" y1="19" x2="12" y2="23"></line>
            <line x1="8" y1="23" x2="16" y2="23"></line>
        </svg>
    ),
    StopCircle: () => (
        <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor" stroke="none">
            <circle cx="12" cy="12" r="10"></circle>
        </svg>
    )
};

// ... inside App component render ...
// Replace 🗑️ with <Icons.Trash />
// Replace 📄 with <Icons.FileText />
// Replace 🎤 with <Icons.Mic />
// Replace 🔴 with <Icons.StopCircle />


export default function App() {
    const [apiStatus, setApiStatus] = useState({ ok: false, message: 'Connecting…' });
    const [messages, setMessages] = useState([]); // Array of { id, role, text, sources? }
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const [isSpeaking, setIsSpeaking] = useState(false);
    const [isListening, setIsListening] = useState(false);

    // Settings
    const [withSources, setWithSources] = useState(true);

    // Refs
    const bottomRef = useRef(null);
    const abortControllerRef = useRef(null);
    const mediaRecorderRef = useRef(null);
    const audioChunksRef = useRef([]);

    const baseUrl = useMemo(
        () => (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, ''),
        []
    );

    // 1. Init & Health Check
    useEffect(() => {
        const ctrl = new AbortController();
        healthCheck(ctrl.signal)
            .then(() => setApiStatus({ ok: true, message: 'Online' }))
            .catch(() => setApiStatus({ ok: false, message: 'Offline' }));

        return () => {
            ctrl.abort();
            window.speechSynthesis.cancel();
        };
    }, []);

    // 2. Auto-scroll
    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, loading]);

    // 3. Handlers
    const handleSend = async (e) => {
        e?.preventDefault();
        const text = input.trim();
        if (!text) return;

        // Add user message
        const userMsg = { id: Date.now(), role: 'user', text };
        setMessages((prev) => [...prev, userMsg]);
        setInput('');
        setLoading(true);

        // Stop TTS if speaking
        window.speechSynthesis.cancel();
        setIsSpeaking(false);

        // Abort previous request
        if (abortControllerRef.current) abortControllerRef.current.abort();
        abortControllerRef.current = new AbortController();

        try {
            const res = await ask(text, withSources, abortControllerRef.current.signal);
            const botMsg = {
                id: Date.now() + 1,
                role: 'bot',
                text: res.answer,
                sources: res.sources // Keep sources in data, visibility handled by UI
            };
            setMessages((prev) => [...prev, botMsg]);
        } catch (err) {
            if (err.name !== 'AbortError') {
                const errorMsg = { id: Date.now() + 2, role: 'bot', text: 'Sorry, something went wrong. Please try again.' };
                setMessages((prev) => [...prev, errorMsg]);
            }
        } finally {
            setLoading(false);
        }
    };

    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const mediaRecorder = new MediaRecorder(stream);
            mediaRecorderRef.current = mediaRecorder;
            audioChunksRef.current = [];

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunksRef.current.push(event.data);
                }
            };

            mediaRecorder.onstop = async () => {
                setIsListening(false);
                const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });

                // Convert to WAV because backend speech_recognition prefers it
                try {
                    const arrayBuffer = await audioBlob.arrayBuffer();
                    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    const decodedData = await audioContext.decodeAudioData(arrayBuffer);
                    const wavBlob = await audioBufferToWav(decodedData);

                    setInput('Transcibing...');
                    const text = await transcribeAudio(wavBlob);
                    setInput(text);
                } catch (err) {
                    alert('Transcription failed: ' + err.message);
                    setInput('');
                }

                // Stop tracks
                stream.getTracks().forEach(track => track.stop());
            };

            mediaRecorder.start();
            setIsListening(true);
        } catch (err) {
            alert('Cannot access microphone: ' + err.message);
        }
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current && isListening) {
            mediaRecorderRef.current.stop();
        }
    };

    const toggleListening = () => {
        if (isListening) stopRecording();
        else startRecording();
    };


    return (
        <div className="layout">
            {/* Header */}
            <header className="header">
                <div className="header-content">
                    <div className="brand">
                        <span className="logo-icon">🎓</span>
                        <div>
                            <h1>UniAssistant</h1>
                            <div className={`status-badge ${apiStatus.ok ? 'online' : 'offline'}`}>
                                {apiStatus.message}
                            </div>
                        </div>
                    </div>
                    <div className="header-actions">
                        {/* Clear Chat */}
                        {messages.length > 0 && (
                            <button className="btn-icon" onClick={() => setMessages([])} title="Clear Chat">
                                <Icons.Trash />
                            </button>
                        )}
                    </div>
                </div>
            </header>

            {/* Chat Area */}
            <main className="chat-feed">
                {messages.length === 0 ? (
                    <div className="welcome-placeholder">
                        <div className="welcome-icon">👋</div>
                        <h2>Hello! I'm your University Assistant.</h2>
                        <p>Ask me about admissions, scholarships, or courses.</p>
                    </div>
                ) : (
                    messages.map((msg) => (
                        <MessageItem
                            key={msg.id}
                            msg={msg}
                            isSpeaking={isSpeaking}
                            setIsSpeaking={setIsSpeaking}
                            withSources={withSources}
                        />
                    ))
                )}

                {loading && (
                    <div className="message bot">
                        <div className="avatar">🤖</div>
                        <div className="bubble loading">
                            <span className="dot">.</span><span className="dot">.</span><span className="dot">.</span>
                        </div>
                    </div>
                )}
                <div ref={bottomRef} />
            </main>

            {/* Input Area */}
            <footer className="input-area">
                <div className="input-container">
                    <form onSubmit={handleSend} className="input-bar">
                        {/* Setting Toggle (Mini) */}
                        <div className="setting-toggle">
                            <input
                                type="checkbox"
                                id="src-toggle"
                                checked={withSources}
                                onChange={(e) => setWithSources(e.target.checked)}
                            />
                            <label htmlFor="src-toggle" title="Toggle Sources">
                                {withSources ? <Icons.FileText /> : <span style={{ opacity: 0.5 }}><Icons.FileText /></span>}
                            </label>
                        </div>

                        <input
                            className="text-input"
                            placeholder="Ask a question..."
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            disabled={loading}
                        />

                        <button type="button" className={`btn-mic ${isListening ? 'active' : ''}`} onClick={toggleListening}>
                            {isListening ? <Icons.StopCircle /> : <Icons.Mic />}
                        </button>

                        <button type="submit" className="btn-send" disabled={!input.trim() || loading}>
                            ➤
                        </button>
                    </form>
                    <div className="disclaimer">AI can make mistakes. Please verify important info.</div>
                </div>
            </footer>
        </div>
    );
}

// Sub-components
function MessageItem({ msg, isSpeaking, setIsSpeaking, withSources }) {
    const isBot = msg.role === 'bot';

    const handleSpeak = () => {
        if (!window.speechSynthesis) return;
        if (isSpeaking) {
            window.speechSynthesis.cancel();
            setIsSpeaking(false);
        } else {
            const ut = new SpeechSynthesisUtterance(msg.text);
            ut.onend = () => setIsSpeaking(false);
            ut.onerror = () => setIsSpeaking(false);
            setIsSpeaking(true);
            window.speechSynthesis.speak(ut);
        }
    };

    return (
        <div className={`message ${msg.role}`}>
            {isBot && <div className="avatar">🤖</div>}
            <div className="content-stack">
                <div className="bubble">
                    <div className="text">{msg.text}</div>
                    {isBot && (
                        <div className="bubble-actions">
                            <button onClick={handleSpeak} className="action-btn" title="Read Aloud">
                                {isSpeaking ? <Icons.StopCircle /> : <span style={{ fontSize: '1.2em' }}>🔊</span>}
                            </button>
                        </div>
                    )}
                </div>

                {/* Sources Section - Only show if valid sources exist AND global toggle is ON */}
                {isBot && withSources && msg.sources && msg.sources.length > 0 && (
                    <div className="sources-section">
                        <div className="sources-title">Sources</div>
                        <div className="sources-list">
                            {msg.sources.map((s, i) => (
                                <a
                                    key={i}
                                    href={s.source?.startsWith('http') ? s.source : '#'}
                                    target="_blank"
                                    rel="noreferrer"
                                    className="source-chip"
                                >
                                    {i + 1}. {s.title || 'Document'}
                                </a>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
