import React, { useEffect, useMemo, useRef, useState } from 'react';
import { ask, healthCheck, transcribeAudio, speak } from './api/client';
import { audioBufferToWav } from './utils/wav_utils';

const Icons = {
    Trash: () => (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M3 6h18"></path>
            <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"></path>
            <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"></path>
        </svg>
    ),
    Plus: () => (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="12" y1="5" x2="12" y2="19"></line>
            <line x1="5" y1="12" x2="19" y2="12"></line>
        </svg>
    ),
    Mic: (props) => (
        <svg {...props} width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
            <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
            <line x1="12" y1="19" x2="12" y2="23"></line>
            <line x1="8" y1="23" x2="16" y2="23"></line>
        </svg>
    ),
    Send: () => (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="22" y1="2" x2="11" y2="13"></line>
            <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
        </svg>
    ),
    GraduationCap: (props) => (
        <svg {...props} width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M22 10v6M2 10l10-5 10 5-10 5z"></path>
            <path d="M6 12v5c3 3 9 3 12 0v-5"></path>
        </svg>
    ),
    StopCircle: () => (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" stroke="none">
            <circle cx="12" cy="12" r="10"></circle>
        </svg>
    ),
    Volume2: (props) => (
        <svg {...props} width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
            <path d="M19.07 4.93a10 10 0 0 1 0 14.14"></path>
            <path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path>
        </svg>
    ),
    FileText: (props) => (
        <svg {...props} width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
            <polyline points="14 2 14 8 20 8"></polyline>
            <line x1="16" y1="13" x2="8" y2="13"></line>
            <line x1="16" y1="17" x2="8" y2="17"></line>
            <polyline points="10 9 9 9 8 9"></polyline>
        </svg>
    ),
    ChevronDown: (props) => (
        <svg {...props} width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="6 9 12 15 18 9"></polyline>
        </svg>
    ),
    ChevronUp: (props) => (
        <svg {...props} width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="18 15 12 9 6 15"></polyline>
        </svg>
    )
};

export default function App() {
    const [apiStatus, setApiStatus] = useState({ ok: false, message: 'Connecting…' });
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const [currentSpeakingId, setCurrentSpeakingId] = useState(null);
    const [isListening, setIsListening] = useState(false);
    const [withSources, setWithSources] = useState(true);
    const [showNewChatModal, setShowNewChatModal] = useState(false);

    const audioRef = useRef(null);
    const bottomRef = useRef(null);
    const abortControllerRef = useRef(null);
    const mediaRecorderRef = useRef(null);
    const audioChunksRef = useRef([]);

    useEffect(() => {
        const ctrl = new AbortController();
        healthCheck(ctrl.signal)
            .then(() => setApiStatus({ ok: true, message: 'Online' }))
            .catch(() => setApiStatus({ ok: false, message: 'Offline' }));

        return () => {
            ctrl.abort();
            if (audioRef.current) audioRef.current.pause();
        };
    }, []);

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, loading]);

    const handleSend = async (e, customText) => {
        e?.preventDefault();
        const text = (customText || input).trim();
        if (!text) return;

        const userMsg = { id: Date.now(), role: 'user', text };
        setMessages((prev) => [...prev, userMsg]);
        setInput('');
        setLoading(true);
        stopTTS();

        if (abortControllerRef.current) abortControllerRef.current.abort();
        abortControllerRef.current = new AbortController();

        try {
            const res = await ask(text, withSources, abortControllerRef.current.signal);
            const botMsg = {
                id: Date.now() + 1,
                role: 'bot',
                text: res.answer,
                sources: res.sources
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
            stopTTS();
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const mediaRecorder = new MediaRecorder(stream);
            mediaRecorderRef.current = mediaRecorder;
            audioChunksRef.current = [];

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) audioChunksRef.current.push(event.data);
            };

            mediaRecorder.onstop = async () => {
                setIsListening(false);
                const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
                try {
                    const arrayBuffer = await audioBlob.arrayBuffer();
                    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    const decodedData = await audioContext.decodeAudioData(arrayBuffer);
                    const wavBlob = await audioBufferToWav(decodedData);
                    setInput('Transcribing...');
                    const text = await transcribeAudio(wavBlob);
                    setInput(text);
                } catch (err) {
                    alert('Transcription failed: ' + err.message);
                    setInput('');
                }
                stream.getTracks().forEach(track => track.stop());
            };

            mediaRecorder.start();
            setIsListening(true);
        } catch (err) {
            alert('Cannot access microphone: ' + err.message);
        }
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current && isListening) mediaRecorderRef.current.stop();
    };

    const toggleListening = () => {
        if (isListening) stopRecording();
        else startRecording();
    };

    const playTTS = async (text, msgId) => {
        if (currentSpeakingId === msgId) {
            stopTTS();
            return;
        }
        stopTTS();
        setCurrentSpeakingId(msgId);
        try {
            const audioBlob = await speak(text);
            const url = URL.createObjectURL(audioBlob);
            const audio = new Audio(url);
            audio.onended = () => {
                setCurrentSpeakingId(null);
                URL.revokeObjectURL(url);
            };
            audio.onerror = () => {
                setCurrentSpeakingId(null);
                alert("Failed to play audio.");
            };
            audioRef.current = audio;
            audio.play();
        } catch (err) {
            console.error(err);
            setCurrentSpeakingId(null);
        }
    };

    const stopTTS = () => {
        if (audioRef.current) {
            audioRef.current.pause();
            audioRef.current.currentTime = 0;
            audioRef.current = null;
        }
        setCurrentSpeakingId(null);
    };

    return (
        <div className="flex flex-col h-screen bg-[#f8fafc] text-[#1e293b]">
            {/* Header */}
            <header className="flex-none px-6 py-4 bg-white/80 backdrop-blur-md border-b border-slate-200">
                <div className="max-w-7xl mx-auto flex justify-between items-center">
                    <div className="flex items-center gap-4">
                        <div className="w-10 h-10 bg-slate-100 rounded-xl flex items-center justify-center text-[#1e293b]">
                            <Icons.GraduationCap />
                        </div>
                        <div>
                            <h1 className="text-xl font-bold tracking-tight">UniQuery</h1>
                            <p className="text-sm text-slate-500">D. K. Bhave Scholarship Assistant</p>
                        </div>
                    </div>
                    <button
                        onClick={() => setShowNewChatModal(true)}
                        className="flex items-center gap-2 px-4 py-2 rounded-lg border border-slate-200 hover:bg-slate-50 transition-colors text-sm font-medium"
                    >
                        <Icons.Plus />
                        New Chat
                    </button>
                </div>
            </header>

            {/* New Chat Modal */}
            {showNewChatModal && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm px-4">
                    <div className="bg-white rounded-3xl shadow-2xl max-w-md w-full p-8 animate-in fade-in zoom-in duration-200">
                        <h3 className="text-2xl font-serif text-[#1e293b] mb-3">Start a new conversation?</h3>
                        <p className="text-slate-500 mb-8 leading-relaxed">
                            This will clear your current chat history. This action cannot be undone.
                        </p>
                        <div className="flex justify-end gap-3">
                            <button
                                onClick={() => setShowNewChatModal(false)}
                                className="px-6 py-3 rounded-xl border border-slate-200 hover:bg-slate-50 transition-colors font-medium text-slate-600"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={() => {
                                    setMessages([]);
                                    setShowNewChatModal(false);
                                }}
                                className="px-6 py-3 rounded-xl bg-[#1e293b] text-white hover:bg-slate-800 transition-colors font-medium"
                            >
                                New Chat
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Main Content */}
            <main className={`flex-1 overflow-y-auto px-4 ${messages.length === 0 ? 'flex flex-col justify-center' : 'py-8'}`}>
                <div className="max-w-3xl mx-auto w-full">
                    {messages.length === 0 ? (
                        <div className="flex flex-col items-center text-center py-4">
                            <div className="w-16 h-16 bg-slate-100 rounded-2xl flex items-center justify-center text-[#1e293b] mb-4">
                                <Icons.GraduationCap className="w-8 h-8" />
                            </div>
                            <h2 className="text-3xl font-serif mb-2">Welcome to UniQuery</h2>
                            <p className="text-base text-slate-500 max-w-xl mb-4 leading-relaxed">
                                Your AI-powered assistant for the D. K. Bhave Scholarship program at Savitribai Phule Pune University.
                            </p>

                            {/* Feature Chips */}
                            <div className="flex flex-wrap justify-center gap-3 mb-6">
                                <div className="px-3 py-1.5 bg-white rounded-full border border-slate-200 text-xs flex items-center gap-2">
                                    <span className="text-amber-500">🎙️</span> Voice Input
                                </div>
                                <div className="px-3 py-1.5 bg-white rounded-full border border-slate-200 text-xs flex items-center gap-2">
                                    <span className="text-emerald-500">🔊</span> Audio Responses
                                </div>
                                <div className="px-3 py-1.5 bg-white rounded-full border border-slate-200 text-xs flex items-center gap-2">
                                    <span className="text-blue-500">📄</span> Cited Sources
                                </div>
                            </div>

                            {/* Try Asking */}
                            <div className="w-full">
                                <p className="text-xs font-bold text-slate-400 tracking-widest uppercase mb-4">Try Asking</p>
                                <div className="flex flex-col gap-3">
                                    {[
                                        "What is the D. K. Bhave Scholarship?",
                                        "Who is eligible for this scholarship?",
                                        "What documents do I need to apply?",
                                        "What is the application deadline?"
                                    ].map((q) => (
                                        <button
                                            key={q}
                                            onClick={(e) => handleSend(e, q)}
                                            className="w-full text-left px-5 py-3 bg-white border border-slate-200 rounded-2xl hover:border-slate-300 hover:shadow-sm transition-all text-[#1e293b] text-sm"
                                        >
                                            {q}
                                        </button>
                                    ))}
                                </div>
                            </div>
                        </div>
                    ) : (
                        <div className="flex flex-col gap-8">
                            {messages.map((msg) => (
                                <MessageItem
                                    key={msg.id}
                                    msg={msg}
                                    isSpeaking={currentSpeakingId === msg.id}
                                    onSpeak={() => playTTS(msg.text, msg.id)}
                                    withSources={withSources}
                                />
                            ))}
                            {loading && (
                                <div className="flex gap-4 items-start">
                                    <div className="w-8 h-8 rounded-full bg-slate-100 flex items-center justify-center text-sm">🤖</div>
                                    <div className="bg-slate-100 px-6 py-4 rounded-3xl rounded-tl-sm animate-pulse">
                                        <div className="flex gap-1">
                                            <div className="w-1.5 h-1.5 bg-slate-400 rounded-full"></div>
                                            <div className="w-1.5 h-1.5 bg-slate-400 rounded-full"></div>
                                            <div className="w-1.5 h-1.5 bg-slate-400 rounded-full"></div>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>
                    )}
                    <div ref={bottomRef} className={messages.length > 0 ? "h-24" : "h-0"} />
                </div>
            </main>

            {/* Input Area */}
            <footer className="flex-none p-4 pb-8 bg-gradient-to-t from-[#f8fafc] via-[#f8fafc] to-transparent">
                <div className="max-w-3xl mx-auto">
                    <form onSubmit={handleSend} className="relative flex items-center gap-2">
                        <div className="flex-1 relative">
                            <input
                                className="w-full bg-white border border-slate-200 rounded-2xl px-6 py-4 pr-32 focus:outline-none focus:border-slate-400 text-[#1e293b] placeholder:text-slate-400 shadow-sm"
                                placeholder="Ask about the D. K. Bhave Scholarship..."
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                                disabled={loading}
                            />
                            <div className="absolute right-4 top-1/2 -translate-y-1/2 flex items-center gap-3">
                                <button
                                    type="button"
                                    className={`p-2 rounded-xl transition-all ${isListening ? 'bg-red-50 text-red-500 scale-110' : 'text-slate-400 hover:bg-slate-100'}`}
                                    onClick={toggleListening}
                                >
                                    <Icons.Mic className={isListening ? 'animate-pulse' : ''} />
                                </button>
                            </div>
                        </div>
                        <button
                            type="submit"
                            disabled={!input.trim() || loading}
                            className="p-4 bg-slate-400 text-white rounded-xl hover:bg-slate-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-sm"
                        >
                            <Icons.Send />
                        </button>
                    </form>
                </div>
            </footer>
        </div>
    );
}

// Sub-components
function MessageItem({ msg, isSpeaking, onSpeak, withSources }) {
    const isBot = msg.role === 'bot';
    const [showSources, setShowSources] = useState(false);

    return (
        <div className={`flex gap-4 ${isBot ? 'items-start' : 'flex-row-reverse items-start'}`}>
            <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm flex-none ${isBot ? 'bg-slate-100' : 'bg-blue-100'}`}>
                {isBot ? '🤖' : '👤'}
            </div>
            <div className={`flex flex-col gap-3 max-w-[85%] ${isBot ? 'items-start' : 'items-end'}`}>
                <div className={`px-6 py-4 rounded-3xl shadow-sm ${isBot
                    ? 'bg-white border border-slate-200 rounded-tl-sm'
                    : 'bg-[#1a365d] text-white rounded-tr-sm'
                    }`}>
                    <div className="text-[15px] leading-relaxed whitespace-pre-wrap">{msg.text}</div>

                    {isBot && (
                        <div className="mt-6 flex flex-wrap items-center gap-6">
                            <button
                                onClick={onSpeak}
                                className={`flex items-center gap-2 px-4 py-2 rounded-2xl transition-all font-medium ${isSpeaking ? 'bg-[#f4a222] text-white shadow-md scale-105' : 'text-slate-600 hover:bg-[#f4a222] hover:text-white hover:shadow-md'}`}
                            >
                                {isSpeaking ? <Icons.StopCircle className="w-5 h-5" /> : <Icons.Volume2 className="w-5 h-5" />}
                                <span>Listen</span>
                            </button>

                            <button
                                onClick={() => setShowSources(!showSources)}
                                className={`flex items-center gap-2 px-3 py-2 rounded-xl transition-all font-medium ${showSources ? 'bg-[#f4a222] text-white shadow-md' : 'text-slate-600 hover:bg-[#f4a222] hover:text-white hover:shadow-md'}`}
                            >
                                <Icons.FileText className="w-5 h-5" />
                                <span>{msg.sources.length} {msg.sources.length === 1 ? 'source' : 'sources'}</span>
                                {showSources ? <Icons.ChevronUp /> : <Icons.ChevronDown />}
                            </button>
                        </div>
                    )}
                </div>

                {/* Detailed Sources Dropdown */}
                {isBot && withSources && msg.sources && msg.sources.length > 0 && showSources && (
                    <div className="w-full flex flex-col gap-3 mt-1 animate-in slide-in-from-top-2 duration-200">
                        {msg.sources.map((s, i) => (
                            <div key={i} className="bg-[#f8fafc] border border-slate-100 rounded-2xl p-5 shadow-sm">
                                <div className="flex items-center gap-2 mb-2">
                                    <span className="text-sm font-bold text-slate-700">Doc #{i + 1}</span>
                                    <span className="px-2.5 py-0.5 bg-purple-100 text-purple-600 text-[11px] font-bold rounded-full uppercase tracking-wider">Guidelines</span>
                                </div>
                                <h4 className="text-[15px] font-bold text-slate-900 mb-1">{s.title || 'Source Document'}</h4>
                                <p className="text-xs text-slate-500 mb-3 font-medium">{s.source || 'document.pdf'}</p>
                                <p className="text-sm text-slate-500 italic leading-relaxed">
                                    "{s.snippet || 'Information relevant to the D.K. Bhave Scholarship program at Savitribai Phule Pune University...'}"
                                </p>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}
