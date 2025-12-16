import React, { useEffect, useMemo, useRef, useState } from 'react';
import { ask, healthCheck } from './api/client';

export default function App() {
    const [apiStatus, setApiStatus] = useState({ ok: false, message: 'Checking…' });
    const [query, setQuery] = useState('');
    const [withSources, setWithSources] = useState(true);
    const [loading, setLoading] = useState(false);
    const [answer, setAnswer] = useState('');
    const [sources, setSources] = useState([]);
    const [error, setError] = useState('');
    const controllerRef = useRef(null);

    const baseUrl = useMemo(
        () => (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, ''),
        []
    );

    useEffect(() => {
        const ctrl = new AbortController();
        healthCheck(ctrl.signal)
            .then((res) => {
                setApiStatus({ ok: true, message: `Online (${res?.version || 'unknown'})` });
            })
            .catch((e) => {
                setApiStatus({ ok: false, message: e.message || 'Offline' });
            });
        return () => ctrl.abort();
    }, []);

    const onSubmit = async (e) => {
        e?.preventDefault();
        const q = query.trim();
        if (!q) {
            setError('Please enter a question.');
            return;
        }
        setError('');
        setLoading(true);
        setAnswer('');
        setSources([]);
        if (controllerRef.current) controllerRef.current.abort();
        controllerRef.current = new AbortController();

        try {
            const res = await ask(q, withSources, controllerRef.current.signal);
            setAnswer(res.answer || '');
            // Ensure UI ignores sources when toggle is off
            setSources(withSources ? (res.sources || []) : []);
        } catch (err) {
            setError(err.message || 'Something went wrong.');
        } finally {
            setLoading(false);
        }
    };

    const onKeyDown = (e) => {
        if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
            onSubmit(e);
        }
    };

    return (
        <div className="app">
            <header className="header">
                <div className="title">
                    <span className="logo">🎓</span>
                    <div>
                        <h1>D. K. Bhave Scholarship Assistant</h1>
                        <p className="subtitle">Ask verified questions from SPPU’s DK Bhave Scholarship pages</p>
                    </div>
                </div>
                <div className={`status ${apiStatus.ok ? 'ok' : 'bad'}`}>
                    <span className="dot" />
                    {apiStatus.message}
                </div>
            </header>

            <section className="card">
                <form onSubmit={onSubmit}>
                    <label htmlFor="query" className="label">Your question</label>
                    <textarea
                        id="query"
                        className="textarea"
                        placeholder="e.g., What are the eligibility criteria and how do I apply?"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        onKeyDown={onKeyDown}
                        rows={4}
                    />
                    <div className="controls">
                        <div className="toggle">
                            <input
                                id="withSources"
                                type="checkbox"
                                checked={withSources}
                                onChange={(e) => {
                                    setWithSources(e.target.checked);
                                }}
                            />
                            <label htmlFor="withSources">Return sources (/api/ask_with_sources)</label>
                        </div>
                        <div className="buttons">
                            <button
                                type="button"
                                className="btn secondary"
                                onClick={() => { setQuery(''); setAnswer(''); setSources([]); setError(''); }}
                            >
                                Clear
                            </button>
                            <button type="submit" className="btn primary" disabled={loading}>
                                {loading ? 'Thinking…' : 'Ask'}
                            </button>
                        </div>
                    </div>
                    <p className="hint">Tip: Ctrl/Cmd + Enter to submit</p>
                </form>
            </section>

            {error && (
                <section className="card error">
                    <strong>⚠️ {error}</strong>
                </section>
            )}

            {!!answer && (
                <section className="card">
                    <h3>Answer</h3>
                    <div className="answer">{answer}</div>
                </section>
            )}

            {withSources && Array.isArray(sources) && sources.length > 0 && (
                <section className="card">
                    <h3>Sources</h3>
                    <ul className="sources">
                        {sources.map((s, i) => (
                            <SourceItem key={s.doc_id ?? i} source={s} />
                        ))}
                    </ul>
                </section>
            )}

            <footer className="footer">
                <span className="muted">
                    API: <code>{baseUrl}</code> • Docs: <a href={`${baseUrl}/docs`} target="_blank" rel="noreferrer">/docs</a>
                </span>
            </footer>
        </div>
    );
}

function SourceItem({ source }) {
    const [expanded, setExpanded] = useState(false);
    const {
        doc_id,
        source: src = '',
        page_type,
        title,
        content_preview,
    } = source || {};

    const displayUrl = typeof src === 'string' ? src : '';
    const isUrl = /^https?:\/\//i.test(displayUrl);

    const preview = content_preview || '';
    const short = preview && preview.length > 260 ? preview.slice(0, 260) + '…' : preview;

    return (
        <li className="source">
            <div className="source-header">
                <div className="badge">#{doc_id ?? '–'}</div>
                <div className="source-meta">
                    {title && <div className="source-title">{title}</div>}
                    <div className="source-sub">
                        {page_type && <span className="tag">{page_type}</span>}
                        {displayUrl && (
                            <>
                                <span className="sep">•</span>
                                {isUrl ? (
                                    <a href={displayUrl} target="_blank" rel="noreferrer" className="link">{displayUrl}</a>
                                ) : (
                                    <span className="muted">{displayUrl}</span>
                                )}
                            </>
                        )}
                    </div>
                </div>
            </div>

            {preview && (
                <div className="source-preview">
                    <div className="preview-text">{expanded ? preview : short}</div>
                    {preview.length > 260 && (
                        <button className="btn linklike" onClick={() => setExpanded((v) => !v)}>
                            {expanded ? 'Show less' : 'Show more'}
                        </button>
                    )}
                </div>
            )}
        </li>
    );
}
