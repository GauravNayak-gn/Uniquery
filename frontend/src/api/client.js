const BASE_URL =
  (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');

export async function healthCheck(signal) {
  const res = await fetch(`${BASE_URL}/`, { signal });
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}

export async function ask(query, withSources = true, signal) {
  const url = `${BASE_URL}${withSources ? '/api/ask_with_sources' : '/api/ask'}`;
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
    signal,
  });

  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const msg = data?.detail || `Request failed: ${res.status}`;
    throw new Error(msg);
  }

  // Ensure no sources are shown when toggle is off,
  // even if the backend happens to return them.
  if (!withSources) {
    data.sources = [];
  } else {
    if (!Array.isArray(data.sources)) data.sources = [];
  }

  return data; // { answer: string, sources: [...] }
}

export async function transcribeAudio(audioBlob) {
  const formData = new FormData();
  formData.append('file', audioBlob, 'recording.wav');

  const res = await fetch(`${BASE_URL}/api/transcribe`, {
    method: 'POST',
    body: formData,
  });

  const data = await res.json();
  if (!res.ok) {
    throw new Error(data?.detail || 'Transcription failed');
  }
  return data.text;
}