// frontend/src/components/Chat.jsx

import React, { useState, useRef, useEffect } from 'react';

function Chat() {
  const [messages, setMessages] = useState([
    { 
      text: "Hello! I'm your AI teaching assistant. I can answer questions about your course materials. What would you like to know?", 
      sender: 'bot' 
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = { text: input.trim(), sender: 'user' };
    setMessages(prev => [...prev, userMessage]);
    
    const currentInput = input.trim();
    setInput('');
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/api/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: currentInput }),
      });

      if (!response.ok) {
        // Handle different error types
        if (response.status === 503) {
          throw new Error('The AI system is not ready yet. Please make sure you\'ve run the setup process.');
        } else if (response.status === 400) {
          const errorData = await response.json();
          throw new Error(errorData.detail || 'Invalid question format.');
        } else {
          throw new Error(`Server error (${response.status}). Please try again.`);
        }
      }

      const data = await response.json();
      const botMessage = { text: data.answer, sender: 'bot' };
      setMessages(prev => [...prev, botMessage]);

    } catch (error) {
      console.error("Error fetching response:", error);
      setError(error.message);
      const errorMessage = { 
        text: `Sorry, there was an error: ${error.message}`, 
        sender: 'bot',
        isError: true 
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <div className="message-list">
        {messages.map((msg, index) => (
          <div 
            key={index} 
            className={`message ${msg.sender} ${msg.isError ? 'error' : ''}`}
          >
            {msg.text}
          </div>
        ))}
        {isLoading && (
          <div className="message bot loading">
            <div className="thinking-indicator">
              <span>Thinking</span>
              <div className="dots">
                <span>.</span>
                <span>.</span>
                <span>.</span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      
      <form onSubmit={handleSubmit} className="input-form">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question about your course notes..."
          disabled={isLoading}
          maxLength={1000}
        />
        <button 
          type="submit" 
          disabled={isLoading || !input.trim()}
        >
          {isLoading ? 'Sending...' : 'Send'}
        </button>
      </form>
      
      {error && (
        <div className="error-banner">
          Connection issue detected. Make sure the backend server is running.
        </div>
      )}
    </div>
  );
}

export default Chat;