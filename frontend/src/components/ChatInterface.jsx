import React, { useState } from 'react';
import { Send, Gavel, Info, Sun, Moon } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import ReferenceCard from './ReferenceCard';

const ChatInterface = ({ darkMode, toggleDarkMode }) => {
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [messages, setMessages] = useState([
    {
      id: 1,
      role: 'assistant',
      content: 'Hello! I am your Hong Kong Legal Assistant. How can I help you today?',
      references: []
    }
  ]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: input
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: input }),
      });

      if (!response.ok) {
        throw new Error('Failed to fetch response from server');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let accumulatedContent = '';
      
      // Add the initial empty assistant message
      setMessages(prev => [...prev, {
        id: 'streaming-msg',
        role: 'assistant',
        content: '',
        references: []
      }]);

      let buffer = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        
        // Keep the last partial line in the buffer
        buffer = lines.pop() || '';
        
        for (const line of lines) {
          const trimmedLine = line.trim();
          if (!trimmedLine || !trimmedLine.startsWith('data: ')) continue;
          
          try {
            const data = JSON.parse(trimmedLine.slice(6));
            if (data.answer) {
              accumulatedContent += data.answer;
              setMessages(prev => {
                const newMessages = [...prev];
                const lastMsg = newMessages[newMessages.length - 1];
                if (lastMsg.id === 'streaming-msg') {
                  newMessages[newMessages.length - 1] = { 
                    ...lastMsg, 
                    content: accumulatedContent 
                  };
                }
                return newMessages;
              });
            } else if (data.error) {
              throw new Error(data.error);
            }
          } catch (e) {
            console.error('Error parsing stream chunk:', e, trimmedLine);
          }
        }
      }

      // Finalize the message ID
      setMessages(prev => {
        const newMessages = [...prev];
        if (newMessages[newMessages.length - 1].id === 'streaming-msg') {
          newMessages[newMessages.length - 1].id = Date.now();
        }
        return newMessages;
      });
    } catch (error) {
      console.error('Error:', error);
      const errorMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: 'Sorry, I encountered an error connecting to the legal database. Please ensure the backend is running.',
        references: []
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-slate-50 dark:bg-slate-900 transition-colors duration-300">
      {/* Header */}
      <header className="bg-white dark:bg-slate-800 border-b border-slate-200 dark:border-slate-700 px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="bg-blue-600 p-1.5 rounded-lg">
            <Gavel className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-slate-800 dark:text-slate-100">HK Legal RAG</h1>
            <p className="text-xs text-slate-500 dark:text-slate-400 flex items-center gap-1">
              <Info className="w-3 h-3" /> Verified Legal Sources Only
            </p>
          </div>
        </div>
        <button 
          onClick={toggleDarkMode}
          className="p-2 rounded-lg bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-600 transition-colors"
        >
          {darkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
        </button>
      </header>

      {/* Chat Area */}
      <main className="flex-1 overflow-y-auto p-6 space-y-6">
        <div className="max-w-4xl mx-auto space-y-6">
          {messages.map((msg) => (
            <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-[80%] ${
                msg.role === 'user' 
                  ? 'bg-blue-600 text-white rounded-l-2xl rounded-tr-2xl' 
                  : 'bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-800 dark:text-slate-100 rounded-r-2xl rounded-tl-2xl'
              } p-4 shadow-sm`}>
                <div className={`text-sm leading-relaxed prose max-w-none ${msg.role === 'user' ? 'text-white' : ''}`}>
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {msg.content}
                  </ReactMarkdown>
                </div>
                
                {msg.references && msg.references.length > 0 && (
                  <div className="mt-4 pt-4 border-t border-slate-100 dark:border-slate-700">
                    <p className="text-[10px] font-bold text-slate-400 dark:text-slate-500 uppercase tracking-wider mb-2">References</p>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                      {msg.references.map(ref => (
                        <ReferenceCard key={ref.id} reference={ref} />
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))}
          {isLoading && messages[messages.length - 1]?.role !== 'assistant' && (
            <div className="flex justify-start">
              <div className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-800 dark:text-slate-100 rounded-r-2xl rounded-tl-2xl p-4 shadow-sm">
                <div className="flex gap-1">
                  <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce [animation-delay:0.2s]"></div>
                  <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce [animation-delay:0.4s]"></div>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Input Area */}
      <footer className="bg-white dark:bg-slate-800 border-t border-slate-200 dark:border-slate-700 p-4">
        <div className="max-w-4xl mx-auto flex gap-4">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Ask about HK Ordinances or Case Law..."
            disabled={isLoading}
            className="flex-1 bg-slate-100 dark:bg-slate-700 border-none rounded-xl px-4 py-3 text-sm text-slate-800 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 focus:ring-2 focus:ring-blue-500 outline-none transition-all disabled:opacity-50"
          />
          <button
            onClick={handleSend}
            disabled={isLoading}
            className="bg-blue-600 hover:bg-blue-700 text-white p-3 rounded-xl transition-colors shadow-lg shadow-blue-200 dark:shadow-none disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Send className={`w-5 h-5 ${isLoading ? 'animate-pulse' : ''}`} />
          </button>
        </div>
      </footer>
    </div>
  );
};

export default ChatInterface;
