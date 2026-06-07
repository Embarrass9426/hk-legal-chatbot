import React, { useState, useRef } from 'react';
import { Send, Square, Brain, ChevronDown, ChevronRight, Gavel, Info, User, Bot, Loader2, Trash2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import ReferenceCard from './ReferenceCard';

const SESSION_STORAGE_KEY = 'hk-legal-chatbot-session-id';

const LOADING_STEPS = {
  en: {
    understanding_question: 'Understanding your question...',
    searching_sources: 'Searching legal sources...',
    analyzing_references: 'Analyzing references...',
    generating_response: 'Generating response...',
  },
  zh: {
    understanding_question: '理解你的問題...',
    searching_sources: '搜索法律來源...',
    analyzing_references: '分析參考資料...',
    generating_response: '生成回應...',
  },
};

const getOrCreateSessionId = () => {
  const existing = window.localStorage.getItem(SESSION_STORAGE_KEY);
  if (existing && existing.trim()) {
    return existing;
  }

  const generated = `session-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
  window.localStorage.setItem(SESSION_STORAGE_KEY, generated);
  return generated;
};

function extractThinking(content) {
  if (!content) return { thinking: null, answer: '' };

  const merged = content.replace(/<\/\s*thinking>(\s*)<thinking>/gi, '$1');

  const matches = [...merged.matchAll(/<\s*thinking>([\s\S]*?)<\/\s*thinking>/gi)];
  if (!matches.length) {
    return { thinking: null, answer: merged };
  }

  const thinking = matches.map(m => m[1].trim()).join('\n\n');
  const answer = merged.replace(/<\s*thinking>[\s\S]*?<\/\s*thinking>/gi, '').trim();

  return {
    thinking: thinking || null,
    answer: answer || '',
  };
}

const ChatInterface = () => {
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [language, setLanguage] = useState('en');
  const [thinkMode, setThinkMode] = useState(false);
  const [expandedThinking, setExpandedThinking] = useState({});
  const abortRef = useRef(null);
  const thinkingPhaseRef = useRef({});
  const userToggledRef = useRef({});
  const [messages, setMessages] = useState([
    {
      id: 1,
      role: 'assistant',
      content: 'Hello! I am your Hong Kong Legal Assistant. How can I help you today?',
      thinking: '',
      references: []
    }
  ]);

  const toggleLanguage = () => {
    const newLang = language === 'en' ? 'zh' : 'en';
    setLanguage(newLang);

    if (messages.length === 1 && messages[0].role === 'assistant') {
      setMessages([{
        ...messages[0],
        content: newLang === 'en'
          ? 'Hello! I am your Hong Kong Legal Assistant. How can I help you today?'
          : '你好！我是你的香港法律助手。今天有什麼可以幫到你？'
      }]);
    }
  };

  const handleStop = () => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
  };

  const handleClearChat = () => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
    setIsLoading(false);
    setExpandedThinking({});
    thinkingPhaseRef.current = {};
    userToggledRef.current = {};
    window.localStorage.removeItem(SESSION_STORAGE_KEY);
    setMessages([{
      id: Date.now(),
      role: 'assistant',
      content: language === 'en'
        ? 'Hello! I am your Hong Kong Legal Assistant. How can I help you today?'
        : '你好！我是你的香港法律助手。今天有什麼可以幫到你？',
      thinking: '',
      references: []
    }]);
  };

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

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: input,
          language: language,
          session_id: getOrCreateSessionId(),
          think: thinkMode,
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error('Failed to fetch response from server');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let accumulatedRaw = '';

      setMessages(prev => [...prev, {
        id: 'streaming-msg',
        role: 'assistant',
        content: '',
        thinking: '',
        references: [],
        status: 'understanding_question'
      }]);

      let buffer = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');

        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmedLine = line.trim();
          if (!trimmedLine || !trimmedLine.startsWith('data: ')) continue;

          const data = JSON.parse(trimmedLine.slice(6));
          if (data.error) {
            throw new Error(data.error);
          }
          if (data.status) {
            setMessages(prev => {
              const newMessages = [...prev];
              const lastMsg = newMessages[newMessages.length - 1];
              if (lastMsg.id === 'streaming-msg') {
                newMessages[newMessages.length - 1] = {
                  ...lastMsg,
                  status: data.status
                };
              }
              return newMessages;
            });
          }
          if (data.references) {
            setMessages(prev => {
              const newMessages = [...prev];
              const lastMsg = newMessages[newMessages.length - 1];
              if (lastMsg.id === 'streaming-msg') {
                newMessages[newMessages.length - 1] = {
                  ...lastMsg,
                  references: data.references
                };
              }
              return newMessages;
            });
          } else if (data.answer) {
            accumulatedRaw += data.answer;

            const extracted = extractThinking(accumulatedRaw);
            const thinking = extracted.thinking || '';
            const answer = extracted.answer;

            const prevPhase = thinkingPhaseRef.current['streaming-msg'];
            const newPhase = answer.length > 0 ? 'answer' : (thinking.length > 0 ? 'thinking' : null);
            thinkingPhaseRef.current['streaming-msg'] = newPhase;

            if (newPhase === 'thinking' && prevPhase !== 'thinking') {
              if (!userToggledRef.current['streaming-msg']) {
                setExpandedThinking(prev => ({ ...prev, ['streaming-msg']: true }));
              }
            }
            if (newPhase === 'answer' && prevPhase === 'thinking') {
              if (!userToggledRef.current['streaming-msg']) {
                setExpandedThinking(prev => ({ ...prev, ['streaming-msg']: false }));
              }
            }

            setMessages(prev => {
              const newMessages = [...prev];
              const lastMsg = newMessages[newMessages.length - 1];
              if (lastMsg.id === 'streaming-msg') {
                newMessages[newMessages.length - 1] = {
                  ...lastMsg,
                  thinking: thinking,
                  content: answer,
                  status: null
                };
              }
              return newMessages;
            });
          }
        }
      }

      setMessages(prev => {
        const newMessages = [...prev];
        if (newMessages[newMessages.length - 1].id === 'streaming-msg') {
          const finalMsg = newMessages[newMessages.length - 1];
          newMessages[newMessages.length - 1] = {
            ...finalMsg,
            id: Date.now(),
            status: null,
          };
        }
        return newMessages;
      });
    } catch (error) {
      if (error.name === 'AbortError') {
        setMessages(prev => {
          const newMessages = [...prev];
          const lastMsg = newMessages[newMessages.length - 1];
          if (lastMsg.id === 'streaming-msg') {
            newMessages[newMessages.length - 1] = {
              ...lastMsg,
              id: Date.now(),
              content: lastMsg.content || 'Generation stopped.',
              status: null,
            };
          }
          return newMessages;
        });
      } else {
        console.error('Error:', error);
        const errorMessage = {
          id: Date.now() + 1,
          role: 'assistant',
          content: language === 'en'
            ? 'Sorry, I encountered an error connecting to the legal database. Please ensure the backend is running.'
            : '抱歉，連接法律數據庫時出錯。請確保後端正在運行。',
          references: []
        };
        setMessages(prev => [...prev, errorMessage]);
      }
    } finally {
      setIsLoading(false);
      abortRef.current = null;
      setTimeout(() => {
        delete thinkingPhaseRef.current['streaming-msg'];
        delete userToggledRef.current['streaming-msg'];
      }, 0);

      setMessages(prev => {
        const lastMsg = prev[prev.length - 1];
        if (lastMsg && lastMsg.role === 'assistant' && lastMsg.thinking?.trim() && !lastMsg.content?.trim() && lastMsg.id !== 'streaming-msg') {
          return [...prev, {
            id: Date.now(),
            role: 'assistant',
            content: language === 'en'
              ? '⚠️ The response was cut off — the thinking process consumed all available tokens. Try disabling thinking mode or rephrasing your question.'
              : '⚠️ 回應被截斷 — 思考過程耗盡了所有可用標記。請嘗試關閉思考模式或重新表述您的問題。',
            references: []
          }];
        }
        return prev;
      });
    }
  };

  const toggleThinkingExpand = (msgId) => {
    userToggledRef.current[msgId] = true;
    setExpandedThinking(prev => ({
      ...prev,
      [msgId]: !prev[msgId],
    }));
  };

  return (
    <div className="flex flex-col h-screen bg-charcoal-900 transition-colors duration-300">
      <header className="bg-charcoal-900 border-b border-charcoal-400 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="bg-gold-400/10 p-1.5 rounded-lg border border-gold-400/20">
            <Gavel className="w-5 h-5 text-gold-400" />
          </div>
          <div>
            <h1 className="text-lg font-semibold text-primary-text tracking-tight">
              {language === 'en' ? 'HK Legal Chatbot' : '香港法律聊天機器人'}
            </h1>
            <p className="text-xs text-muted-text flex items-center gap-1">
              <Info className="w-3 h-3" /> {language === 'en' ? 'Verified legal sources only' : '僅限經核實的法律來源'}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={toggleLanguage}
            className="px-3 py-2 rounded-lg bg-charcoal-500 text-secondary-text hover:bg-charcoal-400 hover:text-primary-text transition-all text-xs font-semibold min-w-[40px] active:scale-[0.98]"
            title={language === 'en' ? 'Switch to traditional chinese' : '切換至英文'}
          >
            {language === 'en' ? 'EN' : '繁'}
          </button>
          <button
            onClick={handleClearChat}
            className="p-2 rounded-lg bg-charcoal-500 text-charcoal-300 hover:bg-red-900/30 hover:text-red-400 transition-all active:scale-[0.98]"
            title={language === 'en' ? 'Clear chat history' : '清除聊天記錄'}
          >
            <Trash2 className="w-5 h-5" />
          </button>
        </div>
      </header>

      <main className="flex-1 overflow-y-auto p-6">
        <div className="max-w-4xl mx-auto space-y-6">
          {messages.map((msg) => {
            if (msg.role === 'assistant' && !msg.content && !msg.thinking?.trim() && (!msg.references || msg.references.length === 0) && !msg.status) {
              return null;
            }

            const stepText = msg.status ? LOADING_STEPS[language][msg.status] || msg.status : null;
            const isStreaming = msg.id === 'streaming-msg';

            return (
              <div key={msg.id} className={`flex items-start gap-3 message-enter ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center mt-1 ${
                  msg.role === 'user'
                    ? 'bg-gold-400/10 text-gold-400 border border-gold-400/20'
                    : 'bg-charcoal-500 text-secondary-text border border-charcoal-400'
                }`}>
                  {msg.role === 'user' ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
                </div>
                <div className={`max-w-[80%] ${
                  msg.role === 'user'
                    ? 'bg-charcoal-500 text-primary-text rounded-l-2xl rounded-tr-2xl border border-charcoal-400'
                    : 'bg-charcoal-600 text-primary-text rounded-r-2xl rounded-tl-2xl border border-charcoal-400'
                } p-5`}>
                  {msg.status && !msg.content && (
                    <div className="flex items-center gap-3 text-sm text-muted-text">
                      <Loader2 className="w-4 h-4 animate-spin text-gold-400" />
                      <span>{stepText}</span>
                    </div>
                  )}

                  {msg.thinking && msg.thinking.trim() && (
                    <div className="mb-3">
                      <button
                        onClick={() => toggleThinkingExpand(msg.id)}
                        className="flex items-center gap-1.5 text-xs font-semibold text-gold-400 hover:text-gold-300 transition-colors"
                      >
                        {expandedThinking[msg.id] ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
                        <Brain className="w-3.5 h-3.5" />
                        {isStreaming
                          ? (language === 'en' ? 'Thinking...' : '思考中...')
                          : (language === 'en' ? 'Thinking' : '思考過程')
                        }
                        {isStreaming && <span className="ml-1 gold-pulse">›</span>}
                      </button>
                      {expandedThinking[msg.id] && (
                        <div className="mt-2 p-3 bg-charcoal-500 rounded-lg text-xs text-secondary-text border-l-2 border-gold-400 whitespace-pre-wrap">
                          {msg.thinking}
                        </div>
                      )}
                    </div>
                  )}

                  {msg.content && (
                    <div className={`text-sm leading-relaxed prose max-w-none ${msg.role === 'user' ? 'text-primary-text' : ''}`}>
                      {isStreaming ? (
                        <div className="whitespace-pre-wrap">{msg.content}</div>
                      ) : (
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {msg.content}
                        </ReactMarkdown>
                      )}
                    </div>
                  )}

                  {msg.references && msg.references.length > 0 && (
                    <div className="mt-4 pt-4 border-t border-charcoal-400">
                      <p className="text-[10px] font-semibold text-muted-text uppercase tracking-wider mb-3">References</p>
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                        {msg.references.map(ref => (
                          <ReferenceCard key={ref.id} reference={ref} />
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </main>

      <footer className="bg-charcoal-900 border-t border-charcoal-400 p-4">
        <div className="max-w-4xl mx-auto flex gap-3 items-end">
          <button
            onClick={() => setThinkMode(prev => !prev)}
            className={`flex-shrink-0 p-3 rounded-xl border transition-all active:scale-[0.98] ${
              thinkMode
                ? 'border-gold-400/50 bg-gold-400/10 text-gold-400'
                : 'border-charcoal-400 bg-charcoal-700 text-muted-text hover:text-secondary-text hover:border-charcoal-300'
            }`}
            title={language === 'en' ? 'Toggle thinking mode' : '切換思考模式'}
          >
            <Brain className="w-5 h-5" />
          </button>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
            placeholder={language === 'en' ? "Ask about HK ordinances or case law..." : "詢問有關香港條例或案例..."}
            disabled={isLoading}
            rows={1}
            className="flex-1 bg-charcoal-700 border border-charcoal-400 rounded-xl px-4 py-3 text-sm text-primary-text placeholder-muted-text focus:ring-2 focus:ring-gold-400/30 focus:border-gold-400/50 outline-none transition-all disabled:opacity-50 resize-none font-sans leading-relaxed max-h-[125px]"
            onInput={(e) => {
              const target = e.target;
              target.style.height = 'auto';
              target.style.height = Math.min(target.scrollHeight, 125) + 'px';
            }}
          />
          <button
            onClick={isLoading ? handleStop : handleSend}
            disabled={!input.trim() && !isLoading}
            className={`flex-shrink-0 p-3 rounded-xl transition-all shadow-lg disabled:opacity-30 disabled:cursor-not-allowed active:scale-[0.98] ${
              isLoading
                ? 'bg-red-600 hover:bg-red-700 shadow-red-900/20'
                : 'bg-gold-400 hover:bg-gold-300 text-charcoal-900 hover:scale-[1.02] shadow-gold-400/10'
            }`}
          >
            {isLoading ? (
              <Square className="w-5 h-5 fill-current" />
            ) : (
              <Send className="w-5 h-5" />
            )}
          </button>
        </div>
      </footer>
    </div>
  );
};

export default ChatInterface;
