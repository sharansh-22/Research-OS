import { useState, useEffect, useCallback } from "react";
import { Settings, Zap } from "lucide-react";
import { fetchHealth, hasApiKey, setApiKey, getApiKey } from "./api";
import { getSessions, getSession, createSession, updateSession, deleteSession, clearAllSessions } from "./chatHistory";
import LeftPane from "./components/LeftPane";
import ChatPane from "./components/ChatPane";
import SourcePane from "./components/SourcePane";
import ApiKeyModal from "./components/ApiKeyModal";

var MAX_HISTORY_TURNS = 3;

export default function App() {
  var [health, setHealth] = useState(null);
  var [showKeyModal, setShowKeyModal] = useState(!hasApiKey());

  // Session state
  var [sessions, setSessions] = useState(function() { return getSessions(); });
  var [activeSessionId, setActiveSessionId] = useState(function() {
    var existing = getSessions();
    if (existing.length > 0) return existing[0].id;
    var fresh = createSession();
    return fresh.id;
  });

  // Chat state derived from active session
  var [messages, setMessages] = useState([]);
  var [history, setHistory] = useState([]);
  var [isStreaming, setIsStreaming] = useState(false);
  var [activeController, setActiveController] = useState(null);

  // Source inspector
  var [activeSources, setActiveSources] = useState([]);
  var [selectedSource, setSelectedSource] = useState(null);
  var [activeIntent, setActiveIntent] = useState(null);

  // Load session data when active session changes
  useEffect(function() {
    var session = getSession(activeSessionId);
    if (session) {
      setMessages(session.messages || []);
      setHistory(session.history || []);
    } else {
      setMessages([]);
      setHistory([]);
    }
    setActiveSources([]);
    setSelectedSource(null);
    setActiveIntent(null);
  }, [activeSessionId]);

  // Persist messages whenever they change (debounced)
  useEffect(function() {
    if (!activeSessionId) return;
    if (isStreaming) return;

    var timeout = setTimeout(function() {
      updateSession(activeSessionId, messages, history);
      setSessions(getSessions());
    }, 500);

    return function() { clearTimeout(timeout); };
  }, [messages, history, activeSessionId, isStreaming]);

  // Health polling
  useEffect(function() {
    var poll = function() { fetchHealth().then(setHealth).catch(function() { setHealth(null); }); };
    poll();
    var interval = setInterval(poll, 30000);
    return function() { clearInterval(interval); };
  }, []);

  // History management (sliding window)
  var pushHistory = useCallback(function(userQuery, assistantResponse) {
    setHistory(function(prev) {
      var next = prev.concat([
        { role: "user", content: userQuery },
        { role: "assistant", content: assistantResponse }
      ]);
      var max = MAX_HISTORY_TURNS * 2;
      return next.length > max ? next.slice(-max) : next;
    });
  }, []);

  // Session management
  var handleNewChat = useCallback(function() {
    if (isStreaming) return;
    var session = createSession();
    setSessions(getSessions());
    setActiveSessionId(session.id);
  }, [isStreaming]);

  var handleSelectSession = useCallback(function(id) {
    if (isStreaming) return;
    setActiveSessionId(id);
  }, [isStreaming]);

  var handleDeleteSession = useCallback(function(id) {
    if (isStreaming) return;
    var remaining = deleteSession(id);
    setSessions(remaining);

    if (id === activeSessionId) {
      if (remaining.length > 0) {
        setActiveSessionId(remaining[0].id);
      } else {
        var fresh = createSession();
        setSessions(getSessions());
        setActiveSessionId(fresh.id);
      }
    }
  }, [activeSessionId, isStreaming]);

  var handleClearAllSessions = useCallback(function() {
    if (isStreaming) return;
    clearAllSessions();
    var fresh = createSession();
    setSessions(getSessions());
    setActiveSessionId(fresh.id);
  }, [isStreaming]);

  var stopStream = useCallback(function() {
    if (activeController) {
      activeController.abort();
      setActiveController(null);
      setIsStreaming(false);
    }
  }, [activeController]);

  var handleApiKey = function(key) {
    setApiKey(key);
    setShowKeyModal(false);
    fetchHealth().then(setHealth).catch(function() { setHealth(null); });
  };

  return (
    <div className="h-screen w-screen flex flex-col overflow-hidden bg-surface-0">
      {/* Top Bar */}
      <header className="h-11 flex items-center justify-between px-4 border-b border-border bg-surface-1 shrink-0">
        <div className="flex items-center gap-2">
          <Zap size={16} className="text-gray-400" />
          <span className="text-sm font-semibold text-gray-300 tracking-wide">RESEARCH-OS</span>
          <span className="text-xs text-gray-600 font-mono ml-2">v2.1.0</span>
        </div>
        <div className="flex items-center gap-3">
          {health && (
            <div className="flex items-center gap-2 text-xs font-mono text-gray-500">
              <span className={"w-1.5 h-1.5 rounded-full " + (health.status === "healthy" ? "bg-accent-code" : "bg-accent-error")} />
              <span>{health.index_chunks} chunks</span>
              <span className="text-gray-700">|</span>
              <span>{health.indexed_files} files</span>
            </div>
          )}
          <button onClick={function() { setShowKeyModal(true); }} className="btn-ghost p-1.5" title="Settings">
            <Settings size={14} />
          </button>
        </div>
      </header>

      {/* Three-Pane Layout */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left Pane */}
        <div className="w-1/5 min-w-[240px] border-r border-border overflow-hidden flex flex-col">
          <LeftPane
            sessions={sessions}
            activeSessionId={activeSessionId}
            onSelectSession={handleSelectSession}
            onNewChat={handleNewChat}
            onDeleteSession={handleDeleteSession}
            onClearAllSessions={handleClearAllSessions}
          />
        </div>

        {/* Center Pane */}
        <div className="flex-1 min-w-0 overflow-hidden flex flex-col">
          <ChatPane
            messages={messages}
            setMessages={setMessages}
            history={history}
            pushHistory={pushHistory}
            isStreaming={isStreaming}
            setIsStreaming={setIsStreaming}
            activeController={activeController}
            setActiveController={setActiveController}
            stopStream={stopStream}
            setActiveSources={setActiveSources}
            setActiveIntent={setActiveIntent}
          />
        </div>

        {/* Right Pane */}
        <div className="w-[30%] min-w-[280px] border-l border-border overflow-hidden flex flex-col">
          <SourcePane
            sources={activeSources}
            selectedSource={selectedSource}
            setSelectedSource={setSelectedSource}
            intent={activeIntent}
          />
        </div>
      </div>

      {/* API Key Modal */}
      {showKeyModal && <ApiKeyModal currentKey={getApiKey()} onSave={handleApiKey} onClose={function() { setShowKeyModal(false); }} />}
    </div>
  );
}
