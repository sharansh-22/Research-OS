import { useState, useRef, useEffect } from "react";
import { Send, Square, Sparkles } from "lucide-react";
import { streamChat } from "../api";
import MessageBubble from "./MessageBubble";

export default function ChatPane(props) {
  var messages = props.messages;
  var setMessages = props.setMessages;
  var history = props.history;
  var pushHistory = props.pushHistory;
  var isStreaming = props.isStreaming;
  var setIsStreaming = props.setIsStreaming;
  var setActiveController = props.setActiveController;
  var stopStream = props.stopStream;
  var setActiveSources = props.setActiveSources;
  var setActiveIntent = props.setActiveIntent;

  var [input, setInput] = useState("");
  var [filterType, setFilterType] = useState(null);
  var scrollRef = useRef(null);
  var inputRef = useRef(null);

  useEffect(function() {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages]);

  useEffect(function() {
    if (inputRef.current) inputRef.current.focus();
  }, [isStreaming]);

  var handleSubmit = function(e) {
    e.preventDefault();
    var query = input.trim();
    if (!query || isStreaming) return;
    setInput("");

    var userMsg = { role: "user", content: query };
    var assistantMsg = { role: "assistant", content: "", intent: null, context: null, sources: [], streaming: true };
    setMessages(function(prev) { return prev.concat([userMsg, assistantMsg]); });
    setIsStreaming(true);
    setActiveSources([]);
    setActiveIntent(null);

    var fullResponse = "";

    var controller = streamChat(query, history, filterType,
      function(event) {
        if (event.event === "start") {
          setActiveIntent(event.intent);
          setMessages(function(prev) { var next = prev.slice(); var last = Object.assign({}, next[next.length - 1]); last.intent = event.intent; next[next.length - 1] = last; return next; });
        } else if (event.event === "context") {
          setMessages(function(prev) { var next = prev.slice(); var last = Object.assign({}, next[next.length - 1]); last.context = { code: event.code || 0, theory: event.theory || 0 }; next[next.length - 1] = last; return next; });
        } else if (event.event === "chunk") {
          fullResponse += event.data || "";
          setMessages(function(prev) { var next = prev.slice(); var last = Object.assign({}, next[next.length - 1]); last.content = fullResponse; next[next.length - 1] = last; return next; });
        } else if (event.event === "sources") {
          var sources = event.sources || [];
          setActiveSources(sources);
          setMessages(function(prev) { var next = prev.slice(); var last = Object.assign({}, next[next.length - 1]); last.sources = sources; next[next.length - 1] = last; return next; });
        } else if (event.event === "error") {
          setMessages(function(prev) { var next = prev.slice(); var last = Object.assign({}, next[next.length - 1]); last.content += "\n\n**Error:** " + event.error; last.streaming = false; next[next.length - 1] = last; return next; });
        }
      },
      function() {
        setIsStreaming(false);
        setActiveController(null);
        setMessages(function(prev) { var next = prev.slice(); var last = Object.assign({}, next[next.length - 1]); last.streaming = false; next[next.length - 1] = last; return next; });
        pushHistory(query, fullResponse);
      },
      function(err) {
        setIsStreaming(false);
        setActiveController(null);
        setMessages(function(prev) { var next = prev.slice(); var last = Object.assign({}, next[next.length - 1]); last.content = "**Connection Error:** " + err; last.streaming = false; next[next.length - 1] = last; return next; });
      }
    );
    setActiveController(controller);
  };

  var intentFilters = [
    { value: null, label: "Auto" },
    { value: "theory", label: "Theory" },
    { value: "code", label: "Code" },
    { value: "hybrid", label: "Hybrid" },
  ];

  return (
    <div className="flex flex-col h-full">
      <div className="pane-header">
        <div className="flex items-center gap-2">
          <Sparkles size={14} />
          <span>Research Chat</span>
          <span className="text-xs text-gray-600 font-mono">{Math.floor(history.length / 2)}/3 turns</span>
        </div>
        <div className="flex items-center gap-1">
          {intentFilters.map(function(f) {
            return (
              <button key={f.label} onClick={function() { setFilterType(f.value); }} className={"px-2 py-0.5 rounded text-xs font-mono transition-colors " + (filterType === f.value ? "bg-surface-4 text-gray-200" : "text-gray-600 hover:text-gray-400")}>
                {f.label}
              </button>
            );
          })}
        </div>
      </div>
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-gray-600">
            <Sparkles size={32} className="mb-3 text-gray-700" />
            <p className="text-sm font-medium">Research-OS</p>
            <p className="text-xs mt-1">Ask a research question to begin</p>
          </div>
        )}
        {messages.map(function(msg, idx) {
          return <MessageBubble key={idx} message={msg} />;
        })}
      </div>
      <div className="p-4 border-t border-border bg-surface-1">
        <form onSubmit={handleSubmit} className="flex gap-2">
          <input ref={inputRef} type="text" value={input} onChange={function(e) { setInput(e.target.value); }} placeholder="Ask a research question..." disabled={isStreaming} className="input-field flex-1" autoFocus />
          {isStreaming ? (
            <button type="button" onClick={stopStream} className="btn-primary flex items-center gap-1.5"><Square size={14} />Stop</button>
          ) : (
            <button type="submit" disabled={!input.trim()} className="btn-primary flex items-center gap-1.5"><Send size={14} />Send</button>
          )}
        </form>
      </div>
    </div>
  );
}
