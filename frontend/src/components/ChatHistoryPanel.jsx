import { useState } from "react";
import { MessageSquare, Plus, Trash2, X, Clock, ChevronDown, ChevronRight } from "lucide-react";
import { formatTime } from "../chatHistory";

export default function ChatHistoryPanel(props) {
  var sessions = props.sessions;
  var activeSessionId = props.activeSessionId;
  var onSelectSession = props.onSelectSession;
  var onNewChat = props.onNewChat;
  var onDeleteSession = props.onDeleteSession;
  var onClearAll = props.onClearAll;

  var [expanded, setExpanded] = useState(true);
  var [confirmClear, setConfirmClear] = useState(false);

  return (
    <div className="border-b border-border">
      {/* New Chat Button */}
      <div className="p-3 border-b border-border">
        <button
          onClick={onNewChat}
          className="btn-primary w-full flex items-center justify-center gap-1.5 text-xs"
        >
          <Plus size={12} />
          New Chat
        </button>
      </div>

      {/* History Header */}
      <button
        onClick={function() { setExpanded(!expanded); }}
        className="w-full px-3 py-2 flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-300"
      >
        {expanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        <Clock size={12} />
        Chat History ({sessions.length})
      </button>

      {/* Session List */}
      {expanded && (
        <div className="px-2 pb-2 space-y-0.5 max-h-[300px] overflow-y-auto">
          {sessions.length === 0 ? (
            <p className="text-xs text-gray-600 py-2 px-1">No conversations yet.</p>
          ) : (
            sessions.map(function(session) {
              var isActive = session.id === activeSessionId;
              var msgCount = 0;
              for (var i = 0; i < session.messages.length; i++) {
                if (session.messages[i].role === "user") msgCount++;
              }

              return (
                <div
                  key={session.id}
                  className={"group flex items-center gap-1.5 px-2 py-1.5 rounded cursor-pointer transition-colors " + (isActive ? "bg-surface-4 text-gray-200" : "hover:bg-surface-2 text-gray-400")}
                >
                  <div
                    onClick={function() { onSelectSession(session.id); }}
                    className="flex-1 min-w-0 flex items-center gap-1.5"
                  >
                    <MessageSquare size={11} className="shrink-0 text-gray-600" />
                    <div className="flex-1 min-w-0">
                      <p className="text-xs truncate">{session.title}</p>
                      <p className="text-[10px] text-gray-600 font-mono">
                        {msgCount}q · {formatTime(session.updatedAt)}
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={function(e) {
                      e.stopPropagation();
                      onDeleteSession(session.id);
                    }}
                    className="opacity-0 group-hover:opacity-100 p-0.5 rounded hover:bg-surface-4 text-gray-600 hover:text-accent-error transition-all"
                    title="Delete"
                  >
                    <X size={10} />
                  </button>
                </div>
              );
            })
          )}
        </div>
      )}

      {/* Clear All */}
      {sessions.length > 0 && expanded && (
        <div className="px-3 pb-2">
          {confirmClear ? (
            <div className="flex items-center gap-1.5">
              <span className="text-[10px] text-accent-error">Delete all?</span>
              <button
                onClick={function() { onClearAll(); setConfirmClear(false); }}
                className="text-[10px] text-accent-error hover:underline"
              >
                Yes
              </button>
              <button
                onClick={function() { setConfirmClear(false); }}
                className="text-[10px] text-gray-500 hover:underline"
              >
                No
              </button>
            </div>
          ) : (
            <button
              onClick={function() { setConfirmClear(true); }}
              className="text-[10px] text-gray-600 hover:text-gray-400 flex items-center gap-1"
            >
              <Trash2 size={9} />
              Clear all history
            </button>
          )}
        </div>
      )}
    </div>
  );
}
