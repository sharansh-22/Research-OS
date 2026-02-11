import { useState, useEffect, useCallback } from "react";
import { Upload, Link, FileText, RefreshCw, Database, ChevronDown, ChevronRight, CheckCircle, Loader, AlertCircle, Clock } from "lucide-react";
import { uploadFile, ingestUrl, fetchIngestionStatus, fetchIndexFiles } from "../api";
import ChatHistoryPanel from "./ChatHistoryPanel";

var STAGE_LABELS = { queued: "Queued", downloading: "Downloading", parsing: "Parsing", embedding: "Embedding", indexing: "Indexing", complete: "Complete", failed: "Failed" };
var STAGE_COLORS = { queued: "text-gray-500", downloading: "text-accent-theory", parsing: "text-accent-math", embedding: "text-accent-theory", indexing: "text-accent-code", complete: "text-accent-code", failed: "text-accent-error" };

function StageIcon(props) {
  var status = props.status;
  var size = props.size || 12;
  if (status === "complete") return <CheckCircle size={size} className="text-accent-code" />;
  if (status === "failed") return <AlertCircle size={size} className="text-accent-error" />;
  if (status === "queued") return <Clock size={size} className="text-gray-500" />;
  return <Loader size={size} className="text-accent-theory animate-spin" />;
}

export default function LeftPane(props) {
  var sessions = props.sessions;
  var activeSessionId = props.activeSessionId;
  var onSelectSession = props.onSelectSession;
  var onNewChat = props.onNewChat;
  var onDeleteSession = props.onDeleteSession;
  var onClearAllSessions = props.onClearAllSessions;

  var [indexFiles, setIndexFiles] = useState([]);
  var [tasks, setTasks] = useState([]);
  var [showUrl, setShowUrl] = useState(false);
  var [urlInput, setUrlInput] = useState("");
  var [uploading, setUploading] = useState(false);
  var [error, setError] = useState(null);
  var [filesExpanded, setFilesExpanded] = useState(false);
  var [tasksExpanded, setTasksExpanded] = useState(true);

  useEffect(function() {
    var poll = function() { fetchIngestionStatus().then(function(data) { setTasks(data.tasks || []); }).catch(function() {}); };
    poll();
    var interval = setInterval(poll, 3000);
    return function() { clearInterval(interval); };
  }, []);

  var loadFiles = useCallback(function() {
    fetchIndexFiles().then(function(data) { setIndexFiles(data.files || []); }).catch(function() {});
  }, []);

  useEffect(function() { loadFiles(); }, [loadFiles]);

  useEffect(function() {
    var hasActive = tasks.some(function(t) { return t.status !== "complete" && t.status !== "failed"; });
    if (!hasActive && tasks.length > 0) loadFiles();
  }, [tasks, loadFiles]);

  var handleFileUpload = async function(e) {
    var files = Array.from(e.target.files);
    if (!files.length) return;
    setUploading(true);
    setError(null);
    try {
      for (var i = 0; i < files.length; i++) { await uploadFile(files[i]); }
    } catch (err) { setError(err.message); }
    setUploading(false);
    e.target.value = "";
  };

  var handleUrlIngest = async function() {
    if (!urlInput.trim()) return;
    setError(null);
    try { await ingestUrl(urlInput.trim()); setUrlInput(""); setShowUrl(false); } catch (err) { setError(err.message); }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="pane-header">
        <div className="flex items-center gap-2"><Database size={14} /><span>Research-OS</span></div>
        <button onClick={loadFiles} className="btn-ghost p-1" title="Refresh"><RefreshCw size={12} /></button>
      </div>

      <div className="flex-1 overflow-y-auto">
        {/* Chat History */}
        <ChatHistoryPanel
          sessions={sessions}
          activeSessionId={activeSessionId}
          onSelectSession={onSelectSession}
          onNewChat={onNewChat}
          onDeleteSession={onDeleteSession}
          onClearAll={onClearAllSessions}
        />

        {/* Ingestion Actions */}
        <div className="p-3 space-y-2 border-b border-border">
          <div className="flex gap-2">
            <label className="btn-primary flex-1 text-center cursor-pointer flex items-center justify-center gap-1.5 text-xs">
              <Upload size={12} />Upload
              <input type="file" multiple onChange={handleFileUpload} className="hidden" accept=".pdf,.py,.ipynb,.md,.tex,.cpp,.cu,.c,.h,.txt,.rst" />
            </label>
            <button onClick={function() { setShowUrl(!showUrl); }} className="btn-ghost flex items-center gap-1.5 text-xs border border-border">
              <Link size={12} />URL
            </button>
          </div>
          {showUrl && (
            <div className="flex gap-1.5">
              <input type="text" value={urlInput} onChange={function(e) { setUrlInput(e.target.value); }} onKeyDown={function(e) { if (e.key === "Enter") handleUrlIngest(); }} placeholder="https://arxiv.org/pdf/..." className="input-field text-xs py-1.5 flex-1" />
              <button onClick={handleUrlIngest} className="btn-primary text-xs py-1.5">Go</button>
            </div>
          )}
          {uploading && <div className="flex items-center gap-2 text-xs text-accent-theory"><Loader size={12} className="animate-spin" />Uploading...</div>}
          {error && <div className="text-xs text-accent-error bg-accent-error/10 rounded p-2">{error}</div>}
        </div>

        {/* Active Tasks */}
        {tasks.length > 0 && (
          <div className="border-b border-border">
            <button onClick={function() { setTasksExpanded(!tasksExpanded); }} className="w-full px-3 py-2 flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-300">
              {tasksExpanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}Tasks ({tasks.length})
            </button>
            {tasksExpanded && (
              <div className="px-3 pb-2 space-y-1.5">
                {tasks.map(function(task) {
                  return (
                    <div key={task.task_id} className="bg-surface-2 rounded p-2 space-y-1">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-mono text-gray-300 truncate max-w-[140px]">{task.filename}</span>
                        <StageIcon status={task.status} />
                      </div>
                      <div className="w-full h-1 bg-surface-4 rounded-full overflow-hidden">
                        <div className={"h-full rounded-full transition-all duration-500 " + (task.status === "failed" ? "bg-accent-error" : task.status === "complete" ? "bg-accent-code" : "bg-accent-theory")} style={{ width: (task.progress * 100) + "%" }} />
                      </div>
                      <div className="flex items-center justify-between">
                        <span className={"text-[10px] font-mono " + STAGE_COLORS[task.status]}>{STAGE_LABELS[task.status]}</span>
                        {task.chunks_added > 0 && <span className="text-[10px] font-mono text-gray-600">{task.chunks_added} chunks</span>}
                      </div>
                      {task.error && <div className="text-[10px] text-accent-error truncate">{task.error}</div>}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {/* Indexed Files */}
        <div>
          <button onClick={function() { setFilesExpanded(!filesExpanded); }} className="w-full px-3 py-2 flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-300">
            {filesExpanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}Indexed Files ({indexFiles.length})
          </button>
          {filesExpanded && (
            <div className="px-3 pb-2 space-y-0.5">
              {indexFiles.length === 0 ? (
                <p className="text-xs text-gray-600 py-2">No files indexed yet.</p>
              ) : (
                indexFiles.map(function(file, idx) {
                  return (
                    <div key={idx} className="flex items-center gap-2 px-2 py-1.5 rounded hover:bg-surface-2 group">
                      <FileText size={12} className="text-gray-600 shrink-0" />
                      <span className="text-xs font-mono text-gray-400 truncate">{file}</span>
                    </div>
                  );
                })
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
