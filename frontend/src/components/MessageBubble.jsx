import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import remarkGfm from "remark-gfm";
import rehypeKatex from "rehype-katex";
import rehypeHighlight from "rehype-highlight";
import { User, Bot } from "lucide-react";

var intentColors = {
  code: "bg-accent-code/15 text-accent-code border-accent-code/30",
  theory: "bg-accent-theory/15 text-accent-theory border-accent-theory/30",
  math: "bg-accent-math/15 text-accent-math border-accent-math/30",
  hybrid: "bg-accent-hybrid/15 text-accent-hybrid border-accent-hybrid/30",
};

function AuditBadge(props) {
  var audit = props.audit;
  if (!audit) return null;

  var f = Math.round((audit.faithfulness || 0) * 100);
  var r = Math.round((audit.relevancy || 0) * 100);

  var getLevel = function (val) {
    if (val >= 80) return "text-green-500 border-green-500/30 bg-green-500/10";
    if (val >= 50) return "text-yellow-500 border-yellow-500/30 bg-yellow-500/10";
    return "text-red-500 border-red-500/30 bg-red-500/10";
  };

  return (
    <div className="mt-3 space-y-2">
      <div className="flex items-center gap-2 p-2 rounded-lg bg-surface-2 border border-border">
        <div className="flex flex-col gap-0.5">
          <span className="text-[10px] uppercase tracking-wider text-gray-500 font-bold">Faithfulness</span>
          <div className={"px-2 py-0.5 rounded text-xs font-mono border " + getLevel(f)}>{f}% Grounded</div>
        </div>
        <div className="w-px h-6 bg-border mx-1" />
        <div className="flex flex-col gap-0.5">
          <span className="text-[10px] uppercase tracking-wider text-gray-500 font-bold">Relevancy</span>
          <div className={"px-2 py-0.5 rounded text-xs font-mono border " + getLevel(r)}>{r}% Intent</div>
        </div>
        <div className="flex-1" />
        <div className="flex flex-col items-end gap-1">
          <div className="text-[10px] text-gray-600 font-mono italic">{audit.sources || 0} Sources</div>
          {audit.cached && <span className="text-[10px] bg-accent-math/10 text-accent-math px-1 rounded border border-accent-math/20">Cached</span>}
        </div>
      </div>
      {audit.reasoning && (
        <div className="px-3 py-2 rounded bg-surface-3 border-l-2 border-accent-theory/40 text-xs text-gray-400 italic">
          "{audit.reasoning}"
        </div>
      )}
    </div>
  );
}

export default function MessageBubble(props) {
  var [showEvidence, setShowEvidence] = useState(false);
  var message = props.message;
  var isUser = message.role === "user";

  var isLowFaithfulness = !isUser && message.audit && message.audit.faithfulness < 0.4;

  return (
    <div className="flex gap-3">
      <div className={"w-7 h-7 rounded-md flex items-center justify-center shrink-0 mt-0.5 " + (isUser ? "bg-surface-4" : "bg-surface-3")}>
        {isUser ? <User size={14} className="text-gray-400" /> : <Bot size={14} className="text-gray-400" />}
      </div>
      <div className="flex-1 min-w-0">
        {!isUser && message.intent && (
          <div className="flex items-center gap-2 mb-2">
            <span className={"intent-badge border " + (intentColors[message.intent] || intentColors.hybrid)}>{message.intent}</span>
            {message.context && <span className="text-xs font-mono text-gray-600">{message.context.theory}T {message.context.code}C</span>}
            {message.streaming && <span className="w-1.5 h-1.5 rounded-full bg-accent-theory animate-pulse" />}
          </div>
        )}

        {isLowFaithfulness && (
          <div className="mb-3 p-2 rounded bg-yellow-500/10 border border-yellow-500/30 text-xs text-yellow-500/80 italic">
            <strong>Synthesis Note:</strong> AI is applying theory to general knowledge.
          </div>
        )}

        <div className="prose prose-invert prose-sm max-w-none prose-p:text-gray-300 prose-p:leading-relaxed prose-headings:text-gray-200 prose-headings:font-semibold prose-strong:text-gray-200 prose-code:text-accent-code prose-code:bg-surface-3 prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-code:text-xs prose-code:font-mono prose-pre:bg-surface-2 prose-pre:border prose-pre:border-border prose-pre:rounded-lg prose-a:text-accent-theory prose-a:no-underline hover:prose-a:underline prose-li:text-gray-300 prose-blockquote:border-accent-theory/40 prose-blockquote:text-gray-400">
          {isUser ? (
            <p className="text-gray-200 text-sm">{message.content}</p>
          ) : (
            <ReactMarkdown remarkPlugins={[remarkMath, remarkGfm]} rehypePlugins={[rehypeKatex, rehypeHighlight]}>
              {message.content || ""}
            </ReactMarkdown>
          )}
        </div>

        {!isUser && message.audit && message.audit.evidence && message.audit.evidence.length > 0 && (
          <div className="mt-3">
            <button
              onClick={() => setShowEvidence(!showEvidence)}
              className="text-[10px] text-gray-500 hover:text-gray-300 font-mono flex items-center gap-1 transition-colors"
            >
              {showEvidence ? "[-]" : "[+]"} View Evidence Snippets ({message.audit.evidence.length})
            </button>
            {showEvidence && (
              <div className="mt-2 space-y-2">
                {message.audit.evidence.map((ev, i) => (
                  <div key={i} className="px-3 py-2 rounded bg-surface-2 border border-border text-[11px] text-gray-400 font-mono">
                    <div className="flex items-center gap-1.5 mb-1 text-[10px] text-accent-theory/60 font-bold uppercase">
                      <FileText size={10} />
                      {ev.file_name || "Unknown Source"}{ev.page ? " (P" + ev.page + ")" : ""}
                    </div>
                    {ev.text || ev}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {!isUser && message.sources && message.sources.length > 0 && !message.streaming && !message.audit && (
          <div className="mt-2 text-xs font-mono text-gray-600">{message.sources.length} source{message.sources.length !== 1 ? "s" : ""} cited</div>
        )}
        {!isUser && message.audit && <AuditBadge audit={message.audit} />}
      </div>
    </div>
  );
}
