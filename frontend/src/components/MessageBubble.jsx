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

export default function MessageBubble(props) {
  var message = props.message;
  var isUser = message.role === "user";

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
        <div className="prose prose-invert prose-sm max-w-none prose-p:text-gray-300 prose-p:leading-relaxed prose-headings:text-gray-200 prose-headings:font-semibold prose-strong:text-gray-200 prose-code:text-accent-code prose-code:bg-surface-3 prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-code:text-xs prose-code:font-mono prose-pre:bg-surface-2 prose-pre:border prose-pre:border-border prose-pre:rounded-lg prose-a:text-accent-theory prose-a:no-underline hover:prose-a:underline prose-li:text-gray-300 prose-blockquote:border-accent-theory/40 prose-blockquote:text-gray-400">
          {isUser ? (
            <p className="text-gray-200 text-sm">{message.content}</p>
          ) : (
            <ReactMarkdown remarkPlugins={[remarkMath, remarkGfm]} rehypePlugins={[rehypeKatex, rehypeHighlight]}>
              {message.content || ""}
            </ReactMarkdown>
          )}
        </div>
        {!isUser && message.sources && message.sources.length > 0 && !message.streaming && (
          <div className="mt-2 text-xs font-mono text-gray-600">{message.sources.length} source{message.sources.length !== 1 ? "s" : ""} cited</div>
        )}
      </div>
    </div>
  );
}
