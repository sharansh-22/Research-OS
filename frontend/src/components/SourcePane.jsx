import { FileText, Hash, BarChart3 } from "lucide-react";

var typeColors = {
  theory: "text-accent-theory border-accent-theory/30 bg-accent-theory/10",
  code: "text-accent-code border-accent-code/30 bg-accent-code/10",
  math: "text-accent-math border-accent-math/30 bg-accent-math/10",
};

function ScoreBar(props) {
  var score = props.score;
  var width = Math.min(score * 100, 100);
  return (
    <div className="flex items-center gap-2">
      <div className="w-16 h-1 bg-surface-4 rounded-full overflow-hidden">
        <div className="h-full rounded-full bg-accent-theory" style={{ width: width + "%" }} />
      </div>
      <span className="text-[10px] font-mono text-gray-500">{score.toFixed(3)}</span>
    </div>
  );
}

export default function SourcePane(props) {
  var sources = props.sources;
  var selectedSource = props.selectedSource;
  var setSelectedSource = props.setSelectedSource;
  var intent = props.intent;

  return (
    <div className="flex flex-col h-full">
      <div className="pane-header">
        <div className="flex items-center gap-2">
          <FileText size={14} /><span>Source Inspector</span>
          {sources.length > 0 && <span className="text-xs font-mono text-gray-600">{sources.length}</span>}
        </div>
        {intent && <span className={"intent-badge border " + (typeColors[intent] || "text-gray-400 border-border bg-surface-3")}>{intent}</span>}
      </div>
      <div className="flex-1 overflow-y-auto">
        {sources.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-gray-600">
            <FileText size={24} className="mb-2 text-gray-700" />
            <p className="text-xs">Sources appear here after a query</p>
          </div>
        ) : (
          <div className="p-3 space-y-2">
            {sources.map(function(src, idx) {
              return (
                <div key={idx} onClick={function() { setSelectedSource(selectedSource === idx ? null : idx); }} className={"source-card " + (selectedSource === idx ? "border-accent-theory/50 bg-surface-3" : "")}>
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex items-center gap-2 min-w-0">
                      <span className={"intent-badge border text-[10px] " + (typeColors[src.type] || "text-gray-400 border-border bg-surface-3")}>{src.type}</span>
                      <span className="text-xs font-mono text-gray-300 truncate">{src.source}</span>
                    </div>
                    <span className="text-[10px] font-mono text-gray-600 shrink-0">#{idx + 1}</span>
                  </div>
                  {src.section && (
                    <div className="mt-1.5 flex items-center gap-1.5">
                      <Hash size={10} className="text-gray-600 shrink-0" />
                      <span className="text-xs text-gray-500 truncate">{src.section}</span>
                    </div>
                  )}
                  <div className="mt-2 flex items-center gap-1.5">
                    <BarChart3 size={10} className="text-gray-600 shrink-0" />
                    <ScoreBar score={src.score} />
                  </div>
                  {selectedSource === idx && (
                    <div className="mt-3 pt-2 border-t border-border space-y-1.5">
                      <div className="flex items-center justify-between">
                        <span className="text-[10px] text-gray-600">Chunk ID</span>
                        <span className="text-[10px] font-mono text-gray-400">{src.chunk_id}</span>
                      </div>
                      {src.source_path && (
                        <div className="flex items-center justify-between">
                          <span className="text-[10px] text-gray-600">Path</span>
                          <span className="text-[10px] font-mono text-gray-500 truncate max-w-[180px]">{src.source_path}</span>
                        </div>
                      )}
                      <div className="flex items-center justify-between">
                        <span className="text-[10px] text-gray-600">Relevance</span>
                        <span className="text-[10px] font-mono text-gray-400">{(src.score * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
