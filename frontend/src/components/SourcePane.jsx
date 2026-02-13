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
        {(!sources || sources.length === 0) ? (
          <div className="flex flex-col items-center justify-center h-full p-6 text-center text-gray-600">
            <BarChart3 size={24} className="mb-2 text-gray-700" />
            <p className="text-xs">
              Synthesized from general knowledge <br />
              <span className="text-[10px] text-gray-700 mt-1 block">(No direct source match found in context)</span>
            </p>
          </div>
        ) : (
          <div className="p-3 space-y-3">
            {sources.map(function (evidence, idx) {
              var fileName = evidence.file_name || "Unknown Source";
              var pageSuffix = evidence.page ? " (Page " + evidence.page + ")" : "";

              return (
                <div key={idx} onClick={function () { setSelectedSource(selectedSource === idx ? null : idx); }} className={"source-card p-3 border rounded-lg transition-all " + (selectedSource === idx ? "border-accent-theory/50 bg-surface-3" : "bg-surface-2 border-border/50")}>
                  <div className="flex items-start justify-between gap-2 mb-2">
                    <div className="flex flex-col">
                      <span className="text-[10px] font-mono text-gray-600 font-bold uppercase tracking-wider">Evidence Chunk #{idx + 1}</span>
                      <div className="flex items-center gap-1.5 mt-0.5">
                        <FileText size={10} className="text-accent-theory/70" />
                        <span className="text-[10px] font-mono text-gray-400 truncate max-w-[150px]">{fileName}{pageSuffix}</span>
                      </div>
                    </div>
                    <BarChart3 size={12} className="text-gray-700" />
                  </div>
                  <div className="text-xs text-gray-300 leading-relaxed font-mono italic">
                    {evidence.text || evidence}
                  </div>
                  {selectedSource === idx && (
                    <div className="mt-3 pt-2 border-t border-border/50">
                      <div className="text-[10px] text-gray-500 italic">
                        Snippet grounded in {fileName}.
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
