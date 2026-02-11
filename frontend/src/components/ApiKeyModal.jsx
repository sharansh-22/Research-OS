import { useState } from "react";
import { Key, X } from "lucide-react";

export default function ApiKeyModal(props) {
  var currentKey = props.currentKey;
  var onSave = props.onSave;
  var onClose = props.onClose;
  var [key, setKey] = useState(currentKey);

  var handleSubmit = function(e) {
    e.preventDefault();
    if (key.trim()) onSave(key.trim());
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="bg-surface-2 border border-border rounded-xl w-[400px] shadow-2xl">
        <div className="flex items-center justify-between px-5 py-4 border-b border-border">
          <div className="flex items-center gap-2">
            <Key size={16} className="text-gray-400" />
            <span className="text-sm font-medium text-gray-200">API Key</span>
          </div>
          <button onClick={onClose} className="btn-ghost p-1"><X size={14} /></button>
        </div>
        <form onSubmit={handleSubmit} className="p-5 space-y-4">
          <div>
            <label className="block text-xs text-gray-500 mb-1.5 font-mono">RESEARCH_OS_API_KEY</label>
            <input type="password" value={key} onChange={function(e) { setKey(e.target.value); }} placeholder="Enter your API key" className="input-field font-mono text-xs" autoFocus />
          </div>
          <p className="text-[11px] text-gray-600 leading-relaxed">This key authenticates requests to your local Research-OS API server on port 8000. Stored in localStorage only.</p>
          <div className="flex gap-2 justify-end">
            <button type="button" onClick={onClose} className="btn-ghost">Cancel</button>
            <button type="submit" disabled={!key.trim()} className="btn-primary">Save Key</button>
          </div>
        </form>
      </div>
    </div>
  );
}
