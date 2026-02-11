var API_BASE = "http://localhost:8000";

function getHeaders(json) {
  var key = getApiKey();
  var h = {};
  if (key) h["X-API-Key"] = key;
  if (json) h["Content-Type"] = "application/json";
  return h;
}

export function getApiKey() {
  return localStorage.getItem("ros_api_key") || "";
}

export function setApiKey(key) {
  localStorage.setItem("ros_api_key", key);
}

export function hasApiKey() {
  return getApiKey().length > 0;
}

export async function fetchHealth() {
  var res = await fetch(API_BASE + "/health");
  if (!res.ok) throw new Error("Health check failed: " + res.status);
  return res.json();
}

export async function fetchIndexFiles() {
  var res = await fetch(API_BASE + "/v1/index/files", {
    method: "GET",
    headers: getHeaders(false),
  });
  if (!res.ok) throw new Error("Index files failed: " + res.status);
  return res.json();
}

export function streamChat(query, history, filterType, onEvent, onDone, onError) {
  var controller = new AbortController();
  var body = { query: query, history: history };
  if (filterType) body.filter_type = filterType;

  fetch(API_BASE + "/v1/chat", {
    method: "POST",
    headers: getHeaders(true),
    body: JSON.stringify(body),
    signal: controller.signal,
  })
    .then(async function(response) {
      if (!response.ok) {
        var text = await response.text();
        onError("HTTP " + response.status + ": " + text);
        return;
      }
      var reader = response.body.getReader();
      var decoder = new TextDecoder();
      var buffer = "";

      while (true) {
        var result = await reader.read();
        if (result.done) break;
        buffer += decoder.decode(result.value, { stream: true });
        var lines = buffer.split("\n");
        buffer = lines.pop() || "";
        for (var i = 0; i < lines.length; i++) {
          var line = lines[i];
          if (!line.startsWith("data:")) continue;
          var dataStr = line.slice(5).trim();
          if (!dataStr) continue;
          try { onEvent(JSON.parse(dataStr)); } catch(e) {}
        }
      }
      if (buffer.startsWith("data:")) {
        var ds = buffer.slice(5).trim();
        if (ds) { try { onEvent(JSON.parse(ds)); } catch(e) {} }
      }
      onDone();
    })
    .catch(function(err) {
      if (err.name === "AbortError") return;
      onError(err.message);
    });

  return controller;
}

export async function uploadFile(file) {
  var formData = new FormData();
  formData.append("file", file);
  var res = await fetch(API_BASE + "/v1/ingest/file", {
    method: "POST",
    headers: { "X-API-Key": getApiKey() },
    body: formData,
  });
  if (!res.ok) {
    var text = await res.text();
    throw new Error("Upload failed (" + res.status + "): " + text);
  }
  return res.json();
}

export async function ingestUrl(url, filename) {
  var body = { url: url };
  if (filename) body.filename = filename;
  var res = await fetch(API_BASE + "/v1/ingest/url", {
    method: "POST",
    headers: getHeaders(true),
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    var text = await res.text();
    throw new Error("URL ingest failed (" + res.status + "): " + text);
  }
  return res.json();
}

export async function fetchIngestionStatus() {
  var res = await fetch(API_BASE + "/v1/ingest/status", {
    method: "GET",
    headers: getHeaders(false),
  });
  if (!res.ok) throw new Error("Status fetch failed: " + res.status);
  return res.json();
}

export async function fetchTaskStatus(taskId) {
  var res = await fetch(API_BASE + "/v1/ingest/status/" + taskId, {
    method: "GET",
    headers: getHeaders(false),
  });
  if (!res.ok) throw new Error("Task status failed: " + res.status);
  return res.json();
}
