var STORAGE_KEY = "ros_chat_sessions";
var MAX_SESSIONS = 50;

function loadSessions() {
  try {
    var raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    return JSON.parse(raw);
  } catch (e) {
    return [];
  }
}

function saveSessions(sessions) {
  try {
    var trimmed = sessions.slice(0, MAX_SESSIONS);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(trimmed));
  } catch (e) {
    console.error("Failed to save sessions:", e);
  }
}

export function getSessions() {
  return loadSessions();
}

export function getSession(id) {
  var sessions = loadSessions();
  return sessions.find(function(s) { return s.id === id; }) || null;
}

export function createSession() {
  var session = {
    id: Date.now().toString(36) + Math.random().toString(36).slice(2, 8),
    title: "New Chat",
    messages: [],
    history: [],
    createdAt: Date.now(),
    updatedAt: Date.now(),
  };
  var sessions = loadSessions();
  sessions.unshift(session);
  saveSessions(sessions);
  return session;
}

export function updateSession(id, messages, history) {
  var sessions = loadSessions();
  var idx = -1;
  for (var i = 0; i < sessions.length; i++) {
    if (sessions[i].id === id) { idx = i; break; }
  }
  if (idx === -1) return;

  sessions[idx].messages = messages;
  sessions[idx].history = history;
  sessions[idx].updatedAt = Date.now();

  // Auto-generate title from first user message
  if (sessions[idx].title === "New Chat" && messages.length > 0) {
    for (var j = 0; j < messages.length; j++) {
      if (messages[j].role === "user") {
        var title = messages[j].content.slice(0, 60);
        if (messages[j].content.length > 60) title += "...";
        sessions[idx].title = title;
        break;
      }
    }
  }

  // Move to top
  var updated = sessions.splice(idx, 1)[0];
  sessions.unshift(updated);
  saveSessions(sessions);
}

export function deleteSession(id) {
  var sessions = loadSessions();
  var filtered = sessions.filter(function(s) { return s.id !== id; });
  saveSessions(filtered);
  return filtered;
}

export function clearAllSessions() {
  saveSessions([]);
}

export function formatTime(timestamp) {
  var now = Date.now();
  var diff = now - timestamp;
  var minutes = Math.floor(diff / 60000);
  var hours = Math.floor(diff / 3600000);
  var days = Math.floor(diff / 86400000);

  if (minutes < 1) return "Just now";
  if (minutes < 60) return minutes + "m ago";
  if (hours < 24) return hours + "h ago";
  if (days < 7) return days + "d ago";
  return new Date(timestamp).toLocaleDateString();
}
