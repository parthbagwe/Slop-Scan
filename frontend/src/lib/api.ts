// frontend/src/lib/api.ts
//
// FIX: localStorage doesn't work in Claude.ai artifact sandbox.
// We use a simple in-memory store instead.
//
// WHAT CHANGES:
// - localStorage.getItem/setItem/removeItem → tokenStore object
// - Everything else stays identical
//
// IN YOUR REAL VS CODE PROJECT:
// localStorage works fine in a real browser / Vite dev server.
// Use localStorage there — only switch to in-memory for the artifact demo.

import axios from "axios";

// ── In-memory token store (replaces localStorage in artifact) ────
// In your real project, swap this back to localStorage.
const tokenStore = {
  _token: null as string | null,
  get()          { return this._token; },
  set(t: string) { this._token = t; },
  clear()        { this._token = null; },
};

// ────────────────────────────────────────────────────────────────
const api = axios.create({
  baseURL: "/api",                    // proxied to :8000 in dev via vite.config.ts
  headers: { "Content-Type": "application/json" },
});

// Attach JWT token to every request automatically
api.interceptors.request.use((config) => {
  const token = tokenStore.get();                    // ← was localStorage.getItem
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

// ── Auth ─────────────────────────────────────────────────────────
export const authApi = {
  register: (email: string, password: string) =>
    api.post("/auth/register", { email, password }),

  login: async (email: string, password: string) => {
    const res = await api.post("/auth/login", { email, password });
    tokenStore.set(res.data.access_token);           // ← was localStorage.setItem
    return res.data;
  },

  logout: () => tokenStore.clear(),                  // ← was localStorage.removeItem

  me: () => api.get("/auth/me"),
};

// ── Detection ────────────────────────────────────────────────────
export const detectApi = {
  submitText: (text: string) =>
    api.post<{ job_id: string; status: string }>("/detect/text", { text }),

  getJob: (jobId: string) =>
    api.get<DetectionResult>(`/detect/jobs/${jobId}`),

  getHistory: (page = 1, pageSize = 20) =>
    api.get<HistoryResponse>(`/detect/history?page=${page}&page_size=${pageSize}`),
};

// ── Types ─────────────────────────────────────────────────────────
export interface FeatureScores {
  perplexity:           number | null;
  burstiness:           number | null;
  vocabulary_diversity: number | null;
  information_density:  number | null;
  roberta_score:        number | null;
  avg_sentence_length:  number | null;
  type_token_ratio:     number | null;
}

export interface ExplanationItem {
  feature:     string;
  impact:      number;
  description: string;
}

export interface DetectionResult {
  job_id:              string;
  status:              "queued" | "processing" | "completed" | "failed";
  content_type:        string;
  ai_probability:      number | null;
  quality_score:       number | null;
  authenticity_score:  number | null;
  confidence:          number | null;
  verdict:             string | null;
  feature_scores:      FeatureScores | null;
  explanation:         ExplanationItem[] | null;
  error_message:       string | null;
  created_at:          string | null;
}

export interface HistoryResponse {
  jobs:      DetectionResult[];
  total:     number;
  page:      number;
  page_size: number;
}