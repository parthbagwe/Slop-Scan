// frontend/src/App.tsx  (single-file demo — split into components in real project)
//
// DESIGN DIRECTION: Industrial terminal aesthetic
// Monospaced type, scan-line textures, amber/green on near-black.
// Feels like a forensic analysis tool, not a consumer app.
// That's intentional — this is a serious detection instrument.

import { useState, useEffect, useRef } from "react";

// ─── MOCK API (replace with real axios calls from lib/api.ts) ────────────────
// In production this becomes:
//   const { data } = await detectApi.submitText(text)  →  returns job_id
//   useDetectionJob(jobId)  →  polls until complete
// For this demo we compute scores locally in JS (mirrors the Python logic).

function mockDetect(text) {
  const words     = text.split(/\s+/).filter(Boolean);
  const sentences = text.split(/(?<=[.!?])\s+/).filter(s => s.trim());

  // Burstiness: coefficient of variation of sentence lengths
  const lens = sentences.map(s => s.split(/\s+/).length);
  const mean = lens.reduce((a, b) => a + b, 0) / (lens.length || 1);
  const std  = Math.sqrt(lens.reduce((a, b) => a + (b - mean) ** 2, 0) / (lens.length || 1));
  const burstiness = Math.min(1, (std / (mean || 1)) / 0.8);

  // Vocabulary diversity (TTR)
  const unique = new Set(words.map(w => w.toLowerCase().replace(/[^a-z]/g, "")));
  const ttr    = words.length ? unique.size / words.length : 0.5;

  // Information density
  const func  = new Set(["the","a","an","and","or","but","in","on","at","to","for","of","with","by","from","is","are","was","were","be","been","have","has","had","do","does","did","will","would","could","should","it","this","that","these","those","i","we","you","he","she","they","my","our","your","his","her","their","not","no","so","also","just","very","really","quite","rather","even","more","most"]);
  const content = words.filter(w => !func.has(w.toLowerCase().replace(/[^a-z]/g, "")));
  const density = words.length ? Math.min(1, Math.max(0, (content.length / words.length - 0.35) / 0.35)) : 0.5;

  // Repetition (4-gram uniqueness)
  const ngrams = [];
  for (let i = 0; i < words.length - 3; i++) ngrams.push(words.slice(i, i+4).join(" ").toLowerCase());
  const repetition = ngrams.length ? Math.min(1, new Set(ngrams).size / ngrams.length) : 1;

  // Perplexity proxy (bigram entropy)
  const t2 = text.toLowerCase();
  const bg = {};
  for (let i = 0; i < t2.length - 1; i++) { const b = t2[i]+t2[i+1]; bg[b] = (bg[b]||0)+1; }
  const tot = Object.values(bg).reduce((a,b)=>a+b,0);
  const ent = Object.values(bg).reduce((a,c)=>{ const p=c/tot; return a - p*Math.log2(p); }, 0);
  const perplexity = Math.min(1, ent / Math.log2(26*26));

  // Ensemble → AI probability
  const humanScore = burstiness*0.30 + ttr*0.25 + perplexity*0.20 + density*0.15 + repetition*0.10;
  let aiProb = Math.max(0, Math.min(1, 1 - humanScore));

  // Boost for known AI phrases
  const aiPhrases = ["it is important to note","in today's","rapidly evolving","leveraging","synerg","stakeholder","in conclusion","furthermore","undoubtedly","holistic approach","it goes without saying","at the end of the day","in the realm of"];
  const phraseHits = aiPhrases.filter(p => text.toLowerCase().includes(p)).length;
  aiProb = Math.min(1, aiProb + phraseHits * 0.06);

  const conf    = Math.min(0.95, 0.5 + Math.abs(aiProb - 0.5) * 0.8 + (words.length > 100 ? 0.1 : 0));
  const quality = Math.min(1, ttr * 0.4 + density * 0.3 + (mean >= 10 && mean <= 30 ? 0.3 : 0.1));
  const auth    = Math.max(0, Math.min(1, 1 - aiProb * 0.7 + 0.1));

  let verdict;
  if (aiProb >= 0.85) verdict = "AI Generated";
  else if (aiProb >= 0.65) verdict = "Likely AI";
  else if (aiProb >= 0.45) verdict = "Uncertain";
  else if (aiProb >= 0.25) verdict = "Likely Human";
  else verdict = "Human";

  const exps = [];
  if (burstiness < 0.35) exps.push({ feature:"Uniform Sentence Lengths", impact: (0.35-burstiness)*2, description:`Sentence length variation is very low (${(burstiness*100).toFixed(0)}%). Human writers naturally vary their rhythm. AI produces uniform sentence lengths like a metronome.` });
  if (burstiness > 0.60) exps.push({ feature:"Natural Sentence Rhythm", impact: -(burstiness-0.60)*1.5, description:`Sentence lengths vary naturally (${(burstiness*100).toFixed(0)}%), a strong indicator of authentic human writing.` });
  if (ttr < 0.42) exps.push({ feature:"Repetitive Vocabulary", impact: (0.42-ttr)*1.5, description:`Words are reused more than expected (TTR: ${(ttr*100).toFixed(0)}%). AI models repeat vocabulary across a passage more than humans.` });
  if (phraseHits > 0) exps.push({ feature:"AI Signature Phrases", impact: phraseHits*0.06, description:`Found ${phraseHits} common AI phrase pattern(s) like "in today's rapidly evolving" or "it is important to note". These are hallmarks of LLM-generated text.` });
  if (density < 0.30) exps.push({ feature:"High Filler Word Ratio", impact: (0.30-density)*1.2, description:`High proportion of function/filler words (${(density*100).toFixed(0)}% density). AI slop inflates word count without adding information.` });
  if (ttr > 0.62) exps.push({ feature:"Rich Vocabulary", impact: -(ttr-0.62)*1.5, description:`Vocabulary is diverse and varied (TTR: ${(ttr*100).toFixed(0)}%), suggesting original human expression.` });

  exps.sort((a,b) => Math.abs(b.impact) - Math.abs(a.impact));

  return {
    ai_probability:     Math.round(aiProb*1000)/1000,
    quality_score:      Math.round(quality*1000)/1000,
    authenticity_score: Math.round(auth*1000)/1000,
    confidence:         Math.round(conf*1000)/1000,
    verdict,
    feature_scores: { burstiness: Math.round(burstiness*100)/100, vocabulary_diversity: Math.round(ttr*100)/100, perplexity: Math.round(perplexity*100)/100, information_density: Math.round(density*100)/100, repetition_score: Math.round(repetition*100)/100, roberta_score: null },
    explanation: exps.slice(0,5),
  };
}

// ─── SAMPLES ─────────────────────────────────────────────────────────────────
const SAMPLES = {
  ai: `In today's rapidly evolving digital landscape, it is crucial for organizations to leverage cutting-edge technologies to maximize their competitive advantage. By implementing robust strategies and harnessing the power of artificial intelligence, businesses can streamline their operations and drive unprecedented value creation. It is important to note that success in this endeavor requires a holistic approach that encompasses both technical and organizational dimensions. Furthermore, it is essential to ensure that all stakeholders are aligned with the overarching vision and mission of the organization. The synergistic integration of these elements will undoubtedly yield transformative outcomes and position the organization for long-term sustainable growth. In conclusion, embracing innovation while maintaining operational excellence is the key to thriving in today's dynamic business environment.`,
  human: `I've been making sourdough for three years now, and honestly the first six months were a disaster. My starter smelled like nail polish remover, my loaves came out dense as bricks, and I threw away probably twenty kilos of flour. The thing nobody tells you is that the windowpane test is basically useless unless you've already developed an intuition for the dough. Last Tuesday my kid knocked the proofing basket off the counter right before I was about to score it. I cried a little. Then I baked it anyway and it somehow came out fine? Better than fine, actually — great ear on it. Maybe the impact degassed it in a useful way. I don't know. Bread is weird.`,
  mixed: `Artificial intelligence has transformed many industries in recent years. Machine learning algorithms can now perform tasks that once required human expertise. However, implementing these systems requires careful consideration. My team spent three months debugging a recommendation system last year, and the weirdest bug we found was that it kept recommending winter coats to people in Mumbai. Turned out a data labeler had tagged "warm" incorrectly. These systems can perpetuate existing biases present in training data. Organizations must prioritize fairness and transparency. Anyway, we fixed it by adding a location temperature feature, which sounds obvious in hindsight.`,
};

// ─── HELPERS ─────────────────────────────────────────────────────────────────
function pct(v) { return v != null ? `${Math.round(v * 100)}%` : "—"; }

function verdictColor(v) {
  if (!v) return "#8b9198";
  if (v === "AI Generated") return "#ff4d4d";
  if (v === "Likely AI")    return "#ffaa00";
  if (v === "Uncertain")    return "#4db8ff";
  if (v === "Likely Human") return "#44cc88";
  return "#00ff88";
}

function probColor(p) {
  if (p == null) return "#8b9198";
  if (p >= 0.65) return "#ff4d4d";
  if (p >= 0.45) return "#ffaa00";
  return "#44cc88";
}

// ─── STYLES ──────────────────────────────────────────────────────────────────
const G = {
  // The entire style block lives here so we can use it in JSX style props
  // without needing a CSS file (works in a single-file component)
  bg:       "#0a0b0e",
  surface:  "#111318",
  surface2: "#181b22",
  border:   "rgba(255,255,255,0.06)",
  border2:  "rgba(255,255,255,0.11)",
  text:     "#d8dce6",
  muted:    "#5a626e",
  amber:    "#ffaa00",
  green:    "#00ff88",
  red:      "#ff4d4d",
  blue:     "#4db8ff",
  mono:     "'Courier New', 'Lucida Console', monospace",
  serif:    "Georgia, 'Times New Roman', serif",
  sans:     "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
};

const css = `
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Serif:ital,wght@0,300;0,400;1,300&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: #0a0b0e;
    color: #d8dce6;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 14px;
    min-height: 100vh;
  }

  /* scan-line overlay */
  body::before {
    content: '';
    position: fixed; inset: 0;
    background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.03) 2px, rgba(0,0,0,0.03) 4px);
    pointer-events: none; z-index: 9999;
  }

  /* subtle grid */
  body::after {
    content: '';
    position: fixed; inset: 0;
    background-image: linear-gradient(rgba(255,170,0,0.02) 1px, transparent 1px), linear-gradient(90deg, rgba(255,170,0,0.02) 1px, transparent 1px);
    background-size: 32px 32px;
    pointer-events: none; z-index: 0;
  }

  @keyframes fadeUp   { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:translateY(0); } }
  @keyframes blink    { 0%,100%{opacity:1} 50%{opacity:0} }
  @keyframes spin     { to { transform: rotate(360deg); } }
  @keyframes scanDown { 0%{transform:translateY(-100%)} 100%{transform:translateY(100%)} }
  @keyframes pulse    { 0%,100%{opacity:1} 50%{opacity:0.4} }
  @keyframes fillBar  { from{width:0} }

  .fade-up { animation: fadeUp 0.5s ease forwards; }
  .blink    { animation: blink 1.1s step-end infinite; }

  ::-webkit-scrollbar { width:4px; }
  ::-webkit-scrollbar-track { background:#0a0b0e; }
  ::-webkit-scrollbar-thumb { background:#222530; border-radius:2px; }

  textarea {
    width:100%; resize:vertical;
    background:#0d0f14; border:1px solid rgba(255,255,255,0.06);
    border-radius:4px; color:#d8dce6;
    font-family:'IBM Plex Mono',monospace; font-size:13px; line-height:1.7;
    padding:14px; outline:none; transition:border-color 0.2s;
    min-height:180px;
  }
  textarea:focus { border-color:rgba(255,170,0,0.35); }
  textarea::placeholder { color:#343840; }

  button { cursor:pointer; border:none; outline:none; }

  .tab-btn {
    padding:7px 16px; border-radius:3px;
    font-family:'IBM Plex Mono',monospace; font-size:11px; letter-spacing:0.08em;
    transition:all 0.15s; background:transparent;
    border:1px solid transparent; color:#5a626e; text-transform:uppercase;
  }
  .tab-btn:hover { color:#d8dce6; border-color:rgba(255,255,255,0.08); }
  .tab-btn.active { color:#ffaa00; border-color:rgba(255,170,0,0.25); background:rgba(255,170,0,0.05); }

  .sample-btn {
    padding:3px 10px; border-radius:2px;
    font-family:'IBM Plex Mono',monospace; font-size:10px; letter-spacing:0.06em;
    border:1px solid rgba(255,255,255,0.08); color:#5a626e; background:transparent;
    transition:all 0.15s; text-transform:uppercase;
  }
  .sample-btn:hover { color:#ffaa00; border-color:rgba(255,170,0,0.25); }

  .analyze-btn {
    padding:10px 24px; border-radius:3px;
    background:rgba(255,170,0,0.12); border:1px solid rgba(255,170,0,0.3);
    color:#ffaa00; font-family:'IBM Plex Mono',monospace; font-size:12px;
    letter-spacing:0.1em; text-transform:uppercase;
    transition:all 0.2s;
  }
  .analyze-btn:hover { background:rgba(255,170,0,0.2); border-color:rgba(255,170,0,0.5); }
  .analyze-btn:disabled { opacity:0.4; cursor:not-allowed; }

  .reset-btn {
    padding:8px 20px; border-radius:3px; width:100%;
    background:transparent; border:1px solid rgba(255,255,255,0.07);
    color:#5a626e; font-family:'IBM Plex Mono',monospace; font-size:11px;
    letter-spacing:0.08em; text-transform:uppercase; margin-top:12px;
    transition:all 0.15s;
  }
  .reset-btn:hover { color:#d8dce6; border-color:rgba(255,255,255,0.12); }

  .bar-track { height:3px; background:rgba(255,255,255,0.06); border-radius:2px; overflow:hidden; }
  .bar-fill { height:100%; border-radius:2px; animation:fillBar 0.8s cubic-bezier(0.16,1,0.3,1) forwards; }
`;

// ─── COMPONENTS ──────────────────────────────────────────────────────────────

function Navbar() {
  return (
    <nav style={{ position:"sticky", top:0, zIndex:100, background:"rgba(10,11,14,0.92)", backdropFilter:"blur(16px)", borderBottom:"1px solid rgba(255,255,255,0.05)", padding:"0 24px", height:52, display:"flex", alignItems:"center", justifyContent:"space-between" }}>
      <div style={{ display:"flex", alignItems:"center", gap:12 }}>
        <div style={{ fontFamily:"'IBM Plex Mono',monospace", fontSize:15, color:"#ffaa00", letterSpacing:"0.05em" }}>
          SLOP<span style={{ color:"#d8dce6" }}>SCAN</span>
        </div>
        <div style={{ width:1, height:14, background:"rgba(255,255,255,0.1)" }} />
        <span style={{ fontFamily:"'IBM Plex Mono',monospace", fontSize:10, color:"#3a4048", letterSpacing:"0.12em", textTransform:"uppercase" }}>AI Content Forensics</span>
      </div>
      <div style={{ display:"flex", alignItems:"center", gap:8 }}>
        <div style={{ width:6, height:6, borderRadius:"50%", background:"#00ff88", boxShadow:"0 0 6px #00ff88", animation:"pulse 2s ease infinite" }} />
        <span style={{ fontFamily:"'IBM Plex Mono',monospace", fontSize:10, color:"#3a4048", letterSpacing:"0.1em" }}>TEXT MODULE ONLINE</span>
      </div>
    </nav>
  );
}

function LoadingSteps({ step }) {
  const steps = ["EXTRACTING FEATURES", "COMPUTING BURSTINESS", "MEASURING VOCAB DIVERSITY", "RUNNING CLASSIFIER", "BUILDING EXPLANATION"];
  return (
    <div style={{ padding:"40px 24px", display:"flex", flexDirection:"column", alignItems:"center", gap:20 }}>
      <div style={{ width:36, height:36, border:"1.5px solid rgba(255,170,0,0.2)", borderTop:"1.5px solid #ffaa00", borderRadius:"50%", animation:"spin 0.7s linear infinite" }} />
      <div style={{ display:"flex", flexDirection:"column", gap:6, width:"100%", maxWidth:320 }}>
        {steps.map((s, i) => (
          <div key={s} style={{ display:"flex", alignItems:"center", gap:10, fontFamily:"'IBM Plex Mono',monospace", fontSize:10, letterSpacing:"0.08em", color: i < step ? "#00ff88" : i === step ? "#ffaa00" : "#2a2f38", transition:"all 0.3s" }}>
            <span>{i < step ? "✓" : i === step ? ">" : "·"}</span>
            <span>{s}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function VerdictBanner({ result }) {
  const c = verdictColor(result.verdict);
  return (
    <div className="fade-up" style={{ border:`1px solid ${c}22`, background:`${c}08`, borderRadius:4, padding:"20px 22px", display:"flex", justifyContent:"space-between", alignItems:"center", gap:16, marginBottom:16 }}>
      <div>
        <div style={{ fontFamily:"'IBM Plex Mono',monospace", fontSize:9, letterSpacing:"0.14em", color:"#5a626e", marginBottom:4, textTransform:"uppercase" }}>Verdict</div>
        <div style={{ fontFamily:"'IBM Plex Serif',Georgia,serif", fontSize:28, fontWeight:300, color:c, lineHeight:1 }}>{result.verdict}</div>
        <div style={{ fontFamily:"'IBM Plex Mono',monospace", fontSize:10, color:"#3a4048", marginTop:5 }}>
          {Math.round(result.confidence * 100)}% confidence · {result.feature_scores?.roberta_score != null ? "model + features" : "statistical only"}
        </div>
      </div>
      <div style={{ textAlign:"right" }}>
        <div style={{ fontFamily:"'IBM Plex Mono',monospace", fontSize:42, fontWeight:600, color:probColor(result.ai_probability), lineHeight:1 }}>
          {pct(result.ai_probability)}
        </div>
        <div style={{ fontFamily:"'IBM Plex Mono',monospace", fontSize:9, color:"#3a4048", letterSpacing:"0.12em", textTransform:"uppercase", marginTop:4 }}>AI PROBABILITY</div>
      </div>
    </div>
  );
}

function ScoreGrid({ result }) {
  const scores = [
    { label:"AI PROBABILITY",  value:result.ai_probability,     color: probColor(result.ai_probability) },
    { label:"QUALITY SCORE",   value:result.quality_score,      color:"#4db8ff" },
    { label:"AUTHENTICITY",    value:result.authenticity_score, color:"#44cc88" },
    { label:"CONFIDENCE",      value:result.confidence,         color:"#ffaa00" },
  ];
  return (
    <div className="fade-up" style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:8, marginBottom:16 }}>
      {scores.map(s => (
        <div key={s.label} style={{ background:"#111318", border:"1px solid rgba(255,255,255,0.05)", borderRadius:4, padding:"12px 14px" }}>
          <div style={{ fontFamily:"'IBM Plex Mono',monospace", fontSize:8, letterSpacing:"0.12em", color:"#3a4048", marginBottom:8, textTransform:"uppercase" }}>{s.label}</div>
          <div style={{ fontFamily:"'IBM Plex Mono',monospace", fontSize:22, fontWeight:600, color:s.color, lineHeight:1, marginBottom:8 }}>{pct(s.value)}</div>
          <div className="bar-track">
            <div className="bar-fill" style={{ width:`${Math.round((s.value||0)*100)}%`, background:s.color }} />
          </div>
        </div>
      ))}
    </div>
  );
}

function FeatureBreakdown({ scores }) {
  if (!scores) return null;
  const rows = [
    { name:"Burstiness",           key:"burstiness",           note:"sentence length variation" },
    { name:"Vocab diversity",      key:"vocabulary_diversity",  note:"MATTR across windows" },
    { name:"Perplexity proxy",     key:"perplexity",            note:"bigram entropy" },
    { name:"Info density",         key:"information_density",   note:"content word ratio" },
    { name:"Repetition score",     key:"repetition_score",      note:"4-gram uniqueness" },
  ];
  return (
    <div className="fade-up" style={{ background:"#111318", border:"1px solid rgba(255,255,255,0.05)", borderRadius:4, padding:"14px 16px", marginBottom:16 }}>
      <div style={{ fontFamily:"'IBM Plex Mono',monospace", fontSize:9, letterSpacing:"0.14em", color:"#3a4048", marginBottom:12, textTransform:"uppercase" }}>Signal Breakdown <span style={{ color:"#ffaa00" }}>// higher = more human-like</span></div>
      {rows.map(r => {
        const v = scores[r.key];
        const pctVal = v != null ? Math.round(v * 100) : null;
        return (
          <div key={r.key} style={{ display:"flex", alignItems:"center", gap:10, padding:"7px 0", borderBottom:"1px solid rgba(255,255,255,0.03)" }}>
            <div style={{ width:130, flexShrink:0 }}>
              <div style={{ fontFamily:"'IBM Plex Mono',monospace", fontSize:11, color:"#8b9198" }}>{r.name}</div>
              <div style={{ fontFamily:"'IBM Plex Mono',monospace", fontSize:9, color:"#2a2f38" }}>{r.note}</div>
            </div>
            <div style={{ flex:1 }}>
              <div className="bar-track">
                <div className="bar-fill" style={{ width:`${pctVal ?? 0}%`, background: pctVal != null && pctVal < 40 ? "#ff4d4d" : pctVal != null && pctVal < 60 ? "#ffaa00" : "#44cc88" }} />
              </div>
            </div>
            <div style={{ fontFamily:"'IBM Plex Mono',monospace", fontSize:11, color:"#5a626e", width:32, textAlign:"right" }}>{pctVal != null ? `${pctVal}%` : "—"}</div>
          </div>
        );
      })}
      {scores.roberta_score != null && (
        <div style={{ marginTop:10, padding:"8px 10px", background:"rgba(255,170,0,0.04)", border:"1px solid rgba(255,170,0,0.12)", borderRadius:3 }}>
          <span style={{ fontFamily:"'IBM Plex Mono',monospace", fontSize:10, color:"#5a626e" }}>RoBERTa fine-tuned score: </span>
          <span style={{ fontFamily:"'IBM Plex Mono',monospace", fontSize:10, color:"#ffaa00" }}>{pct(scores.roberta_score)}</span>
          <span style={{ fontFamily:"'IBM Plex Mono',monospace", fontSize:9, color:"#3a4048" }}> AI probability (from mehddii/ai-text-detector-v2 training)</span>
        </div>
      )}
    </div>
  );
}

function ExplanationPanel({ items }) {
  if (!items || items.length === 0) return null;
  return (
    <div className="fade-up" style={{ background:"#111318", border:"1px solid rgba(255,255,255,0.05)", borderRadius:4, padding:"14px 16px", marginBottom:16 }}>
      <div style={{ fontFamily:"'IBM Plex Mono',monospace", fontSize:9, letterSpacing:"0.14em", color:"#3a4048", marginBottom:12, textTransform:"uppercase" }}>Why this verdict?</div>
      {items.map((item, i) => {
        const isAI = item.impact > 0;
        return (
          <div key={i} style={{ padding:"10px 0", borderBottom:"1px solid rgba(255,255,255,0.03)" }}>
            <div style={{ display:"flex", justifyContent:"space-between", alignItems:"flex-start", marginBottom:5 }}>
              <span style={{ fontFamily:"'IBM Plex Mono',monospace", fontSize:11, color:"#d8dce6", fontWeight:500 }}>{item.feature}</span>
              <span style={{ fontFamily:"'IBM Plex Mono',monospace", fontSize:9, padding:"2px 7px", borderRadius:2, background: isAI ? "rgba(255,77,77,0.1)" : "rgba(68,204,136,0.1)", color: isAI ? "#ff4d4d" : "#44cc88", border:`1px solid ${isAI ? "rgba(255,77,77,0.2)" : "rgba(68,204,136,0.2)"}`, whiteSpace:"nowrap", marginLeft:8 }}>
                {isAI ? "↑ AI signal" : "↓ Human signal"} {(Math.abs(item.impact)*100).toFixed(0)}%
              </span>
            </div>
            <div style={{ fontFamily:"'IBM Plex Sans',sans-serif", fontSize:12, color:"#5a626e", lineHeight:1.6 }}>{item.description}</div>
          </div>
        );
      })}
    </div>
  );
}

function ComingSoon({ icon, title, desc, tag }) {
  return (
    <div style={{ padding:"60px 24px", display:"flex", flexDirection:"column", alignItems:"center", gap:14, textAlign:"center" }}>
      <div style={{ fontSize:32, opacity:0.15 }}>{icon}</div>
      <div style={{ fontFamily:"'IBM Plex Serif',Georgia,serif", fontSize:18, fontWeight:300, color:"#3a4048" }}>{title}</div>
      <div style={{ fontFamily:"'IBM Plex Sans',sans-serif", fontSize:12, color:"#2a2f38", maxWidth:280, lineHeight:1.7 }}>{desc}</div>
      <div style={{ fontFamily:"'IBM Plex Mono',monospace", fontSize:9, padding:"4px 12px", borderRadius:2, border:"1px dashed rgba(255,170,0,0.2)", color:"rgba(255,170,0,0.4)", letterSpacing:"0.1em" }}>{tag}</div>
    </div>
  );
}

function HowItWorks() {
  const signals = [
    { icon:"≈", name:"Burstiness", desc:"Coefficient of variation of sentence lengths. Humans write with natural rhythm; AI is eerily uniform." },
    { icon:"∑", name:"Vocab Diversity", desc:"Moving Average TTR in 100-word windows. AI models repeat vocabulary more than humans." },
    { icon:"H", name:"Perplexity Proxy", desc:"Character bigram Shannon entropy. AI always picks predictable continuations." },
    { icon:"ρ", name:"Info Density", desc:"Content word / total word ratio. AI slop pads text with filler phrases." },
    { icon:"4n", name:"Phrase Repetition", desc:"4-gram uniqueness ratio detects recycled transition phrases common in AI output." },
    { icon:"ΔW", name:"RoBERTa Fine-tuned", desc:"Transformer model trained on mehddii/ai-text-detector-v2 dataset. 60% weight in ensemble." },
  ];
  return (
    <div style={{ marginTop:64, paddingBottom:64 }}>
      <div style={{ display:"flex", alignItems:"center", gap:12, marginBottom:6 }}>
        <div style={{ fontFamily:"'IBM Plex Mono',monospace", fontSize:9, color:"#3a4048", letterSpacing:"0.14em", textTransform:"uppercase" }}>// how it works</div>
        <div style={{ flex:1, height:1, background:"rgba(255,255,255,0.04)" }} />
      </div>
      <div style={{ fontFamily:"'IBM Plex Serif',Georgia,serif", fontSize:22, fontWeight:300, color:"#8b9198", marginBottom:4 }}>Six signals. One ensemble.</div>
      <div style={{ fontFamily:"'IBM Plex Sans',sans-serif", fontSize:13, color:"#3a4048", marginBottom:28 }}>Statistical features (weighted 40%) + fine-tuned RoBERTa model (weighted 60%).</div>
      <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fit,minmax(200px,1fr))", gap:10 }}>
        {signals.map(s => (
          <div key={s.name} style={{ background:"#111318", border:"1px solid rgba(255,255,255,0.04)", borderRadius:4, padding:"16px" }}>
            <div style={{ fontFamily:"'IBM Plex Mono',monospace", fontSize:16, color:"rgba(255,170,0,0.4)", marginBottom:8 }}>{s.icon}</div>
            <div style={{ fontFamily:"'IBM Plex Mono',monospace", fontSize:11, color:"#8b9198", marginBottom:6 }}>{s.name}</div>
            <div style={{ fontFamily:"'IBM Plex Sans',sans-serif", fontSize:12, color:"#3a4048", lineHeight:1.6 }}>{s.desc}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── MAIN APP ─────────────────────────────────────────────────────────────────
export default function App() {
  const [tab,    setTab]    = useState("text");
  const [text,   setText]   = useState("");
  const [phase,  setPhase]  = useState("input");   // input | loading | results
  const [result, setResult] = useState(null);
  const [loadStep, setLoadStep] = useState(0);
  const taRef = useRef(null);

  const charCount = text.length;

  async function analyze() {
    if (text.trim().length < 50) {
      if (taRef.current) { taRef.current.style.borderColor = "rgba(255,77,77,0.5)"; setTimeout(() => { if(taRef.current) taRef.current.style.borderColor = ""; }, 1200); }
      return;
    }
    setPhase("loading");
    setLoadStep(0);

    // Simulate step progression
    const steps = [0,1,2,3,4];
    for (let i = 0; i < steps.length; i++) {
      await new Promise(r => setTimeout(r, 380));
      setLoadStep(i + 1);
    }
    await new Promise(r => setTimeout(r, 200));

    // In production: call detectApi.submitText(text) → poll useDetectionJob(jobId)
    const res = mockDetect(text);
    setResult(res);
    setPhase("results");
  }

  function reset() {
    setPhase("input");
    setResult(null);
    setLoadStep(0);
  }

  return (
    <>
      <style>{css}</style>
      <div style={{ minHeight:"100vh", position:"relative", zIndex:1 }}>
        <Navbar />

        <main style={{ maxWidth:860, margin:"0 auto", padding:"40px 20px 80px" }}>

          {/* HERO */}
          <div className="fade-up" style={{ marginBottom:40 }}>
            <div style={{ fontFamily:"'IBM Plex Mono',monospace", fontSize:9, color:"#3a4048", letterSpacing:"0.18em", textTransform:"uppercase", marginBottom:10 }}>// forensic content analysis</div>
            <h1 style={{ fontFamily:"'IBM Plex Serif',Georgia,serif", fontSize:"clamp(28px,5vw,48px)", fontWeight:300, letterSpacing:"-0.02em", lineHeight:1.1, marginBottom:12 }}>
              Is this text<br/><em style={{ fontStyle:"italic", color:"#ffaa00" }}>machine-made?</em>
            </h1>
            <p style={{ fontFamily:"'IBM Plex Sans',sans-serif", fontSize:13, color:"#3a4048", maxWidth:440, lineHeight:1.7 }}>
              Paste any text. Six linguistic signals + a fine-tuned RoBERTa model trained on{" "}
              <span style={{ color:"#5a626e", fontFamily:"'IBM Plex Mono',monospace", fontSize:11 }}>mehddii/ai-text-detector-v2</span>{" "}
              will analyse it in seconds.
            </p>
          </div>

          {/* MAIN CARD */}
          <div className="fade-up" style={{ background:"#111318", border:"1px solid rgba(255,255,255,0.05)", borderRadius:6, overflow:"hidden" }}>

            {/* TABS */}
            <div style={{ display:"flex", gap:4, padding:"12px 16px 0", borderBottom:"1px solid rgba(255,255,255,0.05)", background:"#0d0f14" }}>
              {[
                { id:"text",  label:"TEXT",  badge:"LIVE"  },
                { id:"audio", label:"AUDIO", badge:"SOON"  },
                { id:"video", label:"VIDEO", badge:"SOON"  },
              ].map(t => (
                <button key={t.id} className={`tab-btn${tab===t.id?" active":""}`} onClick={() => setTab(t.id)} style={{ position:"relative", top:1 }}>
                  {t.label}
                  <span style={{ marginLeft:5, fontFamily:"'IBM Plex Mono',monospace", fontSize:8, padding:"1px 5px", borderRadius:2, background: t.badge==="LIVE" ? (tab===t.id?"rgba(0,255,136,0.12)":"rgba(255,255,255,0.04)") : "rgba(255,255,255,0.03)", color: t.badge==="LIVE" ? (tab===t.id?"#00ff88":"#2a2f38") : "#2a2f38", border:`1px solid ${t.badge==="LIVE"&&tab===t.id?"rgba(0,255,136,0.2)":"rgba(255,255,255,0.04)"}` }}>
                    {t.badge}
                  </span>
                </button>
              ))}
            </div>

            {/* TEXT PANEL */}
            {tab === "text" && (
              <>
                {phase === "input" && (
                  <div style={{ padding:"20px 20px 20px" }}>
                    {/* Sample row */}
                    <div style={{ display:"flex", alignItems:"center", gap:8, marginBottom:10, flexWrap:"wrap" }}>
                      <span style={{ fontFamily:"'IBM Plex Mono',monospace", fontSize:9, color:"#2a2f38", letterSpacing:"0.1em", textTransform:"uppercase" }}>Load sample:</span>
                      {Object.keys(SAMPLES).map(k => (
                        <button key={k} className="sample-btn" onClick={() => setText(SAMPLES[k])}>{k}</button>
                      ))}
                    </div>

                    <textarea
                      ref={taRef}
                      value={text}
                      onChange={e => setText(e.target.value)}
                      placeholder="Paste text here for analysis… (minimum 50 characters)"
                    />

                    <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginTop:10 }}>
                      <span style={{ fontFamily:"'IBM Plex Mono',monospace", fontSize:10, color: charCount > 40000 ? "#ffaa00" : "#2a2f38" }}>
                        {charCount.toLocaleString()} / 50,000 chars
                        {charCount < 50 && charCount > 0 && <span style={{ color:"#ff4d4d", marginLeft:8 }}>min 50</span>}
                      </span>
                      <button className="analyze-btn" onClick={analyze} disabled={charCount < 50}>
                        ANALYZE →
                      </button>
                    </div>
                  </div>
                )}

                {phase === "loading" && <LoadingSteps step={loadStep} />}

                {phase === "results" && result && (
                  <div style={{ padding:"16px 20px" }}>
                    <VerdictBanner result={result} />
                    <ScoreGrid result={result} />
                    <FeatureBreakdown scores={result.feature_scores} />
                    <ExplanationPanel items={result.explanation} />
                    <button className="reset-btn" onClick={reset}>← ANALYZE ANOTHER TEXT</button>
                  </div>
                )}
              </>
            )}

            {tab === "audio" && (
              <ComingSoon icon="🎙️" title="Audio detection coming soon" desc="Deepfake voice detection using Wav2Vec2, MFCC spectrogram analysis, and speaker consistency checks." tag="MODULE 2 — NOT BUILT YET" />
            )}

            {tab === "video" && (
              <ComingSoon icon="🎬" title="Video detection coming soon" desc="Deepfake video detection using CLIP + ViT frame analysis, temporal consistency, and face-sync verification." tag="MODULE 3 — NOT BUILT YET" />
            )}
          </div>

          <HowItWorks />
        </main>
      </div>
    </>
  );
}