// frontend/src/App.tsx
// Full SlopScan UI — Text + Audio + Video + PPT detection
// Audio/Video/PPT use mock detection locally.
// Replace mock functions with real API calls when backend is live.

import { useState, useRef } from "react";

// ─── MOCK DETECTORS ──────────────────────────────────────────────────────────

function mockDetectText(text) {
  const words     = text.split(/\s+/).filter(Boolean);
  const sentences = text.split(/(?<=[.!?])\s+/).filter(s => s.trim());
  const lens      = sentences.map(s => s.split(/\s+/).length);
  const mean      = lens.reduce((a,b)=>a+b,0)/(lens.length||1);
  const std       = Math.sqrt(lens.reduce((a,b)=>a+(b-mean)**2,0)/(lens.length||1));
  const burstiness = Math.min(1,(std/(mean||1))/0.8);
  const unique    = new Set(words.map(w=>w.toLowerCase().replace(/[^a-z]/g,"")));
  const ttr       = words.length ? unique.size/words.length : 0.5;
  const func      = new Set(["the","a","an","and","or","but","in","on","at","to","for","of","with","by","from","is","are","was","were","be","been","have","has","had","do","does","did","will","would","could","should","it","this","that","i","we","you","he","she","they","my","our","your","his","her","their","not","no","so","also","just","very","really","quite"]);
  const content   = words.filter(w=>!func.has(w.toLowerCase().replace(/[^a-z]/g,"")));
  const density   = words.length ? Math.min(1,Math.max(0,(content.length/words.length-0.35)/0.35)) : 0.5;
  const ngrams    = [];
  for(let i=0;i<words.length-3;i++) ngrams.push(words.slice(i,i+4).join(" ").toLowerCase());
  const repetition = ngrams.length ? Math.min(1,new Set(ngrams).size/ngrams.length) : 1;
  const t2        = text.toLowerCase();
  const bg        = {};
  for(let i=0;i<t2.length-1;i++){const b=t2[i]+t2[i+1];bg[b]=(bg[b]||0)+1;}
  const tot       = Object.values(bg).reduce((a,b)=>a+b,0)||1;
  const ent       = Object.values(bg).reduce((a,c)=>{const p=c/tot;return a-p*Math.log2(p);},0);
  const perplexity = Math.min(1,ent/Math.log2(26*26));
  const humanScore = burstiness*0.30+ttr*0.25+perplexity*0.20+density*0.15+repetition*0.10;
  let aiProb = Math.max(0,Math.min(1,1-humanScore));
  const aiPhrases = ["it is important to note","in today's","rapidly evolving","leveraging","synerg","stakeholder","in conclusion","furthermore","undoubtedly","holistic approach"];
  const hits = aiPhrases.filter(p=>text.toLowerCase().includes(p)).length;
  aiProb = Math.min(1,aiProb+hits*0.06);
  const conf    = Math.min(0.95,0.5+Math.abs(aiProb-0.5)*0.8+(words.length>100?0.1:0));
  const quality = Math.min(1,ttr*0.4+density*0.3+(mean>=10&&mean<=30?0.3:0.1));
  const auth    = Math.max(0,Math.min(1,1-aiProb*0.7+0.1));
  let verdict;
  if(aiProb>=0.85) verdict="AI Generated";
  else if(aiProb>=0.65) verdict="Likely AI";
  else if(aiProb>=0.45) verdict="Uncertain";
  else if(aiProb>=0.25) verdict="Likely Human";
  else verdict="Human";
  const exps=[];
  if(burstiness<0.35) exps.push({feature:"Uniform Sentence Lengths",impact:(0.35-burstiness)*2,description:`Sentence lengths are suspiciously uniform (${(burstiness*100).toFixed(0)}%). Human writers naturally vary rhythm.`});
  if(ttr<0.42) exps.push({feature:"Repetitive Vocabulary",impact:(0.42-ttr)*1.5,description:`Words are reused more than expected (TTR: ${(ttr*100).toFixed(0)}%). AI repeats vocabulary.`});
  if(hits>0) exps.push({feature:"AI Signature Phrases",impact:hits*0.06,description:`Found ${hits} common AI phrase pattern(s) like "in today's rapidly evolving" or "it is important to note".`});
  if(density<0.30) exps.push({feature:"High Filler Word Ratio",impact:(0.30-density)*1.2,description:`High proportion of filler words (${(density*100).toFixed(0)}% density). AI slop pads text.`});
  if(ttr>0.62) exps.push({feature:"Rich Vocabulary",impact:-(ttr-0.62)*1.5,description:`Vocabulary is diverse (TTR: ${(ttr*100).toFixed(0)}%), suggesting human writing.`});
  exps.sort((a,b)=>Math.abs(b.impact)-Math.abs(a.impact));
  return {ai_probability:Math.round(aiProb*1000)/1000,quality_score:Math.round(quality*1000)/1000,authenticity_score:Math.round(auth*1000)/1000,confidence:Math.round(conf*1000)/1000,verdict,feature_scores:{burstiness:Math.round(burstiness*100)/100,vocabulary_diversity:Math.round(ttr*100)/100,perplexity:Math.round(perplexity*100)/100,information_density:Math.round(density*100)/100,repetition_score:Math.round(repetition*100)/100,roberta_score:null},explanation:exps.slice(0,5)};
}

function mockDetectAudio(fileName) {
  // Simulates Wav2Vec2 + MFCC analysis
  // In production: send file to POST /api/detect/audio
  const seed = fileName.length % 10;
  const aiProb = Math.min(0.95, Math.max(0.05, 0.3 + seed * 0.07));
  let verdict;
  if(aiProb>=0.85) verdict="AI Generated";
  else if(aiProb>=0.65) verdict="Likely AI";
  else if(aiProb>=0.45) verdict="Uncertain";
  else if(aiProb>=0.25) verdict="Likely Human";
  else verdict="Human";
  return {
    ai_probability: Math.round(aiProb*100)/100,
    quality_score: 0.72,
    authenticity_score: Math.round((1-aiProb*0.7+0.1)*100)/100,
    confidence: 0.68,
    verdict,
    feature_scores: { pitch_std: Math.round((25-seed*2)*10)/10, spectral_bandwidth_std: Math.round((180+seed*20)*10)/10, mfcc_std: Math.round((12+seed)*10)/10, voiced_fraction: Math.round((0.78+seed*0.02)*100)/100, wav2vec_score: null, duration_seconds: 12.4 },
    explanation: [
      { feature:"Pitch Variation", impact: aiProb > 0.5 ? 0.42 : -0.38, description: aiProb > 0.5 ? "Pitch is suspiciously monotone. AI voices lack natural prosodic variation." : "Pitch varies naturally, consistent with authentic human speech." },
      { feature:"Spectral Bandwidth", impact: aiProb > 0.5 ? 0.31 : -0.22, description: aiProb > 0.5 ? "Spectral bandwidth is unnaturally consistent. TTS systems produce uniform timbre." : "Spectral characteristics vary naturally across the recording." },
      { feature:"MFCC Uniformity", impact: aiProb > 0.5 ? 0.28 : -0.18, description: "MFCC coefficients describe vocal tract shape. Low variation suggests synthesis." },
    ]
  };
}

function mockDetectVideo(fileName) {
  // Simulates CLIP + temporal analysis
  const seed = fileName.length % 10;
  const aiProb = Math.min(0.95, Math.max(0.05, 0.25 + seed * 0.08));
  let verdict;
  if(aiProb>=0.85) verdict="AI Generated";
  else if(aiProb>=0.65) verdict="Likely AI";
  else if(aiProb>=0.45) verdict="Uncertain";
  else if(aiProb>=0.25) verdict="Likely Human";
  else verdict="Human";
  return {
    ai_probability: Math.round(aiProb*100)/100,
    quality_score: 0.68,
    authenticity_score: Math.round((1-aiProb*0.7+0.1)*100)/100,
    confidence: 0.61,
    verdict,
    feature_scores: { clip_score: Math.round(aiProb*0.9*100)/100, temporal_inconsistency: Math.round((aiProb*0.6)*100)/100, frame_count: 24, duration_seconds: 18.2, per_frame_mean: Math.round(aiProb*100)/100, per_frame_std: Math.round(aiProb*0.3*100)/100 },
    explanation: [
      { feature:"CLIP Deepfake Score", impact: aiProb > 0.5 ? 0.55 : -0.40, description: aiProb > 0.5 ? "CLIP model detected visual features inconsistent with authentic video recordings." : "Frame-level visual features are consistent with authentic video." },
      { feature:"Temporal Flickering", impact: aiProb > 0.5 ? 0.38 : -0.22, description: aiProb > 0.5 ? "Frame-to-frame inconsistency detected. Deepfake generators produce subtle flickering." : "Smooth temporal consistency between frames — characteristic of real video." },
    ]
  };
}

function mockDetectPPT(fileName, slideCount) {
  // Simulates per-slide text detection
  const slides = Array.from({length: slideCount}, (_, i) => {
    const aiProb = Math.min(0.95, Math.max(0.05, 0.3 + Math.random() * 0.6));
    let verdict;
    if(aiProb>=0.85) verdict="AI Generated";
    else if(aiProb>=0.65) verdict="Likely AI";
    else if(aiProb>=0.45) verdict="Uncertain";
    else verdict="Likely Human";
    return { slide_number: i+1, title: `Slide ${i+1}`, ai_probability: Math.round(aiProb*100)/100, verdict, word_count: Math.floor(50+Math.random()*200), skipped: false };
  });
  const overall = Math.round(slides.reduce((a,s)=>a+s.ai_probability,0)/slides.length*100)/100;
  let verdict;
  if(overall>=0.85) verdict="AI Generated";
  else if(overall>=0.65) verdict="Likely AI";
  else if(overall>=0.45) verdict="Uncertain";
  else if(overall>=0.25) verdict="Likely Human";
  else verdict="Human";
  const mostAI    = slides.reduce((a,b)=>a.ai_probability>b.ai_probability?a:b);
  const mostHuman = slides.reduce((a,b)=>a.ai_probability<b.ai_probability?a:b);
  return { ai_probability: overall, quality_score: 0.70, authenticity_score: Math.round((1-overall*0.7+0.1)*100)/100, confidence: 0.75, verdict, slide_results: slides, most_ai_slide: mostAI.slide_number, most_human_slide: mostHuman.slide_number, slide_count: slideCount, analyzed_slides: slideCount };
}

// ─── SAMPLES ─────────────────────────────────────────────────────────────────
const SAMPLES = {
  ai: `In today's rapidly evolving digital landscape, it is crucial for organizations to leverage cutting-edge technologies to maximize their competitive advantage. By implementing robust strategies and harnessing the power of artificial intelligence, businesses can streamline their operations and drive unprecedented value creation. It is important to note that success in this endeavor requires a holistic approach that encompasses both technical and organizational dimensions. Furthermore, it is essential to ensure that all stakeholders are aligned with the overarching vision. The synergistic integration of these elements will undoubtedly yield transformative outcomes.`,
  human: `I've been making sourdough for three years now, and honestly the first six months were a disaster. My starter smelled like nail polish remover, my loaves came out dense as bricks, and I threw away probably twenty kilos of flour. The thing nobody tells you is that the windowpane test is basically useless unless you've already developed an intuition for the dough. Last Tuesday my kid knocked the proofing basket off the counter right before I was about to score it. I cried a little. Then I baked it anyway and it somehow came out fine? Better than fine — great ear on it.`,
  mixed: `Artificial intelligence has transformed many industries in recent years. Machine learning algorithms can now perform tasks that once required human expertise. However, implementing these systems requires careful consideration. My team spent three months debugging a recommendation system last year, and the weirdest bug we found was that it kept recommending winter coats to people in Mumbai. Turned out a data labeler had tagged "warm" incorrectly. These systems can perpetuate existing biases present in training data.`,
};

// ─── HELPERS ─────────────────────────────────────────────────────────────────
const pct    = v  => v != null ? `${Math.round(v*100)}%` : "—";
const probColor = p => { if(p==null) return "#8b9198"; if(p>=0.65) return "#ff4d4d"; if(p>=0.45) return "#ffaa00"; return "#44cc88"; };
const verdictColor = v => { if(!v) return "#8b9198"; if(v==="AI Generated") return "#ff4d4d"; if(v==="Likely AI") return "#ffaa00"; if(v==="Uncertain") return "#4db8ff"; if(v==="Likely Human") return "#44cc88"; return "#00ff88"; };

// ─── CSS ─────────────────────────────────────────────────────────────────────
const css = `
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Serif:ital,wght@0,300;0,400;1,300&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
  *{box-sizing:border-box;margin:0;padding:0;}
  body{background:#0a0b0e;color:#d8dce6;font-family:'IBM Plex Sans',sans-serif;font-size:14px;min-height:100vh;}
  body::before{content:'';position:fixed;inset:0;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.03) 2px,rgba(0,0,0,0.03) 4px);pointer-events:none;z-index:9999;}
  body::after{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(255,170,0,0.02) 1px,transparent 1px),linear-gradient(90deg,rgba(255,170,0,0.02) 1px,transparent 1px);background-size:32px 32px;pointer-events:none;z-index:0;}
  @keyframes fadeUp{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
  @keyframes spin{to{transform:rotate(360deg)}}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}
  @keyframes fillBar{from{width:0}}
  .fade-up{animation:fadeUp 0.5s ease forwards;}
  ::-webkit-scrollbar{width:4px;}::-webkit-scrollbar-track{background:#0a0b0e;}::-webkit-scrollbar-thumb{background:#222530;border-radius:2px;}
  textarea{width:100%;resize:vertical;background:#0d0f14;border:1px solid rgba(255,255,255,0.06);border-radius:4px;color:#d8dce6;font-family:'IBM Plex Mono',monospace;font-size:13px;line-height:1.7;padding:14px;outline:none;transition:border-color 0.2s;min-height:180px;}
  textarea:focus{border-color:rgba(255,170,0,0.35);}
  textarea::placeholder{color:#343840;}
  button{cursor:pointer;border:none;outline:none;}
  .tab-btn{padding:7px 16px;border-radius:3px;font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:0.08em;transition:all 0.15s;background:transparent;border:1px solid transparent;color:#5a626e;text-transform:uppercase;}
  .tab-btn:hover{color:#d8dce6;border-color:rgba(255,255,255,0.08);}
  .tab-btn.active{color:#ffaa00;border-color:rgba(255,170,0,0.25);background:rgba(255,170,0,0.05);}
  .sample-btn{padding:3px 10px;border-radius:2px;font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:0.06em;border:1px solid rgba(255,255,255,0.08);color:#5a626e;background:transparent;transition:all 0.15s;text-transform:uppercase;}
  .sample-btn:hover{color:#ffaa00;border-color:rgba(255,170,0,0.25);}
  .analyze-btn{padding:10px 24px;border-radius:3px;background:rgba(255,170,0,0.12);border:1px solid rgba(255,170,0,0.3);color:#ffaa00;font-family:'IBM Plex Mono',monospace;font-size:12px;letter-spacing:0.1em;text-transform:uppercase;transition:all 0.2s;}
  .analyze-btn:hover{background:rgba(255,170,0,0.2);border-color:rgba(255,170,0,0.5);}
  .analyze-btn:disabled{opacity:0.4;cursor:not-allowed;}
  .reset-btn{padding:8px 20px;border-radius:3px;width:100%;background:transparent;border:1px solid rgba(255,255,255,0.07);color:#5a626e;font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:0.08em;text-transform:uppercase;margin-top:12px;transition:all 0.15s;}
  .reset-btn:hover{color:#d8dce6;border-color:rgba(255,255,255,0.12);}
  .bar-track{height:3px;background:rgba(255,255,255,0.06);border-radius:2px;overflow:hidden;}
  .bar-fill{height:100%;border-radius:2px;animation:fillBar 0.8s cubic-bezier(0.16,1,0.3,1) forwards;}
  .upload-zone{border:1px dashed rgba(255,255,255,0.1);border-radius:6px;padding:40px 20px;text-align:center;cursor:pointer;transition:all 0.2s;background:rgba(255,255,255,0.01);}
  .upload-zone:hover,.upload-zone.drag{border-color:rgba(255,170,0,0.3);background:rgba(255,170,0,0.03);}
`;

// ─── SHARED COMPONENTS ────────────────────────────────────────────────────────

function Navbar() {
  return (
    <nav style={{position:"sticky",top:0,zIndex:100,background:"rgba(10,11,14,0.92)",backdropFilter:"blur(16px)",borderBottom:"1px solid rgba(255,255,255,0.05)",padding:"0 24px",height:52,display:"flex",alignItems:"center",justifyContent:"space-between"}}>
      <div style={{display:"flex",alignItems:"center",gap:12}}>
        <div style={{fontFamily:"'IBM Plex Mono',monospace",fontSize:15,color:"#ffaa00",letterSpacing:"0.05em"}}>SLOP<span style={{color:"#d8dce6"}}>SCAN</span></div>
        <div style={{width:1,height:14,background:"rgba(255,255,255,0.1)"}}/>
        <span style={{fontFamily:"'IBM Plex Mono',monospace",fontSize:10,color:"#3a4048",letterSpacing:"0.12em",textTransform:"uppercase"}}>AI Content Forensics</span>
      </div>
      <div style={{display:"flex",alignItems:"center",gap:8}}>
        <div style={{width:6,height:6,borderRadius:"50%",background:"#00ff88",boxShadow:"0 0 6px #00ff88",animation:"pulse 2s ease infinite"}}/>
        <span style={{fontFamily:"'IBM Plex Mono',monospace",fontSize:10,color:"#3a4048",letterSpacing:"0.1em"}}>ALL MODULES ONLINE</span>
      </div>
    </nav>
  );
}

function LoadingSteps({step, steps}) {
  return (
    <div style={{padding:"40px 24px",display:"flex",flexDirection:"column",alignItems:"center",gap:20}}>
      <div style={{width:36,height:36,border:"1.5px solid rgba(255,170,0,0.2)",borderTop:"1.5px solid #ffaa00",borderRadius:"50%",animation:"spin 0.7s linear infinite"}}/>
      <div style={{display:"flex",flexDirection:"column",gap:6,width:"100%",maxWidth:320}}>
        {steps.map((s,i)=>(
          <div key={s} style={{display:"flex",alignItems:"center",gap:10,fontFamily:"'IBM Plex Mono',monospace",fontSize:10,letterSpacing:"0.08em",color:i<step?"#00ff88":i===step?"#ffaa00":"#2a2f38",transition:"all 0.3s"}}>
            <span>{i<step?"✓":i===step?">":"·"}</span><span>{s}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function VerdictBanner({result}) {
  const c = verdictColor(result.verdict);
  return (
    <div className="fade-up" style={{border:`1px solid ${c}22`,background:`${c}08`,borderRadius:4,padding:"20px 22px",display:"flex",justifyContent:"space-between",alignItems:"center",gap:16,marginBottom:16}}>
      <div>
        <div style={{fontFamily:"'IBM Plex Mono',monospace",fontSize:9,letterSpacing:"0.14em",color:"#5a626e",marginBottom:4,textTransform:"uppercase"}}>Verdict</div>
        <div style={{fontFamily:"'IBM Plex Serif',Georgia,serif",fontSize:28,fontWeight:300,color:c,lineHeight:1}}>{result.verdict}</div>
        <div style={{fontFamily:"'IBM Plex Mono',monospace",fontSize:10,color:"#3a4048",marginTop:5}}>{Math.round(result.confidence*100)}% confidence</div>
      </div>
      <div style={{textAlign:"right"}}>
        <div style={{fontFamily:"'IBM Plex Mono',monospace",fontSize:42,fontWeight:600,color:probColor(result.ai_probability),lineHeight:1}}>{pct(result.ai_probability)}</div>
        <div style={{fontFamily:"'IBM Plex Mono',monospace",fontSize:9,color:"#3a4048",letterSpacing:"0.12em",textTransform:"uppercase",marginTop:4}}>AI PROBABILITY</div>
      </div>
    </div>
  );
}

function ScoreGrid({result}) {
  const scores=[
    {label:"AI PROBABILITY",value:result.ai_probability,color:probColor(result.ai_probability)},
    {label:"QUALITY SCORE",value:result.quality_score,color:"#4db8ff"},
    {label:"AUTHENTICITY",value:result.authenticity_score,color:"#44cc88"},
    {label:"CONFIDENCE",value:result.confidence,color:"#ffaa00"},
  ];
  return (
    <div className="fade-up" style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:8,marginBottom:16}}>
      {scores.map(s=>(
        <div key={s.label} style={{background:"#111318",border:"1px solid rgba(255,255,255,0.05)",borderRadius:4,padding:"12px 14px"}}>
          <div style={{fontFamily:"'IBM Plex Mono',monospace",fontSize:8,letterSpacing:"0.12em",color:"#3a4048",marginBottom:8,textTransform:"uppercase"}}>{s.label}</div>
          <div style={{fontFamily:"'IBM Plex Mono',monospace",fontSize:22,fontWeight:600,color:s.color,lineHeight:1,marginBottom:8}}>{pct(s.value)}</div>
          <div className="bar-track"><div className="bar-fill" style={{width:`${Math.round((s.value||0)*100)}%`,background:s.color}}/></div>
        </div>
      ))}
    </div>
  );
}

function ExplanationPanel({items}) {
  if(!items||items.length===0) return null;
  return (
    <div className="fade-up" style={{background:"#111318",border:"1px solid rgba(255,255,255,0.05)",borderRadius:4,padding:"14px 16px",marginBottom:16}}>
      <div style={{fontFamily:"'IBM Plex Mono',monospace",fontSize:9,letterSpacing:"0.14em",color:"#3a4048",marginBottom:12,textTransform:"uppercase"}}>Why this verdict?</div>
      {items.map((item,i)=>{
        const isAI=item.impact>0;
        return(
          <div key={i} style={{padding:"10px 0",borderBottom:"1px solid rgba(255,255,255,0.03)"}}>
            <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start",marginBottom:5}}>
              <span style={{fontFamily:"'IBM Plex Mono',monospace",fontSize:11,color:"#d8dce6",fontWeight:500}}>{item.feature}</span>
              <span style={{fontFamily:"'IBM Plex Mono',monospace",fontSize:9,padding:"2px 7px",borderRadius:2,background:isAI?"rgba(255,77,77,0.1)":"rgba(68,204,136,0.1)",color:isAI?"#ff4d4d":"#44cc88",border:`1px solid ${isAI?"rgba(255,77,77,0.2)":"rgba(68,204,136,0.2)"}`,whiteSpace:"nowrap",marginLeft:8}}>
                {isAI?"↑ AI signal":"↓ Human signal"} {(Math.abs(item.impact)*100).toFixed(0)}%
              </span>
            </div>
            <div style={{fontFamily:"'IBM Plex Sans',sans-serif",fontSize:12,color:"#5a626e",lineHeight:1.6}}>{item.description}</div>
          </div>
        );
      })}
    </div>
  );
}

function FeedbackPanel({result, text, feedbackPhase, setFeedbackPhase, feedbackLog, setFeedbackLog}) {
  function handleCorrect() {
    setFeedbackLog(prev=>[...prev,{predicted_label:result.ai_probability>=0.5?"AI":"human",true_label:result.ai_probability>=0.5?"AI":"human",is_correct:true,timestamp:new Date().toISOString()}]);
    setFeedbackPhase("done");
  }
  function handleWrong() { setFeedbackPhase("wrong"); }
  function handleTrueLabel(label) {
    setFeedbackLog(prev=>[...prev,{predicted_label:result.ai_probability>=0.5?"AI":"human",true_label:label,is_correct:false,timestamp:new Date().toISOString()}]);
    setFeedbackPhase("done");
  }

  if(feedbackPhase==="idle") return null;

  if(feedbackPhase==="done") return (
    <div style={{margin:"14px 0",padding:"12px 16px",background:"rgba(0,255,136,0.05)",border:"1px solid rgba(0,255,136,0.15)",borderRadius:4,display:"flex",alignItems:"center",gap:12}}>
      <span style={{color:"#00ff88",fontSize:16}}>✓</span>
      <div>
        <div style={{fontFamily:"'IBM Plex Mono',monospace",fontSize:11,color:"#00ff88",letterSpacing:"0.08em"}}>FEEDBACK RECORDED</div>
        <div style={{fontFamily:"'IBM Plex Sans',sans-serif",fontSize:11,color:"#3a4048",marginTop:2}}>Your correction helps retrain the model. After 50 corrections, retraining triggers automatically via Celery.</div>
      </div>
      <div style={{marginLeft:"auto",fontFamily:"'IBM Plex Mono',monospace",fontSize:10,color:"#2a2f38"}}>{feedbackLog.length} logged</div>
    </div>
  );

  if(feedbackPhase==="wrong") return (
    <div style={{margin:"14px 0",padding:"16px",background:"rgba(255,170,0,0.04)",border:"1px solid rgba(255,170,0,0.15)",borderRadius:4}}>
      <div style={{fontFamily:"'IBM Plex Mono',monospace",fontSize:9,color:"#5a626e",letterSpacing:"0.14em",textTransform:"uppercase",marginBottom:12}}>What was it actually?</div>
      <div style={{display:"flex",gap:8,flexWrap:"wrap"}}>
        <button onClick={()=>handleTrueLabel("human")} style={{padding:"9px 18px",borderRadius:3,border:"1px solid rgba(68,204,136,0.3)",background:"rgba(68,204,136,0.08)",color:"#44cc88",fontFamily:"'IBM Plex Mono',monospace",fontSize:11,cursor:"pointer"}}>👤 HUMAN WRITTEN</button>
        <button onClick={()=>handleTrueLabel("AI")} style={{padding:"9px 18px",borderRadius:3,border:"1px solid rgba(255,77,77,0.3)",background:"rgba(255,77,77,0.08)",color:"#ff4d4d",fontFamily:"'IBM Plex Mono',monospace",fontSize:11,cursor:"pointer"}}>🤖 AI GENERATED</button>
        <button onClick={()=>setFeedbackPhase("idle")} style={{padding:"9px 14px",borderRadius:3,border:"1px solid rgba(255,255,255,0.06)",background:"transparent",color:"#3a4048",fontFamily:"'IBM Plex Mono',monospace",fontSize:11,cursor:"pointer"}}>CANCEL</button>
      </div>
    </div>
  );

  return (
    <div style={{margin:"14px 0",padding:"16px",background:"rgba(255,255,255,0.02)",border:"1px solid rgba(255,255,255,0.06)",borderRadius:4}}>
      <div style={{fontFamily:"'IBM Plex Mono',monospace",fontSize:9,color:"#5a626e",letterSpacing:"0.14em",textTransform:"uppercase",marginBottom:12}}>// RLHF — was this verdict correct?</div>
      <div style={{display:"flex",gap:8,flexWrap:"wrap"}}>
        <button onClick={handleCorrect} style={{padding:"9px 18px",borderRadius:3,border:"1px solid rgba(0,255,136,0.25)",background:"rgba(0,255,136,0.06)",color:"#00ff88",fontFamily:"'IBM Plex Mono',monospace",fontSize:11,cursor:"pointer"}}>✅ YES, CORRECT</button>
        <button onClick={handleWrong} style={{padding:"9px 18px",borderRadius:3,border:"1px solid rgba(255,77,77,0.25)",background:"rgba(255,77,77,0.06)",color:"#ff4d4d",fontFamily:"'IBM Plex Mono',monospace",fontSize:11,cursor:"pointer"}}>❌ NO, IT WAS WRONG</button>
      </div>
      <div style={{fontFamily:"'IBM Plex Sans',sans-serif",fontSize:11,color:"#2a2f38",marginTop:8}}>Your feedback trains the model. After 50 corrections, retraining triggers automatically.</div>
    </div>
  );
}

// ─── PPT SLIDE BREAKDOWN ──────────────────────────────────────────────────────
function PPTSlideBreakdown({pptResult}) {
  if(!pptResult||!pptResult.slide_results) return null;
  return (
    <div className="fade-up" style={{background:"#111318",border:"1px solid rgba(255,255,255,0.05)",borderRadius:4,padding:"14px 16px",marginBottom:16}}>
      <div style={{fontFamily:"'IBM Plex Mono',monospace",fontSize:9,letterSpacing:"0.14em",color:"#3a4048",marginBottom:4,textTransform:"uppercase"}}>
        Per-Slide Breakdown
        <span style={{color:"#ffaa00",marginLeft:8}}>{pptResult.analyzed_slides}/{pptResult.slide_count} slides analyzed</span>
      </div>
      <div style={{fontFamily:"'IBM Plex Sans',sans-serif",fontSize:11,color:"#3a4048",marginBottom:12}}>
        Most AI-like: Slide {pptResult.most_ai_slide} · Most human-like: Slide {pptResult.most_human_slide}
      </div>
      {pptResult.slide_results.map(slide=>{
        const c = probColor(slide.ai_probability);
        return (
          <div key={slide.slide_number} style={{display:"flex",alignItems:"center",gap:10,padding:"7px 0",borderBottom:"1px solid rgba(255,255,255,0.03)"}}>
            <div style={{fontFamily:"'IBM Plex Mono',monospace",fontSize:10,color:"#3a4048",width:24,flexShrink:0}}>#{slide.slide_number}</div>
            <div style={{flex:1,minWidth:0}}>
              <div style={{fontFamily:"'IBM Plex Mono',monospace",fontSize:10,color:"#8b9198",marginBottom:3,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{slide.title||`Slide ${slide.slide_number}`}</div>
              <div className="bar-track"><div className="bar-fill" style={{width:`${Math.round((slide.ai_probability||0)*100)}%`,background:c}}/></div>
            </div>
            <div style={{fontFamily:"'IBM Plex Mono',monospace",fontSize:10,color:c,width:36,textAlign:"right",flexShrink:0}}>{pct(slide.ai_probability)}</div>
            <div style={{fontFamily:"'IBM Plex Mono',monospace",fontSize:9,color:"#2a2f38",width:80,textAlign:"right",flexShrink:0}}>{slide.verdict}</div>
          </div>
        );
      })}
    </div>
  );
}

// ─── UPLOAD ZONE ─────────────────────────────────────────────────────────────
function UploadZone({accept, label, icon, hint, onFile}) {
  const [drag, setDrag] = useState(false);
  const inputRef = useRef(null);

  function handleDrop(e) {
    e.preventDefault();
    setDrag(false);
    const file = e.dataTransfer.files[0];
    if(file) onFile(file);
  }

  return (
    <div
      className={`upload-zone${drag?" drag":""}`}
      onDragOver={e=>{e.preventDefault();setDrag(true);}}
      onDragLeave={()=>setDrag(false)}
      onDrop={handleDrop}
      onClick={()=>inputRef.current?.click()}
    >
      <input ref={inputRef} type="file" accept={accept} style={{display:"none"}} onChange={e=>{if(e.target.files[0])onFile(e.target.files[0]);}}/>
      <div style={{fontSize:28,marginBottom:10,opacity:0.4}}>{icon}</div>
      <div style={{fontFamily:"'IBM Plex Mono',monospace",fontSize:12,color:"#5a626e",marginBottom:6}}>{label}</div>
      <div style={{fontFamily:"'IBM Plex Sans',sans-serif",fontSize:11,color:"#2a2f38"}}>{hint}</div>
    </div>
  );
}

// ─── FEATURE BREAKDOWN (reusable for text+audio+video) ───────────────────────
function FeatureBreakdown({scores, rows}) {
  if(!scores) return null;
  return (
    <div className="fade-up" style={{background:"#111318",border:"1px solid rgba(255,255,255,0.05)",borderRadius:4,padding:"14px 16px",marginBottom:16}}>
      <div style={{fontFamily:"'IBM Plex Mono',monospace",fontSize:9,letterSpacing:"0.14em",color:"#3a4048",marginBottom:12,textTransform:"uppercase"}}>Signal Breakdown</div>
      {rows.map(r=>{
        const v=scores[r.key];
        const pctVal=v!=null?Math.round(v*100):null;
        const displayVal = r.raw ? (v!=null?`${v?.toFixed(1)}${r.unit||""}`:null) : (pctVal!=null?`${pctVal}%`:null);
        return (
          <div key={r.key} style={{display:"flex",alignItems:"center",gap:10,padding:"7px 0",borderBottom:"1px solid rgba(255,255,255,0.03)"}}>
            <div style={{width:150,flexShrink:0}}>
              <div style={{fontFamily:"'IBM Plex Mono',monospace",fontSize:11,color:"#8b9198"}}>{r.name}</div>
              <div style={{fontFamily:"'IBM Plex Mono',monospace",fontSize:9,color:"#2a2f38"}}>{r.note}</div>
            </div>
            {!r.raw && <div style={{flex:1}}><div className="bar-track"><div className="bar-fill" style={{width:`${pctVal||0}%`,background:pctVal!=null&&pctVal<40?"#ff4d4d":pctVal!=null&&pctVal<60?"#ffaa00":"#44cc88"}}/></div></div>}
            <div style={{fontFamily:"'IBM Plex Mono',monospace",fontSize:11,color:"#5a626e",width:50,textAlign:"right"}}>{displayVal||"—"}</div>
          </div>
        );
      })}
    </div>
  );
}

// ─── TEXT PANEL ───────────────────────────────────────────────────────────────
function TextPanel() {
  const [text,          setText]          = useState("");
  const [phase,         setPhase]         = useState("input");
  const [result,        setResult]        = useState(null);
  const [loadStep,      setLoadStep]      = useState(0);
  const [feedbackPhase, setFeedbackPhase] = useState("idle");
  const [feedbackLog,   setFeedbackLog]   = useState([]);
  const taRef = useRef(null);

  const TEXT_STEPS = ["EXTRACTING FEATURES","COMPUTING BURSTINESS","MEASURING VOCAB","RUNNING CLASSIFIER","BUILDING EXPLANATION"];

  async function analyze() {
    if(text.trim().length<50){
      if(taRef.current){taRef.current.style.borderColor="rgba(255,77,77,0.5)";setTimeout(()=>{if(taRef.current)taRef.current.style.borderColor="";},1200);}
      return;
    }
    setPhase("loading");setLoadStep(0);setFeedbackPhase("idle");
    for(let i=0;i<TEXT_STEPS.length;i++){await new Promise(r=>setTimeout(r,380));setLoadStep(i+1);}
    await new Promise(r=>setTimeout(r,200));
    setResult(mockDetectText(text));
    setPhase("results");
    setTimeout(()=>setFeedbackPhase("asking"),1000);
  }

  function reset(){setPhase("input");setResult(null);setLoadStep(0);setFeedbackPhase("idle");}

  const TEXT_FEATURE_ROWS = [
    {name:"Burstiness",key:"burstiness",note:"sentence length CV"},
    {name:"Vocab diversity",key:"vocabulary_diversity",note:"MATTR 100-word window"},
    {name:"Perplexity proxy",key:"perplexity",note:"bigram entropy"},
    {name:"Info density",key:"information_density",note:"content word ratio"},
    {name:"Repetition score",key:"repetition_score",note:"4-gram uniqueness"},
  ];

  return (
    <>
      {phase==="input" && (
        <div style={{padding:"20px"}}>
          <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:10,flexWrap:"wrap"}}>
            <span style={{fontFamily:"'IBM Plex Mono',monospace",fontSize:9,color:"#2a2f38",letterSpacing:"0.1em",textTransform:"uppercase"}}>Load sample:</span>
            {Object.keys(SAMPLES).map(k=><button key={k} className="sample-btn" onClick={()=>setText(SAMPLES[k])}>{k}</button>)}
          </div>
          <textarea ref={taRef} value={text} onChange={e=>setText(e.target.value)} placeholder="Paste text here for analysis… (minimum 50 characters)"/>
          <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginTop:10}}>
            <span style={{fontFamily:"'IBM Plex Mono',monospace",fontSize:10,color:text.length>40000?"#ffaa00":"#2a2f38"}}>
              {text.length.toLocaleString()} / 50,000 chars
              {text.length<50&&text.length>0&&<span style={{color:"#ff4d4d",marginLeft:8}}>min 50</span>}
            </span>
            <button className="analyze-btn" onClick={analyze} disabled={text.length<50}>ANALYZE →</button>
          </div>
        </div>
      )}
      {phase==="loading" && <LoadingSteps step={loadStep} steps={TEXT_STEPS}/>}
      {phase==="results" && result && (
        <div style={{padding:"16px 20px"}}>
          <VerdictBanner result={result}/>
          <FeedbackPanel result={result} text={text} feedbackPhase={feedbackPhase} setFeedbackPhase={setFeedbackPhase} feedbackLog={feedbackLog} setFeedbackLog={setFeedbackLog}/>
          <ScoreGrid result={result}/>
          <FeatureBreakdown scores={result.feature_scores} rows={TEXT_FEATURE_ROWS}/>
          <ExplanationPanel items={result.explanation}/>
          <button className="reset-btn" onClick={reset}>← ANALYZE ANOTHER TEXT</button>
        </div>
      )}
    </>
  );
}

// ─── AUDIO PANEL ─────────────────────────────────────────────────────────────
function AudioPanel() {
  const [file,          setFile]          = useState(null);
  const [phase,         setPhase]         = useState("input");
  const [result,        setResult]        = useState(null);
  const [loadStep,      setLoadStep]      = useState(0);
  const [feedbackPhase, setFeedbackPhase] = useState("idle");
  const [feedbackLog,   setFeedbackLog]   = useState([]);

  const AUDIO_STEPS = ["LOADING AUDIO FILE","EXTRACTING MFCC FEATURES","ANALYSING PITCH PATTERNS","RUNNING WAV2VEC2","COMPUTING VERDICT"];

  async function analyze() {
    if(!file) return;
    setPhase("loading");setLoadStep(0);setFeedbackPhase("idle");
    for(let i=0;i<AUDIO_STEPS.length;i++){await new Promise(r=>setTimeout(r,420));setLoadStep(i+1);}
    await new Promise(r=>setTimeout(r,200));
    setResult(mockDetectAudio(file.name));
    setPhase("results");
    setTimeout(()=>setFeedbackPhase("asking"),1000);
  }

  function reset(){setPhase("input");setResult(null);setFile(null);setLoadStep(0);setFeedbackPhase("idle");}

  const AUDIO_FEATURE_ROWS = [
    {name:"Pitch variation",key:"pitch_std",note:"F0 std deviation",raw:true,unit:" Hz"},
    {name:"Spectral bandwidth",key:"spectral_bandwidth_std",note:"timbre consistency",raw:true,unit:" Hz"},
    {name:"MFCC uniformity",key:"mfcc_std",note:"vocal tract variation",raw:true,unit:""},
    {name:"Voiced fraction",key:"voiced_fraction",note:"speech activity ratio"},
    {name:"Duration",key:"duration_seconds",note:"audio length",raw:true,unit:"s"},
  ];

  return (
    <>
      {phase==="input" && (
        <div style={{padding:"20px"}}>
          <div style={{fontFamily:"'IBM Plex Mono',monospace",fontSize:9,color:"#3a4048",letterSpacing:"0.12em",textTransform:"uppercase",marginBottom:12}}>
            // Wav2Vec2 + MFCC spectrogram analysis
          </div>
          <UploadZone
            accept=".wav,.mp3,.m4a,.ogg,.flac"
            label="Drop audio file here"
            icon="🎙️"
            hint="Supports .wav .mp3 .m4a .ogg .flac · Max 50MB"
            onFile={f=>{setFile(f);}}
          />
          {file && (
            <div style={{marginTop:12,padding:"10px 14px",background:"rgba(255,170,0,0.04)",border:"1px solid rgba(255,170,0,0.12)",borderRadius:4,display:"flex",justifyContent:"space-between",alignItems:"center"}}>
              <div>
                <div style={{fontFamily:"'IBM Plex Mono',monospace",fontSize:11,color:"#ffaa00"}}>{file.name}</div>
                <div style={{fontFamily:"'IBM Plex Mono',monospace",fontSize:10,color:"#3a4048"}}>{(file.size/1024/1024).toFixed(2)} MB</div>
              </div>
              <button className="analyze-btn" onClick={analyze}>ANALYZE →</button>
            </div>
          )}
          <div style={{marginTop:16,padding:"12px 14px",background:"rgba(255,255,255,0.02)",border:"1px solid rgba(255,255,255,0.04)",borderRadius:4}}>
            <div style={{fontFamily:"'IBM Plex Mono',monospace",fontSize:9,color:"#2a2f38",letterSpacing:"0.1em",marginBottom:8,textTransform:"uppercase"}}>What we detect</div>
            {["Monotone pitch (AI voices lack prosodic variation)","Uniform spectral bandwidth (TTS systems produce consistent timbre)","MFCC uniformity (synthetic vocal tract characteristics)","Unnaturally high voiced fraction (no pauses/breaths)"].map(s=>(
              <div key={s} style={{display:"flex",gap:8,marginBottom:4}}>
                <span style={{color:"#3a4048",flexShrink:0}}>→</span>
                <span style={{fontFamily:"'IBM Plex Sans',sans-serif",fontSize:11,color:"#3a4048"}}>{s}</span>
              </div>
            ))}
          </div>
        </div>
      )}
      {phase==="loading" && <LoadingSteps step={loadStep} steps={AUDIO_STEPS}/>}
      {phase==="results" && result && (
        <div style={{padding:"16px 20px"}}>
          <VerdictBanner result={result}/>
          <FeedbackPanel result={result} text={file?.name||""} feedbackPhase={feedbackPhase} setFeedbackPhase={setFeedbackPhase} feedbackLog={feedbackLog} setFeedbackLog={setFeedbackLog}/>
          <ScoreGrid result={result}/>
          <FeatureBreakdown scores={result.feature_scores} rows={AUDIO_FEATURE_ROWS}/>
          <ExplanationPanel items={result.explanation}/>
          <button className="reset-btn" onClick={reset}>← ANALYZE ANOTHER FILE</button>
        </div>
      )}
    </>
  );
}

// ─── VIDEO PANEL ─────────────────────────────────────────────────────────────
function VideoPanel() {
  const [file,          setFile]          = useState(null);
  const [phase,         setPhase]         = useState("input");
  const [result,        setResult]        = useState(null);
  const [loadStep,      setLoadStep]      = useState(0);
  const [feedbackPhase, setFeedbackPhase] = useState("idle");
  const [feedbackLog,   setFeedbackLog]   = useState([]);

  const VIDEO_STEPS = ["EXTRACTING FRAMES","RUNNING CLIP ON FRAMES","COMPUTING TEMPORAL CONSISTENCY","DETECTING DEEPFAKE ARTIFACTS","GENERATING VERDICT"];

  async function analyze() {
    if(!file) return;
    setPhase("loading");setLoadStep(0);setFeedbackPhase("idle");
    for(let i=0;i<VIDEO_STEPS.length;i++){await new Promise(r=>setTimeout(r,500));setLoadStep(i+1);}
    await new Promise(r=>setTimeout(r,200));
    setResult(mockDetectVideo(file.name));
    setPhase("results");
    setTimeout(()=>setFeedbackPhase("asking"),1000);
  }

  function reset(){setPhase("input");setResult(null);setFile(null);setLoadStep(0);setFeedbackPhase("idle");}

  const VIDEO_FEATURE_ROWS = [
    {name:"CLIP score",key:"clip_score",note:"semantic deepfake prob"},
    {name:"Temporal flicker",key:"temporal_inconsistency",note:"frame consistency"},
    {name:"Frame count",key:"frame_count",note:"frames analyzed",raw:true,unit:""},
    {name:"Duration",key:"duration_seconds",note:"video length",raw:true,unit:"s"},
    {name:"Per-frame std",key:"per_frame_std",note:"score variance"},
  ];

  return (
    <>
      {phase==="input" && (
        <div style={{padding:"20px"}}>
          <div style={{fontFamily:"'IBM Plex Mono',monospace",fontSize:9,color:"#3a4048",letterSpacing:"0.12em",textTransform:"uppercase",marginBottom:12}}>
            // CLIP + ViT frame analysis · temporal consistency detection
          </div>
          <UploadZone
            accept=".mp4,.mov,.avi,.mkv,.webm"
            label="Drop video file here"
            icon="🎬"
            hint="Supports .mp4 .mov .avi .mkv .webm · Max 200MB"
            onFile={f=>{setFile(f);}}
          />
          {file && (
            <div style={{marginTop:12,padding:"10px 14px",background:"rgba(255,170,0,0.04)",border:"1px solid rgba(255,170,0,0.12)",borderRadius:4,display:"flex",justifyContent:"space-between",alignItems:"center"}}>
              <div>
                <div style={{fontFamily:"'IBM Plex Mono',monospace",fontSize:11,color:"#ffaa00"}}>{file.name}</div>
                <div style={{fontFamily:"'IBM Plex Mono',monospace",fontSize:10,color:"#3a4048"}}>{(file.size/1024/1024).toFixed(2)} MB</div>
              </div>
              <button className="analyze-btn" onClick={analyze}>ANALYZE →</button>
            </div>
          )}
          <div style={{marginTop:16,padding:"12px 14px",background:"rgba(255,255,255,0.02)",border:"1px solid rgba(255,255,255,0.04)",borderRadius:4}}>
            <div style={{fontFamily:"'IBM Plex Mono',monospace",fontSize:9,color:"#2a2f38",letterSpacing:"0.1em",marginBottom:8,textTransform:"uppercase"}}>What we detect</div>
            {["CLIP semantic analysis — deepfake face textures and lighting","Temporal flickering — frame-to-frame inconsistency in deepfakes","ViT visual feature extraction per sampled frame","Face artifact detection — blurring, warping, boundary artifacts"].map(s=>(
              <div key={s} style={{display:"flex",gap:8,marginBottom:4}}>
                <span style={{color:"#3a4048",flexShrink:0}}>→</span>
                <span style={{fontFamily:"'IBM Plex Sans',sans-serif",fontSize:11,color:"#3a4048"}}>{s}</span>
              </div>
            ))}
          </div>
        </div>
      )}
      {phase==="loading" && <LoadingSteps step={loadStep} steps={VIDEO_STEPS}/>}
      {phase==="results" && result && (
        <div style={{padding:"16px 20px"}}>
          <VerdictBanner result={result}/>
          <FeedbackPanel result={result} text={file?.name||""} feedbackPhase={feedbackPhase} setFeedbackPhase={setFeedbackPhase} feedbackLog={feedbackLog} setFeedbackLog={setFeedbackLog}/>
          <ScoreGrid result={result}/>
          <FeatureBreakdown scores={result.feature_scores} rows={VIDEO_FEATURE_ROWS}/>
          <ExplanationPanel items={result.explanation}/>
          <button className="reset-btn" onClick={reset}>← ANALYZE ANOTHER FILE</button>
        </div>
      )}
    </>
  );
}

// ─── PPT PANEL ────────────────────────────────────────────────────────────────
function PPTPanel() {
  const [file,          setFile]          = useState(null);
  const [phase,         setPhase]         = useState("input");
  const [result,        setResult]        = useState(null);
  const [loadStep,      setLoadStep]      = useState(0);
  const [feedbackPhase, setFeedbackPhase] = useState("idle");
  const [feedbackLog,   setFeedbackLog]   = useState([]);

  const PPT_STEPS = ["PARSING POWERPOINT FILE","EXTRACTING SLIDE TEXT","RUNNING TEXT DETECTOR PER SLIDE","AGGREGATING SCORES","COMPUTING OVERALL VERDICT"];

  async function analyze() {
    if(!file) return;
    setPhase("loading");setLoadStep(0);setFeedbackPhase("idle");
    for(let i=0;i<PPT_STEPS.length;i++){await new Promise(r=>setTimeout(r,450));setLoadStep(i+1);}
    await new Promise(r=>setTimeout(r,200));
    // Mock: simulate 8-15 slides
    const slideCount = 8 + Math.floor(file.name.length % 8);
    setResult(mockDetectPPT(file.name, slideCount));
    setPhase("results");
    setTimeout(()=>setFeedbackPhase("asking"),1000);
  }

  function reset(){setPhase("input");setResult(null);setFile(null);setLoadStep(0);setFeedbackPhase("idle");}

  return (
    <>
      {phase==="input" && (
        <div style={{padding:"20px"}}>
          <div style={{fontFamily:"'IBM Plex Mono',monospace",fontSize:9,color:"#3a4048",letterSpacing:"0.12em",textTransform:"uppercase",marginBottom:12}}>
            // Per-slide text extraction → RoBERTa detection → aggregated score
          </div>
          <UploadZone
            accept=".pptx,.ppt"
            label="Drop PowerPoint file here"
            icon="📊"
            hint="Supports .pptx .ppt · Max 50MB"
            onFile={f=>{setFile(f);}}
          />
          {file && (
            <div style={{marginTop:12,padding:"10px 14px",background:"rgba(255,170,0,0.04)",border:"1px solid rgba(255,170,0,0.12)",borderRadius:4,display:"flex",justifyContent:"space-between",alignItems:"center"}}>
              <div>
                <div style={{fontFamily:"'IBM Plex Mono',monospace",fontSize:11,color:"#ffaa00"}}>{file.name}</div>
                <div style={{fontFamily:"'IBM Plex Mono',monospace",fontSize:10,color:"#3a4048"}}>{(file.size/1024/1024).toFixed(2)} MB</div>
              </div>
              <button className="analyze-btn" onClick={analyze}>ANALYZE →</button>
            </div>
          )}
          <div style={{marginTop:16,padding:"12px 14px",background:"rgba(255,255,255,0.02)",border:"1px solid rgba(255,255,255,0.04)",borderRadius:4}}>
            <div style={{fontFamily:"'IBM Plex Mono',monospace",fontSize:9,color:"#2a2f38",letterSpacing:"0.1em",marginBottom:8,textTransform:"uppercase"}}>How PPT detection works</div>
            {["Extract text from every shape and text box per slide","Run the same RoBERTa + statistical ensemble on each slide","Weight slide scores by word count (more text = more signal)","Show per-slide breakdown — identify which slides are AI-written"].map(s=>(
              <div key={s} style={{display:"flex",gap:8,marginBottom:4}}>
                <span style={{color:"#3a4048",flexShrink:0}}>→</span>
                <span style={{fontFamily:"'IBM Plex Sans',sans-serif",fontSize:11,color:"#3a4048"}}>{s}</span>
              </div>
            ))}
          </div>
        </div>
      )}
      {phase==="loading" && <LoadingSteps step={loadStep} steps={PPT_STEPS}/>}
      {phase==="results" && result && (
        <div style={{padding:"16px 20px"}}>
          <VerdictBanner result={result}/>
          <FeedbackPanel result={result} text={file?.name||""} feedbackPhase={feedbackPhase} setFeedbackPhase={setFeedbackPhase} feedbackLog={feedbackLog} setFeedbackLog={setFeedbackLog}/>
          <ScoreGrid result={result}/>
          <PPTSlideBreakdown pptResult={result}/>
          <button className="reset-btn" onClick={reset}>← ANALYZE ANOTHER FILE</button>
        </div>
      )}
    </>
  );
}

// ─── HOW IT WORKS ─────────────────────────────────────────────────────────────
function HowItWorks() {
  const signals=[
    {icon:"≈",name:"Burstiness",desc:"CV of sentence lengths. Humans vary rhythm naturally; AI is uniform."},
    {icon:"∑",name:"Vocab Diversity",desc:"Moving Average TTR. AI repeats vocabulary more than humans."},
    {icon:"H",name:"Perplexity Proxy",desc:"Bigram Shannon entropy. AI always picks predictable continuations."},
    {icon:"ρ",name:"Info Density",desc:"Content/filler word ratio. AI slop pads text heavily."},
    {icon:"🎙",name:"MFCC + Pitch",desc:"Acoustic features for voice. AI voices are monotone and uniform."},
    {icon:"🎬",name:"CLIP + Temporal",desc:"Frame-level semantics + consistency. Deepfakes flicker between frames."},
    {icon:"📊",name:"PPT Per-Slide",desc:"Text extracted per slide, run through RoBERTa, weighted by length."},
    {icon:"🔁",name:"RLHF Feedback",desc:"Users validate verdicts. 50 corrections triggers model retraining."},
  ];
  return (
    <div style={{marginTop:64,paddingBottom:64}}>
      <div style={{display:"flex",alignItems:"center",gap:12,marginBottom:6}}>
        <div style={{fontFamily:"'IBM Plex Mono',monospace",fontSize:9,color:"#3a4048",letterSpacing:"0.14em",textTransform:"uppercase"}}>// detection signals</div>
        <div style={{flex:1,height:1,background:"rgba(255,255,255,0.04)"}}/>
      </div>
      <div style={{fontFamily:"'IBM Plex Serif',Georgia,serif",fontSize:22,fontWeight:300,color:"#8b9198",marginBottom:4}}>Four modalities. One platform.</div>
      <div style={{fontFamily:"'IBM Plex Sans',sans-serif",fontSize:13,color:"#3a4048",marginBottom:28}}>Text · Audio · Video · Presentations — all powered by the same ensemble architecture.</div>
      <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(180px,1fr))",gap:10}}>
        {signals.map(s=>(
          <div key={s.name} style={{background:"#111318",border:"1px solid rgba(255,255,255,0.04)",borderRadius:4,padding:"16px"}}>
            <div style={{fontFamily:"'IBM Plex Mono',monospace",fontSize:16,color:"rgba(255,170,0,0.4)",marginBottom:8}}>{s.icon}</div>
            <div style={{fontFamily:"'IBM Plex Mono',monospace",fontSize:11,color:"#8b9198",marginBottom:6}}>{s.name}</div>
            <div style={{fontFamily:"'IBM Plex Sans',sans-serif",fontSize:12,color:"#3a4048",lineHeight:1.6}}>{s.desc}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── MAIN APP ─────────────────────────────────────────────────────────────────
export default function App() {
  const [tab, setTab] = useState("text");

  const TABS = [
    {id:"text",  label:"TEXT",  badge:"LIVE", icon:"📝"},
    {id:"audio", label:"AUDIO", badge:"LIVE", icon:"🎙️"},
    {id:"video", label:"VIDEO", badge:"LIVE", icon:"🎬"},
    {id:"ppt",   label:"PPT",   badge:"LIVE", icon:"📊"},
  ];

  return (
    <>
      <style>{css}</style>
      <div style={{minHeight:"100vh",position:"relative",zIndex:1}}>
        <Navbar/>
        <main style={{maxWidth:900,margin:"0 auto",padding:"40px 20px 80px"}}>

          {/* HERO */}
          <div className="fade-up" style={{marginBottom:40}}>
            <div style={{fontFamily:"'IBM Plex Mono',monospace",fontSize:9,color:"#3a4048",letterSpacing:"0.18em",textTransform:"uppercase",marginBottom:10}}>// forensic content analysis</div>
            <h1 style={{fontFamily:"'IBM Plex Serif',Georgia,serif",fontSize:"clamp(28px,5vw,48px)",fontWeight:300,letterSpacing:"-0.02em",lineHeight:1.1,marginBottom:12}}>
              Is this content<br/><em style={{fontStyle:"italic",color:"#ffaa00"}}>machine-made?</em>
            </h1>
            <p style={{fontFamily:"'IBM Plex Sans',sans-serif",fontSize:13,color:"#3a4048",maxWidth:500,lineHeight:1.7}}>
              Text, audio, video, or presentations — paste or upload anything.
              Six signals + fine-tuned RoBERTa trained on{" "}
              <span style={{color:"#5a626e",fontFamily:"'IBM Plex Mono',monospace",fontSize:11}}>mehddii/ai-text-detector-v2</span>.
            </p>
          </div>

          {/* MAIN CARD */}
          <div className="fade-up" style={{background:"#111318",border:"1px solid rgba(255,255,255,0.05)",borderRadius:6,overflow:"hidden"}}>

            {/* TABS */}
            <div style={{display:"flex",gap:4,padding:"12px 16px 0",borderBottom:"1px solid rgba(255,255,255,0.05)",background:"#0d0f14",flexWrap:"wrap"}}>
              {TABS.map(t=>(
                <button key={t.id} className={`tab-btn${tab===t.id?" active":""}`} onClick={()=>setTab(t.id)} style={{position:"relative",top:1}}>
                  {t.icon} {t.label}
                  <span style={{marginLeft:5,fontFamily:"'IBM Plex Mono',monospace",fontSize:8,padding:"1px 5px",borderRadius:2,background:tab===t.id?"rgba(0,255,136,0.12)":"rgba(255,255,255,0.03)",color:tab===t.id?"#00ff88":"#2a2f38",border:`1px solid ${tab===t.id?"rgba(0,255,136,0.2)":"rgba(255,255,255,0.04)"}`}}>
                    {t.badge}
                  </span>
                </button>
              ))}
            </div>

            {tab==="text"  && <TextPanel/>}
            {tab==="audio" && <AudioPanel/>}
            {tab==="video" && <VideoPanel/>}
            {tab==="ppt"   && <PPTPanel/>}
          </div>

          <HowItWorks/>
        </main>
      </div>
    </>
  );
}