# ml/text/detector.py
#
# PURPOSE: Load the SAVED model (from train.py) and run inference.
#
# KEY DESIGN: the model is loaded ONCE into memory when the
# Celery worker process starts. Subsequent jobs reuse it.
# This is called a "warm model" — no cold start cost per request.
#
# ENSEMBLE:
# Final score = 0.40 * statistical_features + 0.60 * roberta_model
# If model file is missing, fall back to statistical only.

import logging
from pathlib import Path
from typing import Optional

from ml.text.features import extract_text_features

logger = logging.getLogger(__name__)

MODEL_DIR = Path("ml/saved_model")

# Module-level globals — loaded once, reused forever
_model     = None
_tokenizer = None
_model_loaded = False


def _load_model():
    """
    Load the fine-tuned model from disk.
    Called only on the first inference request.

    If the saved_model folder doesn't exist yet (i.e. train.py
    hasn't been run), we fall back to statistical-only mode.
    The user sees results, just with lower confidence.
    """
    global _model, _tokenizer, _model_loaded

    if _model_loaded:
        return

    _model_loaded = True  # Set first so we don't retry on failure

    if not MODEL_DIR.exists():
        logger.warning(
            f"No saved model found at {MODEL_DIR}. "
            "Run ml/text/train.py first. Using statistical-only mode."
        )
        return

    try:
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
        )
        import torch

        logger.info(f"Loading fine-tuned model from {MODEL_DIR}...")
        _tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        _model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
        _model.eval()
        logger.info("Model loaded successfully.")

    except Exception as e:
        logger.error(f"Failed to load model: {e}. Falling back to statistical mode.")
        _model = None
        _tokenizer = None


def _roberta_score(text: str) -> Optional[float]:
    """
    Run the fine-tuned RoBERTa model.
    Returns P(AI) in 0.0–1.0, or None if model unavailable.

    HOW INFERENCE WORKS:
    text → tokenizer → token IDs → model → logits → softmax → probability
    logits: raw unnormalized scores, e.g. [-1.2, 2.4]
    softmax: converts to probabilities that sum to 1: [0.04, 0.96]
    We return probability[1] = P(class=AI)
    """
    _load_model()

    if _model is None or _tokenizer is None:
        return None

    try:
        import torch

        inputs = _tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        with torch.no_grad():  # No gradients at inference = faster + less memory
            logits = _model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)
        return float(probs[0][1])  # P(AI)

    except Exception as e:
        logger.error(f"Inference error: {e}")
        return None


def _statistical_ai_probability(features: dict) -> float:
    """
    Convert hand-crafted features into an AI probability.

    All features are 0–1 where HIGHER = more human.
    We compute a weighted 'human score' then invert it.

    WEIGHTS (must sum to 1.0):
    - burstiness: 0.30           strongest signal
    - vocabulary_diversity: 0.25
    - perplexity_proxy: 0.20
    - information_density: 0.15
    - repetition_score: 0.10

    In production: learn these weights via logistic regression
    on your labeled dataset (fit a model where features=X, label=y).
    """
    weights = {
        "burstiness":           0.30,
        "vocabulary_diversity": 0.25,
        "perplexity_proxy":     0.20,
        "information_density":  0.15,
        "repetition_score":     0.10,
    }
    human_score = sum(features.get(k, 0.5) * w for k, w in weights.items())
    return round(max(0.0, min(1.0, 1.0 - human_score)), 4)


def _quality_score(features: dict) -> float:
    """Quality = well-written, regardless of source."""
    signals = []
    avg_len = features.get("avg_sentence_length", 0)
    signals.append(1.0 if 15 <= avg_len <= 25 else 0.7 if 10 <= avg_len <= 30 else 0.3)
    signals.append(features.get("vocabulary_diversity", 0.5))
    signals.append(features.get("information_density", 0.5))
    wc = features.get("word_count", 0)
    signals.append(1.0 if wc >= 200 else 0.7 if wc >= 100 else 0.4)
    return round(sum(signals) / len(signals), 4)


def _confidence(features: dict, stat_prob: float, roberta: Optional[float]) -> float:
    """
    How confident are we?

    Increases when:
    - We have the trained model (not just statistics)
    - Both scores agree
    - Score is far from 0.5 (decisive)
    - Text is long enough to analyze properly

    Decreases when:
    - Model not loaded (statistical only)
    - Scores disagree
    - Score near 0.5 (borderline case)
    - Short text (< 100 words)
    """
    conf = 0.50
    if roberta is not None:
        conf += 0.20
        agreement = 1.0 - abs(stat_prob - roberta)
        conf += 0.10 * agreement

    final = roberta if roberta is not None else stat_prob
    conf += 0.15 * (abs(final - 0.5) * 2)

    wc = features.get("word_count", 0)
    if wc < 100:  conf *= 0.65
    elif wc < 200: conf *= 0.82

    return round(max(0.0, min(0.98, conf)), 4)


def _build_explanations(features: dict, roberta: Optional[float]) -> list:
    """
    Generate plain-English reasons for the verdict.
    Each entry: { feature, impact, description }
    impact > 0 → pushed toward AI
    impact < 0 → pushed toward human
    Sorted by |impact| so strongest reasons appear first.
    """
    exps = []

    b = features.get("burstiness", 0.5)
    if b < 0.35:
        exps.append({
            "feature": "Uniform Sentence Lengths",
            "impact": round((0.35 - b) * 2.0, 3),
            "description": (
                f"Sentence lengths are suspiciously uniform (score: {b:.2f}). "
                "Human writers naturally vary rhythm — short punchy sentences "
                "mixed with longer explanations. AI tends to produce "
                "consistently similar sentence lengths throughout."
            )
        })
    elif b > 0.60:
        exps.append({
            "feature": "Natural Sentence Rhythm",
            "impact": round(-(b - 0.60) * 1.5, 3),
            "description": (
                f"Sentence lengths vary naturally (score: {b:.2f}), "
                "a strong indicator of authentic human writing."
            )
        })

    v = features.get("vocabulary_diversity", 0.5)
    if v < 0.42:
        exps.append({
            "feature": "Repetitive Vocabulary",
            "impact": round((0.42 - v) * 1.5, 3),
            "description": (
                f"Words are reused more than expected (MATTR: {v:.2f}). "
                "AI language models repeat vocabulary within a passage. "
                "Human writers naturally reach for more varied words."
            )
        })
    elif v > 0.62:
        exps.append({
            "feature": "Rich Vocabulary",
            "impact": round(-(v - 0.62) * 1.5, 3),
            "description": (
                f"Vocabulary is diverse and varied (MATTR: {v:.2f}), "
                "suggesting original human expression."
            )
        })

    d = features.get("information_density", 0.5)
    if d < 0.30:
        exps.append({
            "feature": "High Filler Word Ratio",
            "impact": round((0.30 - d) * 1.2, 3),
            "description": (
                f"High ratio of filler/padding words (density: {d:.2f}). "
                "AI slop inflates text with phrases like "
                "'it is important to note that' or "
                "'in today\\'s rapidly evolving landscape' "
                "that add length without information."
            )
        })

    r = features.get("repetition_score", 1.0)
    if r < 0.82:
        exps.append({
            "feature": "Repeated Phrase Patterns",
            "impact": round((0.82 - r) * 1.5, 3),
            "description": (
                f"4-word phrase patterns repeat (uniqueness: {r:.2f}). "
                "AI reuses transition phrases and sentence openers."
            )
        })

    p = features.get("perplexity_proxy", 0.5)
    if p < 0.42:
        exps.append({
            "feature": "Predictable Language Patterns",
            "impact": round((0.42 - p) * 1.0, 3),
            "description": (
                f"Text follows highly predictable patterns "
                f"(entropy: {p:.2f}). AI always picks high-probability "
                "word continuations, making output predictable at every level."
            )
        })

    if roberta is not None:
        exps.append({
            "feature": "Fine-tuned RoBERTa Score",
            "impact": round((roberta - 0.5) * 0.8, 3),
            "description": (
                f"Fine-tuned classifier assigned {roberta:.1%} AI probability. "
                "Trained on mehddii/ai-text-detector-v2 dataset "
                "(real human writing vs AI-generated samples)."
            )
        })

    exps.sort(key=lambda x: abs(x["impact"]), reverse=True)
    return exps[:5]


def _verdict(ai_prob: float) -> str:
    if ai_prob >= 0.85: return "AI Generated"
    if ai_prob >= 0.65: return "Likely AI"
    if ai_prob >= 0.45: return "Uncertain"
    if ai_prob >= 0.25: return "Likely Human"
    return "Human"


# ──────────────────────────────────────────────────────────────
# MAIN FUNCTION — called by the Celery worker
# ──────────────────────────────────────────────────────────────

def detect_ai_text(text: str) -> dict:
    """
    Full pipeline:
    1. Extract hand-crafted features (fast, always runs)
    2. Run fine-tuned RoBERTa (if available)
    3. Ensemble both → final AI probability
    4. Compute quality, authenticity, confidence
    5. Build explanations
    6. Return full result dict

    This dict maps directly to DetectionResult fields in the DB.
    """
    logger.info(f"detect_ai_text: {len(text)} chars")

    features  = extract_text_features(text)
    stat_prob = _statistical_ai_probability(features)
    roberta   = _roberta_score(text)

    if roberta is not None:
        ai_prob = round(0.40 * stat_prob + 0.60 * roberta, 4)
    else:
        logger.warning("Model not available — using statistical-only score")
        ai_prob = stat_prob

    ai_prob = max(0.0, min(1.0, ai_prob))

    quality      = _quality_score(features)
    authenticity = round(max(0.0, min(1.0, 1.0 - ai_prob * 0.70 + 0.10)), 4)
    conf         = _confidence(features, stat_prob, roberta)

    return {
        "ai_probability":     ai_prob,
        "quality_score":      quality,
        "authenticity_score": authenticity,
        "confidence":         conf,
        "verdict":            _verdict(ai_prob),
        "feature_scores": {
            "perplexity":           features["perplexity_proxy"],
            "burstiness":           features["burstiness"],
            "vocabulary_diversity": features["vocabulary_diversity"],
            "information_density":  features["information_density"],
            "roberta_score":        roberta,
            "avg_sentence_length":  features["avg_sentence_length"],
            "type_token_ratio":     features["type_token_ratio"],
        },
        "explanation":   _build_explanations(features, roberta),
        "model_version": "1.0.0",
    }