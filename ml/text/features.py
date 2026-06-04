# ml/text/features.py
#
# PURPOSE: Extract 5 mathematical signals from raw text.
# These run BEFORE the neural model — they're cheap and fast.
#
# All scores: 0.0 to 1.0
# HIGHER = more human-like in every case
# (we invert in detector.py to get AI probability)

import re
import math
import statistics


def calculate_burstiness(text: str) -> float:
    """
    WHAT: Measures how much sentence length VARIES.

    WHY IT CATCHES AI:
    Humans write with rhythm — short punchy lines mixed with
    long detailed ones. AI produces uniform sentence lengths
    like a metronome.

    MATH: Coefficient of Variation = std_dev / mean
    Low CV → uniform → AI. High CV → varied → human.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if s.strip()]
    if len(sentences) < 3:
        return 0.5

    lengths = [len(s.split()) for s in sentences]
    mean = statistics.mean(lengths)
    if mean == 0:
        return 0.5

    cv = statistics.stdev(lengths) / mean
    return round(min(1.0, cv / 0.8), 4)


def calculate_vocabulary_diversity(text: str) -> float:
    """
    WHAT: Moving Average Type-Token Ratio (MATTR).
    unique_words / total_words inside a sliding 100-word window.

    WHY NOT SIMPLE TTR?
    Simple TTR decreases naturally with longer texts, making
    it unfair to compare texts of different lengths.
    MATTR normalizes for length.

    WHY IT CATCHES AI:
    AI models repeat vocabulary within a passage more than humans.
    """
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    if len(words) < 20:
        return 0.5
    if len(words) < 100:
        return round(len(set(words)) / len(words), 4)

    window, ttrs = 100, []
    for i in range(0, len(words) - window + 1, 10):
        w = words[i:i + window]
        ttrs.append(len(set(w)) / len(w))
    return round(statistics.mean(ttrs), 4)


def calculate_perplexity_proxy(text: str) -> float:
    """
    WHAT: Character bigram entropy — a proxy for true perplexity.

    REAL PERPLEXITY: needs a language model, slow.
    OUR PROXY: measure how varied character pairs (bigrams) are.

    "he", "el", "ll", "lo" from "hello"

    Shannon entropy H = -Σ p(x) log2 p(x)
    Low H → text reuses the same pairs → predictable → AI
    High H → diverse pairs → surprising → human

    WHY IT CATCHES AI:
    AI picks high-probability continuations at every step,
    making text predictable down to the character level.
    """
    if len(text) < 100:
        return 0.5

    t = text.lower()
    counts = {}
    for i in range(len(t) - 1):
        bg = t[i:i+2]
        counts[bg] = counts.get(bg, 0) + 1

    total = sum(counts.values())
    entropy = -sum((c/total) * math.log2(c/total) for c in counts.values())
    return round(min(1.0, entropy / math.log2(26 * 26)), 4)


def calculate_information_density(text: str) -> float:
    """
    WHAT: Content word ratio.
    Content words (nouns, verbs, adjectives) carry meaning.
    Function words (the, a, is, of) are grammatical glue.

    WHY IT CATCHES AI SLOP:
    "In today's rapidly evolving landscape, it is crucial
     to leverage synergistic frameworks..."
    That sentence is ~50% function/filler words. Says nothing.
    Human writing with purpose uses more content words.

    Normalized: 0.35 density → 0.0, 0.70 density → 1.0
    """
    FUNC = {
        'the','a','an','and','or','but','in','on','at','to','for',
        'of','with','by','from','up','about','into','through','is',
        'are','was','were','be','been','being','have','has','had',
        'do','does','did','will','would','could','should','may',
        'might','it','its','this','that','these','those','i','we',
        'you','he','she','they','my','our','your','his','her',
        'their','not','no','so','also','just','very','really',
        'quite','rather','such','like','even','more','most','than',
    }
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    if len(words) < 10:
        return 0.5

    density = len([w for w in words if w not in FUNC]) / len(words)
    return round(max(0.0, min(1.0, (density - 0.35) / 0.35)), 4)


def calculate_repetition_score(text: str) -> float:
    """
    WHAT: 4-gram uniqueness ratio.
    unique_4word_phrases / total_4word_phrases

    WHY IT CATCHES AI:
    AI reuses transition phrases:
    "It's important to note...", "In conclusion...",
    "Furthermore...", "It goes without saying..."
    """
    words = text.lower().split()
    if len(words) < 20:
        return 1.0

    ngrams = [' '.join(words[i:i+4]) for i in range(len(words) - 3)]
    if not ngrams:
        return 1.0
    return round(min(1.0, len(set(ngrams)) / len(ngrams)), 4)


def extract_text_features(text: str) -> dict:
    """
    Master function — run all extractors, return one dict.
    Everything downstream (detector.py, explainer) uses this.
    """
    words = text.split()
    sentences = [s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]

    return {
        "burstiness":           calculate_burstiness(text),
        "vocabulary_diversity": calculate_vocabulary_diversity(text),
        "perplexity_proxy":     calculate_perplexity_proxy(text),
        "information_density":  calculate_information_density(text),
        "repetition_score":     calculate_repetition_score(text),
        "avg_sentence_length":  round(
            statistics.mean([len(s.split()) for s in sentences])
            if sentences else 0, 2
        ),
        "word_count":     len(words),
        "sentence_count": len(sentences),
        "type_token_ratio": round(
            len(set(words)) / len(words) if words else 0, 4
        ),
    }