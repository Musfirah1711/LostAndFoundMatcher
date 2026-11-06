"""Campus location normalization with fuzzy matching."""
import json
from rapidfuzz import process, fuzz

_GAZ_CACHE = None
_LOC_PREPS = ["near", "outside", "inside", "at", "in", "on", "beside",
              "behind", "around", "by", "within", "opposite", "adjacent",
              "next to"]
_ARTICLES = ["the", "a", "an"]

def load_gazetteer(path="gazetteer.json"):
    global _GAZ_CACHE
    if _GAZ_CACHE is None:
        with open(path, "r", encoding="utf-8") as f:
            _GAZ_CACHE = [g.lower() for g in json.load(f)]
    return _GAZ_CACHE

def _strip_location(text):
    text = text.lower().strip()
    changed = True
    while changed:
        changed = False
        for p in _LOC_PREPS:
            if text.startswith(p + " "):
                text = text[len(p) + 1:]
                changed = True
        for a in _ARTICLES:
            if text.startswith(a + " "):
                text = text[len(a) + 1:]
                changed = True
    return text.strip()

def normalize_location(location_text, gazetteer):
    """Returns (canonical_name_or_None, confidence_0_to_1)."""
    if not location_text:
        return None, 0.0
    cleaned = _strip_location(location_text)
    if not cleaned:
        return None, 0.0
    # Exact match
    for g in gazetteer:
        if g == cleaned:
            return g, 1.0
    # Fuzzy match
    match = process.extractOne(cleaned, gazetteer, scorer=fuzz.token_set_ratio)
    if match:
        name, score, _ = match
        conf = score / 100.0
        if conf >= 0.7:
            return name, conf
    return cleaned, 0.0
