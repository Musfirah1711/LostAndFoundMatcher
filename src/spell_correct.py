"""Lightweight spell correction with domain-word whitelist."""
from spellchecker import SpellChecker

_spell = None

DOMAIN_WORDS = {
    "wallet", "backpack", "umbrella", "airpods", "earbuds", "notebook",
    "library", "canteen", "cafeteria", "hostel", "campus", "gate",
    "leather", "denim", "canvas", "waterbottle", "charger", "idcard",
}

def _get_spell():
    global _spell
    if _spell is None:
        _spell = SpellChecker()
    return _spell

def correct_text(text: str) -> str:
    if not text:
        return text
    spell = _get_spell()
    out = []
    for raw in text.split():
        core = raw.strip(".,!?;:\"'()[]{}").lower()
        # Skip short words, domain words, and words already valid
        if len(core) < 4 or core in DOMAIN_WORDS or core in spell:
            out.append(raw)
            continue
        suggestion = spell.correction(core)
        # Only correct if suggestion exists and is different
        if suggestion and suggestion != core:
            # Preserve first-letter capitalization
            if raw[:1].isupper():
                suggestion = suggestion.capitalize()
            out.append(raw.replace(raw.strip(".,!?;:\"'()[]{}"), suggestion))
        else:
            out.append(raw)
    return " ".join(out)