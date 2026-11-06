"""Predicate–Argument Structure extraction using spaCy dependency parsing."""
import spacy
from spacy.matcher import Matcher

_nlp = None
_matcher = None

STOPWORDS_NP = {"my", "your", "his", "her", "our", "their",
                "a", "an", "the", "some", "this", "that", "these", "those"}

LOCATION_PREPS = {"near", "outside", "inside", "at", "in", "on", "beside",
                  "behind", "around", "by", "within", "opposite",
                  "adjacent", "next"}

LOST_LEMMAS = {"lose", "lost", "misplace", "misplaced"}
FOUND_LEMMAS = {"find", "found", "pick", "picked", "recover", "recovered"}

COLORS = ["black", "white", "red", "blue", "green", "yellow", "brown",
          "grey", "gray", "pink", "purple", "orange", "silver", "gold",
          "golden", "navy", "maroon", "beige", "khaki", "dark", "light"]
MATERIALS = ["leather", "metal", "plastic", "fabric", "denim", "cotton",
             "wooden", "wood", "steel", "glass", "canvas", "rubber",
             "silk", "wool"]

def get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError(
                "spaCy model missing. Run: python -m spacy download en_core_web_sm"
            )
    return _nlp

def _get_matcher():
    global _matcher
    if _matcher is None:
        nlp = get_nlp()
        m = Matcher(nlp.vocab)
        m.add("COLOR", [[{"LOWER": {"IN": COLORS}}]])
        m.add("MATERIAL", [[{"LOWER": {"IN": MATERIALS}}]])
        _matcher = m
    return _matcher

def _np_tokens(token, toks=None):
    if toks is None:
        toks = []
    toks.append(token)
    for c in token.children:
        if c.dep_ in ("amod", "det", "compound", "nummod", "poss"):
            _np_tokens(c, toks)
    return toks

def _span(token):
    toks = _np_tokens(token)
    toks.sort(key=lambda t: t.i)
    return " ".join(t.text for t in toks
                    if t.text.lower() not in STOPWORDS_NP).strip()

def _empty(text):
    return {
        "raw_text": text,
        "predicate": None,
        "predicate_lemma": None,
        "agent": None,
        "theme": None,
        "location": None,
        "attributes": [],
        "action_type": None,   # will be overridden by form's LOST/FOUND
    }

def extract_pas(text):
    if not text or not text.strip():
        return _empty(text)

    nlp = get_nlp()
    doc = nlp(text)
    result = _empty(text)

    # Process first ROOT verb (reports are single-sentence)
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ in ("VERB", "AUX"):
            result["predicate"] = token.text
            result["predicate_lemma"] = token.lemma_.lower()
            lemma = token.lemma_.lower()
            if lemma in LOST_LEMMAS:
                result["action_type"] = "LOST"
            elif lemma in FOUND_LEMMAS:
                result["action_type"] = "FOUND"

            for c in token.children:
                if c.dep_ == "nsubj":
                    result["agent"] = _span(c)
                elif c.dep_ == "nsubjpass":
                    result["theme"] = _span(c)
                elif c.dep_ in ("dobj", "obj", "attr", "acomp"):
                    result["theme"] = _span(c)
                elif c.dep_ == "agent":
                    pobj = [x for x in c.children if x.dep_ == "pobj"]
                    if pobj:
                        result["agent"] = _span(pobj[0])
                elif c.dep_ == "prep" and c.lemma_.lower() in LOCATION_PREPS:
                    pobj = [x for x in c.children if x.dep_ == "pobj"]
                    if pobj:
                        result["location"] = f"{c.text} {_span(pobj[0])}".strip()
            break

    # Extract attributes
    matches = _get_matcher()(doc)
    attrs = set()
    for mid, s, e in matches:
        attrs.add(doc[s:e].text.lower())
    result["attributes"] = sorted(attrs)

    # Fallback theme: longest noun chunk
    if not result["theme"]:
        ncs = list(doc.noun_chunks)
        if ncs:
            longest = max(ncs, key=lambda nc: len(nc.text))
            result["theme"] = " ".join(
                t.text for t in longest if t.text.lower() not in STOPWORDS_NP
            ).strip()

    return result