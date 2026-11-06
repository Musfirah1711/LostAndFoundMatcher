"""SBERT embeddings with in-memory caching."""
import numpy as np

_model = None
_cache = {}

def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def embed(text):
    """Return a single normalized embedding for text (with cache)."""
    if text in _cache:
        return _cache[text]
    model = _get_model()
    vec = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
    _cache[text] = vec
    return vec

def cosine_sim(a, b):
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b))