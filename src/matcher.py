"""Hybrid composite scoring across PAS components with optional CLIP."""
import os
from .embeddings import embed, cosine_sim
from .gazetteer import normalize_location

DEFAULT_WEIGHTS = {"theme": 0.40, "attributes": 0.20, "location": 0.30, "predicate": 0.10}
WEIGHTS_WITH_IMAGE = {"theme": 0.35, "attributes": 0.15, "location": 0.25,
                      "predicate": 0.10, "image": 0.15}

_clip = None
_clip_failed = False

def _get_clip():
    """Lazily load CLIP. If it fails, cache the failure and never retry."""
    global _clip, _clip_failed
    if _clip_failed:
        return None
    if _clip is None:
        try:
            from sentence_transformers import SentenceTransformer
            _clip = SentenceTransformer("clip-ViT-B-32")
        except Exception as e:
            print(f"[warn] CLIP unavailable: {e}")
            _clip_failed = True
            return None
    return _clip

def image_similarity(path_a, path_b):
    """Return CLIP cosine similarity in [0,1], or None if unavailable."""
    if not path_a or not path_b:
        return None
    if not (os.path.exists(path_a) and os.path.exists(path_b)):
        return None
    model = _get_clip()
    if model is None:
        return None
    try:
        from PIL import Image
        img_a = Image.open(path_a).convert("RGB")
        img_b = Image.open(path_b).convert("RGB")
        embs = model.encode([img_a, img_b], convert_to_numpy=True,
                            normalize_embeddings=True)
        return float(embs[0] @ embs[1])
    except Exception as e:
        print(f"[warn] image_similarity error: {e}")
        return None

def _jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def _predicate_compat(p1, p2):
    return 1.0 if {p1, p2} == {"LOST", "FOUND"} else 0.0

def _confidence(score):
    if score >= 0.65:
        return "HIGH"
    if score >= 0.55:
        return "MEDIUM"
    if score >= 0.35:
        return "LOW"
    return "NONE"

def score_pair(lost_pas, found_pas, gazetteer, weights=None,
               lost_photo=None, found_photo=None, use_image=False):
    # Compute image similarity first to know if we should include its weight
    img_sim = None
    if use_image:
        img_sim = image_similarity(lost_photo, found_photo)

    # Pick weights: image weight only counts if we actually have an image score
    if weights is None:
        if img_sim is not None:
            weights = WEIGHTS_WITH_IMAGE
        else:
            weights = DEFAULT_WEIGHTS

    # 1. Theme
    theme_sim = 0.0
    if lost_pas["theme"] and found_pas["theme"]:
        theme_sim = max(0.0, cosine_sim(
            embed(lost_pas["theme"]), embed(found_pas["theme"])
        ))

    # 2. Attributes
    attr_sim = _jaccard(lost_pas["attributes"], found_pas["attributes"])
    matched_attrs = sorted(set(lost_pas["attributes"]) & set(found_pas["attributes"]))

    # 3. Location
    loc_sim = 0.0
    loc_norm_l = loc_norm_f = None
    if lost_pas["location"] and found_pas["location"]:
        loc_norm_l, conf_l = normalize_location(lost_pas["location"], gazetteer)
        loc_norm_f, conf_f = normalize_location(found_pas["location"], gazetteer)
        if loc_norm_l and loc_norm_f and loc_norm_l == loc_norm_f:
            loc_sim = 0.9 * min(max(conf_l, 0.8), max(conf_f, 0.8))
        else:
            raw = max(0.0, cosine_sim(
                embed(lost_pas["location"]), embed(found_pas["location"])
            ))
            loc_sim = 0.7 * raw

    # 4. Predicate
    pred_sim = _predicate_compat(lost_pas["action_type"], found_pas["action_type"])

    final = (weights["theme"] * theme_sim
             + weights["attributes"] * attr_sim
             + weights["location"] * loc_sim
             + weights["predicate"] * pred_sim)

    if img_sim is not None:
        final += weights["image"] * img_sim

    explanation = {
        "theme_similarity": round(theme_sim, 3),
        "attribute_similarity": round(attr_sim, 3),
        "location_similarity": round(loc_sim, 3),
        "predicate_compatibility": round(pred_sim, 3),
        "image_similarity": round(img_sim, 3) if img_sim is not None else None,
        "matched_attributes": matched_attrs,
        "lost_theme": lost_pas["theme"],
        "found_theme": found_pas["theme"],
        "lost_location": lost_pas["location"],
        "found_location": found_pas["location"],
        "normalized_lost_location": loc_norm_l,
        "normalized_found_location": loc_norm_f,
        "final_score": round(final, 3),
        "confidence": _confidence(final),
        "weights_used": weights,
    }
    return final, explanation

def rank_matches(lost_pas, found_reports, gazetteer, top_k=5,
                 lost_photo=None, use_image=False):
    scored = []
    for fr in found_reports:
        score, exp = score_pair(
            lost_pas, fr["pas"], gazetteer,
            lost_photo=lost_photo,
            found_photo=fr.get("photo_path"),
            use_image=use_image,
        )
        scored.append((score, fr, exp))
    scored.sort(key=lambda x: -x[0])
    return scored[:top_k]