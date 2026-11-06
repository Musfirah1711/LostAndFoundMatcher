"""Ablation evaluation: full PAS pipeline vs theme-only baseline."""
import json
from .pas_extractor import extract_pas
from .matcher import score_pair
from .gazetteer import load_gazetteer
from .embeddings import embed, cosine_sim

def _load_pairs(path="data/test_pairs.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_eval(use_pas=True, weights=None, verbose=False):
    data = _load_pairs()
    gaz = load_gazetteer()

    lost_pas = {r["id"]: extract_pas(r["text"]) for r in data["lost_reports"]}
    found_pas = {r["id"]: extract_pas(r["text"]) for r in data["found_reports"]}

    positives = {}
    for lab in data["labels"]:
        if lab["label"] == 1:
            positives.setdefault(lab["lost_id"], set()).add(lab["found_id"])

    p_at_1, rr_sum, n = 0, 0.0, 0

    for lost in data["lost_reports"]:
        lid = lost["id"]
        if lid not in positives:
            continue
        n += 1
        scored = []
        for fr in data["found_reports"]:
            fid = fr["id"]
            if use_pas:
                score, _ = score_pair(lost_pas[lid], found_pas[fid], gaz, weights)
            else:
                t1 = lost_pas[lid]["theme"] or lost["text"]
                t2 = found_pas[fid]["theme"] or fr["text"]
                score = max(0.0, cosine_sim(embed(t1), embed(t2)))
            scored.append((score, fid))
        scored.sort(key=lambda x: -x[0])

        top = scored[0][1]
        if top in positives[lid]:
            p_at_1 += 1
        for rank, (_, fid) in enumerate(scored, 1):
            if fid in positives[lid]:
                rr_sum += 1.0 / rank
                break

        if verbose:
            print(f"[{lid}] ranked: {[f for _, f in scored[:3]]}  gold: {positives[lid]}")

    return {
        "n_queries": n,
        "precision_at_1": round(p_at_1 / n, 3) if n else 0.0,
        "mrr": round(rr_sum / n, 3) if n else 0.0,
    }

def ablation():
    return {
        "with_PAS": run_eval(use_pas=True),
        "theme_only_baseline": run_eval(use_pas=False),
    }

if __name__ == "__main__":
    print("Full PAS pipeline:")
    print(run_eval(use_pas=True, verbose=True))
    print("\nTheme-only baseline:")
    print(run_eval(use_pas=False))
    print("\nAblation summary:")
    print(json.dumps(ablation(), indent=2))