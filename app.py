import json
import os
import streamlit as st

from src.database import (init_db, insert_report, get_reports,
                          check_duplicate, save_upload)
from src.spell_correct import correct_text
from src.pas_extractor import extract_pas
from src.matcher import rank_matches
from src.gazetteer import load_gazetteer

st.set_page_config(page_title="Smart Lost-and-Found Matcher",
                   page_icon="🔍", layout="wide")
init_db()
GAZ = load_gazetteer()

st.title("🔍 Smart Lost-and-Found Matcher")
st.caption("Predicate–Argument Structure + NLP powered matching")

tab_submit, tab_view, tab_match, tab_eval = st.tabs(
    ["📝 Submit Report", "📊 All Reports", "🎯 Find Matches", "📈 Evaluation"]
)

# ---------------- Submit ----------------
with tab_submit:
    st.subheader("Submit a Lost or Found report")
    col1, col2 = st.columns([1, 3])
    with col1:
        report_type = st.selectbox("Report Type", ["LOST", "FOUND"])
        contact = st.text_input("Contact (optional)")
        uploaded = st.file_uploader("Photo (optional)", type=["jpg", "jpeg", "png"])
    with col2:
        raw_text = st.text_area(
            "Describe the item (natural language)",
            placeholder="e.g., I lost my black wallet near the college library."
        )

    if st.button("Submit Report", type="primary"):
        if not raw_text.strip():
            st.error("Please describe the item.")
        else:
            corrected = correct_text(raw_text)
            if corrected != raw_text:
                st.info(f"✏️ Spell-corrected: {corrected}")

            dup = check_duplicate(corrected)
            if dup:
                st.warning(
                    f"⚠️ Possible duplicate of report #{dup['id']} "
                    f"(similarity {dup['similarity']})."
                )

            photo_path = None
            if uploaded is not None:
                photo_path = save_upload(uploaded.getbuffer(), uploaded.name)

            pas = extract_pas(corrected)
            pas["action_type"] = report_type
            rid = insert_report(report_type, corrected, pas, contact, photo_path)
            st.success(f"Report #{rid} stored.")
            with st.expander("🔎 Extracted PAS structure"):
                st.json(pas)
            if photo_path:
                st.image(photo_path, caption="Uploaded photo", width=240)

# ---------------- View ----------------
with tab_view:
    st.subheader("All reports")
    reports = get_reports()
    if not reports:
        st.info("No reports yet.")
    for r in reports:
        header = f"#{r['id']} — {r['report_type']} — {r['raw_text'][:60]}"
        with st.expander(header):
            left, right = st.columns([2, 1])
            with left:
                st.write(r["raw_text"])
                st.json(json.loads(r["pas_json"]))
            with right:
                if r.get("photo_path") and os.path.exists(r["photo_path"]):
                    st.image(r["photo_path"], caption="Photo", width=240)
                else:
                    st.caption("No photo attached")

# ---------------- Matching ----------------
with tab_match:
    st.subheader("Match lost reports against found reports")
    use_image = st.toggle(
        "Use CLIP image similarity (slower on first run, requires photos on both sides)",
        value=False,
    )
    lost = get_reports("LOST")
    found = get_reports("FOUND")
    st.write(f"**{len(lost)}** lost, **{len(found)}** found.")

    if st.button("Run Matching", type="primary"):
        if not lost or not found:
            st.warning("Need at least one LOST and one FOUND report.")
        else:
            found_payload = [{
                "id": f["id"],
                "raw_text": f["raw_text"],
                "pas": json.loads(f["pas_json"]),
                "photo_path": f.get("photo_path"),
            } for f in found]

            for L in lost:
                lpas = json.loads(L["pas_json"])
                ranked = rank_matches(
                    lpas, found_payload, GAZ, top_k=3,
                    lost_photo=L.get("photo_path"), use_image=use_image,
                )
                st.markdown(f"### Lost #{L['id']}: *{L['raw_text']}*")
                for score, fr, exp in ranked:
                    with st.expander(
                        f"[{exp['confidence']}] Found #{fr['id']} — "
                        f"score {score:.3f} — {fr['raw_text'][:60]}"
                    ):
                        lcol, rcol = st.columns([3, 1])
                        with lcol:
                            st.write(f"**Found report:** {fr['raw_text']}")
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Theme", exp["theme_similarity"])
                            c2.metric("Attributes", exp["attribute_similarity"])
                            c3.metric("Location", exp["location_similarity"])
                            c4.metric("Predicate", exp["predicate_compatibility"])
                            if exp["image_similarity"] is not None:
                                st.metric("Image similarity",
                                          exp["image_similarity"])
                            st.write(
                                f"**Matched attributes:** "
                                f"{exp['matched_attributes'] or '—'}"
                            )
                            st.write(
                                f"**Theme:** {exp['lost_theme']} ↔ "
                                f"{exp['found_theme']}"
                            )
                            st.write(
                                f"**Location:** {exp['lost_location']} ↔ "
                                f"{exp['found_location']}"
                            )
                        with rcol:
                            if fr.get("photo_path") and os.path.exists(fr["photo_path"]):
                                st.image(fr["photo_path"],
                                         caption="Found item photo", width=200)

# ---------------- Evaluation ----------------
with tab_eval:
    st.subheader("Ablation evaluation on the gold-standard set")
    st.write("Compares the full PAS pipeline against a theme-only baseline.")
    if st.button("Run Evaluation"):
        with st.spinner("Running evaluation..."):
            from src.evaluate import ablation
            results = ablation()
        st.json(results)
        pas_p1 = results["with_PAS"]["precision_at_1"]
        base_p1 = results["theme_only_baseline"]["precision_at_1"]
        st.success(
            f"Precision@1 — PAS: **{pas_p1}** vs. theme-only: **{base_p1}** "
            f"(improvement: {round((pas_p1 - base_p1) * 100, 1)} pts)"
        )