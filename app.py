"""
Candidate-Role Alignment - Streamlit front end.

Runs in two modes:
  * LLaMA 3 70B via Groq, when GROQ_API_KEY is present
  * deterministic heuristic scorer, when it is not

The app is deliberately usable in heuristic mode so the deployed demo works for
anyone, with no key and no cost.
"""
import json
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from parsing import Candidate, from_plain_text, from_json_profile, from_html  # noqa: E402
from scoring import DIMENSIONS, HeuristicScorer, LLMScorer, RoleSpec  # noqa: E402

SAMPLE_DIR = "sample_data"


def _secret_present(key: str) -> bool:
    """st.secrets raises when no secrets file exists, so probe defensively."""
    try:
        return bool(st.secrets.get(key, ""))
    except Exception:
        return False

st.set_page_config(page_title="Candidate-Role Alignment", page_icon="📊", layout="wide")


# ---------------------------------------------------------------------------
# Sidebar - role definition and engine choice
# ---------------------------------------------------------------------------

def load_default_role() -> dict:
    with open(os.path.join(SAMPLE_DIR, "role_spec.json")) as fh:
        return json.load(fh)


with st.sidebar:
    st.header("Role definition")
    default = load_default_role()

    title = st.text_input("Title", default["title"])
    seniority = st.selectbox("Seniority", ["Intern", "Junior", "Mid", "Senior"],
                             index=["Intern", "Junior", "Mid", "Senior"].index(default["seniority"]))
    min_years = st.number_input("Minimum years", 0.0, 20.0, float(default["min_years"]), 0.5)
    domain = st.text_input("Domain", default["domain"])
    all_skills = ["sql", "python", "analytics", "visualisation", "data_engineering",
                  "product", "ml", "cloud", "stakeholder"]
    must = st.multiselect("Must-have skills", all_skills, default["must_have_skills"])
    nice = st.multiselect("Nice-to-have skills", all_skills, default["nice_to_have_skills"])
    description = st.text_area("Description", default["description"], height=110)

    st.divider()
    st.header("Scoring engine")
    has_key = bool(os.getenv("GROQ_API_KEY")) or _secret_present("GROQ_API_KEY")
    engine_choice = st.radio(
        "Engine",
        ["Heuristic (no key needed)", "LLaMA 3 70B (Groq)"],
        index=1 if has_key else 0,
        help="The heuristic scorer is deterministic and runs offline. "
             "The LLM scorer needs a Groq API key.",
    )
    if "LLaMA" in engine_choice and not has_key:
        st.warning("No GROQ_API_KEY found. Add it to Streamlit secrets or your .env file.")

role = RoleSpec(
    title=title, seniority=seniority, must_have_skills=must,
    nice_to_have_skills=nice, min_years=min_years, domain=domain, description=description,
)


# ---------------------------------------------------------------------------
# Main - candidate intake
# ---------------------------------------------------------------------------

st.title("Candidate-Role Alignment Model")
st.caption(
    "Scores how well each candidate aligns with an open role on five dimensions, 1-10. "
    "A decision-support tool, not a decision-making one."
)

tab_samples, tab_upload, tab_paste = st.tabs(["Sample candidates", "Upload files", "Paste text"])
candidates: list[Candidate] = []

with tab_samples:
    files = sorted(f for f in os.listdir(SAMPLE_DIR) if f.endswith(".txt"))
    picked = st.multiselect("Sample resumes", files, default=files)
    for filename in picked:
        with open(os.path.join(SAMPLE_DIR, filename)) as fh:
            candidates.append(from_plain_text(filename.replace(".txt", ""), fh.read()))

with tab_upload:
    uploads = st.file_uploader("Resume files (.txt, .json, .html)",
                               type=["txt", "json", "html"], accept_multiple_files=True)
    for upload in uploads or []:
        raw = upload.read().decode("utf-8", errors="ignore")
        name = upload.name.rsplit(".", 1)[0]
        if upload.name.endswith(".json"):
            candidates.append(from_json_profile(raw))
        elif upload.name.endswith(".html"):
            candidates.append(from_html(name, raw))
        else:
            candidates.append(from_plain_text(name, raw))

with tab_paste:
    pasted = st.text_area("Paste a single resume", height=200, key="pasted")
    pasted_name = st.text_input("Candidate label", "Pasted candidate")
    if pasted.strip():
        candidates.append(from_plain_text(pasted_name, pasted))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

if st.button("Score candidates", type="primary", disabled=not candidates):
    try:
        scorer = LLMScorer() if "LLaMA" in engine_choice else HeuristicScorer()
    except RuntimeError as exc:
        st.error(f"{exc} — falling back to the heuristic scorer.")
        scorer = HeuristicScorer()

    control = HeuristicScorer()
    rows, details = [], {}

    progress = st.progress(0.0, text="Scoring...")
    for i, candidate in enumerate(candidates, start=1):
        result = scorer.score(candidate, role)
        baseline = control.score(candidate, role)
        details[candidate.name] = (result, candidate)
        rows.append({
            "Candidate": candidate.name,
            "Alignment (1-10)": result.overall,
            "Heuristic control": baseline.overall,
            "Divergence": round(abs(result.overall - baseline.overall), 1),
            "Years": candidate.years_experience,
            "Skills detected": ", ".join(candidate.skills) or "—",
            **{label: result.dimensions[key] for key, (label, _) in DIMENSIONS.items()},
        })
        progress.progress(i / len(candidates), text=f"Scored {i}/{len(candidates)}")
    progress.empty()

    table = pd.DataFrame(rows).sort_values("Alignment (1-10)", ascending=False)
    st.session_state["table"] = table
    st.session_state["details"] = details
    st.session_state["engine"] = scorer.engine


if "table" in st.session_state:
    table = st.session_state["table"]
    st.subheader("Ranked candidates")
    st.caption(f"Engine: `{st.session_state['engine']}`")

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Alignment (1-10)": st.column_config.ProgressColumn(
                "Alignment (1-10)", min_value=0, max_value=10, format="%.1f"),
        },
    )

    diverged = table[table["Divergence"] >= 2.0]
    if not diverged.empty:
        st.warning(
            "Model and heuristic control disagree by 2+ points for: "
            + ", ".join(diverged["Candidate"])
            + ". Read these resumes manually before acting on the score."
        )

    st.subheader("Dimension breakdown")
    labels = [label for _, (label, _) in DIMENSIONS.items()]
    st.bar_chart(table.set_index("Candidate")[labels], height=320)

    st.subheader("Rationale")
    for name in table["Candidate"]:
        result, candidate = st.session_state["details"][name]
        with st.expander(f"{name} — {result.overall}/10"):
            st.write(result.rationale)
            if result.flags:
                st.markdown("**Evidence gaps**")
                for flag in result.flags:
                    st.markdown(f"- {flag}")
            st.markdown("**Parsed record**")
            st.json(candidate.to_dict() | {"raw_text": candidate.raw_text[:400] + "..."})

    st.download_button(
        "Download scores as CSV",
        table.to_csv(index=False).encode(),
        file_name="alignment_scores.csv",
        mime="text/csv",
    )

with st.expander("How this should and should not be used"):
    st.markdown(
        """
- **Ranking, not rejecting.** The score orders a shortlist for human review. It is not
  wired to auto-reject anyone and should not be.
- **Evidence only.** The prompt instructs the model to score what is written in the
  resume and to mark a dimension as neutral when evidence is absent.
- **Protected attributes excluded.** Name, gender, nationality, age and institution
  prestige are explicitly out of scope in the prompt. Parsing does not extract them.
- **A control is always run.** Every LLM score is shown beside a deterministic
  heuristic score. Large divergence is surfaced as a warning rather than hidden.
- **Known limits.** Resume text rewards people who write good resumes. That is a real
  bias and this tool does not remove it.
        """
    )
