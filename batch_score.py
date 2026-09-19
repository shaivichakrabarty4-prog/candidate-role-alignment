"""
Score every sample resume from the command line and publish the results.

Uses LLaMA 3 70B when GROQ_API_KEY is set, otherwise the deterministic
heuristic scorer, so this runs in CI with no secrets.

Run:  python src/batch_score.py
Out:  results/sample_scores.csv, results/sample_scores.md
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from parsing import from_plain_text  # noqa: E402
from scoring import DIMENSIONS, HeuristicScorer, RoleSpec, get_scorer  # noqa: E402

SAMPLE_DIR = "sample_data"
RESULTS = "results"


def main():
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(SAMPLE_DIR, "role_spec.json")) as fh:
        role = RoleSpec(**json.load(fh))

    scorer = get_scorer()
    control = HeuristicScorer()
    rows, rationales = [], []

    for filename in sorted(f for f in os.listdir(SAMPLE_DIR) if f.endswith(".txt")):
        with open(os.path.join(SAMPLE_DIR, filename)) as fh:
            candidate = from_plain_text(filename.replace(".txt", ""), fh.read())
        result = scorer.score(candidate, role)
        baseline = control.score(candidate, role)
        rows.append({
            "candidate": candidate.name,
            "alignment": result.overall,
            "heuristic_control": baseline.overall,
            "divergence": round(abs(result.overall - baseline.overall), 1),
            "years": candidate.years_experience,
            "skills": ", ".join(candidate.skills),
            **{key: result.dimensions[key] for key in DIMENSIONS},
        })
        rationales.append((candidate.name, result.overall, result.rationale, result.flags))

    df = pd.DataFrame(rows).sort_values("alignment", ascending=False)
    df.to_csv(os.path.join(RESULTS, "sample_scores.csv"), index=False)

    lines = [
        "# Sample scoring run",
        "",
        f"Role: **{role.title}** ({role.seniority}, {role.min_years}+ years, {role.domain})",
        f"Engine: `{scorer.engine}`",
        "",
        df.to_markdown(index=False),
        "",
        "## Rationale",
        "",
    ]
    for name, overall, rationale, flags in sorted(rationales, key=lambda r: -r[1]):
        lines.append(f"**{name} — {overall}/10**  ")
        lines.append(rationale)
        for flag in flags:
            lines.append(f"- ⚠️ {flag}")
        lines.append("")

    with open(os.path.join(RESULTS, "sample_scores.md"), "w") as fh:
        fh.write("\n".join(lines))

    print(df.to_string(index=False))
    print(f"\nWrote {RESULTS}/sample_scores.csv and {RESULTS}/sample_scores.md")


if __name__ == "__main__":
    main()
