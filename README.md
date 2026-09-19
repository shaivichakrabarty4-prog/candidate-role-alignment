# Candidate-Role Alignment Model

**Replacing "this one feels like a fit" with a score you can argue with.**

An end-to-end application that parses resumes from inconsistent sources, scores candidate-role
alignment on five dimensions (1–10) using **LLaMA 3 70B**, and shows every score next to a
deterministic control so a human can see when the model is drifting.

![python](https://img.shields.io/badge/python-3.9%2B-blue)
![streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B)
![llama3](https://img.shields.io/badge/LLaMA%203-70B-6b46c1)
![license](https://img.shields.io/badge/license-MIT-green)

### ▶️ [Try the live app](https://YOUR-APP-NAME.streamlit.app)

*The demo runs with no API key required — it falls back to the deterministic scorer, so anyone can
open it and get results.*

---

## What it does

| | |
|---|---|
| **Input** | Resumes as plain text, JSON profile exports, or scraped HTML — three different DOM shapes, one normalised record |
| **Scoring** | Five weighted dimensions → a single 1–10 alignment score, with a written rationale and explicit evidence gaps |
| **Control** | Every LLM score is reported beside a deterministic heuristic score; divergence of 2+ points raises a flag |
| **Output** | Ranked shortlist, per-dimension breakdown, downloadable CSV |

Sample run against 5 candidates: **[results/sample_scores.md](results/sample_scores.md)**

## Scoring dimensions

| Dimension | Weight | What it measures |
|---|---|---|
| Skills match | 30% | Must-have and nice-to-have skill coverage |
| Experience depth | 25% | Years against the role's bar, with diminishing returns above it |
| Domain relevance | 20% | Evidence of working in the role's actual problem space |
| Seniority fit | 15% | Distance between implied and required seniority, penalised in both directions |
| Career trajectory | 10% | Progression signal across titles and education |

Over-qualification is penalised as well as under-qualification. A senior candidate scored against a
junior opening is a fit problem, not a bonus.

## Architecture

```
resume (.txt / .json / .html)
        │
        ▼
  src/parsing.py         adapters per source shape → one Candidate dataclass
        │
        ▼
  src/scoring.py         ┌── LLMScorer       LLaMA 3 70B via Groq, strict-JSON prompt
                         └── HeuristicScorer deterministic, offline, always runs as control
        │
        ▼
  app.py (Streamlit)     ranked table · dimension chart · rationale · CSV export
  src/batch_score.py     same pipeline, headless, for CI
```

### Why there are two engines

The heuristic scorer is not a fallback afterthought. It is the **control**. LLM scores are fluent
and confident whether or not they are grounded, so a plausible-sounding rationale is not evidence
that the model read the resume properly. Running a deterministic scorer alongside and surfacing the
gap turns silent drift into a visible warning.

### Prompt design

The system prompt constrains the model in three ways that matter:

```
- Judge ONLY evidence present in the resume text. Never infer or invent experience.
- Ignore name, gender, nationality, age, photographs, school prestige and any
  other protected or proxy attribute. Score the work, not the person.
- If evidence for a dimension is absent, score it 5 and say so in the rationale.
```

Output is forced to strict JSON, with a fence-stripping and brace-extraction fallback in
`_parse_json` because models ignore formatting instructions often enough to need one.

## Responsible use

This is **decision support, not decision making**. Concretely:

- It ranks a shortlist for human review. Nothing in the codebase auto-rejects a candidate, and
  nothing should be built on top of it that does.
- Protected and proxy attributes are excluded in the prompt and never extracted during parsing.
- Absent evidence scores neutral (5) rather than zero, so a thin resume is not treated as a
  disqualifying one.
- Every score carries a written rationale and a list of evidence gaps, so a reviewer can check the
  reasoning rather than inherit the number.
- **A real bias remains**: resume text rewards people who are good at writing resumes. This tool
  does not fix that, and pretending otherwise would be the more dangerous claim.

## Run it locally

```bash
git clone https://github.com/YOUR-USERNAME/candidate-role-alignment.git
cd candidate-role-alignment
pip install -r requirements.txt

streamlit run app.py                # works immediately, heuristic mode
```

To enable LLaMA 3 70B:

```bash
cp .env.example .env                # add a free key from https://console.groq.com
export GROQ_API_KEY=your_key_here
streamlit run app.py
```

Headless batch scoring:

```bash
python src/batch_score.py           # writes results/sample_scores.{csv,md}
```

## Deploying your own copy

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → pick the repo, branch
   `main`, main file `app.py`.
3. Under **Advanced settings → Secrets**, paste:
   ```toml
   GROQ_API_KEY = "your_key_here"
   ```
4. Deploy. You get a permanent `https://<name>.streamlit.app` URL.

## Repo structure

```
candidate-role-alignment/
├── app.py                  # Streamlit UI
├── src/
│   ├── parsing.py          # source adapters → Candidate dataclass
│   ├── scoring.py          # LLM + heuristic engines, prompt, JSON guards
│   └── batch_score.py      # headless CLI runner
├── sample_data/            # 5 sample resumes + role spec
├── results/                # committed sample output
└── .streamlit/config.toml
```

## Known limitations

- **Keyword-based skill detection over-credits breadth.** In the sample run, a backend engineer who
  mentions SQL in passing scores 8.5 — close to a dedicated analyst. Detecting that a skill is
  *named* is not detecting that it was *used in the relevant context*, and that is the single
  biggest weakness in the current scorer.
- Years of experience is regex-derived and will miss non-standard date formats.
- Scores are not calibrated against hiring outcomes, because no outcome data exists here. Without
  that, weights are reasoned choices rather than fitted ones.
- Sample resumes are fabricated for demonstration.

## License

MIT
