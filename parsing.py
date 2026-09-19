"""
Resume parsing.

Resumes arrive in inconsistent shapes: plain text exports, PDF text layers, and
scraped HTML where every source uses a different DOM structure. This module
normalises all of them into one `Candidate` record so the scoring layer only
ever sees a single, predictable schema.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Iterable

# ---------------------------------------------------------------------------
# Canonical record
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    name: str
    headline: str = ""
    years_experience: float = 0.0
    skills: list[str] = field(default_factory=list)
    titles: list[str] = field(default_factory=list)
    education: str = ""
    raw_text: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Skill vocabulary
# ---------------------------------------------------------------------------

SKILL_VOCAB = {
    "sql": ["sql", "sqlite", "postgres", "postgresql", "mysql", "bigquery", "snowflake", "t-sql"],
    "python": ["python", "pandas", "numpy", "scikit-learn", "streamlit"],
    "analytics": ["analytics", "a/b testing", "experimentation", "cohort", "funnel", "retention"],
    "visualisation": ["tableau", "power bi", "looker", "matplotlib", "dashboard"],
    "data_engineering": ["airflow", "dbt", "etl", "elt", "pipeline", "spark"],
    "product": ["product management", "roadmap", "prd", "user research", "kpi"],
    "ml": ["machine learning", "regression", "classification", "model", "llm", "nlp"],
    "cloud": ["aws", "gcp", "azure", "databricks"],
    "stakeholder": ["stakeholder", "executive", "cross-functional", "presented"],
}

TITLE_PATTERN = re.compile(
    r"\b(data|business|product|marketing|financial|research)?\s*"
    r"(analyst|scientist|engineer|manager|associate|consultant|intern)\b",
    re.I,
)
YEARS_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)", re.I)
DATE_RANGE = re.compile(r"(19|20)\d{2}\s*[-–—to]+\s*((19|20)\d{2}|present|current)", re.I)


def _extract_skills(text: str) -> list[str]:
    lowered = text.lower()
    found = []
    for canonical, variants in SKILL_VOCAB.items():
        if any(v in lowered for v in variants):
            found.append(canonical)
    return found


def _extract_years(text: str) -> float:
    explicit = YEARS_PATTERN.findall(text)
    if explicit:
        return max(float(y) for y in explicit)

    # fall back to summing date ranges found in the experience section
    spans = []
    for match in DATE_RANGE.finditer(text):
        raw = match.group(0)
        years = re.findall(r"(19|20)\d{2}", raw)
        start = int(re.findall(r"(?:19|20)\d{2}", raw)[0])
        end = 2026 if re.search(r"present|current", raw, re.I) else int(
            re.findall(r"(?:19|20)\d{2}", raw)[-1]
        )
        if end >= start:
            spans.append(end - start)
    return float(sum(spans)) if spans else 0.0


def _extract_titles(text: str) -> list[str]:
    titles = {m.group(0).strip().title() for m in TITLE_PATTERN.finditer(text)}
    return sorted(t for t in titles if len(t) > 4)


# ---------------------------------------------------------------------------
# Source adapters - one per DOM / file shape, all returning Candidate
# ---------------------------------------------------------------------------


def from_plain_text(name: str, text: str) -> Candidate:
    """Plain-text or PDF-text-layer resume."""
    first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    return Candidate(
        name=name,
        headline=first_line[:120],
        years_experience=_extract_years(text),
        skills=_extract_skills(text),
        titles=_extract_titles(text),
        education=_extract_education(text),
        raw_text=text,
    )


def from_json_profile(payload: dict | str) -> Candidate:
    """Structured profile export - keys vary by source, so we probe aliases."""
    data = json.loads(payload) if isinstance(payload, str) else payload

    def pick(*keys, default=""):
        for key in keys:
            if data.get(key):
                return data[key]
        return default

    text = pick("summary", "about", "bio", default="")
    skills = pick("skills", "skill_list", default=[])
    if isinstance(skills, str):
        skills = [s.strip() for s in skills.split(",") if s.strip()]

    return Candidate(
        name=pick("name", "full_name", "candidate_name", default="Unknown"),
        headline=pick("headline", "title", "current_role", default=""),
        years_experience=float(pick("years_experience", "yoe", default=0) or 0),
        skills=_extract_skills(" ".join(skills) + " " + text) or [s.lower() for s in skills],
        titles=pick("titles", "roles", default=[]),
        education=pick("education", "degree", default=""),
        raw_text=text,
    )


def from_html(name: str, html: str) -> Candidate:
    """Scraped profile page. Strips tags rather than assuming any one DOM shape,
    which is what makes this survive across different source sites."""
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", "\n", text)
    text = re.sub(r"&nbsp;?", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return from_plain_text(name, text.strip())


def _extract_education(text: str) -> str:
    for line in text.splitlines():
        if re.search(r"\b(b\.?tech|b\.?sc|b\.?a\b|m\.?sc|m\.?tech|mba|bachelor|master|phd)\b",
                     line, re.I):
            return line.strip()[:160]
    return ""


def parse_many(items: Iterable[tuple[str, str]]) -> list[Candidate]:
    """Convenience: parse a batch of (name, text) pairs."""
    return [from_plain_text(name, text) for name, text in items]
