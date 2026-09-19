"""
Candidate-role alignment scoring.

Produces a 1-10 alignment score plus per-dimension sub-scores and a written
rationale. Two engines:

  - `LLMScorer`       : LLaMA 3 70B via the Groq API, prompted to return strict JSON
  - `HeuristicScorer` : deterministic, no network, no API key

The heuristic scorer is not a toy. It is the control: every LLM score is
reported next to it, so a reviewer can see when the model is drifting away from
the evidence in the resume. If the two disagree sharply, that is a signal to
read the resume rather than trust the number.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict

try:                                  # optional - only needed for LLM mode
    from groq import Groq
except ImportError:                   # pragma: no cover
    Groq = None

from parsing import Candidate

MODEL = "llama3-70b-8192"

DIMENSIONS = {
    "skills_match":      ("Skills match",      0.30),
    "experience_depth":  ("Experience depth",  0.25),
    "domain_relevance":  ("Domain relevance",  0.20),
    "seniority_fit":     ("Seniority fit",     0.15),
    "trajectory":        ("Career trajectory", 0.10),
}


@dataclass
class RoleSpec:
    title: str
    seniority: str                 # Intern / Junior / Mid / Senior
    must_have_skills: list[str]
    nice_to_have_skills: list[str]
    min_years: float
    domain: str
    description: str = ""


@dataclass
class Score:
    candidate: str
    overall: float
    dimensions: dict
    rationale: str
    engine: str
    flags: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Heuristic engine
# ---------------------------------------------------------------------------


class HeuristicScorer:
    engine = "heuristic"

    def score(self, candidate: Candidate, role: RoleSpec) -> Score:
        have = {s.lower() for s in candidate.skills}
        must = {s.lower() for s in role.must_have_skills}
        nice = {s.lower() for s in role.nice_to_have_skills}

        must_hit = len(have & must) / len(must) if must else 1.0
        nice_hit = len(have & nice) / len(nice) if nice else 0.0
        skills_match = 10 * (0.75 * must_hit + 0.25 * nice_hit)

        if role.min_years <= 0:
            experience_depth = 7.0
        else:
            ratio = candidate.years_experience / role.min_years
            experience_depth = max(1.0, min(10.0, 3 + 6 * min(ratio, 1.6) / 1.6))

        domain_hits = sum(
            1 for token in re.split(r"\W+", role.domain.lower())
            if len(token) > 3 and token in candidate.raw_text.lower()
        )
        domain_relevance = max(1.0, min(10.0, 3 + 2.0 * domain_hits))

        ladder = {"intern": 0, "junior": 1, "mid": 2, "senior": 3}
        wanted = ladder.get(role.seniority.lower(), 2)
        implied = 0 if candidate.years_experience < 1 else (
            1 if candidate.years_experience < 3 else (
                2 if candidate.years_experience < 6 else 3))
        seniority_fit = max(1.0, 10.0 - 3.0 * abs(wanted - implied))

        trajectory = 6.0 + min(2.0, len(candidate.titles) * 0.7) + (
            1.5 if candidate.education else 0.0)
        trajectory = min(10.0, trajectory)

        dims = {
            "skills_match": round(skills_match, 1),
            "experience_depth": round(experience_depth, 1),
            "domain_relevance": round(domain_relevance, 1),
            "seniority_fit": round(seniority_fit, 1),
            "trajectory": round(trajectory, 1),
        }
        overall = round(sum(dims[k] * w for k, (_, w) in DIMENSIONS.items()), 1)

        missing = sorted(must - have)
        flags = []
        if missing:
            flags.append(f"Missing must-have: {', '.join(missing)}")
        if candidate.years_experience == 0:
            flags.append("No years of experience detected - parser may have missed it")

        rationale = (
            f"Matches {len(have & must)}/{len(must) or 1} must-have skills. "
            f"{candidate.years_experience:.0f} years against a {role.min_years:.0f}-year bar. "
            f"Seniority signal reads as {['intern', 'junior', 'mid', 'senior'][implied]} "
            f"against a {role.seniority.lower()} opening."
        )
        return Score(candidate.name, overall, dims, rationale, self.engine, flags)


# ---------------------------------------------------------------------------
# LLM engine
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an evaluation component inside a hiring-support tool.

Score how well a candidate aligns with a role on five dimensions, each 1-10:
  skills_match, experience_depth, domain_relevance, seniority_fit, trajectory

Rules you must follow:
- Judge ONLY evidence present in the resume text. Never infer or invent experience.
- Ignore name, gender, nationality, age, photographs, school prestige and any
  other protected or proxy attribute. Score the work, not the person.
- If evidence for a dimension is absent, score it 5 and say so in the rationale.
- Return STRICT JSON only. No markdown, no code fences, no commentary.

JSON shape:
{"skills_match": int, "experience_depth": int, "domain_relevance": int,
 "seniority_fit": int, "trajectory": int, "rationale": "2-3 sentences",
 "evidence_gaps": ["..."]}"""

USER_TEMPLATE = """ROLE
Title: {title}
Seniority: {seniority}
Minimum years: {min_years}
Domain: {domain}
Must-have skills: {must}
Nice-to-have skills: {nice}
Description: {description}

CANDIDATE
Name: {name}
Headline: {headline}
Detected years of experience: {years}
Detected skills: {skills}
Previous titles: {titles}
Education: {education}

RESUME TEXT
\"\"\"
{raw}
\"\"\""""


class LLMScorer:
    engine = f"llama-3-70b ({MODEL})"

    def __init__(self, api_key: str | None = None, temperature: float = 0.1):
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY not set - use HeuristicScorer instead")
        if Groq is None:
            raise RuntimeError("groq package not installed - pip install groq")
        self.client = Groq(api_key=key)
        self.temperature = temperature

    def score(self, candidate: Candidate, role: RoleSpec) -> Score:
        prompt = USER_TEMPLATE.format(
            title=role.title, seniority=role.seniority, min_years=role.min_years,
            domain=role.domain, must=", ".join(role.must_have_skills),
            nice=", ".join(role.nice_to_have_skills), description=role.description,
            name=candidate.name, headline=candidate.headline,
            years=candidate.years_experience, skills=", ".join(candidate.skills),
            titles=", ".join(candidate.titles), education=candidate.education,
            raw=candidate.raw_text[:6000],
        )
        response = self.client.chat.completions.create(
            model=MODEL,
            temperature=self.temperature,
            max_tokens=700,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        payload = _parse_json(response.choices[0].message.content)

        dims = {k: float(payload.get(k, 5)) for k in DIMENSIONS}
        overall = round(sum(dims[k] * w for k, (_, w) in DIMENSIONS.items()), 1)
        return Score(
            candidate=candidate.name,
            overall=overall,
            dimensions={k: round(v, 1) for k, v in dims.items()},
            rationale=payload.get("rationale", ""),
            engine=self.engine,
            flags=list(payload.get("evidence_gaps", [])),
        )


def _parse_json(text: str) -> dict:
    """LLMs sometimes wrap JSON in fences or prose despite instructions."""
    cleaned = re.sub(r"```(?:json)?|```", "", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.S)
        if match:
            return json.loads(match.group(0))
        raise ValueError(f"Model did not return JSON: {text[:200]}")


def get_scorer(prefer_llm: bool = True):
    """Return the LLM scorer when a key is available, else the heuristic one."""
    if prefer_llm and os.getenv("GROQ_API_KEY") and Groq is not None:
        try:
            return LLMScorer()
        except RuntimeError:
            pass
    return HeuristicScorer()
