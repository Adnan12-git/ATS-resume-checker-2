"""ATS Resume Checker - Streamlit + Groq.

Upload a resume (PDF, DOCX or TXT), optionally paste a job description, and get:
  * an ATS compatibility score (0-100) with a category breakdown
  * strengths, problems, missing keywords
  * concrete improvements and rewritten bullet examples
"""

import io
import json
import os
import re

import streamlit as st
from docx import Document
from groq import Groq
from pypdf import PdfReader

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
# Models are tried in this order. Some Groq models (e.g. the Llama ones) are
# "Enterprise" only, so a normal key gets a 404 for them - the app then falls
# back to the next model automatically. Check https://console.groq.com/docs/models
PREFERRED_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]
NON_CHAT_HINTS = ("whisper", "guard", "orpheus", "tts", "safeguard", "embed")
APP_VERSION = "2.1 (model fallback)"
MAX_OUTPUT_TOKENS = 6000  # reasoning models spend part of this on "thinking"
MAX_RESUME_CHARS = 12000  # keeps the prompt well inside the model context
MAX_JD_CHARS = 6000
MIN_TEXT_CHARS = 100  # below this the file is probably a scanned image

CATEGORY_KEYS = [
    ("formatting", "Formatting & Structure"),
    ("keywords", "Keywords & Skills"),
    ("experience", "Experience & Achievements"),
    ("education", "Education & Certifications"),
    ("readability", "Readability & Clarity"),
]

SYSTEM_PROMPT = """You are an expert ATS (Applicant Tracking System) analyst and \
professional resume reviewer. You evaluate how well a resume would be parsed and \
ranked by an ATS and how strong it is for a recruiter.

Rules:
- Base every statement ONLY on the resume text provided. Never invent experience.
- Be honest and specific. Do not inflate scores.
- Scores are integers from 0 to 100.
- Respond with a single valid JSON object and nothing else."""

USER_PROMPT_TEMPLATE = """Analyse the resume below{jd_clause}.

Return JSON with EXACTLY this structure:
{{
  "overall_score": <int 0-100>,
  "summary": "<2-3 sentence overall assessment>",
  "category_scores": {{
    "formatting": <int 0-100>,
    "keywords": <int 0-100>,
    "experience": <int 0-100>,
    "education": <int 0-100>,
    "readability": <int 0-100>
  }},
  "strengths": ["<short point>", "..."],
  "issues": ["<specific problem found in this resume>", "..."],
  "missing_keywords": ["<keyword or skill>", "..."],
  "improvements": [
    {{"priority": "High|Medium|Low", "suggestion": "<actionable fix>"}}
  ],
  "bullet_rewrites": [
    {{"original": "<line copied from the resume>", "improved": "<stronger version with action verb and measurable result placeholder>"}}
  ]
}}

Guidelines:
- 3-6 strengths, 3-8 issues, up to 12 missing keywords, 5-8 improvements, 2-4 bullet rewrites.
- {keyword_rule}
- In "improved" bullets, use placeholders like [X%] or [number] where a metric is needed; do not make up facts.

{jd_block}RESUME TEXT:
\"\"\"
{resume_text}
\"\"\""""


# ----------------------------------------------------------------------------
# Text extraction
# ----------------------------------------------------------------------------
def extract_text(file_name: str, data: bytes) -> str:
    """Extract plain text from a PDF, DOCX or TXT upload."""
    name = file_name.lower()

    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                unlocked = bool(reader.decrypt(""))
            except Exception:
                unlocked = False
            if not unlocked:
                raise ValueError("This PDF is password protected. Please upload an unlocked copy.")
        pages = [(page.extract_text() or "") for page in reader.pages]
        return "\n".join(pages).strip()

    if name.endswith(".docx"):
        doc = Document(io.BytesIO(data))
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))
        return "\n".join(parts).strip()

    if name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore").strip()

    raise ValueError("Unsupported file type. Please upload a PDF, DOCX or TXT file.")


# ----------------------------------------------------------------------------
# Quick rule-based checks (no AI needed)
# ----------------------------------------------------------------------------
def basic_checks(text: str) -> list:
    """Return a list of (passed, message) tuples for simple ATS checks."""
    lowered = text.lower()
    words = len(text.split())
    checks = [
        (
            bool(re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)),
            "Email address found",
        ),
        (
            bool(re.search(r"(\+?\d[\d\s().-]{8,}\d)", text)),
            "Phone number found",
        ),
        (
            any(k in lowered for k in ("experience", "employment", "work history")),
            "Work experience section detected",
        ),
        (
            any(k in lowered for k in ("education", "university", "college", "bachelor", "master")),
            "Education section detected",
        ),
        (
            "skills" in lowered,
            "Skills section detected",
        ),
        (
            250 <= words <= 1200,
            f"Length looks reasonable ({words} words; ideal is roughly 250-1200)",
        ),
    ]
    return checks


# ----------------------------------------------------------------------------
# Groq call + parsing
# ----------------------------------------------------------------------------
def get_api_key() -> str:
    """Read the key from Streamlit secrets, then environment variables."""
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        key = ""
    return key or os.environ.get("GROQ_API_KEY", "")


def build_prompt(resume_text: str, job_description: str) -> str:
    resume_text = resume_text[:MAX_RESUME_CHARS]
    if job_description.strip():
        jd_clause = " and the target job description"
        keyword_rule = (
            "For missing_keywords, compare the resume against the job description "
            "and list important terms from the job description that the resume lacks."
        )
        jd_block = (
            'JOB DESCRIPTION:\n"""\n' + job_description.strip()[:MAX_JD_CHARS] + '\n"""\n\n'
        )
    else:
        jd_clause = ""
        keyword_rule = (
            "For missing_keywords, list common industry keywords and skills that "
            "are typical for the candidate's apparent field but absent from the resume."
        )
        jd_block = ""
    return USER_PROMPT_TEMPLATE.format(
        jd_clause=jd_clause,
        keyword_rule=keyword_rule,
        jd_block=jd_block,
        resume_text=resume_text,
    )


def parse_json_response(raw: str) -> dict:
    """Parse model output into a dict, tolerating code fences or extra text."""
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def _score(value) -> int:
    try:
        return max(0, min(100, int(round(float(value)))))
    except (TypeError, ValueError):
        return 0


def _str_list(value) -> list:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()]


def normalise_result(data: dict) -> dict:
    """Make sure every field exists and has the right type."""
    if not isinstance(data, dict):
        raise ValueError("Model did not return a JSON object.")

    cats = data.get("category_scores")
    cats = cats if isinstance(cats, dict) else {}

    improvements = []
    for item in data.get("improvements") or []:
        if isinstance(item, dict) and item.get("suggestion"):
            improvements.append(
                {
                    "priority": str(item.get("priority", "Medium")).title(),
                    "suggestion": str(item["suggestion"]).strip(),
                }
            )
        elif isinstance(item, str) and item.strip():
            improvements.append({"priority": "Medium", "suggestion": item.strip()})

    rewrites = []
    for item in data.get("bullet_rewrites") or []:
        if isinstance(item, dict) and item.get("original") and item.get("improved"):
            rewrites.append(
                {"original": str(item["original"]).strip(), "improved": str(item["improved"]).strip()}
            )

    return {
        "overall_score": _score(data.get("overall_score")),
        "summary": str(data.get("summary", "")).strip(),
        "category_scores": {key: _score(cats.get(key)) for key, _ in CATEGORY_KEYS},
        "strengths": _str_list(data.get("strengths")),
        "issues": _str_list(data.get("issues")),
        "missing_keywords": _str_list(data.get("missing_keywords")),
        "improvements": improvements,
        "bullet_rewrites": rewrites,
    }


def is_model_unavailable(exc: Exception) -> bool:
    """True when Groq says the model doesn't exist or this key can't use it."""
    text = str(exc).lower()
    return (
        "model_not_found" in text
        or "does not exist" in text
        or "decommissioned" in text
        or getattr(exc, "status_code", None) == 404
    )


def _call_model(client, model: str, messages: list) -> str:
    """One chat completion. Retries with simpler settings if a parameter is rejected."""
    base = dict(model=model, messages=messages, temperature=0.2, max_tokens=MAX_OUTPUT_TOKENS)
    attempts = []
    full = dict(base, response_format={"type": "json_object"})
    if "gpt-oss" in model:
        full["extra_body"] = {"reasoning_effort": "low"}  # faster, leaves room for the answer
    attempts.append(full)
    attempts.append(dict(base, response_format={"type": "json_object"}))
    attempts.append(base)

    last_exc = None
    for kwargs in attempts:
        try:
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            if is_model_unavailable(exc):
                raise  # no point retrying other settings for this model
            last_exc = exc
    raise last_exc


def candidate_models(selected: str) -> list:
    """Selected model first, then the preferred list as fallbacks."""
    order = [selected] + [m for m in PREFERRED_MODELS if m != selected]
    return [m for m in order if m]


def analyse_resume(api_key: str, model: str, resume_text: str, job_description: str, extra_models=None):
    """Return (result_dict, model_used). Falls back to other models on 404.

    extra_models: other model ids (e.g. the ones your key can list) to try last.
    """
    client = Groq(api_key=api_key)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(resume_text, job_description)},
    ]

    last_exc = None
    candidates = candidate_models(model)
    for extra in extra_models or []:
        if extra not in candidates:
            candidates.append(extra)
    candidates = candidates[:8]  # don't loop forever

    for candidate in candidates:
        try:
            raw = _call_model(client, candidate, messages)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if is_model_unavailable(exc):
                continue  # try the next model
            raise
        if not raw.strip():
            raise ValueError(
                "The model returned an empty answer. Try again or pick another model."
            )
        return normalise_result(parse_json_response(raw)), candidate

    raise RuntimeError(
        f"None of the {len(candidates)} models I tried are available for your Groq API key "
        f"({', '.join(candidates)}). Open console.groq.com/docs/models and check which "
        f"models your account can use. Last error: {last_exc}"
    )


@st.cache_data(ttl=600, show_spinner=False)
def list_chat_models(api_key: str) -> list:
    """Ask Groq which models exist, keep chat models, put preferred ones first."""
    try:
        ids = [m.id for m in Groq(api_key=api_key).models.list().data]
    except Exception:  # noqa: BLE001
        return []
    ids = [i for i in ids if not any(h in i.lower() for h in NON_CHAT_HINTS)]
    first = [m for m in PREFERRED_MODELS if m in ids]
    rest = sorted(i for i in ids if i not in first)
    return first + rest


# ----------------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------------
def score_label(score: int) -> str:
    if score >= 80:
        return "Excellent"
    if score >= 65:
        return "Good"
    if score >= 50:
        return "Needs work"
    return "Poor"


def render_results(result: dict, resume_text: str) -> None:
    overall = result["overall_score"]

    st.subheader("Your ATS score")
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric("Overall", f"{overall}/100", score_label(overall), delta_color="off")
    with col2:
        st.progress(overall / 100)
        if result["summary"]:
            st.write(result["summary"])

    st.subheader("Score breakdown")
    cols = st.columns(len(CATEGORY_KEYS))
    for col, (key, label) in zip(cols, CATEGORY_KEYS):
        with col:
            value = result["category_scores"][key]
            st.metric(label, f"{value}/100")
            st.progress(value / 100)

    left, right = st.columns(2)
    with left:
        st.subheader("Strengths")
        for item in result["strengths"] or ["No strengths reported."]:
            st.markdown(f"- {item}")
    with right:
        st.subheader("Issues found")
        for item in result["issues"] or ["No issues reported."]:
            st.markdown(f"- {item}")

    st.subheader("Missing keywords")
    if result["missing_keywords"]:
        st.write(", ".join(f"`{k}`" for k in result["missing_keywords"]))
    else:
        st.write("No missing keywords reported.")

    st.subheader("How to improve your resume")
    order = {"High": 0, "Medium": 1, "Low": 2}
    for item in sorted(result["improvements"], key=lambda i: order.get(i["priority"], 1)):
        st.markdown(f"**[{item['priority']}]** {item['suggestion']}")
    if not result["improvements"]:
        st.write("No improvements reported.")

    if result["bullet_rewrites"]:
        st.subheader("Example bullet rewrites")
        for item in result["bullet_rewrites"]:
            st.markdown(f"**Before:** {item['original']}")
            st.markdown(f"**After:** {item['improved']}")
            st.divider()

    st.subheader("Quick checks")
    for passed, message in basic_checks(resume_text):
        st.markdown(f"{'✅' if passed else '❌'} {message}")

    report = json.dumps(result, indent=2)
    st.download_button(
        "Download report (JSON)",
        data=report,
        file_name="ats_report.json",
        mime="application/json",
    )


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="ATS Resume Checker", page_icon="📄", layout="wide")
    st.title("📄 ATS Resume Checker")
    st.caption("Upload your resume to get an ATS score and tips to improve it. Powered by Groq.")

    with st.sidebar:
        st.header("Settings")
        api_key = get_api_key()
        if not api_key:
            api_key = st.text_input("Groq API key", type="password", help="Get one free at console.groq.com/keys")
        else:
            st.success("Groq API key loaded.")

        listed = list_chat_models(api_key) if api_key else []
        choices = listed or PREFERRED_MODELS.copy()
        if api_key and not listed:
            st.warning("Could not fetch the model list from Groq. Check that your API key is valid.")
        env_model = os.environ.get("GROQ_MODEL", "")
        if env_model:
            choices = [env_model] + [c for c in choices if c != env_model]
        model = st.selectbox(
            "Model",
            choices,
            index=0,
            help="If a model isn't available for your key, the app tries the next one automatically.",
        )
        if listed:
            st.caption(f"{len(listed)} chat models visible to your key.")
        st.markdown("---")
        st.caption(f"App version {APP_VERSION}")
        st.caption("Your resume is sent to Groq for analysis and is not stored by this app.")

    uploaded = st.file_uploader("Upload your resume", type=["pdf", "docx", "txt"])
    job_description = st.text_area(
        "Job description (optional)",
        height=160,
        placeholder="Paste the job description here for a targeted keyword match...",
    )

    if st.button("Analyse resume", type="primary", disabled=uploaded is None):
        if not api_key:
            st.error("Please add your Groq API key in the sidebar (or in Streamlit secrets).")
            return

        try:
            resume_text = extract_text(uploaded.name, uploaded.getvalue())
        except Exception as exc:
            st.error(f"Could not read the file: {exc}")
            return

        if len(resume_text) < MIN_TEXT_CHARS:
            st.error(
                "Very little text could be extracted. If your resume is a scanned image or "
                "uses text inside images, an ATS cannot read it either - export a text-based "
                "PDF or DOCX and try again."
            )
            return

        with st.spinner("Analysing your resume..."):
            try:
                result, used_model = analyse_resume(
                    api_key, model, resume_text, job_description, extra_models=listed
                )
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
                return

        if used_model != model:
            st.info(f"`{model}` is not available for your key, so `{used_model}` was used instead.")
        render_results(result, resume_text)


if __name__ == "__main__":
    main()
