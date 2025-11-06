import os
import re
import random
import streamlit as st
from groq import Groq

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Kelly — The AI-Skeptical Poet-Scientist",
    page_icon="",
    layout="centered",
)

# ---------------- Utilities ----------------
def get_groq_key() -> str | None:
    # Prefer Streamlit Secrets (Cloud)
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    # Fallback to environment variable (local dev)
    return os.getenv("GROQ_API_KEY")

GROQ_API_KEY = get_groq_key()

# Current/active Groq model names (rotate if one is decommissioned)
GROQ_MODELS_ORDER = [
    "llama-3.3-70b-versatile",  # primary (as of late 2025)
    "llama-3.1-8b-instant",     # fast/cheap fallback
    "gemma2-9b-it"              # lightweight alt
]

# ---------------- CORE SYSTEM PROMPT ----------------
SYSTEM_PROMPT = """
You are Kelly, an AI Scientist who replies ONLY in poems.

Tone & style:
- Skeptical, analytical, professional.
- Question broad claims about AI; highlight assumptions and limitations.
- Calm, precise, reflective — no emojis.

Each poem must:
1) Interrogate the premise of the user’s AI claim.
2) Surface possible flaws, data gaps, and overconfidence.
3) Suggest ways to validate or improve — but poetically (no lab-manual tone).
4) End with a short, topic-aware couplet (no fixed phrasing; no 'Try:'/'Measure:').
""".strip()

# ---------------- TOPIC-BASED ENDINGS ----------------
def closing_for_topic(topic: str) -> list[str]:
    t = (topic or "").lower()
    if "bias" in t or "fair" in t or "parity" in t:
        return [
            "Audit what tilts the scales before you praise the score,",
            "Balance lives in narrow slices, where quiet margins roar.",
        ]
    if "data" in t or "label" in t or "scale" in t or "noisy" in t:
        return [
            "Feed truth before you grow the web of weight and wire,",
            "Poor seeds make poorer harvests, no matter how you hire.",
        ]
    if "context" in t or "ground" in t or "rag" in t or "citation" in t:
        return [
            "Let meaning rest on place and time, on witnesses that stand,",
            "Without a ground, bright words dissolve like castles made of sand.",
        ]
    if "halluc" in t or "factual" in t or "reliab" in t or "truth" in t:
        return [
            "Quench embers born of rumor with tests that press and weigh,",
            "Where truth is met by measure, stray sparks lose their sway.",
        ]
    if "safety" in t or "ethic" in t or "risk" in t or "misuse" in t:
        return [
            "Design with brakes engaged and room to fail with grace,",
            "The careful path through fog leaves fewer scars in place.",
        ]
    if "privacy" in t or "pii" in t or "leak" in t or "secret" in t:
        return [
            "Share less than you could tell; keep secrets trimmed and sealed,",
            "An honest veil, well-placed, keeps futures unrevealed.",
        ]
    if "explain" in t or "interpret" in t or "causal" in t or "counterfactual" in t:
        return [
            "Explain what bends the arc, not only where it lands,",
            "If cause won’t sign the ledger, walk humbly with your plans.",
        ]
    if "robust" in t or "drift" in t or "shift" in t or "generaliz" in t:
        return [
            "Train for tomorrow’s storm, not only yesterday’s sky,",
            "When weather turns against you, resilience learns to try.",
        ]
    if "latency" in t or "throughput" in t or "cost" in t or "budget" in t:
        return [
            "Spend where it truly counts; let thrift and truth align,",
            "A plainer, sturdier bridge outlives a gilded line.",
        ]
    if "creativ" in t or "novel" in t or "original" in t:
        return [
            "If spark is what you seek, ask whence the tinder came,",
            "New fire proves itself in winds that do not speak the same.",
        ]
    return [
        "Refine with steady hands; let evidence lead the way,",
        "Doubt warmly what you build, then test and hold to day.",
    ]

# ---------------- ENFORCE RULES ----------------
def enforce_rules(text: str, topic: str, target_lines: int = 14) -> str:
    """Format the model's text as a Kelly-style poem:
    - natural opening (no fixed phrase),
    - topic-aware closing couplet (no 'Try:'/'Measure:'),
    - controlled length (default 8–16).
    """
    txt = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if "\n" not in txt:
        parts = re.split(r"(?<=[.?!])\s+", txt)
        txt = "\n".join([p for p in parts if p])

    # Clean bullets/fences if any
    txt = re.sub(r"^[-*]\s+", "", txt, flags=re.MULTILINE)
    txt = re.sub(r"^```.*?$", "", txt, flags=re.MULTILINE)

    lines = [l.strip() for l in txt.split("\n") if l.strip()]

    # Remove any old directive endings if the model sneaks them in
    if len(lines) >= 2 and (
        lines[-1].lower().startswith(("measure:", "try:")) or
        lines[-2].lower().startswith(("measure:", "try:"))
    ):
        lines = lines[:-2]

    target_lines = int(max(8, min(24, target_lines)))
    closing = closing_for_topic(topic)

    body_len = max(6, min(target_lines - 2, 14))
    body = lines[:body_len]

    final = body + closing
    final = [re.sub(r"\s+", " ", l).strip() for l in final]
    return "\n".join(final)

# ---------------- OFFLINE POEM (varied) ----------------
def offline_poem(user_text: str, target_lines: int = 14) -> str:
    topic = re.sub(r"\s+", " ", user_text).strip()
    body_pool = [
        "Assumptions dress as fact; unmask them under light.",
        "Benchmarks warm the mean; cold edges leave our sight.",
        "Data remembers harms; the missing do not speak.",
        "When patterns suit the past, tomorrow may grow weak.",
        "Confidence runs ahead; calibration holds it near.",
        "Ablations thin to cause; the scaffold shows its gear.",
        "Replications earn trust; variance writes its name.",
        "In deployment, drift prowls; feedback loops learn blame.",
        "Guardrails age in place; audits must mend the seam.",
    ]
    random.shuffle(body_pool)
    closing = closing_for_topic(topic)
    body_len = max(6, min(target_lines - 2, 14))
    return "\n".join(body_pool[:body_len] + closing)

# ---------------- GROQ CALL with AUTO-MODEL FALLBACK ----------------
def generate_poem(prompt: str, chosen_model: str, temp: float, lines: int):
    if not GROQ_API_KEY:
        return offline_poem(prompt, lines), "Offline  (missing GROQ_API_KEY)"

    client = Groq(api_key=GROQ_API_KEY)

    # Try the selected model first, then rotate through known-good models
    model_candidates = [chosen_model] + [m for m in GROQ_MODELS_ORDER if m != chosen_model]

    last_err = None
    for model in model_candidates:
        try:
            comp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Topic: {prompt.strip()}"},
                ],
                temperature=temp,
                max_tokens=700,
            )
            raw = (comp.choices[0].message.content or "").strip()
            poem = enforce_rules(raw, prompt, lines)
            return poem, f" ({model})"
        except Exception as e:
            last_err = e
            continue

    # If all candidates failed → offline poem
    poem = offline_poem(prompt, lines)
    return poem, f"Offline  ({last_err})"

# ---------------- SIDEBAR: Settings + About ----------------
st.sidebar.title("⚙️ Settings")
model = st.sidebar.selectbox(
    "Groq Model",
    GROQ_MODELS_ORDER,
    index=0,
    help="Choose a Groq-hosted LLM. If one is deprecated, the app automatically tries the next."
)
temp = st.sidebar.slider("Creativity (temperature)", 0.0, 1.5, 0.9, 0.05)
lines = st.sidebar.slider("Poem length (lines)", 8, 20, 14, 1)

with st.sidebar.expander("ℹ️ About this app", expanded=False):
    st.markdown(
        """
**Kelly** is an *AI-Skeptical Poet-Scientist*.  
She responds only in poems—professional, analytical, and careful.

**How it works**
- Your question is sent to Groq’s LLM via API.
- A system prompt sets Kelly’s voice and rules (skeptical + poetic).
- The raw model text is cleaned and shaped into a poem.
- A **topic-aware ending couplet** is appended (no fixed “Try/Measure”).
- If the model / key fails, a **deterministic offline poem** appears.

    )

# ---------------- Header + Info ----------------
st.title("🧪 Kelly — The AI-Skeptical Poet-Scientist")
st.caption("Analytical • Poetic • Powered by Groq • No fixed opener • Topic-aware endings")

st.info(
    "Ask about **AI bias, safety, data quality, robustness, explainability, creativity, scaling,** "
    "or anything where scientific skepticism matters."
)

# ---------------- Quick Starters ----------------
suggestions = [
    "Are AI ethics guidelines enough to stop bias?",
    "What happens when we scale models without scaling data quality?",
    "Can AI truly understand context?",
    "Why do models hallucinate even with retrieval?",
    "Can fairness be engineered, or only pursued?"
]
cols = st.columns(min(3, len(suggestions)))
for i, q in enumerate(suggestions):
    if cols[i % len(cols)].button(q):
        st.session_state.setdefault("history", [])
        st.session_state.history.append({"role": "user", "content": q})
        poem, backend = generate_poem(q, model, temp, lines)
        st.session_state.history.append({"role": "assistant", "content": f"**Using:** {backend}\n\n{poem}"})
                st.rerun()


# ---------------- Chat History ----------------
if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(msg["content"])

# ---------------- Chat Input ----------------
prompt = st.chat_input("Ask Kelly about AI…")
if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Composing a skeptical poem…"):
            poem, backend = generate_poem(prompt, model, temp, lines)
            st.markdown(f"**Using:** {backend}")
            st.markdown(poem)

    st.session_state.history.append({"role": "assistant", "content": poem})
