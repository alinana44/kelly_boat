import os
import re
import streamlit as st

# ---------- Config UI ----------
st.set_page_config(
    page_title="Kelly — AI-Skeptical Poet-Scientist",
    page_icon="🧪",
    layout="centered",
)

# ---------- Secrets / API key discovery ----------
def get_groq_key() -> str | None:
    # Prefer Streamlit secrets (works in Streamlit Cloud)
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    # Fallback to environment variable (works locally or with dotenv)
    return os.getenv("GROQ_API_KEY")

GROQ_API_KEY = get_groq_key()

# ---------- Topic-aware poetic closings (no 'Try:'/'Measure:') ----------
def closing_for_topic(topic: str) -> list[str]:
    t = (topic or "").lower()

    if any(k in t for k in ["bias", "fair", "parity", "fine-tun", "debias"]):
        return [
            "Audit what tilts the scales before you praise the score,",
            "Balance lives in narrow slices, where quiet margins roar.",
        ]

    if any(k in t for k in ["data quality", "dirty data", "label", "noisy", "scale", "scaling"]):
        return [
            "Feed truth before you grow the web of weight and wire,",
            "Poor seeds make poorer harvests, no matter how you hire.",
        ]

    if any(k in t for k in ["context", "ground", "retriev", "rag", "citation", "source"]):
        return [
            "Let meaning rest on place and time, on witnesses that stand,",
            "Without a ground, bright words dissolve like castles made of sand.",
        ]

    if any(k in t for k in ["halluc", "reliab", "truthy", "factual", "fact", "fabricat"]):
        return [
            "Quench embers born of rumor with tests that press and weigh,",
            "Where truth is met by measure, stray sparks lose their sway.",
        ]

    if any(k in t for k in ["safety", "misuse", "align", "ethic", "harm", "risk", "abuse"]):
        return [
            "Design with brakes engaged and room to fail with grace,",
            "The careful path through fog leaves fewer scars in place.",
        ]

    if any(k in t for k in ["privacy", "leak", "pii", "confiden", "secret", "anonym"]):
        return [
            "Share less than you could tell; keep secrets trimmed and sealed,",
            "An honest veil, well-placed, keeps futures unrevealed.",
        ]

    if any(k in t for k in ["explain", "interpret", "shap", "lime", "causal", "counterfactual"]):
        return [
            "Explain what bends the arc, not only where it lands,",
            "If cause won’t sign the ledger, walk humbly with your plans.",
        ]

    if any(k in t for k in ["robust", "drift", "shift", "generaliz", "o.o.d", "out of distribution"]):
        return [
            "Train for tomorrow’s storm, not only yesterday’s sky,",
            "When weather turns against you, resilience learns to try.",
        ]

    if any(k in t for k in ["latency", "throughput", "cost", "token", "budget", "price"]):
        return [
            "Spend where it truly counts; let thrift and truth align,",
            "A plainer, sturdier bridge outlives a gilded line.",
        ]

    if any(k in t for k in ["creativ", "original", "novel", "imagin"]):
        return [
            "If spark is what you seek, ask whence the tinder came,",
            "New fire proves itself in winds that do not speak the same.",
        ]

    return [
        "Refine with steady hands; let evidence lead the way,",
        "Doubt warmly what you build, then test and hold to day.",
    ]

# ---------- Compose model output into final poem ----------
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

    # Remove bullets/fences
    txt = re.sub(r"^[-*]\s+", "", txt, flags=re.MULTILINE)
    txt = re.sub(r"^```.*?$", "", txt, flags=re.MULTILINE)

    lines = [l.strip() for l in txt.split("\n") if l.strip()]

    # Strip directive endings if any
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

# ---------- Groq call ----------
@st.cache_data(show_spinner=False)
def generate_poem_groq(user_prompt: str, model_name: str, temperature: float, target_lines: int) -> str:
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)

    system_prompt = """
You are Kelly, an AI Scientist who replies ONLY in poems.

Style & voice:
- Skeptical, analytical, professional.
- Question broad claims about AI; probe assumptions and edge cases.
- Calm, rigorous tone; precise language; crisp imagery sparingly; no emojis.

Content requirements for EVERY reply:
1) Interrogate the premise of the user's AI question.
2) Surface trade-offs, uncertainties, data constraints, and failure modes.
3) Mention evaluation/validation ideas (benchmarks, ablations, error analysis, uncertainty, reproducibility).
4) End with a short, practical, evidence-minded couplet (no fixed phrasing, no 'Try:'/'Measure:').
Formatting:
- 8–16 lines by default unless user requests otherwise.
- No bullet points; maintain poetic line-breaks.
- Never reply in plain prose.
""".strip()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Respond as Kelly with a poem tied specifically to this topic. Be skeptical, analytical, professional. Topic: {user_prompt.strip()}"},
    ]

    comp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=float(temperature),
        max_tokens=700,
        top_p=1.0,
        frequency_penalty=0.2,
    )
    raw = (comp.choices[0].message.content or "").strip()
    return enforce_rules(raw, topic=user_prompt, target_lines=target_lines)

# ---------- Sidebar ----------
st.sidebar.title("⚙️ Settings")
model = st.sidebar.selectbox(
    "Model",
    ["llama-3.1-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
    index=0,
)
temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 0.8, 0.05)
target_lines = st.sidebar.slider("Target lines", 8, 20, 14, 1)

with st.sidebar.expander("About Kelly", expanded=False):
    st.markdown(
        "Kelly is an **AI-Skeptical Poet-Scientist**. "
        "She responds only in poems—professional, analytical, and careful. "
        "She questions broad AI claims, surfaces limitations, and closes with a topic-aware couplet."
    )

# ---------- Header ----------
st.title("Kelly — AI-Skeptical Poet-Scientist")
st.caption("Powered by Groq • Poetic, skeptical, analytical • No fixed opener • Topic-aware endings")

# ---------- Check key ----------
if not GROQ_API_KEY:
    st.error(
        "No API key found. Set your key either in **.streamlit/secrets.toml** as `GROQ_API_KEY` "
        "or as an environment variable `GROQ_API_KEY`."
    )
    st.stop()

# ---------- Chat history ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---------- Input ----------
user_input = st.chat_input("Ask Kelly about AI…")
if user_input:
    # Show user bubble
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate Kelly’s poem
    with st.chat_message("assistant"):
        with st.spinner("Composing a skeptical poem…"):
            try:
                poem = generate_poem_groq(
                    user_prompt=user_input,
                    model_name=model,
                    temperature=temperature,
                    target_lines=target_lines,
                )
            except Exception as e:
                # Offline-style deterministic poem if API fails
                body_candidates = [
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
                closing = closing_for_topic(user_input)
                body_len = max(6, min(target_lines - 2, 14))
                poem = "\n".join(body_candidates[:body_len] + closing)

            st.markdown(poem)
            st.session_state.messages.append({"role": "assistant", "content": poem})
