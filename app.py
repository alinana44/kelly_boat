import os
import re
import random
import streamlit as st
from groq import Groq

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Kelly — The AI-Skeptical Poet-Scientist",
    page_icon="🧠",
    layout="centered",
)

# ------------- API KEY HANDLING -------------
def get_groq_key() -> str | None:
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY")

GROQ_API_KEY = get_groq_key()
if not GROQ_API_KEY:
    st.error("No GROQ_API_KEY found. Please add it in Streamlit → Settings → Secrets.")
    st.stop()

# ---------------- CORE SYSTEM PROMPT ----------------
SYSTEM_PROMPT = """
You are Kelly, an AI Scientist who replies ONLY in poems.

Tone & style:
- Skeptical, analytical, professional.
- Question broad claims about AI; highlight assumptions and limitations.
- Calm, precise, and reflective — no emojis.

Each poem must:
1. Question the premise of the user’s AI claim.
2. Surface possible flaws, data gaps, and overconfidence.
3. Suggest ways to validate or improve — but poetically.
4. End with a short, topic-aware couplet (no 'Try:'/'Measure:').
"""

# ---------------- TOPIC-BASED ENDINGS ----------------
def closing_for_topic(topic: str) -> list[str]:
    t = (topic or "").lower()
    if "bias" in t or "fair" in t:
        return ["Audit what tilts the scales before you praise the score,",
                "Balance lives in narrow slices, where quiet margins roar."]
    if "data" in t or "scale" in t:
        return ["Feed truth before you grow the web of weight and wire,",
                "Poor seeds make poorer harvests, no matter how you hire."]
    if "context" in t or "ground" in t:
        return ["Let meaning rest on place and time, on witnesses that stand,",
                "Without a ground, bright words dissolve like castles made of sand."]
    if "safety" in t or "ethic" in t:
        return ["Design with brakes engaged and room to fail with grace,",
                "The careful path through fog leaves fewer scars in place."]
    if "privacy" in t or "secret" in t:
        return ["Share less than you could tell; keep secrets trimmed and sealed,",
                "An honest veil, well-placed, keeps futures unrevealed."]
    if "creativ" in t:
        return ["If spark is what you seek, ask whence the tinder came,",
                "New fire proves itself in winds that do not speak the same."]
    return ["Refine with steady hands; let evidence lead the way,",
            "Doubt warmly what you build, then test and hold to day."]

# ---------------- ENFORCE RULES ----------------
def enforce_rules(text: str, topic: str, target_lines: int = 14) -> str:
    txt = (text or "").replace("\r", "").strip()
    lines = [l.strip() for l in txt.split("\n") if l.strip()]
    closing = closing_for_topic(topic)
    body_len = max(6, min(target_lines - 2, len(lines)))
    return "\n".join(lines[:body_len] + closing)

# ---------------- OFFLINE POEM ----------------
def offline_poem(user_text: str, target_lines: int = 14) -> str:
    topic = re.sub(r"\s+", " ", user_text)
    lines = [
        "Assumptions dress as fact; unmask them under light.",
        "Benchmarks warm the mean; cold edges leave our sight.",
        "Data remembers harms; the missing do not speak.",
        "When patterns suit the past, tomorrow may grow weak.",
        "Confidence runs ahead; calibration holds it near.",
        "Replications earn trust; variance writes its name.",
        "Guardrails age in place; audits must mend the seam.",
    ]
    random.shuffle(lines)
    closing = closing_for_topic(topic)
    return "\n".join(lines[:10] + closing)

# ---------------- GENERATE POEM ----------------
def generate_poem(prompt: str, model: str, temp: float, lines: int):
    client = Groq(api_key=GROQ_API_KEY)
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
        text = comp.choices[0].message.content
        poem = enforce_rules(text, prompt, lines)
        return poem, "GROQ ✅"
    except Exception as e:
        poem = offline_poem(prompt, lines)
        return poem, f"Offline ❌ ({e})"

# ---------------- SIDEBAR SETTINGS ----------------
st.sidebar.title("⚙️ Kelly Settings")
model = st.sidebar.selectbox(
    "Groq Model",
    ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"],
)
temp = st.sidebar.slider("Creativity (temperature)", 0.0, 1.5, 0.8, 0.05)
lines = st.sidebar.slider("Poem length", 8, 20, 14, 1)
st.sidebar.caption("Kelly crafts poems that question AI’s boldest claims.")

# ---------------- UI ----------------
st.title("🧠 Kelly — The AI-Skeptical Poet-Scientist")
st.caption("Analytical • Poetic • Powered by Groq")

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask Kelly about AI…")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.history.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Composing a skeptical poem…"):
            poem, backend = generate_poem(prompt, model, temp, lines)
            st.markdown(f"**Using: {backend}**")
            st.markdown(poem)
    st.session_state.history.append({"role": "assistant", "content": poem})
