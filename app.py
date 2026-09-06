import os
import re
import html
import asyncio
import requests

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# Free Google Gemini model that supports function/tool calling on OpenRouter.
# Swap for any other OpenRouter slug that supports tools if you prefer.
MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-exp:free")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

st.set_page_config(page_title="MediVoice — Health answers, read aloud", page_icon="🩺")

SYSTEM_PROMPT = """You are a medical information assistant.

Your job is to provide general educational medical information, grounded in the medical information tool.

Rules:
- Do not diagnose diseases.
- Do not prescribe medicines.
- Do not replace a healthcare professional.
- Always call the medical_information tool for the topic asked about before answering, and base the answer on what it returns.
- Keep the answer concise: at most 3 short paragraphs.
- Use simple language.
- Do not use Markdown, headings, bold, or bullet lists. Write plain flowing sentences, because the answer may be spoken aloud.
- End with one gentle sentence reminding the user to consult a healthcare professional for personal advice."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "medical_information",
            "description": "Search MedlinePlus for general medical information about a topic. Educational information only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "The medical topic to search for"},
                },
                "required": ["topic"],
                "additionalProperties": False,
            },
        },
    }
]


def decode_entities(value: str) -> str:
    return html.unescape(value)


def extract_content(body: str, name: str) -> str:
    m = re.search(
        rf'<content[^>]*name="{name}"[^>]*>(.*?)</content>', body, re.IGNORECASE | re.DOTALL
    )
    if not m:
        return ""
    return decode_entities(m.group(1)).strip()


def fetch_medlineplus(topic: str):
    """Mirror of the website's MedlinePlus lookup. Returns (text, sources)."""
    url = "https://wsearch.nlm.nih.gov/ws/query"
    params = {"db": "healthTopics", "term": topic, "retmax": "3", "rettype": "brief"}
    try:
        res = requests.get(url, params=params, timeout=15)
    except requests.RequestException:
        return "Unable to access MedlinePlus right now.", []

    if not res.ok:
        return f"Unable to access MedlinePlus (status {res.status_code}).", []

    xml = res.text
    sources = []
    parts = []
    for m in re.finditer(
        r'<document[^>]*url="([^"]*)"[^>]*>(.*?)</document>', xml, re.DOTALL
    ):
        doc_url = decode_entities(m.group(1))
        body = m.group(2)
        title = extract_content(body, "title")
        summary = (
            extract_content(body, "FullSummary") or extract_content(body, "full-summary")
        )
        if title or summary:
            sources.append({"title": title or topic, "url": doc_url})
            parts.append(f"{title}\n{summary}\nSource: {doc_url}")

    if not parts:
        return f"No medical information found for: {topic}", []

    text = (
        f"Medical information from MedlinePlus for '{topic}':\n\n"
        + "\n\n".join(parts)
        + "\n\nImportant: This information is for educational purposes only."
    )
    return text, sources


def call_openrouter(messages):
    """Call OpenRouter (OpenAI-compatible) with tools, return the assistant message."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://lovable.dev",
        "X-Title": "MediVoice",
    }
    payload = {"model": MODEL, "messages": messages, "tools": TOOLS}
    res = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
    if not res.ok:
        body = res.text
        if res.status_code == 429:
            raise RuntimeError("The assistant is busy right now. Please try again in a moment.")
        if res.status_code == 402:
            raise RuntimeError("The assistant is out of credits. Check your OpenRouter account.")
        raise RuntimeError(f"The assistant could not answer right now ({res.status_code}). {body[:200]}")
    data = res.json()
    msg = data.get("choices", [{}])[0].get("message")
    if not msg:
        raise RuntimeError("The assistant returned an empty response.")
    return msg


def answer_question(user_text: str):
    """Run the same tool-calling loop as the website (ask-medical.functions.ts)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    sources = []

    for _ in range(3):
        message = call_openrouter(messages)
        tool_calls = message.get("tool_calls")

        if tool_calls:
            messages.append(message)
            for call in tool_calls:
                topic = user_text
                try:
                    topic = (
                        call.get("function", {}).get("arguments", "{}")
                    )
                    import json
                    topic = json.loads(topic).get("topic", user_text)
                except Exception:
                    pass
                text, src = fetch_medlineplus(topic)
                sources.extend(src)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id"),
                        "content": text,
                    }
                )
            continue

        answer = (message.get("content") or "").strip()
        # de-duplicate sources by url
        seen = set()
        unique = []
        for s in sources:
            if s["url"] and s["url"] not in seen:
                seen.add(s["url"])
                unique.append(s)
        return answer, unique

    raise RuntimeError("The assistant took too many steps. Please rephrase your question.")


def generate_voice(text: str, path: str) -> None:
    import edge_tts

    communicate = edge_tts.Communicate(text=text, voice="en-IN-NeerjaNeural")
    asyncio.run(communicate.save(path))


# ---- UI (matches the previewed website) ----
st.markdown(
    "<h1 style='margin-bottom:0.2em'>Ask a health question,<br>hear a plain answer.</h1>",
    unsafe_allow_html=True,
)
st.caption(
    "Answers are drawn from MedlinePlus, the U.S. National Library of Medicine's "
    "consumer health library, then explained in simple words and read aloud. "
    "General educational information only — not medical advice."
)

if not OPENROUTER_API_KEY:
    st.error(
        "OPENROUTER_API_KEY is not configured. Add it in the app's "
        "Settings → Secrets as `OPENROUTER_API_KEY`."
    )
    st.stop()

examples = [
    "What is hypertension?",
    "How can I manage type 2 diabetes?",
    "What causes migraines?",
    "Why is vitamin D important?",
]

cols = st.columns(len(examples))
for col, ex in zip(cols, examples):
    if col.button(ex, key=f"ex_{ex}"):
        st.session_state["question"] = ex
        st.session_state["run"] = ex

question = st.text_input(
    "Your question",
    value=st.session_state.get("question", ""),
    placeholder="e.g. What is hypertension?",
    label_visibility="collapsed",
)

voice_on = st.toggle("Voice on", value=True)

run_q = st.session_state.get("run")
if st.button("Get answer", type="primary") or run_q:
    q = (run_q or question).strip()
    st.session_state.pop("run", None)
    if not q:
        st.warning("Please type a question first.")
        st.stop()
    with st.spinner("Looking it up…"):
        try:
            answer, sources = answer_question(q)
        except Exception as e:
            st.error(str(e))
            st.stop()

    st.subheader("Answer")
    for paragraph in re.split(r"\n+", answer):
        if paragraph.strip():
            st.write(paragraph)

    if sources:
        st.markdown("**Sources**")
        for s in sources:
            st.markdown(f"- [{s['title']}]({s['url']})")

    if voice_on and answer:
        audio_file = "medical_response.mp3"
        try:
            with st.spinner("Generating voice…"):
                generate_voice(answer, audio_file)
            st.audio(audio_file, format="audio/mp3")
        except Exception as e:
            st.caption(f"Voice playback unavailable: {e}")

st.divider()
st.info("Always consult a qualified healthcare professional for personal medical advice.")
