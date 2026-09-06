"""Streamlit interface for the medical information voice assistant."""

from __future__ import annotations

import asyncio
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import requests
import streamlit as st


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "z-ai/glm-5.3-flash"
MEDLINEPLUS_URL = "https://wsearch.nlm.nih.gov/ws/query"
VOICE_NAME = "en-IN-NeerjaNeural"

SYSTEM_PROMPT = """You are a medical information voice assistant.

Your job is to provide general educational medical information.

Important rules:
- Do not diagnose diseases.
- Do not prescribe medicines.
- Do not replace a healthcare professional.
- Use the retrieved MedlinePlus information when it is available.
- Keep the answer concise and easy to understand.
- Use simple language.
- Do not use Markdown tables.
- Mention urgent care when the question suggests a possible emergency.
"""


def get_api_key() -> str:
    """Read the OpenRouter key from Streamlit secrets or the environment."""
    try:
        configured_key = st.secrets.get("OPENROUTER_API_KEY", "")
    except FileNotFoundError:
        configured_key = ""
    return str(configured_key or os.getenv("OPENROUTER_API_KEY", "")).strip()


@st.cache_data(ttl=900, show_spinner=False)
def search_medlineplus(topic: str) -> list[dict[str, str]]:
    """Fetch a small set of educational results from MedlinePlus."""
    response = requests.get(
        MEDLINEPLUS_URL,
        params={"db": "healthTopics", "term": topic, "retmax": 3, "rettype": "brief"},
        timeout=15,
    )
    response.raise_for_status()
    root = ET.fromstring(response.text)

    results: list[dict[str, str]] = []
    for document in root.findall(".//document"):
        title = ""
        summary = ""
        for content in document.findall("content"):
            name = content.attrib.get("name")
            text = "".join(content.itertext()).strip()
            if name == "title":
                title = text
            elif name == "full-summary":
                summary = text

        if title or summary:
            results.append(
                {
                    "title": title or "MedlinePlus health topic",
                    "summary": summary,
                    "url": document.attrib.get("url", ""),
                }
            )

    return results


def build_reference_context(results: list[dict[str, str]]) -> str:
    if not results:
        return "No MedlinePlus results were found for this question."

    chunks = []
    for index, result in enumerate(results, start=1):
        chunks.append(
            f"Source {index}: {result['title']}\n"
            f"{result['summary']}\n"
            f"URL: {result['url']}"
        )
    return "\n\n".join(chunks)


def call_openrouter(question: str, results: list[dict[str, str]]) -> str:
    """Ask the configured OpenRouter model for a concise educational answer."""
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not configured. Add it to Streamlit secrets before asking a question."
        )

    user_prompt = (
        f"Patient question:\n{question}\n\n"
        "Retrieved educational references from MedlinePlus:\n"
        f"{build_reference_context(results)}\n\n"
        "Answer the patient's question using the references where relevant. "
        "Do not claim certainty beyond the references. End with a brief reminder "
        "that this is educational information, not a diagnosis."
    )

    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://streamlit.io",
            "X-Title": "Medical Information Voice Assistant",
        },
        json={
            "model": OPENROUTER_MODEL,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        },
        timeout=90,
    )
    response.raise_for_status()
    payload = response.json()

    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("OpenRouter returned an unexpected response.") from exc

    if isinstance(content, list):
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    answer = str(content).strip()
    if not answer:
        raise RuntimeError("OpenRouter returned an empty answer.")
    return answer


@st.cache_resource(show_spinner=False)
def load_whisper_model() -> Any:
    """Load Whisper only when a user chooses voice input."""
    from faster_whisper import WhisperModel

    return WhisperModel("small", device="cpu", compute_type="int8")


def transcribe_audio(audio_bytes: bytes) -> str:
    """Transcribe browser-recorded WAV audio with faster-whisper."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
        audio_file.write(audio_bytes)
        audio_path = Path(audio_file.name)

    try:
        model = load_whisper_model()
        segments, _ = model.transcribe(str(audio_path), vad_filter=True)
        return " ".join(segment.text.strip() for segment in segments).strip()
    finally:
        audio_path.unlink(missing_ok=True)


def generate_speech(text: str) -> bytes:
    """Generate an MP3 response with Edge TTS."""

    async def synthesize() -> bytes:
        import edge_tts

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as audio_file:
            audio_path = Path(audio_file.name)
        try:
            communicator = edge_tts.Communicate(text=text, voice=VOICE_NAME)
            await communicator.save(str(audio_path))
            return audio_path.read_bytes()
        finally:
            audio_path.unlink(missing_ok=True)

    return asyncio.run(synthesize())


def ask_question(question: str) -> None:
    cleaned_question = question.strip()
    if not cleaned_question:
        return

    st.session_state.messages.append({"role": "user", "content": cleaned_question})
    with st.spinner("Checking MedlinePlus and preparing an answer..."):
        try:
            sources = search_medlineplus(cleaned_question)
            answer = call_openrouter(cleaned_question, sources)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                }
            )
        except requests.RequestException as exc:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"I couldn't reach a required service right now: {exc}",
                    "sources": [],
                }
            )
        except (ET.ParseError, RuntimeError) as exc:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": str(exc),
                    "sources": [],
                }
            )


st.set_page_config(
    page_title="Medical Information Voice Assistant",
    page_icon="🩺",
    layout="centered",
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello. I can explain general health topics using information "
                "from MedlinePlus. What would you like to know?"
            ),
            "sources": [],
        }
    ]

st.title("Medical Information Voice Assistant")
st.caption("General educational information, grounded in MedlinePlus references.")

with st.sidebar:
    st.header("About this assistant")
    st.write(
        "Ask a health question in plain language. The assistant checks "
        "MedlinePlus and uses the notebook's OpenRouter model to summarize "
        "the information."
    )
    st.warning(
        "This is not a doctor. It does not diagnose conditions or prescribe "
        "medicines. For urgent symptoms, contact local emergency services."
    )
    st.divider()
    st.subheader("Voice input")
    st.write("Record a question in your browser, then transcribe it locally.")
    audio_input = getattr(st, "audio_input", None)
    if audio_input is None:
        st.info("Upgrade Streamlit to enable browser audio recording.")
    else:
        recording = audio_input("Record a question")
        if recording is not None:
            if st.button("Transcribe recording", use_container_width=True):
                with st.spinner("Transcribing with Whisper..."):
                    try:
                        transcript = transcribe_audio(recording.getvalue())
                        st.session_state.voice_transcript = transcript
                    except Exception as exc:
                        st.error(f"Transcription failed: {exc}")

        transcript = st.session_state.get("voice_transcript", "")
        if transcript:
            st.text_area("Transcription", value=transcript, height=100)
            if st.button("Ask using transcription", use_container_width=True):
                st.session_state.voice_transcript = ""
                ask_question(transcript)
                st.rerun()

for message_index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant" and message["content"]:
            if st.button(
                "Play spoken response",
                key=f"speech-{message_index}",
                use_container_width=False,
            ):
                with st.spinner("Generating spoken response..."):
                    try:
                        message["audio"] = generate_speech(message["content"])
                    except Exception as exc:
                        st.error(f"Audio generation failed: {exc}")
            if message.get("audio"):
                st.audio(message["audio"], format="audio/mp3")
        sources = message.get("sources", [])
        if sources:
            with st.expander("MedlinePlus sources"):
                for source in sources:
                    st.markdown(f"**{source['title']}**")
                    if source["summary"]:
                        st.write(source["summary"])
                    if source["url"]:
                        st.markdown(f"[Open source]({source['url']})")

typed_question = st.chat_input("Ask a general health question...")
if typed_question:
    ask_question(typed_question)
    st.rerun()