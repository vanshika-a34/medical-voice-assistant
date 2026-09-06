import asyncio
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import edge_tts
import requests
import streamlit as st
from faster_whisper import WhisperModel
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openrouter import ChatOpenRouter


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Medical Voice Assistant",
    page_icon="🩺",
    layout="centered",
)


# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.4rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #666;
            margin-bottom: 1.5rem;
        }

        .disclaimer {
            padding: 1rem;
            border-radius: 10px;
            background: #fff4e5;
            border: 1px solid #ffd699;
            margin-top: 1rem;
        }

        .source-box {
            padding: 0.8rem;
            border-left: 4px solid #4b7bec;
            background: #f7f9fc;
            margin-bottom: 0.7rem;
            border-radius: 4px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# API key
# ---------------------------------------------------------
def get_api_key():
    """
    Streamlit Cloud:
        Add OPENROUTER_API_KEY under App Settings > Secrets.

    Local:
        You can also create a .env file with:
        OPENROUTER_API_KEY=your_key
    """
    try:
        key = st.secrets.get("OPENROUTER_API_KEY")
    except Exception:
        key = None

    if key:
        return key

    # Optional local .env support
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    return os.getenv("OPENROUTER_API_KEY")


OPENROUTER_API_KEY = get_api_key()

if not OPENROUTER_API_KEY:
    st.error(
        "OPENROUTER_API_KEY is not configured. "
        "For Streamlit Cloud, add it in App Settings → Secrets."
    )
    st.stop()


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------
@st.cache_resource
def load_llm():
    return ChatOpenRouter(
        model="z-ai/glm-5.3-flash",
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        temperature=0,
    )


model = load_llm()


# ---------------------------------------------------------
# MedlinePlus tool
# ---------------------------------------------------------
@tool
def medical_information(topic: str) -> str:
    """
    Search MedlinePlus for general medical information about a topic.
    Provides educational information only and does not diagnose or prescribe.
    """
    url = "https://wsearch.nlm.nih.gov/ws/query"

    params = {
        "db": "healthTopics",
        "term": topic,
        "retmax": 3,
        "rettype": "brief",
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        root = ET.fromstring(response.text)
        results = []

        for document in root.findall(".//document"):
            title = ""
            summary = ""
            page_url = document.attrib.get("url", "")

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
                        "title": title,
                        "summary": summary,
                        "url": page_url,
                    }
                )

        if not results:
            return f"No medical information found for: {topic}"

        output = f"Medical information from MedlinePlus for '{topic}':\n\n"

        for i, result in enumerate(results, 1):
            output += f"{i}. {result['title']}\n"
            output += f"{result['summary']}\n"
            output += f"Source: {result['url']}\n\n"

        output += (
            "Important: This information is for educational purposes only. "
            "It does not provide a diagnosis or medical prescription."
        )

        return output

    except requests.RequestException as e:
        return f"Unable to access MedlinePlus: {str(e)}"

    except ET.ParseError:
        return "Unable to process the medical information returned by MedlinePlus."


model_with_tools = model.bind_tools([medical_information])


# ---------------------------------------------------------
# Whisper model
# ---------------------------------------------------------
@st.cache_resource
def load_whisper():
    return WhisperModel(
        "small",
        device="cpu",
        compute_type="int8",
    )


# ---------------------------------------------------------
# Speech-to-text
# ---------------------------------------------------------
def transcribe_audio(audio_bytes):
    whisper_model = load_whisper()

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav",
    ) as temp_file:
        temp_file.write(audio_bytes)
        temp_path = temp_file.name

    try:
        segments, _ = whisper_model.transcribe(
            temp_path,
            beam_size=5,
        )

        text = " ".join(segment.text.strip() for segment in segments).strip()
        return text

    finally:
        Path(temp_path).unlink(missing_ok=True)


# ---------------------------------------------------------
# Text-to-speech
# ---------------------------------------------------------
async def generate_voice(text, output_path):
    voice = "en-IN-NeerjaNeural"

    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
    )

    await communicate.save(output_path)


def text_to_speech(text):
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp3",
    ) as temp_file:
        output_path = temp_file.name

    asyncio.run(generate_voice(text, output_path))

    with open(output_path, "rb") as audio_file:
        audio_bytes = audio_file.read()

    Path(output_path).unlink(missing_ok=True)

    return audio_bytes


# ---------------------------------------------------------
# AI response
# ---------------------------------------------------------
def get_medical_response(user_text):
    system_prompt = """
You are a medical information voice assistant.

Your job is to provide general educational medical information.

Important rules:
- Do not diagnose diseases.
- Do not prescribe medicines.
- Do not replace a healthcare professional.
- Use information retrieved from the medical information tool.
- Since your response will be spoken aloud, keep the answer concise.
- Use simple language.
- Do not use Markdown.
- Avoid long lists.
"""

    messages = [
        HumanMessage(content=system_prompt),
        HumanMessage(content=user_text),
    ]

    response = model_with_tools.invoke(messages)

    tool_results = []

    if response.tool_calls:
        messages.append(response)

        for tool_call in response.tool_calls:
            if tool_call["name"] == "medical_information":
                tool_result = medical_information.invoke(tool_call["args"])

                tool_results.append(tool_result)

                messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call["id"],
                    )
                )

        final_response = model_with_tools.invoke(messages)
        answer = final_response.content

    else:
        answer = response.content

    return answer, tool_results


# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.markdown(
    '<div class="main-title">🩺 Medical Voice Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Ask a general medical-information question using your voice or text."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="disclaimer">
        <b>⚠️ Important:</b> This assistant provides general educational
        information only. It does not diagnose conditions or prescribe
        medicines. For medical concerns, consult a qualified healthcare
        professional.
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

if "pending_audio_text" not in st.session_state:
    st.session_state.pending_audio_text = None

if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None


# ---------------------------------------------------------
# Previous conversation
# ---------------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# ---------------------------------------------------------
# Voice input
# ---------------------------------------------------------
st.subheader("🎙️ Ask using your voice")

audio_value = st.audio_input(
    "Record your medical question",
)

if audio_value is not None:
    if st.button("🔎 Transcribe and Ask", use_container_width=True):
        with st.spinner("Converting your speech to text..."):
            try:
                transcribed_text = transcribe_audio(audio_value.getvalue())
            except Exception as e:
                st.error(f"Speech recognition failed: {e}")
                st.stop()

        if not transcribed_text:
            st.warning("I could not understand the recording. Please try again.")
            st.stop()

        st.info(f"**You said:** {transcribed_text}")

        st.session_state.messages.append(
            {
                "role": "user",
                "content": transcribed_text,
            }
        )

        with st.chat_message("user"):
            st.write(transcribed_text)

        with st.chat_message("assistant"):
            with st.spinner("Searching medical information and preparing a response..."):
                try:
                    answer, sources = get_medical_response(transcribed_text)
                except Exception as e:
                    st.error(f"Unable to generate a response: {e}")
                    st.stop()

                st.write(answer)

                st.session_state.pending_audio_text = answer

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        st.session_state.last_sources = sources


# ---------------------------------------------------------
# Text input
# ---------------------------------------------------------
st.divider()
st.subheader("⌨️ Or type your question")

text_question = st.chat_input(
    "Example: Can you explain hypertension in simple words?"
)

if text_question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": text_question,
        }
    )

    with st.chat_message("user"):
        st.write(text_question)

    with st.chat_message("assistant"):
        with st.spinner("Searching medical information and preparing a response..."):
            try:
                answer, sources = get_medical_response(text_question)
            except Exception as e:
                st.error(f"Unable to generate a response: {e}")
                st.stop()

            st.write(answer)

            st.session_state.pending_audio_text = answer

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    st.session_state.last_sources = sources


# ---------------------------------------------------------
# Read aloud controls
# ---------------------------------------------------------
if st.session_state.pending_audio_text:
    st.divider()
    st.subheader("🔊 Read response aloud")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("▶️ Read Aloud", use_container_width=True):
            with st.spinner("Generating voice..."):
                try:
                    st.session_state.audio_bytes = text_to_speech(
                        st.session_state.pending_audio_text
                    )
                except Exception as e:
                    st.session_state.audio_bytes = None
                    st.error(f"Voice generation failed: {e}")

    with col2:
        if st.button("⏹️ Stop", use_container_width=True):
            st.session_state.audio_bytes = None
            st.rerun()

    if st.session_state.audio_bytes:
        st.audio(
            st.session_state.audio_bytes,
            format="audio/mp3",
        )

        st.caption(
            "Use the audio player's controls to pause or stop playback."
        )


# ---------------------------------------------------------
# MedlinePlus sources
# ---------------------------------------------------------
if st.session_state.last_sources:
    st.divider()
    st.subheader("🔎 MedlinePlus information used")

    for source_text in st.session_state.last_sources:
        st.markdown(
            f'<div class="source-box"><pre>{source_text}</pre></div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.header("About")

    st.write(
        "This application is based on the notebook's "
        "OpenRouter + LangChain + MedlinePlus + Faster-Whisper + Edge-TTS workflow."
    )

    st.markdown("### Components")
    st.write("• OpenRouter / GLM 5.3 Flash")
    st.write("• LangChain tool calling")
    st.write("• MedlinePlus")
    st.write("• Faster-Whisper")
    st.write("• Edge-TTS")

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.session_state.pending_audio_text = None
        st.session_state.audio_bytes = None
        st.rerun()

    st.caption(
        "Educational use only. Not a substitute for professional medical advice."
    )
