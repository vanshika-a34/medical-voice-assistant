import os
import requests
import xml.etree.ElementTree as ET

import streamlit as st
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

st.set_page_config(page_title="Medical Info Voice Assistant", page_icon="🩺")

SYSTEM_PROMPT = """
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


@tool
def medical_information(topic: str) -> str:
    """
    Search MedlinePlus for general medical information about a topic.
    Provides educational information only and does not diagnose or prescribe.
    """
    url = "https://wsearch.nlm.nih.gov/ws/query"
    params = {"db": "healthTopics", "term": topic, "retmax": 3, "rettype": "brief"}

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
                results.append({"title": title, "summary": summary, "url": page_url})

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


@st.cache_resource
def get_model():
    return ChatOpenRouter(
        model="z-ai/glm-5.3-flash",
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        temperature=0,
    )


def answer_question(user_text: str) -> str:
    model_with_tools = get_model().bind_tools([medical_information])
    messages = [
        HumanMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_text),
    ]
    response = model_with_tools.invoke(messages)

    if response.tool_calls:
        messages.append(response)
        for tool_call in response.tool_calls:
            if tool_call["name"] == "medical_information":
                tool_result = medical_information.invoke(tool_call["args"])
                messages.append(
                    ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
                )
        return model_with_tools.invoke(messages).content

    return response.content


async def generate_voice(text: str, path: str) -> None:
    import edge_tts

    communicate = edge_tts.Communicate(text=text, voice="en-IN-NeerjaNeural")
    await communicate.save(path)


st.title("🩺 Medical Information Voice Assistant")
st.caption(
    "General educational health information only — not a diagnosis or a prescription."
)

if not OPENROUTER_API_KEY:
    st.error("OPENROUTER_API_KEY is not configured. Add it to your .env file or Streamlit secrets.")
    st.stop()

question = st.text_input("Ask a health question:", placeholder="What is hypertension?")

if st.button("Get answer", type="primary") and question.strip():
    with st.spinner("Looking it up…"):
        answer = answer_question(question)

    st.subheader("Answer")
    st.write(answer)

    audio_file = "medical_response.mp3"
    import asyncio

    asyncio.run(generate_voice(answer, audio_file))
    st.audio(audio_file, format="audio/mp3")

st.divider()
st.info(
    "Always consult a qualified healthcare professional for personal medical advice."
)
