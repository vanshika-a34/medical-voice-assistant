# 🩺 Medical Voice Assistant

An AI-powered medical information assistant that lets users **type or speak medical questions** and receive answers in **text**, with an optional **Read Aloud** feature.

**Live:** https://medical-voice-assistant.streamlit.app/

## ✨ Features

* ⌨️ **Text Input** — Type your medical question.
* 🎙️ **Voice Input** — Speak your question using **Faster-Whisper (STT)**.
* 🤖 **AI Responses** — Uses **GLM-5.3-Flash via OpenRouter**.
* 🏥 **Medical Information** — Retrieves relevant information from **MedlinePlus** using a LangChain tool.
* 📝 **Text Answers** — Displays the final response on screen.
* 🔊 **Read Aloud** — Converts the response to speech using **Edge-TTS (TTS)**.
* ⏹️ **Stop Audio** — Stop/clear the generated audio.

## 🔄 How It Works

```text
⌨️ Type OR 🎙️ Speak
        ↓
   Question Text
        ↓
     AI Agent
        ↓
   MedlinePlus Tool
        ↓
   📝 Text Response
        ↓
   ▶️ Read Aloud
        ↓
     🔊 Edge-TTS
```

### Speech Features

```text
🎙️ Speech → Faster-Whisper → Text        (STT)

📝 Text Response → Edge-TTS → Speech     (TTS)
```

## 🛠️ Tech Stack

| Technology     | Purpose                 |
| -------------- | ----------------------- |
| Python         | Core development        |
| Streamlit      | Web interface           |
| LangChain      | AI agent & tool calling |
| OpenRouter     | LLM API                 |
| GLM-5.3-Flash  | Language model          |
| Faster-Whisper | Speech-to-Text          |
| Edge-TTS       | Text-to-Speech          |
| MedlinePlus    | Medical information     |
| Requests       | API requests            |

## 📁 Project Structure

```text
medical-voice-assistant/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── medical_voice_assistant.ipynb
├── .env.example

```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd medical-voice-assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add API Key

Create:

```text
.env
```

Add:

```text
OPENROUTER_API_KEY = your_api_key_here
```

> Never commit your API key to GitHub.

## ▶️ Run the App

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## 🩺 Example

**User:**
"What are the symptoms of diabetes?"

**Assistant:**
Displays a medical information response in text.

The user can then click **▶️ Read Aloud** to listen to the response.

## ⚠️ Disclaimer

This project is for **educational and informational purposes only**. It is not a substitute for a doctor, medical diagnosis, treatment, or emergency medical advice. Always consult a qualified healthcare professional for personal medical concerns.
