# Medical Information Voice Assistant

A Streamlit app adapted from `medical_voice_assistant.ipynb`. It answers general health questions with an OpenRouter model, retrieves educational references from MedlinePlus, and can transcribe browser-recorded questions with Whisper.

## Streamlit deployment

1. Deploy this repository as a Streamlit app with `app.py` as the main file.
2. Add the secret below in the app's secrets settings:

```toml
OPENROUTER_API_KEY = "your-key"
```

3. The app will start with the default Streamlit command:

```bash
streamlit run app.py
```

## Features

- Text chat for general medical-information questions.
- MedlinePlus search results shown with source links.
- Optional browser audio recording and local `faster-whisper` transcription.
- Optional Edge TTS response playback is available from each assistant message.
- The original notebook is included as `medical_voice_assistant.ipynb`.

This app is educational only. It does not diagnose conditions, prescribe medicines, or replace a qualified healthcare professional.