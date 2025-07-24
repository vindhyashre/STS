# app.py (Simulated Real-Time Version)

import streamlit as st
import numpy as np
import pyaudio
import queue
import torch
import soundfile as sf
import time

from faster_whisper import WhisperModel
from transformers import pipeline

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Speech Translator",
    page_icon="🎙️",
    layout="wide"
)

# --- CONFIGURATION & CONSTANTS ---
# You can add more languages here if you find the codes for them
LANGUAGE_CODES = {
    "English": {"whisper": "en", "nllb": "eng_Latn", "mms": "eng"},
    "Hindi":   {"whisper": "hi", "nllb": "hin_Deva", "mms": "hin"},
    "Marathi": {"whisper": "mr", "nllb": "mar_Deva", "mms": "mar"},
    "Telugu":  {"whisper": "te", "nllb": "tel_Telu", "mms": "tel"},
    "Tamil":   {"whisper": "ta", "nllb": "tam_Taml", "mms": "tam"},
    "Kannada": {"whisper": "kn", "nllb": "kan_Knda", "mms": "kan"},
}
# How many seconds of audio to process at a time
RECORD_SECONDS = 5 

# --- MODEL LOADING (CACHED) ---
# This function loads all AI models and is cached by Streamlit for performance.
@st.cache_resource
def load_models():
    """Loads and caches all AI models."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if torch.cuda.is_available() else "int8"
    
    st.write(f"Loading models on: {device.upper()}")
    
    # Use a smaller model if you don't have a strong GPU for faster performance
    # Options: "tiny", "base", "small", "medium", "large-v3"
    stt_model = WhisperModel("large-v3", device=device, compute_type=compute_type)
    
    # Use a smaller translation model for faster performance
    # Options: "facebook/nllb-200-distilled-600M", "facebook/nllb-200-1.3B"
    translator = pipeline(
        'translation', 
        model="facebook/nllb-200-1.3B", 
        device=device, 
        torch_dtype=torch.float16 if device=="cuda" else torch.float32
    )
    
    tts_model_cache = {}
    return stt_model, translator, tts_model_cache, device

with st.spinner("Loading AI models... This may take a while the first time you run it."):
    stt_model, translator, tts_model_cache, device = load_models()

# --- SESSION STATE ---
# This is used to keep track of whether the app is recording or not.
if "run" not in st.session_state:
    st.session_state.run = False

def start_listening():
    st.session_state.run = True

def stop_listening():
    st.session_state.run = False

# --- UI LAYOUT ---
st.title("Multilingual Speech-to-Speech Translator 🏃")
st.markdown("Select your languages, press 'Start Translating', and begin speaking. The app will process your speech every few seconds.")

col1, col2 = st.columns(2)
with col1:
    source_lang_name = st.selectbox("I am speaking in...", options=list(LANGUAGE_CODES.keys()), index=0)
with col2:
    target_lang_name = st.selectbox("Translate to...", options=list(LANGUAGE_CODES.keys()), index=1)

# Centered Start/Stop buttons
st.columns(3)[1].button("Start Translating", on_click=start_listening, type="primary", use_container_width=True, disabled=st.session_state.run)
st.columns(3)[1].button("Stop Translating", on_click=stop_listening, use_container_width=True, disabled=not st.session_state.run)

# --- MAIN PROCESSING LOOP ---
if st.session_state.run:
    st.info(f"Listening in {source_lang_name}... Speak now.")
    
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    
    # Placeholders to display the results dynamically
    transcription_placeholder = st.empty()
    translation_placeholder = st.empty()
    audio_placeholder = st.empty()

    while st.session_state.run:
        frames = []
        for _ in range(0, int(16000 / 1024 * RECORD_SECONDS)):
            if not st.session_state.run: break
            try:
                data = stream.read(1024)
                frames.append(data)
            except IOError:
                # This can happen if the audio device changes
                pass

        if not st.session_state.run: break

        audio_data = b''.join(frames)
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        if len(audio_np) > 16000 * 0.5: # Only process if audio is longer than 0.5 seconds
            with st.spinner("Translating..."):
                # 1. Speech-to-Text
                segments, _ = stt_model.transcribe(audio_np, language=LANGUAGE_CODES[source_lang_name]["whisper"], beam_size=5)
                transcribed_text = "".join(seg.text for seg in segments).strip()
                
                if transcribed_text:
                    transcription_placeholder.info(f"Heard ({source_lang_name}): {transcribed_text}")

                    # 2. Translation
                    translation_result = translator(transcribed_text, src_lang=LANGUAGE_CODES[source_lang_name]["nllb"], tgt_lang=LANGUAGE_CODES[target_lang_name]["nllb"])
                    translated_text = translation_result[0]['translation_text']
                    translation_placeholder.success(f"Translation ({target_lang_name}): {translated_text}")

                    # 3. Text-to-Speech
                    tgt_lang_mms = LANGUAGE_CODES[target_lang_name]["mms"]
                    tts_model_name = f"facebook/mms-tts-{tgt_lang_mms}"
                    if tts_model_name not in tts_model_cache:
                        tts_model_cache[tts_model_name] = pipeline("text-to-speech", model=tts_model_name, device=device)
                    
                    speech = tts_model_cache[tts_model_name](translated_text)
                    audio_placeholder.audio(speech["audio"], sample_rate=speech["sampling_rate"], autoplay=True)
                else:
                    transcription_placeholder.warning("No speech detected in the last few seconds.")
        
    # Cleanup after stopping
    stream.stop_stream()
    stream.close()
    p.terminate()

else:
    st.info("Press 'Start Translating' to begin.")

st.markdown("---")
st.markdown("Powered by Open Source AI models from Meta and OpenAI.")