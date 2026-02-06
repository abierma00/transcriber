import streamlit as st
import whisper
import os
import tempfile
from whisper.tokenizer import LANGUAGES
from deep_translator import GoogleTranslator
from gtts import gTTS

# --- FFmpeg Check ---
try:
    import imageio_ffmpeg
    ffmpeg_cmd = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    pass

# Helper 1: Map Whisper languages (for the "Source" dropdown)
WHISPER_LANGUAGES = {v.capitalize(): k for k, v in LANGUAGES.items()}

# Helper 2: Map Google Translator languages (Name -> Code)
# We fetch this as a dictionary { 'arabic': 'ar', 'french': 'fr' ... }
try:
    LANG_CODES = GoogleTranslator().get_supported_languages(as_dict=True)
except:
    # Fallback if internet fails
    LANG_CODES = {"english": "en", "spanish": "es", "french": "fr", "german": "de", "arabic": "ar"}

def main():
    st.title("🌍 Universal Audio Transcriber & Translator")

    # --- 1. Audio Settings ---
    st.subheader("1. Audio Settings")
    source_options = ["Auto-Detect"] + list(WHISPER_LANGUAGES.keys())
    source_lang_name = st.selectbox("What language is the audio speaking?", source_options, index=0)

    # --- 2. Translation Settings ---
    st.subheader("2. Translation Settings")
    enable_translation = st.checkbox("Translate the result to another language?")
    
    target_lang_code = "en" # Default to English
    target_lang_name = "English"

    if enable_translation:
        # Create a list of Capitalized names for the dropdown
        display_names = [name.capitalize() for name in LANG_CODES.keys()]
        
        target_lang_name = st.selectbox(
            "Translate text into:",
            display_names,
            index=display_names.index("English") if "English" in display_names else 0
        )
        # Get the 2-letter code for the selected name
        target_lang_code = LANG_CODES.get(target_lang_name.lower(), "en")

    # --- 3. Upload File ---
    st.subheader("3. Upload File")
    uploaded_file = st.file_uploader("Drop your audio file here", type=["mp3", "wav", "m4a", "mp4"])

    if uploaded_file is not None:
        if st.button("Start Processing"):
            # Save uploaded file momentarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            try:
                # --- PHASE 1: TRANSCRIPTION ---
                with st.spinner("Loading AI Model (Whisper)..."):
                    model = whisper.load_model("base")

                with st.spinner("Transcribing audio..."):
                    whisper_lang_code = None
                    if source_lang_name != "Auto-Detect":
                        whisper_lang_code = WHISPER_LANGUAGES[source_lang_name]

                    result = model.transcribe(tmp_path, language=whisper_lang_code)
                    original_text = result["text"]
                    
                    st.success("Transcription Complete!")
                    st.markdown("### 📝 Original Transcript")
                    st.text_area("Original", original_text, height=150)

                # --- PHASE 2: TRANSLATION & AUDIO ---
                if enable_translation:
                    with st.spinner(f"Translating to {target_lang_name}..."):
                        # Translate
                        translator = GoogleTranslator(source='auto', target=target_lang_code)
                        translated_text = translator.translate(original_text)
                        
                        st.markdown(f"### 🌐 Translated Transcript ({target_lang_name})")
                        st.text_area("Translated", translated_text, height=150)
                        
                        # --- AUDIO GENERATION ---
                        st.markdown(f"### 🔊 Listen in {target_lang_name}")
                        try:
                            # NOW we use the correct 2-letter code (e.g., 'ar') instead of the full name
                            tts = gTTS(text=translated_text, lang=target_lang_code)
                            
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                                tts.save(fp.name)
                                st.audio(fp.name, format="audio/mp3")
                                
                        except Exception as e:
                            st.error(f"Could not generate audio: {e}")

            except Exception as e:
                st.error(f"An error occurred: {e}")
            
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

if __name__ == "__main__":
    main()
