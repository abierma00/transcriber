import streamlit as st
import whisper
import os
import tempfile
from whisper.tokenizer import LANGUAGES
from deep_translator import GoogleTranslator

# --- FFmpeg Check ---
# On Streamlit Cloud, FFmpeg is installed via packages.txt, so we just check if it's there.
# If running locally without a path set, this ensures we don't crash immediately.
try:
    import imageio_ffmpeg
    ffmpeg_cmd = imageio_ffmpeg.get_ffmpeg_exe()
    print(f"FFmpeg located at: {ffmpeg_cmd}")
except ImportError:
    print("imageio-ffmpeg not installed. Relying on system PATH.")

# Helper: Map language codes
WHISPER_LANGUAGES = {v.capitalize(): k for k, v in LANGUAGES.items()}

def main():
    st.title("🌍 Universal Audio Transcriber & Translator")
    st.markdown("Transcribe audio from **any language** and translate it to **any language**.")

    # --- 1. Audio Settings ---
    st.subheader("1. Audio Settings")
    source_options = ["Auto-Detect"] + list(WHISPER_LANGUAGES.keys())
    source_lang_name = st.selectbox("What language is the audio speaking?", source_options, index=0)

    # --- 2. Translation Settings ---
    st.subheader("2. Translation Settings")
    enable_translation = st.checkbox("Translate the result to another language?")
    
    target_lang_name = None
    if enable_translation:
        try:
            translator_langs = GoogleTranslator().get_supported_languages()
            translator_langs = [lang.capitalize() for lang in translator_langs]
            target_lang_name = st.selectbox(
                "Translate text into:",
                translator_langs,
                index=translator_langs.index("English") if "English" in translator_langs else 0
            )
        except Exception as e:
            st.warning(f"Could not load languages from Google Translator: {e}")

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
                    # Use 'base' model. If it crashes on Cloud (Out of Memory), change to 'tiny'
                    model = whisper.load_model("base")

                with st.spinner("Transcribing audio..."):
                    whisper_lang_code = None
                    if source_lang_name != "Auto-Detect":
                        whisper_lang_code = WHISPER_LANGUAGES[source_lang_name]

                    result = model.transcribe(tmp_path, language=whisper_lang_code)
                    original_text = result["text"]
                    
                    st.success("Transcription Complete!")
                    st.markdown("### 📝 Original Transcript")
                    st.text_area("Original", original_text, height=200)
                    
                    st.download_button(
                        label="Download Original",
                        data=original_text,
                        file_name="transcript_original.txt",
                        mime="text/plain"
                    )

                # --- PHASE 2: TRANSLATION ---
                if enable_translation and target_lang_name:
                    with st.spinner(f"Translating to {target_lang_name}..."):
                        translator = GoogleTranslator(source='auto', target=target_lang_name.lower())
                        translated_text = translator.translate(original_text)
                        
                        st.markdown(f"### 🌐 Translated Transcript ({target_lang_name})")
                        st.text_area("Translated", translated_text, height=200)
                        
                        st.download_button(
                            label=f"Download {target_lang_name} Translation",
                            data=translated_text,
                            file_name=f"transcript_{target_lang_name}.txt",
                            mime="text/plain"
                        )

            except Exception as e:
                st.error(f"An error occurred: {e}")
            
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

if __name__ == "__main__":
    main()
