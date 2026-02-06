import streamlit as st
import whisper
import os
import tempfile
from whisper.tokenizer import LANGUAGES # Import Whisper's huge list of languages
from deep_translator import GoogleTranslator

# Helper: Create a dictionary of languages for the dropdown
# Format: {"English": "en", "Spanish": "es", ...}
# We capitalize the language names for a better look
WHISPER_LANGUAGES = {v.capitalize(): k for k, v in LANGUAGES.items()}

def main():
    st.title("🌍 Universal Audio Transcriber & Translator")
    st.markdown("Transcribe audio from **any language** and translate it to **any language**.")

    # --- 1. Source Language Settings ---
    st.subheader("1. Audio Settings")
    
    # Add "Auto-Detect" to the top of the list
    source_options = ["Auto-Detect"] + list(WHISPER_LANGUAGES.keys())
    
    source_lang_name = st.selectbox(
        "What language is the audio speaking?",
        source_options,
        index=0 # Default to Auto-Detect
    )

    # --- 2. Translation Settings ---
    st.subheader("2. Translation Settings")
    enable_translation = st.checkbox("Translate the result to another language?")
    
    target_lang_name = None
    if enable_translation:
        # Get list of supported languages from Google Translator
        translator_langs = GoogleTranslator().get_supported_languages()
        # Capitalize them for the dropdown
        translator_langs = [lang.capitalize() for lang in translator_langs]
        
        target_lang_name = st.selectbox(
            "Translate text into:",
            translator_langs,
            index=translator_langs.index("English") if "English" in translator_langs else 0
        )

    # --- 3. File Upload ---
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
                    # Determine language code
                    whisper_lang_code = None
                    if source_lang_name != "Auto-Detect":
                        whisper_lang_code = WHISPER_LANGUAGES[source_lang_name]

                    # Run Transcription
                    result = model.transcribe(tmp_path, language=whisper_lang_code)
                    original_text = result["text"]
                    
                    st.success("Transcription Complete!")
                    
                    # Show Original Text
                    st.markdown("### 📝 Original Transcript")
                    st.text_area("Original", original_text, height=200)
                    
                    # Download Original
                    st.download_button(
                        label="Download Original Transcript",
                        data=original_text,
                        file_name="transcript_original.txt",
                        mime="text/plain"
                    )

                # --- PHASE 2: TRANSLATION ---
                if enable_translation and target_lang_name:
                    with st.spinner(f"Translating to {target_lang_name}..."):
                        # Initialize the translator
                        # We map the capitalized name back to lower case for the tool
                        translator = GoogleTranslator(source='auto', target=target_lang_name.lower())
                        
                        # Translate the text (handling long texts by splitting if necessary is handled by the lib usually, 
                        # but for very long audio, chunking might be needed. This works for standard usage.)
                        translated_text = translator.translate(original_text)
                        
                        st.markdown(f"### 🌐 Translated Transcript ({target_lang_name})")
                        st.text_area("Translated", translated_text, height=200)
                        
                        # Download Translated
                        st.download_button(
                            label=f"Download {target_lang_name} Translation",
                            data=translated_text,
                            file_name=f"transcript_{target_lang_name}.txt",
                            mime="text/plain"
                        )

            except Exception as e:
                st.error(f"An error occurred: {e}")
            
            finally:
                # Clean up
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

if __name__ == "__main__":
    main()
