import streamlit as st
import whisper
import os
import tempfile

def main():
    # 1. Title and Description
    st.title("🎙️Audio Transcriber")
    st.write("Upload an audio file (MP3, WAV, M4A) to transcribe it into text.")

    # 2. File Uploader
    uploaded_file = st.file_uploader("Choose a file", type=["mp3", "wav", "m4a", "mp4"])

    if uploaded_file is not None:
        # Create a button to start processing
        if st.button("Start Transcription"):
            
            # Show a spinner while loading (since it takes a few seconds)
            with st.spinner("Loading AI model... (This might take a minute the first time)"):
                model = whisper.load_model("base")

            with st.spinner("Transcribing audio..."):
                # Streamlit keeps files in memory, but Whisper needs a file path.
                # We save the uploaded file to a temporary location.
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                # Run the transcription
                try:
                    # 'language="es"' forces Spanish detection
                    result = model.transcribe(tmp_path, language="es")
                    transcription = result["text"]
                    
                    st.success("Transcription Complete!")
                    
                    # 3. Display Result
                    st.text_area("Transcription:", transcription, height=300)
                    
                    # 4. Download Button
                    st.download_button(
                        label="Download Text File",
                        data=transcription,
                        file_name="transcription.txt",
                        mime="text/plain"
                    )
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                
                finally:
                    # Clean up the temporary file
                    os.remove(tmp_path)

if __name__ == "__main__":
    main()
