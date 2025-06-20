from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile
import os

# Load the Whisper model (use "base", "small", etc. depending on your preference)
model = WhisperModel("base", compute_type="int8")

DURATION = 5  # duration in seconds
SAMPLE_RATE = 16000  # sampling rate

def record_audio(duration=DURATION, sample_rate=SAMPLE_RATE):
    print("Recording audio...")
    try:
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()
        print("Audio recording complete.")
        return audio.squeeze()
    except Exception as e:
        print(f"Error during recording: {e}")
        return np.array([])

def save_temp_wav(audio_data, sample_rate=SAMPLE_RATE):
    print("Saving audio to temp WAV file...")
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        wav.write(temp_file.name, sample_rate, audio_data)
        print(f"File saved at: {temp_file.name}")
        return temp_file.name
    except Exception as e:
        print(f"Error saving WAV file: {e}")
        return None

def transcribe_audio_whisper(audio_path):
    print("Transcribing audio with Faster-Whisper...")
    try:
        segments, _ = model.transcribe(audio_path)
        result_text = " ".join([segment.text for segment in segments])
        print("Transcription result:", result_text)
        return result_text
    except Exception as e:
        print(f"Error during transcription: {e}")
        return ""

def listen_and_transcribe():
    audio_data = record_audio()
    if audio_data is None or len(audio_data) == 0:
        print("No audio data captured.")
        return ""

    audio_file = save_temp_wav(audio_data)
    if not audio_file:
        print("Failed to save audio file.")
        return ""

    try:
        transcription = transcribe_audio_whisper(audio_file)
        return transcription
    finally:
        if os.path.exists(audio_file):
            os.remove(audio_file)
            print("Temporary file removed.")

# ✅ FIXED: Correct Python main entry point
if __name__ == "__main__":
    transcription = listen_and_transcribe()
    if transcription:
        print(f"Transcription: {transcription}")
    else:
        print("No transcription available.")
