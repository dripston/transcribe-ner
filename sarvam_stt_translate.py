import pyaudio
import wave
import requests
import time
import threading
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Sarvam AI configuration
API_KEY = os.getenv('SARVAM_API_KEY')
API_URL = os.getenv('SARVAM_API_URL', 'https://api.sarvam.ai/speech-to-text-translate')
MODEL = os.getenv('SARVAM_MODEL', 'saaras:v2.5')

# Audio recording configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "recording.wav"

class AudioRecorder:
    def __init__(self):
        self.recording = False
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []

    def start_recording(self):
        """Start recording audio"""
        self.frames = []
        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        self.recording = True
        print("Recording started... Press Ctrl+C to stop")
        
        while self.recording:
            try:
                data = self.stream.read(CHUNK)
                self.frames.append(data)
            except Exception as e:
                print(f"Error recording: {e}")
                break

    def stop_recording(self):
        """Stop recording audio"""
        self.recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        print("Recording stopped")

    def save_recording(self):
        """Save the recorded audio to a WAV file"""
        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        print(f"Audio saved as {WAVE_OUTPUT_FILENAME}")

def send_to_sarvam():
    """Send recorded audio to Sarvam AI for translation"""
    try:
        with open(WAVE_OUTPUT_FILENAME, 'rb') as audio_file:
            files = {
                'file': (WAVE_OUTPUT_FILENAME, audio_file, 'audio/wav')
            }
            data = {
                'model': MODEL
            }
            headers = {
                'api-subscription-key': API_KEY
            }
            
            print("Sending audio to Sarvam AI for translation...")
            response = requests.post(API_URL, files=files, data=data, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                print("Translation result:")
                # Extract the transcript from the response
                transcript = result.get('transcript', 'No transcript found')
                language_code = result.get('language_code', 'Unknown')
                print(f"Text: {transcript}")
                print(f"Detected Language: {language_code}")
                return result
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
                return None
    except Exception as e:
        print(f"Error sending to Sarvam AI: {e}")
        return None

def record_audio():
    """Record audio for a specified duration"""
    recorder = AudioRecorder()
    
    # Start recording in a separate thread
    record_thread = threading.Thread(target=recorder.start_recording)
    record_thread.start()
    
    # Record for specified duration
    time.sleep(RECORD_SECONDS)
    
    # Stop recording
    recorder.stop_recording()
    record_thread.join()
    
    # Save the recording
    recorder.save_recording()

def main():
    """Main function to record audio and translate it"""
    # Check if API key is available
    if not API_KEY:
        print("Error: SARVAM_API_KEY environment variable not set")
        print("Please set your Sarvam API key in the environment variables")
        print("Example: export SARVAM_API_KEY=your_key_here")
        return
    
    print("Sarvam AI Speech-to-Text Translation")
    print("=====================================")
    
    try:
        # Record audio
        record_audio()
        
        # Send to Sarvam AI
        result = send_to_sarvam()
        
        if result:
            print("\nTranslation successful!")
        else:
            print("\nTranslation failed!")
            
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()