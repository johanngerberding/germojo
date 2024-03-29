import sys
import wave

import pyaudio
import whisper
from TTS.api import TTS

device = "cpu"
# Load models

model = whisper.load_model("base")
tts = TTS("tts_models/de/thorsten/tacotron2-DDC").to(device)

FORMAT = pyaudio.paInt16 
CHANNELS = 1 if sys.platform == 'darwin' else 2 
CHUNK = 1024 
RECORD_SECONDS = 5 

with wave.open("output.wav", "wb") as wf: 
    p = pyaudio.PyAudio()
    sample_rate = p.get_default_input_device_info()['defaultSampleRate']
    
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(int(sample_rate))

    stream = p.open(format=FORMAT, channels=CHANNELS, rate=int(sample_rate), input=True)

    print("Recording...")
    for _ in range(0, int(sample_rate) // CHUNK * RECORD_SECONDS): 
        wf.writeframes(stream.read(CHUNK))
    print("Done.")

    stream.close()
    p.terminate()


result = model.transcribe("output.wav")
print(result["text"])

tts.tts_with_vc_to_file(
    result["text"], 
    speaker_wav="voice.wav",
    file_path="generated_output.wav"
)