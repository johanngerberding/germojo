import sys
import wave
import requests
from string import Template

import pyaudio
import whisper
from transformers import AutoTokenizer, pipeline
from TTS.api import TTS

llm_url = "localhost:8081/v1/completions"

# pyaudio config
FORMAT = pyaudio.paInt16 
CHANNELS = 1 if sys.platform == 'darwin' else 2 
CHUNK = 1024 
RECORD_SECONDS = 5 

template = Template("""$personality\nUser: $prompt\nAsssistant:""")
personality = "Du bist ein hilfreicher Assistent mit dem Namen Mojo. Du bist hilfreich und humorvoll."
device = "cpu"
model = "flozi00/Mistral-7B-german-assistant-v4"
tokenizer = AutoTokenizer.from_pretrained(model)

# init models
speech_to_text = whisper.load_model("base")
text_to_speech = TTS("tts_models/de/thorsten/tacotron2-DDC").to(device)

p = pyaudio.PyAudio()
sample_rate = p.get_default_input_device_info()['defaultSampleRate']

with wave.open("output.wav", "wb") as wf: 
    
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(int(sample_rate))

    stream = p.open(format=FORMAT, channels=CHANNELS, rate=int(sample_rate), input=True)

    print("Recording...")
    for _ in range(0, int(sample_rate) // CHUNK * RECORD_SECONDS): 
        wf.writeframes(stream.read(CHUNK))
    print("Done.")

    stream.close()


result = speech_to_text.transcribe("output.wav")
transcription = result["text"]

# call llm api
response = requests.post(
    url=llm_url, 
    data={
        "model": "flozi00/Mistral-7B-german-assistant-v4", 
        "prompt": str(transcription), 
        "temperature": 0.0, 
        "max_tokens": 512
    },
)

llm_output = response 

text_to_speech.tts_with_vc_to_file(
    llm_output, 
    speaker_wav="voice.wav",
    file_path="generated_output.wav"
)

with wave.open("generated_output.wav", "rb") as wf: 
    stream = p.open(
        format=p.get_format_from_width(wf.getsampwidth()),
        channels=wf.getnchannels(),
        rate=wf.getframerate(),
        output=True,
    )
    data = wf.readframes(CHUNK)
    while data: 
        stream.write(data)
        data = wf.readframes(CHUNK)

    stream.close()

p.terminate()
