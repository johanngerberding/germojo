import sys
import wave
from string import Template

import pyaudio
import whisper
from TTS.api import TTS
from vllm import LLM, SamplingParams

# pyaudio config
FORMAT = pyaudio.paInt16 
CHANNELS = 1 if sys.platform == 'darwin' else 2 
CHUNK = 1024 
RECORD_SECONDS = 5 

# LLM params
sampling_params = SamplingParams(temperature=0.25, top_p=0.95, max_tokens=256)
template = Template("""$personality\nUser: $prompt\nAsssistant:""")
personality = "Du bist ein hilfreicher Assistent mit dem Namen Mojo. Du bist hilfreich und humorvoll."
device = "cpu"
model = "TheBloke/Llama-2-13B-German-Assistant-v4-AWQ"

# Init models
llm = LLM(model=model, quantization="awq")
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

outputs = llm.generate(
    prompts=template.substitute(personality=personality, prompt=transcription),
    sampling_params=sampling_params,
)

text_to_speech.tts_with_vc_to_file(
    outputs[0].outputs[0].text, 
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
