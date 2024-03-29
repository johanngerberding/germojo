import pyaudio
import wave 
import sys 


FORMAT = pyaudio.paInt8 
CHANNELS = 1 if sys.platform == 'darwin' else 2 
RATE = 44100 
CHUNK = 1024 
RECORD_SECONDS = 5 
WAVE_OUTPUT_FILENAME = "test.wav"

with wave.open("output.wav", "wb") as wf: 
    p = pyaudio.PyAudio()
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)

    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True)

    print("Recording...")
    for _ in range(0, RATE // CHUNK * RECORD_SECONDS): 
        wf.writeframes(stream.read(CHUNK))
    print("Done.")

    stream.close()
    p.terminate()
