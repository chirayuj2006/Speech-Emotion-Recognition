import sounddevice as sd
import queue

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_DURATION = 0.1
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION)

audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_queue.put(indata.copy().flatten())

def start_stream():
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        blocksize=BLOCK_SIZE,
        callback=audio_callback
    )
    stream.start()
    return stream, audio_queue
