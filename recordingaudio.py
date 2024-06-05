import os
import wave
import pyaudio


def process(fname):
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 5

    audio = pyaudio.PyAudio()

    try:
        # Start recording
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )
        print("Recording...")
        frames = []

        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        print("Finished recording")

        # Stop the stream
        stream.stop_stream()
        stream.close()

        # Write to WAV file
        with wave.open(fname, "wb") as waveFile:
            waveFile.setnchannels(CHANNELS)
            waveFile.setsampwidth(audio.get_sample_size(FORMAT))
            waveFile.setframerate(RATE)
            waveFile.writeframes(b"".join(frames))

    except Exception as e:
        print(f"Error during recording: {e}")

    finally:
        audio.terminate()
