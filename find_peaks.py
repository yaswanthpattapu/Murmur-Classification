import numpy as np
import scipy.signal as signal
import soundfile as sf
import matplotlib.pyplot as plt

def find_peaks_scipy(audio, fs=4000, height=0, distance=0.15):
    peaks, _ = signal.find_peaks(audio, distance=fs*distance, height=height)
    return peaks

def find_peaks(audio, fs=4000, height=0, distance=0.15):
    distance = int(distance*fs)
    peaks = []
    peak = 0
    for i in range(0, len(audio)):
        if i == 0:
            if audio[i] > audio[i+1] and audio[i] > height:
                peaks.append(i)
                peak = i
        elif i == len(audio)-1:
            if audio[i] > audio[i-1] and audio[i] > height and i-peak > distance:
                peaks.append(i)
                peak = i
        elif audio[i] > audio[i-1] and audio[i] > audio[i+1] and audio[i] > height and i-peak > distance:
            peaks.append(i)
            peak = i+1

    return peaks

file = "unsynced/raw_data/physionet_2022/50354_TV.wav"
height = 0.1
distance = 0.15

# Load the audio file
audio, fs = sf.read(file)
# audio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.5, 0.4, 0.3, 0.2, 0.3, 0.2, 0.3, 0.4, 0.5, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
# audio = np.array(audio)
# print(audio[1:5])

peaks_s = find_peaks_scipy(audio, fs=fs, height=height, distance=distance)
peaks = find_peaks(audio, fs=fs, height=height, distance=distance)

# Plot the audio waveform
plt.figure(figsize=(15, 6))
plt.plot(audio)
plt.plot(peaks, audio[peaks], "X", color="red")
plt.plot(peaks_s, audio[peaks_s], ".", color="green")
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.title('Audio Waveform')
plt.legend(["Audio", "Peaks (Implemented by me)", "Peaks (scipy)"])
plt.savefig("unsynced/pics/audio_waveform.png")
plt.show()

print(len(peaks_s), len(peaks))

# peaks_s = [round(a/fs, 3) for a in peaks_s]
# print(peaks_s)

# peaks = [round(a/fs, 3) for a in peaks]
# print(peaks)