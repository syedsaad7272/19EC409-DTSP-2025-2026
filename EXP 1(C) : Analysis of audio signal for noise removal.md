# EXP 1(C) : Analysis of audio signal for noise removal
## SYED SAAD 212223060283

# AIM: 

To analyse an audio signal and remove noise

# APPARATUS REQUIRED: 

PC installed with SCILAB. 

# PROGRAM: 
~~~
# ==============================
# AUDIO NOISE MIXING + NOISE REDUCTION
# ==============================

!pip install -q librosa noisereduce soundfile

import librosa
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio, display
import noisereduce as nr
from google.colab import files
import soundfile as sf

# ——— Step 1: Upload files ———
print("Upload *clean* (speech / music) audio:")
uploaded = files.upload()
clean_file = next(iter(uploaded.keys()))

print("Upload *noise-only* (background) audio:")
uploaded = files.upload()
noise_file = next(iter(uploaded.keys()))

# ——— Step 2: Load audio ———
clean, sr_c = librosa.load(clean_file, sr=None, mono=True)
noise, sr_n = librosa.load(noise_file, sr=None, mono=True)

# Resample noise if needed
if sr_c != sr_n:
    print(f"Resampling noise from {sr_n} → {sr_c}")
    noise = librosa.resample(noise, orig_sr=sr_n, target_sr=sr_c)
    sr_n = sr_c

sr = sr_c
print("Sampling rate:", sr)
print("Clean duration (s):", len(clean)/sr)
print("Noise duration (s):", len(noise)/sr)

# ——— Step 3: Align / shift / tile noise ———

# Optionally, circular‑shift noise by random amount so you don’t always start at same point
shift = np.random.randint(0, len(noise))
noise = np.roll(noise, shift)

# Tile or truncate so noise length matches clean
if len(noise) < len(clean):
    reps = int(np.ceil(len(clean) / len(noise)))
    noise = np.tile(noise, reps)
noise = noise[:len(clean)]

# ——— Step 4: Mix with a scaling factor ———
scale = 0.7  # you can increase this if you don't hear noise
noisy = clean + noise * scale

# Optional: mild normalization to prevent clipping
max_abs = np.max(np.abs(noisy))
if max_abs > 1.0:
    noisy = noisy / max_abs

print("Generated noisy mixture.")

# ——— Step 5: Diagnostics / plotting & inspection ———

# Function to plot waveform
def plot_waveforms(signals, labels, sr, title="Waveforms"):
    plt.figure(figsize=(12, 4))
    t = np.arange(signals[0].shape[0]) / sr
    for sig, lab in zip(signals, labels):
        plt.plot(t, sig, label=lab, alpha=0.7)
    plt.legend()
    plt.xlabel("Time (s)")
    plt.title(title)
    plt.tight_layout()
    plt.show()

print("\nWaveform overlay (clean + scaled noise + mixture):")
plot_waveforms(
    [clean, noise * scale, noisy],
    ["clean", "noise × scale", "mixture"],
    sr,
    title="Clean vs Scaled Noise vs Mixture"
)

# Spectral / frequency analysis
def plot_spectrum(signal, sr, title="Spectrum"):
    n_fft = 2**14
    Y = np.fft.rfft(signal, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1.0/sr)
    mag = np.abs(Y)
    plt.figure(figsize=(10, 4))
    plt.semilogy(freqs, mag + 1e-12)
    plt.xlim(0, sr/2)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (log)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_spectrum(clean, sr, "Spectrum: Clean")
plot_spectrum(noise * scale, sr, "Spectrum: Noise × scale")
plot_spectrum(noisy, sr, "Spectrum: Mixture (noisy)")

# ——— Step 6: Playback (raw mixture + components) ———
print("\n--- Clean Audio ---")
display(Audio(clean, rate=sr))
print("\n--- Noise (scaled) ---")
display(Audio(noise * scale, rate=sr))
print("\n--- Noisy Mixture ---")
display(Audio(noisy, rate=sr))

# ——— Step 7: Noise reduction / denoising ———
reduced = nr.reduce_noise(y=noisy, y_noise=noise * scale, sr=sr)
estimated_noise = noisy - reduced

print("\n--- Denoised (cleaned) Audio ---")
display(Audio(reduced, rate=sr))
print("\n--- Extracted Noise Component ---")
display(Audio(estimated_noise, rate=sr))

# ——— Step 8: Spectra of outputs ———
plot_spectrum(reduced, sr, "Spectrum: Denoised Output")
plot_spectrum(estimated_noise, sr, "Spectrum: Extracted Noise")

# ——— Step 9: Save outputs ———
sf.write("noisy_mixture.wav", noisy, sr)
sf.write("denoised_output.wav", reduced, sr)
sf.write("extracted_noise.wav", estimated_noise, sr)

print("Saved files: noisy_mixture.wav, denoised_output.wav, extracted_noise.wav")

~~~

# clean Audio
[download.wav](https://github.com/user-attachments/files/22882144/download.wav)

# Noise (scaled)
[download (1).wav](https://github.com/user-attachments/files/22882155/download.1.wav)

# Noisy Mixture
[download (2).wav](https://github.com/user-attachments/files/22882192/download.2.wav)

# Denoised (cleaned) Audio
[download (3).wav](https://github.com/user-attachments/files/22882196/download.3.wav)

# Extracted Noise Component
[download (4).wav](https://github.com/user-attachments/files/22882200/download.4.wav)


# Output:

<img width="1277" height="559" alt="Screenshot 2025-10-13 153637" src="https://github.com/user-attachments/assets/d8dddd12-bed5-416a-94e6-fb38d8ecadf6" />
<img width="1099" height="793" alt="Screenshot 2025-10-13 153645" src="https://github.com/user-attachments/assets/58a64121-2361-40e8-8403-08a48fd67f86" />
<img width="971" height="414" alt="Screenshot 2025-10-13 153651" src="https://github.com/user-attachments/assets/47763fa3-c280-4784-bcfe-d1f47fd1b4a4" />
<img width="1085" height="378" alt="Screenshot 2025-10-13 153657" src="https://github.com/user-attachments/assets/b10c99dc-f194-4bc7-b2ba-33173b2ae7be" />
<img width="1034" height="390" alt="Screenshot 2025-10-13 153711" src="https://github.com/user-attachments/assets/aa6a0ddc-a825-4066-b806-86fcdf12ce37" />

# RESULT: 

Analysis of audio signal for noise removal is successfully executed in co lab
