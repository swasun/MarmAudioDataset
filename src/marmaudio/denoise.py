# Authors: Charly Lamothe <charlylmth_AT_gmail_DOT_com>; Paul Best <paulobest25_AT_@gmail_DOT_com>

# License: BSD (3-clause)

import noisereduce as nr
from scipy.signal import butter, lfilter


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    if highcut > int(fs / 2):
        print("[WARNING] Highcut is too high for bandpass filter. Setting to nyquist")
        highcut = int(fs / 2)
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def denoise(signal, sampling_rate):
    denoised_signal = butter_bandpass_filter(signal, 1000, 47000, sampling_rate, order=5)
    denoised_signal = nr.reduce_noise(y=denoised_signal, sr=sampling_rate) # Denoise the signal if needed
    return denoised_signal
