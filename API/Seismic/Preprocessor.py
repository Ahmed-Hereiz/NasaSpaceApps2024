import pandas as pd
import numpy as np
from scipy.signal import stft, butter, filtfilt

class SignalPreprocessor:
    def __init__(self, csv_file, signal_column):

        self.data = pd.read_csv(csv_file)
        self.signal = self.data[signal_column].values
    
    def perform_stft(self, fs, nperseg):

        f, t, Zxx = stft(self.signal, fs=fs, nperseg=nperseg)
        return f, t, Zxx
    
    def filter_bandwidth(self, lowcut, highcut, fs, order=5):

        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, self.signal)
        return filtered_signal


