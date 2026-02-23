import mne
import os
import numpy as np

# Remove dot (.) from channels name
def ch_rename(path, ch_names):
    """
        Remove dots on channels name.
    """
    raw = mne.io.read_raw_edf(os.path.join(path,ch_names), preload=True, verbose=False)
    ch_names = raw.ch_names
    ch_names = [ch.replace('.', '') for ch in ch_names]
    raw.rename_channels(dict(zip(raw.ch_names, ch_names)))

    return raw

def sliding_windows(raw_data, sampling_rate=250, window_sec=2, overlap_sec=1):
    """
    Sliding windows with stride.
    """
    all_segments = []

    for data in raw_data:
        sfreq = int(data.info['sfreq'])

        eeg_data = data.get_data()

        window_size = window_sec * sfreq
        stride_size = (window_size - overlap_sec) * sfreq

        for start in range(0, eeg_data.shape[1] - window_size + 1, stride_size):
            end = start + window_size
            all_segments.append(eeg_data[:, start:end])

    return all_segments