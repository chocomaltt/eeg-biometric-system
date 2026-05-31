import mne
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv("../.env")

WINDOWS_SETUP=[
    [1, 0.5],
    [1, 1],
    [1.5, 0.5],
    [1.5, 1],
    [1.5, 1.5],
    [2, 0.5],
    [2, 1],
    [2, 1.5],
    [2, 2]
]

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

def sliding_windows(raw_data, window_size, stride, sfreq):
    """
    Sliding windows with stride.
    """
    all_segments = []
    all_labels = []

    for i in range (int(os.getenv("NUM_CLASSES"))):
        # print(sfreq)
        # sfreq = int(sfreq)
        eeg_data = raw_data[i]

        window_samples = int(window_size * int(sfreq))
        stride_samples = int(stride * int(sfreq))

        for start in range(0, eeg_data.shape[1] - window_samples + 1, stride_samples):
            end = start + window_samples
            all_segments.append(eeg_data[:, start:end])
            all_labels.append(i)

    return all_segments, all_labels