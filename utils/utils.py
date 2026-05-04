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

def sliding_windows(raw_data):
    """
    Sliding windows with stride.
    """
    all_segments = []
    all_labels = []

    for subject_id, data in enumerate(raw_data):
        sfreq = int(data.info['sfreq'])

        eeg_data = data.get_data()

        window_size = int(float(os.getenv("WINDOW_SIZE")) * sfreq)
        stride_size = int(float(os.getenv("STRIDE")) * sfreq)

        for start in range(0, eeg_data.shape[1] - window_size + 1, stride_size):
            end = start + window_size
            all_segments.append(eeg_data[:, start:end])
            all_labels.append(subject_id)

    return all_segments, all_labels