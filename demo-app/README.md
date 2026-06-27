# EEG Biometric Demo App

FastAPI web demo for EEG biometric identification and verification using preprocessed `.npy` signals.

## Input

Upload a NumPy `.npy` file containing either:

- one preprocessed EEG window shaped `(64, 320)`, or
- a batch shaped `(N, 64, 320)`.

The app does not preprocess raw EEG files in this first version.

## Prediction

The form lets you select a discovered local configuration from:

- `../Dataset/preprocessed_research_final_v4_90/`
- `../models/`

The app returns:

- predicted subject ID,
- vote count,
- mean and max similarity,
- top matches per uploaded window,
- verification accept/reject when a claimed subject ID is provided.

Verification uses fixed threshold `0.6216`.

## Run

```bash
cd demo-app
uvicorn main:app --reload
```

Open <http://127.0.0.1:8000/>.

## Create a sample upload from the test set

```bash
cd /home/chocomaltt/Kuliah/eeg-biometric-system
python - <<'PY'
from pathlib import Path
import numpy as np
base = Path('Dataset/preprocessed_research_final_v4_90')
X = np.load(base / 'X_eo_test_2_1_seed42.npy')
out = Path('demo-app/sample_eo_2_1_seed42.npy')
np.save(out, X[0])
print(out)
PY
```

Upload `demo-app/sample_eo_2_1_seed42.npy` and select `EO | window 2s | stride 1s | seed 42`.

The first prediction for a configuration can take longer because the app builds the local embedding gallery cache.
