# EEG Biometric Demo App

FastAPI web demo for EEG biometric identification and verification. The app supports registered raw EO/R01 EDF recordings and unregistered BCICIV MATLAB calibration files, generates EEG embeddings, and searches enrolled embeddings in Qdrant using Euclidean distance.

## Public accounts and internal labels

Users do not enter internal numeric subject labels. The form provides an account selector populated from the registered `S###` folders under `../Dataset/files`.

Public identities are one-based while Qdrant labels remain zero-based:

```text
Subject 1 (S001)   ↔ Qdrant label 0
Subject 2 (S002)   ↔ Qdrant label 1
...
Subject 109 (S109) ↔ Qdrant label 108
```

The browser submits the public account code, such as `S001`. The backend validates it against the available account list and resolves it to Qdrant label `0`. Internal Qdrant labels are not shown as account identities.

Leaving the account selector at **No account claim — identification only** runs identification without producing an accepted/rejected verification decision.

## Input paths

### Registered raw EEG

Select a subject folder such as `S001`. The app loads:

```text
../Dataset/files/<subject>/<subject>R01.edf
```

Raw preprocessing:

- remove `.` from channel names;
- crop to the first 60 seconds when the recording is longer;
- create 2-second windows with 2-second stride;
- apply a 4–40 Hz order-5 Butterworth bandpass;
- apply per-window, per-channel z-score normalization;
- validate model input shape `(N, 64, 320)`.

The fixed raw configuration is:

- model: `../models/embedding_v4_eo_train_80_0_2_2_b128_e100_margin_0.2.pth`;
- display name: `EO | window 2s | stride 2s | seed 0`;
- default collection: `embedding_v4_eo_train_80_0_2_2_b128_e100_margin_0.2`.

Override the raw collection with `QDRANT_COLLECTION`.

### Unregistered BCICIV access attempts

The BCICIV section loads calibration files from:

```text
../Dataset/BCICIV_1calib_1000Hz_mat
```

BCICIV files are treated as unregistered users attempting to authenticate against a selected registered account. Choose one evaluation checkpoint:

- EO: `embedding_v4_eo_train_80_0_2_1_b128_e100_margin_0.2.pth`;
- EC: `embedding_v4_ec_train_80_0_2_1_b128_e100_margin_0.2.pth`.

EO/EC selects the model and Qdrant collection. It does not split the BCICIV input recording into EO and EC data.

BCICIV preprocessing:

- load continuous `cnt` data from MATLAB;
- create 2-second windows with 1-second stride;
- apply a 4–40 Hz bandpass;
- resample each window to 320 samples;
- apply z-score normalization;
- align 59 BCICIV channels to the model's 64-channel order;
- zero-fill unavailable model channels.

Marker labels are loaded as metadata but are not used to segment authentication input.

## Prediction and verification

For each preprocessed window, the app:

1. generates an embedding;
2. queries the selected Euclidean Qdrant collection;
3. keeps the top matches and distance;
4. selects the most frequent top-1 subject across all windows;
5. calculates mean and minimum distance.

The result displays public account identities such as `Subject 1 (S001)` for the final prediction and every per-window top match.

When an account is selected, verification uses internal labels and accepts only when:

```python
predicted_internal_label == claimed_internal_label and mean_distance <= threshold
```

The default threshold is `0.5828`. Override it with:

```bash
VERIFICATION_DISTANCE_THRESHOLD=0.5
```

Lower Euclidean distance means a closer match.

## Qdrant configuration

Default URL:

```text
http://localhost:6333
```

Override it with `QDRANT_URL`. The app validates that the selected collection exists and uses Euclidean distance before querying it. Search requests use HNSW with `exact=False`.

## Run

Start Qdrant and ensure the required collections exist, then run:

```bash
cd demo-app
uv run uvicorn main:app --reload
```

Open <http://127.0.0.1:8000/>.

## Tests

```bash
cd demo-app
uv run pytest -v
```
