# EEG Biometric Demo App

FastAPI web demo for EEG biometric identification and verification using raw EO EEG files from `../Dataset/files/`. Prediction uses an existing Qdrant collection for HNSW nearest-neighbor search with Euclidean distance.

## Input

Select a subject folder such as `S001` from the form. The app loads that subject's EO/R01 file from:

```text
../Dataset/files/<subject>/<subject>R01.edf
```

The app preprocesses the raw EDF during prediction:

- remove `.` from channel names,
- crop to the first 60 seconds when longer than 60 seconds,
- create 1.5 second windows with 0.5 second stride,
- apply Butterworth bandpass filtering from 4–40 Hz with order 5,
- apply z-score normalization along each window's time axis.

## Prediction

The app uses one fixed local model/configuration and queries Qdrant for enrollment matches:

- model: `../models/embedding_v4_eo_train_80_0_1.5_0.5_b128_e100_margin_0.2.pth`
- Qdrant URL: `http://localhost:6333` by default, override with `QDRANT_URL`
- Qdrant collection: `embedding_v4_eo_train_80_0_1.5_0.5_b128_e100_margin_0.2` by default, override with `QDRANT_COLLECTION`
- Qdrant collection distance: Euclidean (`Distance.EUCLID`)
- display name: `EO | window 1.5s | stride 0.5s | seed 0`

The app validates that the configured Qdrant collection exists and uses Euclidean distance before querying it. Search requests use Qdrant HNSW search parameters with `exact=False`.

The app returns:

- predicted subject ID,
- vote count,
- mean and minimum Euclidean distance,
- top matches per preprocessed window,
- verification accept/reject when a claimed subject ID is provided.

For Euclidean distance, lower is better. Verification accepts when the predicted subject matches the claim and the minimum distance is at or below the threshold. The default distance threshold is `0.6216`; override it with `VERIFICATION_DISTANCE_THRESHOLD`.

## Run

Start Qdrant and ensure the collection exists before running the app.

```bash
cd demo-app
uv run uvicorn main:app --reload
```

Open <http://127.0.0.1:8000/>.
