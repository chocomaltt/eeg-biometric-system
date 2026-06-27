# Demo App Design: EEG Biometric Identification and Verification

Date: 2026-06-27

## Goal

Create a simple local demo app for EEG biometric prediction using existing preprocessed `.npy` signals from `../Dataset/preprocessed_research_final_v4_90/` and trained embedding models from `../models/`.

The first version accepts uploaded preprocessed `.npy` files, not raw EEG files. It supports both:

- **Identification:** predict the nearest subject ID.
- **Verification:** optionally compare the prediction against a claimed subject ID and accept/reject with a fixed threshold.

## Chosen Approach

Use a local FastAPI web form with local model loading and a local enrollment gallery built from the matching train arrays.

This avoids Qdrant for the demo. The app can run from files already present in the repository and dataset/model folders.

## User Decisions

- Upload format: preprocessed `.npy`.
- Prediction mode: both identification and verification.
- Model configuration: selectable from available dataset/model combinations.
- Interface: web form.
- Verification threshold: fixed at `0.6216`.

## Architecture

The app stays inside `demo-app/` and has three layers.

### Web/UI Layer

Serves a simple HTML form at `/` with:

- `.npy` file upload field.
- Dropdowns for available `data_type`, `window_size`, `stride`, and `seed` configurations.
- Optional claimed subject ID input for verification.
- Submit button.
- Result display for prediction, similarity, top matches, and verification decision.

### Prediction Service Layer

Responsible for:

- Loading the selected embedding model.
- Loading matching train arrays.
- Computing and caching enrollment embeddings.
- Validating uploaded `.npy` inputs.
- Running model inference.
- Computing cosine similarities.
- Returning identification and verification results.

### Configuration Discovery Layer

Scans available dataset and model filenames and only exposes complete configurations where all required files exist:

- Matching model file.
- Matching `X_*_train_*.npy` file.
- Matching `y_*_train_*.npy` file.

## Data Flow

1. User opens `/`.
2. App shows selectable configurations discovered from files.
3. User uploads a preprocessed `.npy` signal.
4. App validates the upload shape:
   - single window: `(64, 320)`, or
   - batch: `(N, 64, 320)`.
5. App loads the selected model from `../models/`.
6. App loads matching enrollment arrays from `../Dataset/preprocessed_research_final_v4_90/`:
   - `X_{data_type}_train_{window}_{stride}_seed{seed}.npy`
   - `y_{data_type}_train_{window}_{stride}_seed{seed}.npy`
7. App computes gallery embeddings from train data on first use.
8. App embeds uploaded samples.
9. App computes cosine similarity between uploaded embeddings and gallery embeddings.
10. App identifies the nearest subject by top similarity.
11. If a claimed subject ID is supplied, app verifies using:
    - predicted subject equals claimed subject, and
    - similarity is at least `0.6216`.

For batch uploads, the app returns per-window predictions and an aggregate result using majority vote over predicted subjects plus summary similarity values.

## Planned Files

### `demo-app/main.py`

FastAPI application and routes:

- `GET /` renders the upload form.
- `POST /predict` processes the upload and renders results.
- `GET /health` returns a simple health check.

Routing should stay thin and delegate prediction work to `prediction.py`.

### `demo-app/prediction.py`

Prediction logic:

- Discover valid model/dataset configurations.
- Load selected model.
- Load or build enrollment gallery embeddings.
- Validate `.npy` uploads.
- Run identification.
- Run optional verification.

### `demo-app/templates/index.html`

Minimal HTML UI:

- Upload field.
- Config dropdowns.
- Optional claimed subject field.
- Result section.
- Error section.

### `demo-app/pyproject.toml`

Add app dependencies:

- `fastapi`
- `uvicorn`
- `python-multipart`
- `jinja2`
- `numpy`
- `torch`

### `demo-app/.gitignore`

Ignore generated local cache files if embedding caches are written under `demo-app/.cache/`.

## Model and Dataset Naming

The app should use the existing naming conventions.

Dataset arrays:

```text
X_{data_type}_train_{window}_{stride}_seed{seed}.npy
y_{data_type}_train_{window}_{stride}_seed{seed}.npy
```

Model files:

```text
embedding_v3.1_{data_type}_train_80_{seed}_{window}_{stride}_b32_e100_margin_0.2.pth
```

The app should convert decimal window/stride values to the filename format already used in dataset arrays, for example:

- `2` + `1` -> `2_1`
- `1.5` + `0.5` -> `15_05`

Model filenames use decimal-style values, for example:

- `2_1`
- `1.5_0.5`

## Checkpoint Loading

The discovered `.pth` files are saved full PyTorch model objects, not only plain `state_dict` checkpoints. Because PyTorch 2.6 defaults `torch.load` to `weights_only=True`, implementation should load these trusted local project artifacts with `weights_only=False` after making the repository root importable so `utils.embedding_model.embedding_model` can be resolved.

The app must not accept uploaded model files. Only local model files discovered under `../models/` are loaded.

## Error Handling

The app should return clear user-facing errors for:

- No valid configurations found.
- Uploaded file is not `.npy`.
- Uploaded array has an unsupported shape.
- Selected configuration has missing dataset/model files.
- Model cannot be loaded.
- Prediction fails.

Errors should be shown in the web form rather than as an unformatted traceback.

## Performance and Caching

The first prediction for a configuration may be slow because the app must compute gallery embeddings from the train array.

To keep the demo usable:

- Cache loaded models in memory by configuration.
- Cache gallery embeddings in memory by configuration.
- Optionally persist gallery embeddings to `demo-app/.cache/` as compressed `.npz` files so subsequent app restarts are faster.

The initial implementation can use CPU by default and CUDA automatically if available.

## Testing and Verification

Implementation should include at least lightweight checks that verify:

- Config discovery finds expected complete configurations.
- Upload validation accepts `(64, 320)` and `(N, 64, 320)` arrays.
- Upload validation rejects unsupported shapes.
- Prediction works with one known test sample from `preprocessed_research_final_v4_90`.
- Web form route loads successfully.

Manual verification should run the app and test uploading a sample `.npy` file.

## Out of Scope for First Version

- Raw EEG preprocessing from EDF/CSV.
- Qdrant integration.
- User authentication.
- Database persistence.
- Advanced charts or dashboards.
- Training or retraining models.
