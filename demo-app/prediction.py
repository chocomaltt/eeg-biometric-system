from __future__ import annotations

import io
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import mne
import numpy as np
import torch
from qdrant_client import QdrantClient, models
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, resample
from scipy.stats import zscore

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DATASET_DIR = PROJECT_ROOT / "Dataset" / "preprocessed_research_final_v4_90"
RAW_DATASET_DIR = PROJECT_ROOT / "Dataset" / "files"
BCICIV_DATASET_DIR = PROJECT_ROOT / "Dataset" / "BCICIV_1calib_1000Hz_mat"
BCICIV_MAT_PATTERN = re.compile(r"^BCICIV_calib_ds1[a-g]_1000Hz\.mat$")
RAW_PREPROCESSED_DIR = PROJECT_ROOT / "Dataset" / "preprocessed_research_final_v4_80"
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = APP_DIR / ".cache"
MODEL_PREFIX = "embedding_v4"
TRAIN_SPLIT = "80"
MODEL_SUFFIX = "b128_e100_margin_0.2.pth"
FIXED_MODEL_FILENAME = "embedding_v4_eo_train_80_0_2_2_b128_e100_margin_0.2.pth"
QDRANT_COLLECTION_NAME = "embedding_v4_eo_train_80_0_2_2_b128_e100_margin_0.2"
BCICIV_MODEL_OPTIONS = {
    "eo": "embedding_v4_eo_train_80_0_2_1_b128_e100_margin_0.2.pth",
    "ec": "embedding_v4_ec_train_80_0_2_1_b128_e100_margin_0.2.pth",
}
BCICIV_QDRANT_COLLECTIONS = {
    "eo": "embedding_v4_eo_train_80_0_2_1_b128_e100_margin_0.2",
    "ec": "embedding_v4_ec_train_80_0_2_1_b128_e100_margin_0.2",
}
QDRANT_URL = "http://localhost:6333"
SIGNAL_CHANNELS = 64
SIGNAL_SAMPLES = 320
FIXED_RAW_SIGNAL_SAMPLES = 320
VERIFICATION_THRESHOLD = 0.5828

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class PredictionError(Exception):
    """User-facing prediction error shown in the demo form."""


SUBJECT_CODE_PATTERN = re.compile(r"^S(?P<number>\d{3})$")


def subject_code_to_number(subject_code: str) -> int:
    match = SUBJECT_CODE_PATTERN.fullmatch(subject_code)
    if match is None:
        raise PredictionError(f"Invalid account code: {subject_code}")
    subject_number = int(match.group("number"))
    if subject_number < 1:
        raise PredictionError(f"Invalid account code: {subject_code}")
    return subject_number


def subject_code_to_qdrant_label(subject_code: str) -> int:
    return subject_code_to_number(subject_code) - 1


def qdrant_label_to_subject_number(subject_label: int) -> int:
    if isinstance(subject_label, bool):
        raise PredictionError(f"Invalid Qdrant subject label: {subject_label}")
    try:
        label = int(subject_label)
    except (TypeError, ValueError) as exc:
        raise PredictionError(f"Invalid Qdrant subject label: {subject_label}") from exc
    if label < 0 or label > 998:
        raise PredictionError(f"Invalid Qdrant subject label: {subject_label}")
    return label + 1


def qdrant_label_to_subject_code(subject_label: int) -> str:
    return f"S{qdrant_label_to_subject_number(subject_label):03d}"


def subject_display_from_qdrant_label(subject_label: int) -> str:
    number = qdrant_label_to_subject_number(subject_label)
    return f"Subject {number} ({qdrant_label_to_subject_code(subject_label)})"


def resolve_claimed_account(
    account_code: str,
    available_accounts: Sequence[str],
) -> int | None:
    normalized = account_code.strip()
    if not normalized:
        return None
    label = subject_code_to_qdrant_label(normalized)
    if normalized not in set(available_accounts):
        raise PredictionError(f"Account is not available: {normalized}")
    return label


def add_public_subject_fields(
    result: dict,
    available_accounts: Sequence[str] | None = None,
) -> dict:
    available = set(available_accounts) if available_accounts is not None else None

    def identity(label: int) -> tuple[int, str, str]:
        number = qdrant_label_to_subject_number(label)
        code = qdrant_label_to_subject_code(label)
        if available is not None and code not in available:
            raise PredictionError(f"Predicted account is not available: {code}")
        return number, code, f"Subject {number} ({code})"

    predicted_number, predicted_code, predicted_display = identity(result["predicted_subject_id"])
    result["predicted_subject_number"] = predicted_number
    result["predicted_subject_code"] = predicted_code
    result["predicted_subject_display"] = predicted_display

    if "claimed_subject_id" in result:
        claimed_number, claimed_code, claimed_display = identity(result["claimed_subject_id"])
        result["claimed_subject_number"] = claimed_number
        result["claimed_subject_code"] = claimed_code
        result["claimed_subject_display"] = claimed_display

    for window in result.get("windows", []):
        number, code, display = identity(window["predicted_subject_id"])
        window["predicted_subject_number"] = number
        window["predicted_subject_code"] = code
        window["predicted_subject_display"] = display
        for match in window.get("top_matches", []):
            match_number, match_code, match_display = identity(match["subject_id"])
            match["subject_number"] = match_number
            match["subject_code"] = match_code
            match["subject_display"] = match_display
    return result


@dataclass(frozen=True)
class ModelConfig:
    data_type: str
    window_token: str
    stride_token: str
    seed: int
    x_train_path: Path
    y_train_path: Path
    model_path: Path

    @property
    def config_id(self) -> str:
        return f"{self.data_type}:{self.window_token}:{self.stride_token}:seed{self.seed}"

    @property
    def window_label(self) -> str:
        return token_to_model_value(self.window_token)

    @property
    def stride_label(self) -> str:
        return token_to_model_value(self.stride_token)

    @property
    def display_name(self) -> str:
        return (
            f"{self.data_type.upper()} | window {self.window_label}s | "
            f"stride {self.stride_label}s | seed {self.seed}"
        )


@dataclass(frozen=True)
class BCICIVMatData:
    cnt: np.ndarray
    sfreq: float
    channel_names: list[str]
    marker_positions: np.ndarray
    marker_labels: np.ndarray
    classes: list[str]


def token_to_model_value(token: str) -> str:
    if token == "05":
        return "0.5"
    if token == "15":
        return "1.5"
    return token


def format_dataset_token(value: str) -> str:
    return value.replace(".", "")


def model_filename(data_type: str, seed: int, window_token: str, stride_token: str) -> str:
    window_value = token_to_model_value(window_token)
    stride_value = token_to_model_value(stride_token)
    return (
        f"{MODEL_PREFIX}_{data_type}_train_{TRAIN_SPLIT}_{seed}_"
        f"{window_value}_{stride_value}_{MODEL_SUFFIX}"
    )


def find_model_path(
    models_dir: Path,
    data_type: str,
    seed: int,
    window_token: str,
    stride_token: str,
) -> Path | None:
    exact_path = models_dir / model_filename(data_type, seed, window_token, stride_token)
    if exact_path.exists():
        return exact_path

    window_value = token_to_model_value(window_token)
    stride_value = token_to_model_value(stride_token)
    pattern = f"*_{data_type}_train_*_{seed}_{window_value}_{stride_value}_*.pth"
    candidates = sorted(models_dir.glob(pattern))
    if not candidates:
        return None

    preferred = [path for path in candidates if path.name.startswith(f"{MODEL_PREFIX}_")]
    return (preferred or candidates)[0]


def discover_configurations(
    dataset_dir: Path = DATASET_DIR,
    models_dir: Path = MODELS_DIR,
) -> list[ModelConfig]:
    pattern = re.compile(
        r"^X_(?P<data_type>eo|ec)_train_"
        r"(?P<window>[^_]+)_(?P<stride>[^_]+)_seed(?P<seed>\d+)\.npy$"
    )
    configs: list[ModelConfig] = []

    if not dataset_dir.exists() or not models_dir.exists():
        return []

    for x_path in sorted(dataset_dir.glob("X_*_train_*_seed*.npy")):
        match = pattern.match(x_path.name)
        if not match:
            continue

        data_type = match.group("data_type")
        window_token = match.group("window")
        stride_token = match.group("stride")
        seed = int(match.group("seed"))
        y_path = dataset_dir / f"y_{data_type}_train_{window_token}_{stride_token}_seed{seed}.npy"
        m_path = find_model_path(models_dir, data_type, seed, window_token, stride_token)

        if y_path.exists() and m_path is not None:
            configs.append(
                ModelConfig(
                    data_type=data_type,
                    window_token=window_token,
                    stride_token=stride_token,
                    seed=seed,
                    x_train_path=y_path.with_name(x_path.name),
                    y_train_path=y_path,
                    model_path=m_path,
                )
            )

    return sorted(
        configs,
        key=lambda c: (c.data_type, float(c.window_label), float(c.stride_label), c.seed),
    )


def get_config(
    config_id: str,
    configs: Sequence[ModelConfig] | None = None,
) -> ModelConfig:
    candidates = list(configs) if configs is not None else discover_configurations()
    for config in candidates:
        if config.config_id == config_id:
            return config
    raise PredictionError(f"Unknown configuration: {config_id}")


def fixed_eo_v4_config(
    dataset_dir: Path = RAW_PREPROCESSED_DIR,
    models_dir: Path = MODELS_DIR,
) -> ModelConfig:
    return ModelConfig(
        data_type="eo",
        window_token="2",
        stride_token="2",
        seed=0,
        x_train_path=dataset_dir / "X_eo_train_2_2_seed0.npy",
        y_train_path=dataset_dir / "y_eo_train_2_2_seed0.npy",
        model_path=models_dir / FIXED_MODEL_FILENAME,
    )


def bciciv_v4_config(
    data_type: str,
    dataset_dir: Path = RAW_PREPROCESSED_DIR,
    models_dir: Path = MODELS_DIR,
) -> ModelConfig:
    if data_type not in BCICIV_MODEL_OPTIONS:
        raise PredictionError(f"Unknown BCICIV model data type: {data_type}")

    return ModelConfig(
        data_type=data_type,
        window_token="2",
        stride_token="1",
        seed=0,
        x_train_path=dataset_dir / f"X_{data_type}_train_2_1_seed0.npy",
        y_train_path=dataset_dir / f"y_{data_type}_train_2_1_seed0.npy",
        model_path=models_dir / BCICIV_MODEL_OPTIONS[data_type],
    )


def bciciv_model_options() -> list[dict[str, str]]:
    return [
        {"value": "eo", "label": "EO checkpoint | window 2s | stride 1s | seed 0"},
        {"value": "ec", "label": "EC checkpoint | window 2s | stride 1s | seed 0"},
    ]


def list_raw_subjects(dataset_files_dir: Path = RAW_DATASET_DIR) -> list[str]:
    if not dataset_files_dir.exists():
        return []

    return sorted(
        path.name
        for path in dataset_files_dir.iterdir()
        if path.is_dir() and re.fullmatch(r"S\d{3}", path.name)
    )


def list_bciciv_mat_files(dataset_dir: Path = BCICIV_DATASET_DIR) -> list[str]:
    if not dataset_dir.exists():
        return []

    return sorted(
        path.name
        for path in dataset_dir.iterdir()
        if path.is_file() and BCICIV_MAT_PATTERN.fullmatch(path.name)
    )


def resolve_bciciv_mat_path(file_name: str, dataset_dir: Path = BCICIV_DATASET_DIR) -> Path:
    if not BCICIV_MAT_PATTERN.fullmatch(file_name):
        raise PredictionError(f"Invalid BCICIV file name: {file_name}")

    mat_path = dataset_dir / file_name
    if not mat_path.exists():
        raise PredictionError(f"BCICIV file not found: {file_name}")

    return mat_path


def _mat_field(obj: Any, field: str) -> Any:
    if hasattr(obj, field):
        return getattr(obj, field)
    if isinstance(obj, np.ndarray) and obj.dtype.names and field in obj.dtype.names:
        return obj[field].item()
    raise PredictionError(f"BCICIV .mat metadata is missing required field '{field}'")


def _as_string_list(value: Any) -> list[str]:
    array = np.asarray(value, dtype=object).ravel()
    return [str(item.item() if hasattr(item, "item") else item) for item in array]


def _as_int_array(value: Any) -> np.ndarray:
    if value is None:
        return np.array([], dtype=np.int64)
    return np.asarray(value).ravel().astype(np.int64, copy=False)


def load_bciciv_mat(file_path: Path) -> BCICIVMatData:
    try:
        mat = loadmat(file_path, squeeze_me=True, struct_as_record=False)
    except Exception as exc:
        raise PredictionError(f"Could not load BCICIV MATLAB file: {file_path.name}") from exc

    if "cnt" not in mat:
        raise PredictionError("BCICIV .mat file is missing required field 'cnt'")
    if "nfo" not in mat:
        raise PredictionError("BCICIV .mat file is missing required field 'nfo'")

    cnt = np.asarray(mat["cnt"], dtype=np.float32)
    if cnt.ndim != 2:
        raise PredictionError(f"BCICIV cnt must be 2D samples x channels; got {tuple(cnt.shape)}")

    nfo = mat["nfo"]
    sfreq = float(np.asarray(_mat_field(nfo, "fs")).squeeze())
    channel_names = _as_string_list(_mat_field(nfo, "clab"))
    if len(channel_names) != cnt.shape[1]:
        raise PredictionError(
            f"BCICIV channel metadata count {len(channel_names)} does not match cnt channels {cnt.shape[1]}"
        )

    classes = _as_string_list(_mat_field(nfo, "classes")) if hasattr(nfo, "classes") else []
    mrk = mat.get("mrk")
    marker_positions = _as_int_array(_mat_field(mrk, "pos")) if mrk is not None else np.array([], dtype=np.int64)
    marker_labels = _as_int_array(_mat_field(mrk, "y")) if mrk is not None else np.array([], dtype=np.int64)

    return BCICIVMatData(
        cnt=cnt,
        sfreq=sfreq,
        channel_names=channel_names,
        marker_positions=marker_positions,
        marker_labels=marker_labels,
        classes=classes,
    )


def subject_edf_path(subject_id: str, dataset_files_dir: Path = RAW_DATASET_DIR) -> Path:
    if not re.fullmatch(r"S\d{3}", subject_id):
        raise PredictionError(f"Invalid subject ID: {subject_id}")

    subject_dir = dataset_files_dir / subject_id
    if not subject_dir.is_dir():
        raise PredictionError(f"Subject folder not found: {subject_id}")

    edf_path = subject_dir / f"{subject_id}R01.edf"
    if not edf_path.exists():
        raise PredictionError(f"Raw EO file not found: {edf_path.name}")

    return edf_path


def load_raw_eo_subject(subject_id: str, dataset_files_dir: Path = RAW_DATASET_DIR):
    edf_path = subject_edf_path(subject_id, dataset_files_dir)
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except Exception as exc:
        raise PredictionError(f"Could not load raw EO file: {edf_path.name}") from exc

    renamed = {channel: channel.replace(".", "") for channel in raw.ch_names}
    raw.rename_channels(renamed)
    return raw


def normalize_channel_name(channel_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", channel_name).upper()


def model_channel_order(dataset_files_dir: Path = RAW_DATASET_DIR) -> list[str]:
    global _CHANNEL_ORDER_CACHE
    if _CHANNEL_ORDER_CACHE is not None and dataset_files_dir == RAW_DATASET_DIR:
        return list(_CHANNEL_ORDER_CACHE)

    subjects = list_raw_subjects(dataset_files_dir)
    if not subjects:
        raise PredictionError("Cannot infer model channel order because no raw subject folders were found")

    raw = load_raw_eo_subject(subjects[0], dataset_files_dir)
    channel_names = list(raw.ch_names)
    if len(channel_names) != SIGNAL_CHANNELS:
        raise PredictionError(
            f"Expected {SIGNAL_CHANNELS} channels while inferring model channel order; got {len(channel_names)}"
        )

    if dataset_files_dir == RAW_DATASET_DIR:
        _CHANNEL_ORDER_CACHE = channel_names
    return channel_names


def align_windows_to_model_channels(
    windows: np.ndarray,
    source_channel_names: Sequence[str],
    target_channel_names: Sequence[str],
) -> np.ndarray:
    if windows.ndim != 3:
        raise PredictionError(f"Expected BCICIV windows with shape (N, channels, samples); got {tuple(windows.shape)}")
    if windows.shape[1] != len(source_channel_names):
        raise PredictionError(
            f"BCICIV window channel count {windows.shape[1]} does not match metadata count {len(source_channel_names)}"
        )

    source_lookup = {normalize_channel_name(name): index for index, name in enumerate(source_channel_names)}
    aligned = np.zeros((windows.shape[0], len(target_channel_names), windows.shape[2]), dtype=np.float32)
    for target_index, target_name in enumerate(target_channel_names):
        source_index = source_lookup.get(normalize_channel_name(target_name))
        if source_index is not None:
            aligned[:, target_index, :] = windows[:, source_index, :]
    return aligned


def window_raw_signal(raw, window_size: float = 2, stride: float = 2) -> np.ndarray:
    sfreq = float(raw.info["sfreq"])
    window_samples = int(window_size * sfreq)
    stride_samples = int(stride * sfreq)
    data = raw.get_data()

    windows = [
        data[:, start : start + window_samples]
        for start in range(0, data.shape[1] - window_samples + 1, stride_samples)
    ]
    if not windows:
        raise PredictionError("Preprocessing produced no windows")

    return np.asarray(windows)


def butter_bandpass_filter(
    data: np.ndarray,
    lowcut: float = 4,
    highcut: float = 40,
    fs: float = 160,
    order: int = 5,
) -> np.ndarray:
    nyquist = 0.5 * fs
    b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype="band")
    filtered = np.zeros_like(data)
    for index in range(data.shape[0]):
        for channel in range(data.shape[1]):
            filtered[index, channel, :] = filtfilt(b, a, data[index, channel, :])
    return filtered


def preprocess_raw_eo_subject(
    subject_id: str,
    dataset_files_dir: Path = RAW_DATASET_DIR,
) -> np.ndarray:
    raw = load_raw_eo_subject(subject_id, dataset_files_dir)
    sfreq = float(raw.info["sfreq"])
    duration = raw.n_times / sfreq
    if duration > 60:
        raw.crop(tmin=0.0, tmax=60.0, include_tmax=False)

    windows = window_raw_signal(raw, window_size=2, stride=2)
    if windows.shape[1] != SIGNAL_CHANNELS:
        raise PredictionError(
            f"Expected {SIGNAL_CHANNELS} EEG channels after preprocessing; got {windows.shape[1]}"
        )

    filtered = butter_bandpass_filter(windows, lowcut=4, highcut=40, fs=sfreq, order=5)
    normalized = zscore(filtered, axis=2)
    normalized = np.nan_to_num(normalized, copy=False)
    return validate_signal_array(normalized.astype(np.float32, copy=False), expected_samples=FIXED_RAW_SIGNAL_SAMPLES)


def preprocess_bciciv_mat(
    file_name: str,
    dataset_dir: Path = BCICIV_DATASET_DIR,
    target_channel_names: Sequence[str] | None = None,
) -> np.ndarray:
    mat_path = resolve_bciciv_mat_path(file_name, dataset_dir)
    data = load_bciciv_mat(mat_path)
    sfreq = float(data.sfreq)
    if sfreq <= 0:
        raise PredictionError(f"BCICIV sampling rate must be positive; got {sfreq}")

    continuous = data.cnt.T
    window_samples = int(2 * sfreq)
    stride_samples = int(1 * sfreq)
    windows = [
        continuous[:, start : start + window_samples]
        for start in range(0, continuous.shape[1] - window_samples + 1, stride_samples)
    ]
    if not windows:
        raise PredictionError("Preprocessing produced no windows")

    window_array = np.asarray(windows, dtype=np.float32)
    filtered = butter_bandpass_filter(window_array, lowcut=4, highcut=40, fs=sfreq, order=5)
    if filtered.shape[2] != SIGNAL_SAMPLES:
        filtered = resample(filtered, SIGNAL_SAMPLES, axis=2).astype(np.float32, copy=False)

    normalized = zscore(filtered, axis=2)
    normalized = np.nan_to_num(normalized, copy=False).astype(np.float32, copy=False)
    aligned = align_windows_to_model_channels(
        normalized,
        source_channel_names=data.channel_names,
        target_channel_names=target_channel_names or model_channel_order(),
    )
    return validate_signal_array(aligned, expected_samples=SIGNAL_SAMPLES)


def array_from_npy_bytes(content: bytes) -> np.ndarray:
    try:
        return np.load(io.BytesIO(content), allow_pickle=False)
    except Exception as exc:
        raise PredictionError("Uploaded file must be a valid .npy file") from exc


def validate_signal_array(array: np.ndarray, expected_samples: int = SIGNAL_SAMPLES) -> np.ndarray:
    expected_shape = (SIGNAL_CHANNELS, expected_samples)
    if array.ndim == 2 and array.shape == expected_shape:
        return array[np.newaxis, :, :].astype(np.float32, copy=False)

    if array.ndim == 3 and array.shape[1:] == expected_shape:
        return array.astype(np.float32, copy=False)

    raise PredictionError(
        f"Expected array shape {expected_shape} for one window or (N, {SIGNAL_CHANNELS}, {expected_samples}) "
        f"for a batch; got {tuple(array.shape)}"
    )


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2:
        raise PredictionError(f"Expected 2D embeddings, got shape {tuple(values.shape)}")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return values / norms


def identify_from_embeddings(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    gallery_labels: np.ndarray,
    top_k: int = 5,
) -> dict:
    if len(gallery_embeddings) == 0:
        raise PredictionError("Enrollment gallery is empty")

    query = normalize_embeddings(query_embeddings)
    gallery = normalize_embeddings(gallery_embeddings)
    labels = np.asarray(gallery_labels)
    limit = max(1, min(top_k, len(gallery)))
    similarity_matrix = query @ gallery.T
    windows: list[dict] = []

    for row in similarity_matrix:
        top_indices = np.argsort(row)[::-1][:limit]
        best_index = int(top_indices[0])
        top_matches = [
            {
                "subject_id": int(labels[index]),
                "similarity": float(row[index]),
            }
            for index in top_indices
        ]
        windows.append(
            {
                "predicted_subject_id": int(labels[best_index]),
                "similarity": float(row[best_index]),
                "top_matches": top_matches,
            }
        )

    return {"windows": windows}


def qdrant_collection_name() -> str:
    return os.getenv("QDRANT_COLLECTION", QDRANT_COLLECTION_NAME)


def qdrant_url() -> str:
    return os.getenv("QDRANT_URL", QDRANT_URL)


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=qdrant_url())


def _qdrant_distance_name(distance: Any) -> str:
    value = getattr(distance, "value", distance)
    return str(value).lower()


def validate_qdrant_collection(client: QdrantClient, collection_name: str) -> None:
    try:
        collection = client.get_collection(collection_name)
    except Exception as exc:
        raise PredictionError(f"Qdrant collection '{collection_name}' is not available") from exc

    vectors = collection.config.params.vectors
    distance = getattr(vectors, "distance", None)
    if distance is None and isinstance(vectors, dict):
        distance = next((getattr(vector, "distance", None) for vector in vectors.values()), None)

    if "euclid" not in _qdrant_distance_name(distance):
        raise PredictionError(f"Qdrant collection '{collection_name}' must use Euclidean distance")


def identify_from_qdrant_embeddings(
    query_embeddings: np.ndarray,
    client: QdrantClient,
    collection_name: str,
    top_k: int = 5,
) -> dict:
    limit = max(1, top_k)
    search_params = models.SearchParams(hnsw_ef=128, exact=False)
    windows: list[dict] = []

    for embedding in np.asarray(query_embeddings, dtype=np.float32):
        try:
            result = client.query_points(
                collection_name=collection_name,
                query=embedding.tolist(),
                limit=limit,
                with_payload=True,
                search_params=search_params,
            )
        except Exception as exc:
            raise PredictionError(f"Qdrant search failed for collection '{collection_name}'") from exc

        points = result.points
        if not points:
            raise PredictionError(f"Qdrant collection '{collection_name}' returned no matches")

        top_matches = []
        for point in points:
            payload = point.payload or {}
            if "subject_id" not in payload:
                raise PredictionError("Qdrant match is missing subject_id payload")
            try:
                subject_label = int(payload["subject_id"])
            except (TypeError, ValueError) as exc:
                raise PredictionError("Qdrant subject_id payload must be an integer") from exc
            qdrant_label_to_subject_number(subject_label)
            top_matches.append({"subject_id": subject_label, "distance": float(point.score)})

        windows.append(
            {
                "predicted_subject_id": top_matches[0]["subject_id"],
                "distance": top_matches[0]["distance"],
                "top_matches": top_matches,
            }
        )

    return {"windows": windows}


def verification_threshold() -> float:
    return float(os.getenv("VERIFICATION_DISTANCE_THRESHOLD", VERIFICATION_THRESHOLD))


def aggregate_window_results(
    window_results: list[dict],
    claimed_subject_id: int | None = None,
) -> dict:
    if not window_results:
        raise PredictionError("No prediction windows were produced")

    vote_counts = Counter(int(window["predicted_subject_id"]) for window in window_results)
    predicted_subject_id, vote_count = vote_counts.most_common(1)[0]
    result = {
        "predicted_subject_id": int(predicted_subject_id),
        "vote_count": int(vote_count),
        "window_count": len(window_results),
        "windows": window_results,
    }

    if "distance" in window_results[0]:
        distances = np.array([float(window["distance"]) for window in window_results], dtype=np.float32)
        min_distance = float(np.min(distances))
        mean_distance = float(np.mean(distances))
        threshold = verification_threshold()
        result.update(
            {
                "mean_distance": mean_distance,
                "min_distance": min_distance,
            }
        )
        if claimed_subject_id is not None:
            accepted = predicted_subject_id == claimed_subject_id and mean_distance <= threshold
            result.update(
                {
                    "claimed_subject_id": int(claimed_subject_id),
                    "verification_threshold": threshold,
                    "accepted": bool(accepted),
                }
            )
        return result

    similarities = np.array([float(window["similarity"]) for window in window_results], dtype=np.float32)
    max_similarity = float(np.max(similarities))
    mean_similarity = float(np.mean(similarities))
    result.update(
        {
            "mean_similarity": float(np.mean(similarities)),
            "max_similarity": max_similarity,
        }
    )

    if claimed_subject_id is not None:
        threshold = verification_threshold()
        accepted = predicted_subject_id == claimed_subject_id and max_similarity >= threshold
        result.update(
            {
                "claimed_subject_id": int(claimed_subject_id),
                "verification_threshold": threshold,
                "accepted": bool(accepted),
            }
        )

    return result


@dataclass(frozen=True)
class Gallery:
    embeddings: np.ndarray
    labels: np.ndarray


_MODEL_CACHE: dict[str, tuple[Any, str]] = {}
_GALLERY_CACHE: dict[str, Gallery] = {}
_CHANNEL_ORDER_CACHE: list[str] | None = None


def clear_caches() -> None:
    global _CHANNEL_ORDER_CACHE
    _MODEL_CACHE.clear()
    _GALLERY_CACHE.clear()
    _CHANNEL_ORDER_CACHE = None


def _cache_path(config: ModelConfig) -> Path:
    safe_name = config.config_id.replace(":", "_")
    return CACHE_DIR / f"gallery_{safe_name}.npz"


def expected_samples_for_config(config: ModelConfig) -> int:
    try:
        train_array = np.load(config.x_train_path, allow_pickle=False, mmap_mode="r")
    except Exception as exc:
        raise PredictionError(f"Could not read training array shape: {config.x_train_path.name}") from exc

    if train_array.ndim != 3 or train_array.shape[1] != SIGNAL_CHANNELS:
        raise PredictionError(
            f"Training array for selected configuration must have shape (N, {SIGNAL_CHANNELS}, samples); "
            f"got {tuple(train_array.shape)}"
        )
    return int(train_array.shape[2])


def load_model(config: ModelConfig, device: str | None = None):
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cache_key = f"{config.config_id}:{selected_device}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if not config.model_path.exists():
        raise PredictionError(f"Model file not found: {config.model_path}")

    try:
        model = torch.load(
            config.model_path,
            map_location=selected_device,
            weights_only=False,
        )
        model.to(selected_device)
        model.eval()
    except Exception as exc:
        raise PredictionError(f"Could not load model: {config.model_path.name}") from exc

    _MODEL_CACHE[cache_key] = (model, selected_device)
    return model, selected_device


def embed_signals(
    model,
    signals: np.ndarray,
    device: str,
    batch_size: int = 128,
    expected_samples: int = SIGNAL_SAMPLES,
) -> np.ndarray:
    validated = validate_signal_array(signals, expected_samples=expected_samples)
    chunks: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(validated), batch_size):
            batch = torch.from_numpy(validated[start : start + batch_size]).float().to(device)
            embeddings = model(batch).detach().cpu().numpy()
            chunks.append(embeddings.astype(np.float32, copy=False))

    if not chunks:
        raise PredictionError("No signals were available for embedding")
    return np.concatenate(chunks, axis=0)


def load_or_build_gallery(config: ModelConfig, model, device: str) -> Gallery:
    if config.config_id in _GALLERY_CACHE:
        return _GALLERY_CACHE[config.config_id]

    cache_path = _cache_path(config)
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        gallery = Gallery(
            embeddings=normalize_embeddings(cached["embeddings"]),
            labels=cached["labels"].astype(np.int64, copy=False),
        )
        _GALLERY_CACHE[config.config_id] = gallery
        return gallery

    if not config.x_train_path.exists() or not config.y_train_path.exists():
        raise PredictionError("Training arrays for the selected configuration are missing")

    signals = validate_signal_array(
        np.load(config.x_train_path, allow_pickle=False),
        expected_samples=expected_samples_for_config(config),
    )
    labels = np.load(config.y_train_path, allow_pickle=False).astype(np.int64, copy=False)
    if len(signals) != len(labels):
        raise PredictionError(
            f"Training signal count {len(signals)} does not match label count {len(labels)}"
        )

    embeddings = embed_signals(model, signals, device, expected_samples=signals.shape[2])
    gallery = Gallery(embeddings=embeddings, labels=labels)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, embeddings=gallery.embeddings, labels=gallery.labels)
    _GALLERY_CACHE[config.config_id] = gallery
    return gallery


def predict_signal(
    config_id: str,
    array: np.ndarray,
    claimed_subject_id: int | None = None,
    top_k: int = 5,
) -> dict:
    config = get_config(config_id)
    expected_samples = expected_samples_for_config(config)
    signals = validate_signal_array(array, expected_samples=expected_samples)
    model, device = load_model(config)
    gallery = load_or_build_gallery(config, model, device)
    query_embeddings = embed_signals(model, signals, device, expected_samples=expected_samples)
    identified = identify_from_embeddings(
        query_embeddings=query_embeddings,
        gallery_embeddings=gallery.embeddings,
        gallery_labels=gallery.labels,
        top_k=top_k,
    )
    result = aggregate_window_results(
        identified["windows"],
        claimed_subject_id=claimed_subject_id,
    )
    result = add_public_subject_fields(result)
    result.update(
        {
            "config_id": config.config_id,
            "config_name": config.display_name,
            "device": device,
        }
    )
    return result


def predict_raw_subject(
    subject_id: str,
    claimed_subject_id: int | None = None,
    top_k: int = 5,
) -> dict:
    config = fixed_eo_v4_config()
    expected_samples = expected_samples_for_config(config)
    signals = preprocess_raw_eo_subject(subject_id)
    signals = validate_signal_array(signals, expected_samples=expected_samples)
    model, device = load_model(config)
    query_embeddings = embed_signals(model, signals, device, expected_samples=expected_samples)
    qdrant_client = get_qdrant_client()
    collection_name = qdrant_collection_name()
    validate_qdrant_collection(qdrant_client, collection_name)
    identified = identify_from_qdrant_embeddings(
        query_embeddings=query_embeddings,
        client=qdrant_client,
        collection_name=collection_name,
        top_k=top_k,
    )
    result = aggregate_window_results(
        identified["windows"],
        claimed_subject_id=claimed_subject_id,
    )
    result = add_public_subject_fields(result, list_raw_subjects())
    result.update(
        {
            "config_id": config.config_id,
            "config_name": config.display_name,
            "device": device,
            "subject_id": subject_id,
            "qdrant_collection": collection_name,
        }
    )
    return result


def predict_bciciv_mat(
    file_name: str,
    claimed_subject_id: int | None = None,
    data_type: str = "eo",
    top_k: int = 5,
) -> dict:
    config = bciciv_v4_config(data_type)
    expected_samples = expected_samples_for_config(config)
    signals = preprocess_bciciv_mat(file_name)
    signals = validate_signal_array(signals, expected_samples=expected_samples)
    model, device = load_model(config)
    query_embeddings = embed_signals(model, signals, device, expected_samples=expected_samples)
    qdrant_client = get_qdrant_client()
    collection_name = BCICIV_QDRANT_COLLECTIONS[data_type]
    validate_qdrant_collection(qdrant_client, collection_name)
    identified = identify_from_qdrant_embeddings(
        query_embeddings=query_embeddings,
        client=qdrant_client,
        collection_name=collection_name,
        top_k=top_k,
    )
    result = aggregate_window_results(
        identified["windows"],
        claimed_subject_id=claimed_subject_id,
    )
    result = add_public_subject_fields(result, list_raw_subjects())
    result.update(
        {
            "config_id": config.config_id,
            "config_name": config.display_name,
            "device": device,
            "bciciv_file": file_name,
            "input_dataset": "BCICIV unregistered access attempt",
            "qdrant_collection": collection_name,
        }
    )
    return result
