from __future__ import annotations

import io
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DATASET_DIR = PROJECT_ROOT / "Dataset" / "preprocessed_research_final_v4_90"
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = APP_DIR / ".cache"
MODEL_PREFIX = "embedding_v3.1"
TRAIN_SPLIT = "80"
MODEL_SUFFIX = "b32_e100_margin_0.2.pth"
SIGNAL_CHANNELS = 64
SIGNAL_SAMPLES = 320
VERIFICATION_THRESHOLD = 0.6216

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class PredictionError(Exception):
    """User-facing prediction error shown in the demo form."""


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
        m_path = models_dir / model_filename(data_type, seed, window_token, stride_token)

        if y_path.exists() and m_path.exists():
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


def array_from_npy_bytes(content: bytes) -> np.ndarray:
    try:
        return np.load(io.BytesIO(content), allow_pickle=False)
    except Exception as exc:
        raise PredictionError("Uploaded file must be a valid .npy file") from exc


def validate_signal_array(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2 and array.shape == (SIGNAL_CHANNELS, SIGNAL_SAMPLES):
        return array[np.newaxis, :, :].astype(np.float32, copy=False)

    if array.ndim == 3 and array.shape[1:] == (SIGNAL_CHANNELS, SIGNAL_SAMPLES):
        return array.astype(np.float32, copy=False)

    raise PredictionError(
        "Expected array shape (64, 320) for one window or (N, 64, 320) for a batch; "
        f"got {tuple(array.shape)}"
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


def aggregate_window_results(
    window_results: list[dict],
    claimed_subject_id: int | None = None,
) -> dict:
    if not window_results:
        raise PredictionError("No prediction windows were produced")

    vote_counts = Counter(int(window["predicted_subject_id"]) for window in window_results)
    predicted_subject_id, vote_count = vote_counts.most_common(1)[0]
    similarities = np.array([float(window["similarity"]) for window in window_results], dtype=np.float32)
    max_similarity = float(np.max(similarities))
    result = {
        "predicted_subject_id": int(predicted_subject_id),
        "vote_count": int(vote_count),
        "window_count": len(window_results),
        "mean_similarity": float(np.mean(similarities)),
        "max_similarity": max_similarity,
        "windows": window_results,
    }

    if claimed_subject_id is not None:
        accepted = predicted_subject_id == claimed_subject_id and max_similarity >= VERIFICATION_THRESHOLD
        result.update(
            {
                "claimed_subject_id": int(claimed_subject_id),
                "verification_threshold": VERIFICATION_THRESHOLD,
                "accepted": bool(accepted),
            }
        )

    return result
