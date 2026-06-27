from pathlib import Path

import numpy as np
import pytest

from prediction import (
    PredictionError,
    aggregate_window_results,
    array_from_npy_bytes,
    discover_configurations,
    get_config,
    identify_from_embeddings,
    normalize_embeddings,
    token_to_model_value,
    validate_signal_array,
)


def test_token_to_model_value_converts_dataset_tokens():
    assert token_to_model_value("05") == "0.5"
    assert token_to_model_value("15") == "1.5"
    assert token_to_model_value("1") == "1"
    assert token_to_model_value("2") == "2"


def test_discover_configurations_returns_only_complete_configs(tmp_path):
    dataset_dir = tmp_path / "dataset"
    models_dir = tmp_path / "models"
    dataset_dir.mkdir()
    models_dir.mkdir()

    (dataset_dir / "X_eo_train_2_1_seed42.npy").write_bytes(b"x")
    (dataset_dir / "y_eo_train_2_1_seed42.npy").write_bytes(b"y")
    (models_dir / "embedding_v3.1_eo_train_80_42_2_1_b32_e100_margin_0.2.pth").write_bytes(b"model")

    (dataset_dir / "X_ec_train_2_1_seed42.npy").write_bytes(b"x")
    (dataset_dir / "y_ec_train_2_1_seed42.npy").write_bytes(b"y")

    configs = discover_configurations(dataset_dir=dataset_dir, models_dir=models_dir)

    assert len(configs) == 1
    config = configs[0]
    assert config.config_id == "eo:2:1:seed42"
    assert config.data_type == "eo"
    assert config.window_token == "2"
    assert config.stride_token == "1"
    assert config.window_label == "2"
    assert config.stride_label == "1"
    assert config.seed == 42
    assert config.x_train_path == dataset_dir / "X_eo_train_2_1_seed42.npy"
    assert config.y_train_path == dataset_dir / "y_eo_train_2_1_seed42.npy"
    assert config.model_path == models_dir / "embedding_v3.1_eo_train_80_42_2_1_b32_e100_margin_0.2.pth"


def test_get_config_finds_config_by_id(tmp_path):
    dataset_dir = tmp_path / "dataset"
    models_dir = tmp_path / "models"
    dataset_dir.mkdir()
    models_dir.mkdir()
    (dataset_dir / "X_eo_train_15_05_seed123.npy").write_bytes(b"x")
    (dataset_dir / "y_eo_train_15_05_seed123.npy").write_bytes(b"y")
    (models_dir / "embedding_v3.1_eo_train_80_123_1.5_0.5_b32_e100_margin_0.2.pth").write_bytes(b"model")
    configs = discover_configurations(dataset_dir=dataset_dir, models_dir=models_dir)

    config = get_config("eo:15:05:seed123", configs=configs)

    assert config.window_label == "1.5"
    assert config.stride_label == "0.5"


def test_get_config_raises_for_unknown_id():
    with pytest.raises(PredictionError, match="Unknown configuration"):
        get_config("eo:2:1:seed42", configs=[])


def test_array_from_npy_bytes_loads_numpy_without_pickle():
    import io

    buffer = io.BytesIO()
    np.save(buffer, np.zeros((64, 320), dtype=np.float32))

    array = array_from_npy_bytes(buffer.getvalue())

    assert array.shape == (64, 320)
    assert array.dtype == np.float32


def test_validate_signal_array_accepts_single_window():
    array = np.zeros((64, 320), dtype=np.float64)

    result = validate_signal_array(array)

    assert result.shape == (1, 64, 320)
    assert result.dtype == np.float32


def test_validate_signal_array_accepts_batch():
    array = np.zeros((3, 64, 320), dtype=np.float64)

    result = validate_signal_array(array)

    assert result.shape == (3, 64, 320)
    assert result.dtype == np.float32


def test_validate_signal_array_rejects_wrong_shape():
    with pytest.raises(PredictionError, match="Expected array shape"):
        validate_signal_array(np.zeros((32, 320), dtype=np.float32))


def test_normalize_embeddings_returns_unit_vectors():
    embeddings = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)

    result = normalize_embeddings(embeddings)

    assert np.allclose(np.linalg.norm(result, axis=1), np.array([1.0, 1.0]))


def test_identify_from_embeddings_returns_top_matches():
    query = np.array([[1.0, 0.0]], dtype=np.float32)
    gallery = np.array(
        [
            [0.0, 1.0],
            [0.8, 0.2],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )
    labels = np.array([7, 8, 9], dtype=np.int64)

    result = identify_from_embeddings(query, gallery, labels, top_k=2)

    assert result["windows"][0]["predicted_subject_id"] == 9
    assert result["windows"][0]["similarity"] == pytest.approx(1.0)
    assert result["windows"][0]["top_matches"] == [
        {"subject_id": 9, "similarity": pytest.approx(1.0)},
        {"subject_id": 8, "similarity": pytest.approx(0.9701425)},
    ]


def test_aggregate_window_results_uses_majority_vote_and_threshold():
    windows = [
        {"predicted_subject_id": 4, "similarity": 0.70, "top_matches": []},
        {"predicted_subject_id": 4, "similarity": 0.80, "top_matches": []},
        {"predicted_subject_id": 5, "similarity": 0.95, "top_matches": []},
    ]

    result = aggregate_window_results(windows, claimed_subject_id=4)

    assert result["predicted_subject_id"] == 4
    assert result["window_count"] == 3
    assert result["mean_similarity"] == pytest.approx(0.8166666)
    assert result["max_similarity"] == pytest.approx(0.95)
    assert result["claimed_subject_id"] == 4
    assert result["verification_threshold"] == pytest.approx(0.6216)
    assert result["accepted"] is True


def test_aggregate_window_results_rejects_wrong_claimed_subject():
    windows = [
        {"predicted_subject_id": 4, "similarity": 0.90, "top_matches": []},
        {"predicted_subject_id": 4, "similarity": 0.91, "top_matches": []},
    ]

    result = aggregate_window_results(windows, claimed_subject_id=3)

    assert result["predicted_subject_id"] == 4
    assert result["claimed_subject_id"] == 3
    assert result["accepted"] is False
