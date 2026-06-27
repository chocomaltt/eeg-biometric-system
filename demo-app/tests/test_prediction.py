from pathlib import Path

import numpy as np
import pytest

from prediction import (
    PredictionError,
    array_from_npy_bytes,
    discover_configurations,
    get_config,
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
