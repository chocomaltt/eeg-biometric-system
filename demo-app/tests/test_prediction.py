from pathlib import Path

import numpy as np
import pytest
import torch

import prediction
from prediction import (
    Gallery,
    PredictionError,
    aggregate_window_results,
    array_from_npy_bytes,
    clear_caches,
    discover_configurations,
    fixed_eo_v4_config,
    get_config,
    identify_from_embeddings,
    list_raw_subjects,
    model_filename,
    normalize_embeddings,
    predict_raw_subject,
    predict_signal,
    preprocess_raw_eo_subject,
    subject_edf_path,
    token_to_model_value,
    validate_signal_array,
    window_raw_signal,
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
    (models_dir / model_filename("eo", 42, "2", "1")).write_bytes(b"model")

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
    assert config.model_path == models_dir / model_filename("eo", 42, "2", "1")


def test_discover_configurations_accepts_available_v4_model_naming(tmp_path):
    dataset_dir = tmp_path / "dataset"
    models_dir = tmp_path / "models"
    dataset_dir.mkdir()
    models_dir.mkdir()

    (dataset_dir / "X_eo_train_15_1_seed42.npy").write_bytes(b"x")
    (dataset_dir / "y_eo_train_15_1_seed42.npy").write_bytes(b"y")
    model_path = models_dir / "embedding_v4_eo_train_65_42_1.5_1_b128_e100_margin_0.2.pth"
    model_path.write_bytes(b"model")

    configs = discover_configurations(dataset_dir=dataset_dir, models_dir=models_dir)

    assert len(configs) == 1
    assert configs[0].config_id == "eo:15:1:seed42"
    assert configs[0].model_path == model_path


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


def test_list_raw_subjects_returns_sorted_subject_directories(tmp_path):
    (tmp_path / "S002").mkdir()
    (tmp_path / "S001").mkdir()
    (tmp_path / "S010").mkdir()
    (tmp_path / "README.txt").write_text("not a subject")
    (tmp_path / "S01").mkdir()
    (tmp_path / "SABC").mkdir()

    assert list_raw_subjects(tmp_path) == ["S001", "S002", "S010"]


def test_list_raw_subjects_returns_empty_list_when_dataset_missing(tmp_path):
    assert list_raw_subjects(tmp_path / "missing") == []


def test_fixed_eo_v4_config_points_to_requested_model_and_gallery(tmp_path):
    dataset_dir = tmp_path / "preprocessed_research_final_v4_80"
    models_dir = tmp_path / "models"
    dataset_dir.mkdir()
    models_dir.mkdir()

    config = fixed_eo_v4_config(dataset_dir=dataset_dir, models_dir=models_dir)

    assert config.config_id == "eo:15:05:seed0"
    assert config.display_name == "EO | window 1.5s | stride 0.5s | seed 0"
    assert config.x_train_path == dataset_dir / "X_eo_train_15_05_seed0.npy"
    assert config.y_train_path == dataset_dir / "y_eo_train_15_05_seed0.npy"
    assert config.model_path == models_dir / "embedding_v4_eo_train_80_0_1.5_0.5_b128_e100_margin_0.2.pth"


class FakeRaw:
    def __init__(self, data: np.ndarray, sfreq: float = 160.0):
        self._data = data
        self.info = {"sfreq": sfreq}
        self.n_times = data.shape[1]
        self.ch_names = [f"EEG.{index:03d}" for index in range(data.shape[0])]
        self.crop_args = None

    def rename_channels(self, mapping):
        self.ch_names = [mapping.get(name, name) for name in self.ch_names]

    def crop(self, tmin, tmax, include_tmax=False):
        self.crop_args = (tmin, tmax, include_tmax)
        sfreq = int(self.info["sfreq"])
        end = int((tmax - tmin) * sfreq)
        self._data = self._data[:, :end]
        self.n_times = self._data.shape[1]
        return self

    def get_data(self):
        return self._data


def test_subject_edf_path_validates_subject_id(tmp_path):
    with pytest.raises(PredictionError, match="Invalid subject ID"):
        subject_edf_path("../S001", tmp_path)


def test_subject_edf_path_requires_existing_eo_file(tmp_path):
    (tmp_path / "S001").mkdir()

    with pytest.raises(PredictionError, match="Raw EO file not found"):
        subject_edf_path("S001", tmp_path)


def test_subject_edf_path_returns_s001_r01_file(tmp_path):
    subject_dir = tmp_path / "S001"
    subject_dir.mkdir()
    edf_path = subject_dir / "S001R01.edf"
    edf_path.write_text("edf")

    assert subject_edf_path("S001", tmp_path) == edf_path


def test_window_raw_signal_creates_expected_15s_05s_windows():
    data = np.arange(64 * 400, dtype=np.float64).reshape(64, 400)
    raw = FakeRaw(data)

    windows = window_raw_signal(raw)

    assert windows.shape == (3, 64, 240)
    np.testing.assert_array_equal(windows[0], data[:, 0:240])
    np.testing.assert_array_equal(windows[1], data[:, 80:320])
    np.testing.assert_array_equal(windows[2], data[:, 160:400])


def test_preprocess_raw_eo_subject_loads_crops_windows_filters_and_zscores(monkeypatch, tmp_path):
    subject_dir = tmp_path / "S001"
    subject_dir.mkdir()
    (subject_dir / "S001R01.edf").write_text("edf")
    timeline = np.linspace(0, 8 * np.pi, 400)
    data = np.vstack([np.sin(timeline + channel) for channel in range(64)]).astype(np.float64)
    fake_raw = FakeRaw(data)

    import prediction

    monkeypatch.setattr(prediction.mne.io, "read_raw_edf", lambda *args, **kwargs: fake_raw)

    windows = preprocess_raw_eo_subject("S001", tmp_path)

    assert windows.shape == (3, 64, 240)
    assert windows.dtype == np.float32
    assert fake_raw.crop_args is None
    assert all("." not in channel for channel in fake_raw.ch_names)
    assert np.allclose(np.mean(windows, axis=2), 0.0, atol=1e-5)
    assert np.allclose(np.std(windows, axis=2), 1.0, atol=1e-5)


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


def test_validate_signal_array_accepts_config_specific_sample_count():
    array = np.zeros((2398, 64, 240), dtype=np.float64)

    result = validate_signal_array(array, expected_samples=240)

    assert result.shape == (2398, 64, 240)
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


def test_embed_signals_returns_raw_embeddings_for_distance_search():
    class FixedEmbeddingModel:
        def __call__(self, batch):
            return torch.tensor([[3.0, 4.0]], dtype=torch.float32)

    signals = np.zeros((1, 64, 320), dtype=np.float32)

    result = prediction.embed_signals(FixedEmbeddingModel(), signals, "cpu")

    np.testing.assert_array_equal(result, np.array([[3.0, 4.0]], dtype=np.float32))


class FakeCollectionConfig:
    def __init__(self, distance):
        self.config = type(
            "Config",
            (),
            {
                "params": type(
                    "Params",
                    (),
                    {
                        "vectors": type("Vectors", (), {"distance": distance})(),
                    },
                )(),
            },
        )()


class FakeCollectionClient:
    def __init__(self, distance):
        self.distance = distance
        self.collection_name = None

    def get_collection(self, collection_name):
        self.collection_name = collection_name
        return FakeCollectionConfig(self.distance)


def test_validate_qdrant_collection_accepts_euclidean_distance():
    client = FakeCollectionClient("Euclid")

    assert hasattr(prediction, "validate_qdrant_collection")
    prediction.validate_qdrant_collection(client, "demo_collection")

    assert client.collection_name == "demo_collection"


def test_validate_qdrant_collection_rejects_non_euclidean_distance():
    client = FakeCollectionClient("Cosine")

    with pytest.raises(PredictionError, match="must use Euclidean distance"):
        prediction.validate_qdrant_collection(client, "demo_collection")


class FakeScoredPoint:
    def __init__(self, subject_id, score):
        self.payload = {"subject_id": subject_id}
        self.score = score


class FakeQueryResult:
    def __init__(self, points):
        self.points = points


class FakeQdrantClient:
    def __init__(self):
        self.calls = []

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
        return FakeQueryResult(
            [
                FakeScoredPoint(subject_id=4, score=0.20),
                FakeScoredPoint(subject_id=7, score=0.35),
            ]
        )


def test_identify_from_qdrant_embeddings_queries_hnsw_euclidean_collection():
    client = FakeQdrantClient()
    query = np.array([[1.0, 2.0]], dtype=np.float32)

    assert hasattr(prediction, "identify_from_qdrant_embeddings")
    result = prediction.identify_from_qdrant_embeddings(
        query,
        client=client,
        collection_name="embedding_v4_eo_train_80_0_1.5_0.5_b128_e100_margin_0.2",
        top_k=2,
    )

    assert result["windows"] == [
        {
            "predicted_subject_id": 4,
            "distance": pytest.approx(0.20),
            "top_matches": [
                {"subject_id": 4, "distance": pytest.approx(0.20)},
                {"subject_id": 7, "distance": pytest.approx(0.35)},
            ],
        }
    ]
    assert client.calls[0]["collection_name"] == "embedding_v4_eo_train_80_0_1.5_0.5_b128_e100_margin_0.2"
    assert client.calls[0]["query"] == pytest.approx([1.0, 2.0])
    assert client.calls[0]["limit"] == 2
    assert client.calls[0]["with_payload"] is True
    assert client.calls[0]["search_params"].exact is False


def test_aggregate_window_results_accepts_claimed_subject_below_distance_threshold():
    windows = [
        {"predicted_subject_id": 4, "distance": 0.40, "top_matches": []},
        {"predicted_subject_id": 4, "distance": 0.55, "top_matches": []},
        {"predicted_subject_id": 5, "distance": 0.20, "top_matches": []},
    ]

    result = aggregate_window_results(windows, claimed_subject_id=4)

    assert result["predicted_subject_id"] == 4
    assert result["window_count"] == 3
    assert result["mean_distance"] == pytest.approx(0.3833333)
    assert result["min_distance"] == pytest.approx(0.20)
    assert result["claimed_subject_id"] == 4
    assert result["verification_threshold"] == pytest.approx(0.6216)
    assert result["accepted"] is True
    assert "mean_similarity" not in result
    assert "max_similarity" not in result


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


class FakeModel:
    def eval(self):
        return self

    def to(self, device):
        return self


def test_predict_signal_uses_gallery_and_returns_aggregate(monkeypatch, tmp_path):
    clear_caches()
    dataset_dir = tmp_path / "dataset"
    models_dir = tmp_path / "models"
    dataset_dir.mkdir()
    models_dir.mkdir()

    x_path = dataset_dir / "X_eo_train_2_1_seed42.npy"
    y_path = dataset_dir / "y_eo_train_2_1_seed42.npy"
    model_path = models_dir / "embedding_v3.1_eo_train_80_42_2_1_b32_e100_margin_0.2.pth"
    np.save(x_path, np.zeros((2, 64, 320), dtype=np.float32))
    np.save(y_path, np.array([4, 5], dtype=np.int64))
    model_path.write_bytes(b"model")

    monkeypatch.setattr("prediction.DATASET_DIR", dataset_dir)
    monkeypatch.setattr("prediction.MODELS_DIR", models_dir)
    monkeypatch.setattr(
        "prediction.discover_configurations",
        lambda: discover_configurations(dataset_dir=dataset_dir, models_dir=models_dir),
    )
    monkeypatch.setattr("prediction.load_model", lambda config, device=None: (FakeModel(), "cpu"))
    monkeypatch.setattr(
        "prediction.load_or_build_gallery",
        lambda config, model, device: Gallery(
            embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            labels=np.array([4, 5], dtype=np.int64),
        ),
    )
    monkeypatch.setattr(
        "prediction.embed_signals",
        lambda model, signals, device, batch_size=128, expected_samples=320: np.array([[1.0, 0.0]], dtype=np.float32),
    )

    result = predict_signal(
        "eo:2:1:seed42",
        np.zeros((64, 320), dtype=np.float32),
        claimed_subject_id=4,
    )

    assert result["config_id"] == "eo:2:1:seed42"
    assert result["config_name"] == "EO | window 2s | stride 1s | seed 42"
    assert result["predicted_subject_id"] == 4
    assert result["accepted"] is True
    assert result["windows"][0]["top_matches"][0]["subject_id"] == 4


def test_predict_raw_subject_preprocesses_and_queries_qdrant_collection(monkeypatch, tmp_path):
    clear_caches()
    config = fixed_eo_v4_config(dataset_dir=tmp_path, models_dir=tmp_path)
    calls = {}
    client = object()

    def fake_preprocess_raw_eo_subject(subject_id):
        calls["subject_id"] = subject_id
        return np.zeros((1, 64, 240), dtype=np.float32)

    def fake_identify_from_qdrant_embeddings(query_embeddings, client, collection_name, top_k=5):
        calls["query_embeddings"] = query_embeddings
        calls["client"] = client
        calls["collection_name"] = collection_name
        calls["top_k"] = top_k
        return {
            "windows": [
                {
                    "predicted_subject_id": 0,
                    "distance": 0.40,
                    "top_matches": [{"subject_id": 0, "distance": 0.40}],
                }
            ]
        }

    monkeypatch.setattr("prediction.fixed_eo_v4_config", lambda: config)
    monkeypatch.setattr("prediction.preprocess_raw_eo_subject", fake_preprocess_raw_eo_subject)
    monkeypatch.setattr("prediction.expected_samples_for_config", lambda received_config: 240)
    monkeypatch.setattr("prediction.load_model", lambda received_config, device=None: (FakeModel(), "cpu"))
    monkeypatch.setattr(
        "prediction.load_or_build_gallery",
        lambda *args, **kwargs: pytest.fail("raw subject prediction should use Qdrant, not local gallery"),
    )
    monkeypatch.setattr("prediction.get_qdrant_client", lambda: client)
    monkeypatch.setattr("prediction.qdrant_collection_name", lambda: "embedding_v4_eo_train_80_0_1.5_0.5_b128_e100_margin_0.2")
    monkeypatch.setattr(
        "prediction.validate_qdrant_collection",
        lambda received_client, collection_name: calls.update(
            {"validated_client": received_client, "validated_collection": collection_name}
        ),
    )
    monkeypatch.setattr("prediction.identify_from_qdrant_embeddings", fake_identify_from_qdrant_embeddings)
    monkeypatch.setattr(
        "prediction.embed_signals",
        lambda model, signals, device, batch_size=128, expected_samples=240: np.array([[1.0, 0.0]], dtype=np.float32),
    )

    result = predict_raw_subject("S001", claimed_subject_id=0)

    assert calls["subject_id"] == "S001"
    assert calls["client"] is client
    assert calls["validated_client"] is client
    assert calls["collection_name"] == "embedding_v4_eo_train_80_0_1.5_0.5_b128_e100_margin_0.2"
    assert calls["validated_collection"] == "embedding_v4_eo_train_80_0_1.5_0.5_b128_e100_margin_0.2"
    np.testing.assert_array_equal(calls["query_embeddings"], np.array([[1.0, 0.0]], dtype=np.float32))
    assert result["config_id"] == "eo:15:05:seed0"
    assert result["config_name"] == "EO | window 1.5s | stride 0.5s | seed 0"
    assert result["predicted_subject_id"] == 0
    assert result["min_distance"] == pytest.approx(0.40)
    assert result["accepted"] is True


def test_predict_signal_accepts_sample_count_for_selected_config(monkeypatch, tmp_path):
    clear_caches()
    dataset_dir = tmp_path / "dataset"
    models_dir = tmp_path / "models"
    dataset_dir.mkdir()
    models_dir.mkdir()

    x_path = dataset_dir / "X_eo_train_15_1_seed42.npy"
    y_path = dataset_dir / "y_eo_train_15_1_seed42.npy"
    model_path = models_dir / model_filename("eo", 42, "15", "1")
    np.save(x_path, np.zeros((2, 64, 240), dtype=np.float32))
    np.save(y_path, np.array([4, 5], dtype=np.int64))
    model_path.write_bytes(b"model")

    monkeypatch.setattr("prediction.DATASET_DIR", dataset_dir)
    monkeypatch.setattr("prediction.MODELS_DIR", models_dir)
    monkeypatch.setattr(
        "prediction.discover_configurations",
        lambda: discover_configurations(dataset_dir=dataset_dir, models_dir=models_dir),
    )
    monkeypatch.setattr("prediction.load_model", lambda config, device=None: (FakeModel(), "cpu"))
    monkeypatch.setattr(
        "prediction.load_or_build_gallery",
        lambda config, model, device: Gallery(
            embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            labels=np.array([4, 5], dtype=np.int64),
        ),
    )
    monkeypatch.setattr(
        "prediction.embed_signals",
        lambda model, signals, device, batch_size=128, expected_samples=240: np.array([[1.0, 0.0]], dtype=np.float32),
    )

    result = predict_signal(
        "eo:15:1:seed42",
        np.zeros((2398, 64, 240), dtype=np.float32),
    )

    assert result["config_id"] == "eo:15:1:seed42"
    assert result["config_name"] == "EO | window 1.5s | stride 1s | seed 42"
    assert result["predicted_subject_id"] == 4
