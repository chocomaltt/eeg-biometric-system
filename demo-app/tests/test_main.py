import io

import numpy as np
from fastapi.testclient import TestClient

import main
from prediction import ModelConfig


client = TestClient(main.app)


def test_health_route():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_index_route_renders_upload_form(monkeypatch, tmp_path):
    config = ModelConfig(
        data_type="eo",
        window_token="2",
        stride_token="1",
        seed=42,
        x_train_path=tmp_path / "x.npy",
        y_train_path=tmp_path / "y.npy",
        model_path=tmp_path / "model.pth",
    )
    monkeypatch.setattr(main, "discover_configurations", lambda: [config])

    response = client.get("/")

    assert response.status_code == 200
    assert "EEG Biometric Demo" in response.text
    assert "EO | window 2s | stride 1s | seed 42" in response.text
    assert "name=\"signal_file\"" in response.text


def test_predict_route_renders_prediction_result(monkeypatch):
    monkeypatch.setattr(main, "discover_configurations", lambda: [])
    monkeypatch.setattr(
        main,
        "predict_signal",
        lambda config_id, array, claimed_subject_id=None: {
            "config_id": config_id,
            "config_name": "EO | window 2s | stride 1s | seed 42",
            "device": "cpu",
            "predicted_subject_id": 4,
            "vote_count": 1,
            "window_count": 1,
            "mean_similarity": 0.91,
            "max_similarity": 0.91,
            "claimed_subject_id": claimed_subject_id,
            "verification_threshold": 0.6216,
            "accepted": True,
            "windows": [
                {
                    "predicted_subject_id": 4,
                    "similarity": 0.91,
                    "top_matches": [{"subject_id": 4, "similarity": 0.91}],
                }
            ],
        },
    )
    buffer = io.BytesIO()
    np.save(buffer, np.zeros((64, 320), dtype=np.float32))
    buffer.seek(0)

    response = client.post(
        "/predict",
        data={"config_id": "eo:2:1:seed42", "claimed_subject_id": "4"},
        files={"signal_file": ("sample.npy", buffer, "application/octet-stream")},
    )

    assert response.status_code == 200
    assert "Predicted Subject" in response.text
    assert "4" in response.text
    assert "Accepted" in response.text


def test_predict_route_rejects_non_npy_upload():
    response = client.post(
        "/predict",
        data={"config_id": "eo:2:1:seed42", "claimed_subject_id": ""},
        files={"signal_file": ("sample.txt", b"not numpy", "text/plain")},
    )

    assert response.status_code == 200
    assert "Please upload a .npy file" in response.text
