from fastapi.testclient import TestClient

import main


client = TestClient(main.app)


def test_health_route():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_index_route_renders_subject_form(monkeypatch):
    monkeypatch.setattr(main, "list_raw_subjects", lambda: ["S001", "S002"])

    response = client.get("/")

    assert response.status_code == 200
    assert "EEG Biometric Demo" in response.text
    assert "name=\"subject_id\"" in response.text
    assert "S001" in response.text
    assert "embedding_v4_eo_train_80_0_1.5_0.5_b128_e100_margin_0.2.pth" in response.text
    assert "embedding_v4_eo_train_80_0_1.5_0.5_b128_e100_margin_0.2" in response.text


def test_predict_route_renders_prediction_result(monkeypatch):
    monkeypatch.setattr(main, "list_raw_subjects", lambda: ["S001"])
    calls = {}

    def fake_predict_raw_subject(subject_id, claimed_subject_id=None):
        calls["subject_id"] = subject_id
        calls["claimed_subject_id"] = claimed_subject_id
        return {
            "config_id": "eo:15:05:seed0",
            "config_name": "EO | window 1.5s | stride 0.5s | seed 0",
            "device": "cpu",
            "subject_id": subject_id,
            "predicted_subject_id": 0,
            "vote_count": 1,
            "window_count": 1,
            "mean_distance": 0.40,
            "min_distance": 0.40,
            "claimed_subject_id": claimed_subject_id,
            "verification_threshold": 0.6216,
            "accepted": True,
            "windows": [
                {
                    "predicted_subject_id": 0,
                    "distance": 0.40,
                    "top_matches": [{"subject_id": 0, "distance": 0.40}],
                }
            ],
        }

    monkeypatch.setattr(main, "predict_raw_subject", fake_predict_raw_subject)

    response = client.post(
        "/predict",
        data={"subject_id": "S001", "claimed_subject_id": "0"},
    )

    assert response.status_code == 200
    assert calls == {"subject_id": "S001", "claimed_subject_id": 0}
    assert "Predicted Subject" in response.text
    assert "0" in response.text
    assert "Accepted" in response.text
    assert "Distance" in response.text


def test_predict_route_rejects_non_integer_claimed_subject(monkeypatch):
    monkeypatch.setattr(main, "list_raw_subjects", lambda: ["S001"])

    response = client.post(
        "/predict",
        data={"subject_id": "S001", "claimed_subject_id": "not-an-int"},
    )

    assert response.status_code == 200
    assert "Claimed subject ID must be an integer" in response.text
