import pytest
from fastapi.testclient import TestClient

import main


client = TestClient(main.app)


def prediction_result(
    *,
    subject_label=0,
    subject_display="Subject 1 (S001)",
    claimed_subject_id=None,
    claimed_subject_display=None,
    accepted=None,
    **extra,
):
    result = {
        "config_id": "eo:2:2:seed0",
        "config_name": "EO | window 2s | stride 2s | seed 0",
        "device": "cpu",
        "predicted_subject_id": subject_label,
        "predicted_subject_display": subject_display,
        "vote_count": 1,
        "window_count": 1,
        "mean_distance": 0.40,
        "min_distance": 0.40,
        "windows": [
            {
                "predicted_subject_id": subject_label,
                "predicted_subject_display": subject_display,
                "distance": 0.40,
                "top_matches": [
                    {
                        "subject_id": subject_label,
                        "subject_display": subject_display,
                        "distance": 0.40,
                    }
                ],
            }
        ],
    }
    if claimed_subject_id is not None:
        result.update(
            {
                "claimed_subject_id": claimed_subject_id,
                "claimed_subject_display": claimed_subject_display,
                "verification_threshold": 0.5828,
                "accepted": accepted,
            }
        )
    result.update(extra)
    return result


def test_health_route():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_index_route_renders_subject_bciciv_and_account_forms(monkeypatch):
    monkeypatch.setattr(main, "list_raw_subjects", lambda: ["S001", "S002"])
    monkeypatch.setattr(main, "list_bciciv_mat_files", lambda: ["BCICIV_calib_ds1a_1000Hz.mat"])
    monkeypatch.setattr(
        main,
        "bciciv_model_options",
        lambda: [{"value": "eo", "label": "EO checkpoint | window 2s | stride 1s | seed 0"}],
    )

    response = client.get("/")

    assert response.status_code == 200
    assert "EEG Biometric Demo" in response.text
    assert "Registered sample subjects" in response.text
    assert "Unregistered BCICIV access attempts" in response.text
    assert 'name="dataset_type"' in response.text
    assert 'value="raw_subject"' in response.text
    assert 'value="bciciv"' in response.text
    assert 'name="subject_id"' in response.text
    assert 'name="bciciv_file"' in response.text
    assert "BCICIV_calib_ds1a_1000Hz.mat" in response.text
    assert 'name="bciciv_model_type"' in response.text
    assert "EO checkpoint | window 2s | stride 1s | seed 0" in response.text
    assert 'name="claimed_account"' in response.text
    assert 'value="S001"' in response.text
    assert "Subject 1 (S001)" in response.text
    assert "Subject 2 (S002)" in response.text
    assert 'name="claimed_subject_id"' not in response.text


def test_predict_route_resolves_raw_subject_account_to_internal_label(monkeypatch):
    monkeypatch.setattr(main, "list_raw_subjects", lambda: ["S001"])
    monkeypatch.setattr(main, "list_bciciv_mat_files", lambda: ["BCICIV_calib_ds1a_1000Hz.mat"])
    monkeypatch.setattr(main, "bciciv_model_options", lambda: [{"value": "eo", "label": "EO checkpoint"}])
    calls = {}

    def fake_predict_raw_subject(subject_id, claimed_subject_id=None):
        calls["raw"] = {"subject_id": subject_id, "claimed_subject_id": claimed_subject_id}
        return prediction_result(
            subject_id=subject_id,
            claimed_subject_id=claimed_subject_id,
            claimed_subject_display="Subject 1 (S001)",
            accepted=True,
        )

    monkeypatch.setattr(main, "predict_raw_subject", fake_predict_raw_subject)
    monkeypatch.setattr(main, "predict_bciciv_mat", lambda *args, **kwargs: pytest.fail("BCICIV path should not run"))

    response = client.post(
        "/predict",
        data={
            "dataset_type": "raw_subject",
            "subject_id": "S001",
            "claimed_account": "S001",
        },
    )

    assert response.status_code == 200
    assert calls == {"raw": {"subject_id": "S001", "claimed_subject_id": 0}}
    assert "Predicted Account" in response.text
    assert "Subject 1 (S001)" in response.text
    assert "Acceptance Status" in response.text
    assert "Accepted" in response.text


def test_predict_route_shows_not_evaluated_status_without_account(monkeypatch):
    monkeypatch.setattr(main, "list_raw_subjects", lambda: ["S001"])
    monkeypatch.setattr(main, "list_bciciv_mat_files", lambda: ["BCICIV_calib_ds1a_1000Hz.mat"])
    monkeypatch.setattr(main, "bciciv_model_options", lambda: [{"value": "eo", "label": "EO checkpoint"}])
    calls = {}

    def fake_predict_raw_subject(subject_id, claimed_subject_id=None):
        calls["claimed_subject_id"] = claimed_subject_id
        return prediction_result(subject_id=subject_id)

    monkeypatch.setattr(main, "predict_raw_subject", fake_predict_raw_subject)

    response = client.post(
        "/predict",
        data={"dataset_type": "raw_subject", "subject_id": "S001", "claimed_account": ""},
    )

    assert response.status_code == 200
    assert calls["claimed_subject_id"] is None
    assert "Acceptance Status" in response.text
    assert "Not evaluated" in response.text
    assert "select an account for verification" in response.text


def test_predict_route_resolves_bciciv_account_to_internal_label(monkeypatch):
    monkeypatch.setattr(main, "list_raw_subjects", lambda: ["S001", "S002", "S003", "S004"])
    monkeypatch.setattr(main, "list_bciciv_mat_files", lambda: ["BCICIV_calib_ds1a_1000Hz.mat"])
    monkeypatch.setattr(main, "bciciv_model_options", lambda: [{"value": "ec", "label": "EC checkpoint"}])
    calls = {}

    def fake_predict_bciciv_mat(file_name, claimed_subject_id=None, data_type="eo"):
        calls["bciciv"] = {
            "file_name": file_name,
            "claimed_subject_id": claimed_subject_id,
            "data_type": data_type,
        }
        return prediction_result(
            subject_label=3,
            subject_display="Subject 4 (S004)",
            claimed_subject_id=claimed_subject_id,
            claimed_subject_display="Subject 4 (S004)",
            accepted=False,
            bciciv_file=file_name,
            input_dataset="BCICIV unregistered access attempt",
        )

    monkeypatch.setattr(main, "predict_raw_subject", lambda *args, **kwargs: pytest.fail("Raw path should not run"))
    monkeypatch.setattr(main, "predict_bciciv_mat", fake_predict_bciciv_mat)

    response = client.post(
        "/predict",
        data={
            "dataset_type": "bciciv",
            "bciciv_file": "BCICIV_calib_ds1a_1000Hz.mat",
            "bciciv_model_type": "ec",
            "claimed_account": "S004",
        },
    )

    assert response.status_code == 200
    assert calls == {
        "bciciv": {
            "file_name": "BCICIV_calib_ds1a_1000Hz.mat",
            "claimed_subject_id": 3,
            "data_type": "ec",
        }
    }
    assert "BCICIV_calib_ds1a_1000Hz.mat" in response.text
    assert "BCICIV unregistered access attempt" in response.text
    assert "Subject 4 (S004)" in response.text
    assert "Rejected" in response.text


def test_predict_route_rejects_unavailable_account(monkeypatch):
    monkeypatch.setattr(main, "list_raw_subjects", lambda: ["S001"])
    monkeypatch.setattr(main, "list_bciciv_mat_files", lambda: [])
    monkeypatch.setattr(main, "bciciv_model_options", lambda: [])

    response = client.post(
        "/predict",
        data={
            "dataset_type": "raw_subject",
            "subject_id": "S001",
            "claimed_account": "S002",
        },
    )

    assert response.status_code == 200
    assert "Account is not available: S002" in response.text


def test_predict_route_rejects_malformed_account(monkeypatch):
    monkeypatch.setattr(main, "list_raw_subjects", lambda: ["S001"])
    monkeypatch.setattr(main, "list_bciciv_mat_files", lambda: [])
    monkeypatch.setattr(main, "bciciv_model_options", lambda: [])

    response = client.post(
        "/predict",
        data={
            "dataset_type": "raw_subject",
            "subject_id": "S001",
            "claimed_account": "not-an-account",
        },
    )

    assert response.status_code == 200
    assert "Invalid account code: not-an-account" in response.text
