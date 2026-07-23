from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from prediction import (
    FIXED_MODEL_FILENAME,
    PredictionError,
    bciciv_model_options,
    list_bciciv_mat_files,
    list_raw_subjects,
    predict_bciciv_mat,
    predict_raw_subject,
    resolve_claimed_account,
    subject_code_to_number,
)

APP_DIR = Path(__file__).resolve().parent

app = FastAPI(title="EEG Biometric Demo")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


def _template_context(
    request: Request,
    result: dict | None = None,
    error: str | None = None,
    selected_subject_id: str | None = None,
    selected_bciciv_file: str | None = None,
    selected_dataset_type: str = "raw_subject",
    selected_bciciv_model_type: str = "eo",
    selected_claimed_account: str | None = None,
    subjects: list[str] | None = None,
) -> dict:
    available_subjects = list_raw_subjects() if subjects is None else subjects
    bciciv_files = list_bciciv_mat_files()
    accounts = [
        {
            "code": subject,
            "label": f"Subject {subject_code_to_number(subject)} ({subject})",
        }
        for subject in available_subjects
    ]
    return {
        "request": request,
        "subjects": available_subjects,
        "accounts": accounts,
        "bciciv_files": bciciv_files,
        "bciciv_model_options": bciciv_model_options(),
        "result": result,
        "error": error,
        "selected_subject_id": selected_subject_id,
        "selected_bciciv_file": selected_bciciv_file,
        "selected_dataset_type": selected_dataset_type,
        "selected_bciciv_model_type": selected_bciciv_model_type,
        "selected_claimed_account": selected_claimed_account,
        "fixed_model_filename": FIXED_MODEL_FILENAME,
        "fixed_config_name": "EO | window 2s | stride 2s | seed 0",
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", _template_context(request))


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    dataset_type: str = Form("raw_subject"),
    subject_id: str = Form(""),
    bciciv_file: str = Form(""),
    bciciv_model_type: str = Form("eo"),
    claimed_account: str = Form(""),
):
    selected_subject_id = subject_id
    selected_bciciv_file = bciciv_file
    selected_dataset_type = dataset_type
    selected_bciciv_model_type = bciciv_model_type
    selected_claimed_account = claimed_account
    subjects = list_raw_subjects()
    try:
        claimed_id = resolve_claimed_account(claimed_account, subjects)
        if dataset_type == "raw_subject":
            result = predict_raw_subject(subject_id, claimed_subject_id=claimed_id)
        elif dataset_type == "bciciv":
            result = predict_bciciv_mat(
                bciciv_file,
                claimed_subject_id=claimed_id,
                data_type=bciciv_model_type,
            )
        else:
            raise PredictionError(f"Unknown data section: {dataset_type}")

        return templates.TemplateResponse(
            request,
            "index.html",
            _template_context(
                request,
                result=result,
                selected_subject_id=selected_subject_id,
                selected_bciciv_file=selected_bciciv_file,
                selected_dataset_type=selected_dataset_type,
                selected_bciciv_model_type=selected_bciciv_model_type,
                selected_claimed_account=selected_claimed_account,
                subjects=subjects,
            ),
        )
    except PredictionError as exc:
        error = str(exc)
    except Exception as exc:
        error = f"Prediction failed: {exc}"

    return templates.TemplateResponse(
        request,
        "index.html",
        _template_context(
            request,
            error=error,
            selected_subject_id=selected_subject_id,
            selected_bciciv_file=selected_bciciv_file,
            selected_dataset_type=selected_dataset_type,
            selected_bciciv_model_type=selected_bciciv_model_type,
            selected_claimed_account=selected_claimed_account,
            subjects=subjects,
        ),
    )


def main():
    print("Run with: uvicorn main:app --reload")


if __name__ == "__main__":
    main()
