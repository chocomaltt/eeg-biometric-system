from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from prediction import (
    FIXED_MODEL_FILENAME,
    PredictionError,
    list_raw_subjects,
    predict_raw_subject,
)

APP_DIR = Path(__file__).resolve().parent

app = FastAPI(title="EEG Biometric Demo")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


def _template_context(
    request: Request,
    result: dict | None = None,
    error: str | None = None,
    selected_subject_id: str | None = None,
) -> dict:
    subjects = list_raw_subjects()
    return {
        "request": request,
        "subjects": subjects,
        "result": result,
        "error": error,
        "selected_subject_id": selected_subject_id,
        "fixed_model_filename": FIXED_MODEL_FILENAME,
        "fixed_config_name": "EO | window 1.5s | stride 0.5s | seed 0",
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
    subject_id: str = Form(...),
    claimed_subject_id: str = Form(""),
):
    selected_subject_id = subject_id
    try:
        claimed_id = int(claimed_subject_id) if claimed_subject_id.strip() else None
        result = predict_raw_subject(subject_id, claimed_subject_id=claimed_id)
        return templates.TemplateResponse(
            request,
            "index.html",
            _template_context(
                request,
                result=result,
                selected_subject_id=selected_subject_id,
            ),
        )
    except ValueError:
        error = "Claimed subject ID must be an integer"
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
        ),
    )


def main():
    print("Run with: uvicorn main:app --reload")


if __name__ == "__main__":
    main()
