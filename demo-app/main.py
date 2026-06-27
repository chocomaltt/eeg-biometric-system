from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from prediction import (
    PredictionError,
    array_from_npy_bytes,
    discover_configurations,
    predict_signal,
)

APP_DIR = Path(__file__).resolve().parent

app = FastAPI(title="EEG Biometric Demo")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


def _template_context(
    request: Request,
    result: dict | None = None,
    error: str | None = None,
    selected_config_id: str | None = None,
) -> dict:
    configs = discover_configurations()
    return {
        "request": request,
        "configs": configs,
        "result": result,
        "error": error,
        "selected_config_id": selected_config_id,
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
    config_id: str = Form(...),
    claimed_subject_id: str = Form(""),
    signal_file: UploadFile = File(...),
):
    selected_config_id = config_id
    try:
        filename = signal_file.filename or ""
        if not filename.endswith(".npy"):
            raise PredictionError("Please upload a .npy file")

        claimed_id = int(claimed_subject_id) if claimed_subject_id.strip() else None
        content = await signal_file.read()
        array = array_from_npy_bytes(content)
        result = predict_signal(config_id, array, claimed_subject_id=claimed_id)
        return templates.TemplateResponse(
            request,
            "index.html",
            _template_context(
                request,
                result=result,
                selected_config_id=selected_config_id,
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
            selected_config_id=selected_config_id,
        ),
    )


def main():
    print("Run with: uvicorn main:app --reload")


if __name__ == "__main__":
    main()
