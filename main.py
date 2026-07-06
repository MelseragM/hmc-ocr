import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("FLAGS_enable_pir_api", "0")

from fastapi import FastAPI, File, HTTPException, UploadFile
from paddleocr import PaddleOCR

app = FastAPI(title="HMC PaddleOCR API")


def create_ocr_engine() -> PaddleOCR:
    lang = os.getenv("PADDLE_OCR_LANG", "en")

    try:
        return PaddleOCR(
            lang=lang,
            use_textline_orientation=True,
            enable_mkldnn=False,
        )
    except (TypeError, ValueError):
        # PaddleOCR versions differ on accepted constructor options.
        return PaddleOCR(lang=lang)


ocr_engine = create_ocr_engine()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ocr")
async def run_ocr(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are supported")

    suffix = Path(file.filename or "upload").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_path = Path(temp_file.name)
        temp_file.write(await file.read())

    try:
        result = run_paddle(temp_path)
        lines = extract_lines(result)
        raw_text = "\n".join(line["text"] for line in lines if line["text"])

        return {
            "text": raw_text,
            "rawText": raw_text,
            "result": lines,
        }
    finally:
        temp_path.unlink(missing_ok=True)


def run_paddle(image_path: Path) -> Any:
    try:
        return ocr_engine.predict(str(image_path))
    except AttributeError:
        return ocr_engine.ocr(str(image_path))


def extract_lines(value: Any) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    collect_lines(value, lines)
    return lines


def collect_lines(value: Any, lines: list[dict[str, Any]]) -> None:
    if value is None:
        return

    if isinstance(value, dict):
        rec_texts = value.get("rec_texts")
        if isinstance(rec_texts, list):
            rec_scores = value.get("rec_scores")
            for index, text in enumerate(rec_texts):
                if isinstance(text, str) and text.strip():
                    score = None
                    if isinstance(rec_scores, list) and index < len(rec_scores):
                        score = rec_scores[index]
                    lines.append({"text": text.strip(), "confidence": score})
            return

        for key in ("text", "rawText", "transcription"):
            text = value.get(key)
            if isinstance(text, str) and text.strip():
                lines.append({"text": text.strip(), "confidence": value.get("confidence")})
                return

        for nested in value.values():
            collect_lines(nested, lines)
        return

    if isinstance(value, (list, tuple)):
        if looks_like_ocr_line(value):
            text = value[1][0]
            confidence = value[1][1] if len(value[1]) > 1 else None
            lines.append({"text": text.strip(), "confidence": confidence, "box": value[0]})
            return

        for item in value:
            collect_lines(item, lines)


def looks_like_ocr_line(value: list[Any] | tuple[Any, ...]) -> bool:
    return (
        len(value) >= 2
        and isinstance(value[1], (list, tuple))
        and len(value[1]) >= 1
        and isinstance(value[1][0], str)
    )
