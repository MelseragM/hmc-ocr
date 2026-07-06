# PaddleOCR API

This service exposes PaddleOCR as a local HTTP API for the NestJS backend.

## Run With Docker

From the repository root:

```powershell
docker compose up --build paddle-ocr
```

The OCR URL for `hmc-app/.env` is:

```env
OCR_PROVIDER=paddle
PADDLE_OCR_URL="http://localhost:8000/ocr"
PADDLE_OCR_FILE_FIELD=file
```

## Test Directly

```powershell
curl.exe http://localhost:8000/health
curl.exe -X POST http://localhost:8000/ocr -F "file=@C:\path\to\id-or-passport.jpg"
```

The API returns:

```json
{
  "text": "all extracted text",
  "rawText": "all extracted text",
  "result": [
    {
      "text": "single OCR line",
      "confidence": 0.98
    }
  ]
}
```
