# -*- coding: utf-8 -*-
"""
CAPTCHA Recognition API Service

Inference interface based on the core CAPTCHA recognition model

Startup:
    python -m api.app
    or
    uvicorn api.app:app --host 0.0.0.0 --port 8000

API Endpoints:
    POST /predict      - Upload image, return recognition result (multipart/form-data)
    POST /predict/batch - Batch upload multiple images
    GET  /health       - Health check
"""

import logging
import time
from pathlib import Path
from contextlib import asynccontextmanager

import torch
import numpy as np
from PIL import Image
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from models.model import CaptchaModel
from models.transforms import get_val_transform
from utils.config_loader import get_config
from utils.chars import CharMapper
from utils.decoder import beam_search_decode as _beam_search_decode, postprocess_captcha

logger = logging.getLogger(__name__)

config = get_config()
IMAGE_H, IMAGE_W = config.get_preprocessed_image_size()
CHECKPOINT_PATH = str(Path(config.get_project_root()) / "checkpoints" / "best_model.pth")
BEAM_WIDTH = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
char_mapper = None
val_transform = None


def load_model():
    global model, char_mapper, val_transform
    if model is not None:
        return

    logger.info(f"Loading model weights: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    fc_weight_shape = state_dict.get("fc.weight", state_dict.get("decoder.fc.weight")).shape
    num_classes = fc_weight_shape[0]

    model = CaptchaModel(num_chars=num_classes - 1, pretrained=False).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    char_mapper = CharMapper.get_instance()
    val_transform = get_val_transform()

    logger.info(f"Model loaded successfully | Device: {DEVICE} | Classes: {num_classes}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    logger.info("API service shutdown")


app = FastAPI(
    title="CAPTCHA Recognition API",
    description="CAPTCHA recognition service based on ConvNeXt V2-Tiny + BiLSTM",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictResponse(BaseModel):
    text: str
    confidence: float = None
    inference_time_ms: float


class BatchPredictRequest(BaseModel):
    pass


class BatchPredictResponse(BaseModel):
    results: list[dict]
    total: int
    total_time_ms: float


@app.get("/health")
async def health():
    return {"status": "ok", "device": str(DEVICE), "model_loaded": model is not None}


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(img)
    transformed = val_transform(image=img_np)
    tensor = transformed['image'].unsqueeze(0).to(DEVICE)
    return tensor


def predict_single(tensor: torch.Tensor) -> tuple[str, float]:
    with torch.inference_mode():
        encoder_out, _ = model(tensor)
        decoder_out = encoder_out.permute(1, 0, 2)

        log_probs = decoder_out[0]
        max_log_prob = log_probs.max().item()
        confidence = np.exp(max_log_prob) if max_log_prob > -50 else 0.0

        texts = _beam_search_decode(decoder_out, char_mapper, beam_width=BEAM_WIDTH)
        raw_text = texts[0] if texts else ""
        text = postprocess_captcha(raw_text)

    return text, confidence


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        tensor = preprocess_image(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {str(e)}")

    start = time.perf_counter()
    text, confidence = predict_single(tensor)
    elapsed_ms = (time.perf_counter() - start) * 1000

    return PredictResponse(text=text, confidence=round(confidence, 4), inference_time_ms=round(elapsed_ms, 2))


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    results = []
    start = time.perf_counter()

    for file in files:
        if not file.filename:
            results.append({"filename": file.filename or "unknown", "error": "No filename"})
            continue

        try:
            content = await file.read()
            if len(content) == 0:
                results.append({"filename": file.filename, "error": "Empty file"})
                continue

            tensor = preprocess_image(content)
            t0 = time.perf_counter()
            text, confidence = predict_single(tensor)
            ms = (time.perf_counter() - t0) * 1000

            results.append({
                "filename": file.filename,
                "text": text,
                "confidence": round(confidence, 4),
                "inference_time_ms": round(ms, 2),
            })
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})

    total_ms = (time.perf_counter() - start) * 1000
    return BatchPredictResponse(results=results, total=len(results), total_time_ms=round(total_ms, 2))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
