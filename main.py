from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import json
import os

# ==================== 1. SENTRY ====================
import sentry_sdk

SENTRY_DSN = os.getenv("SENTRY_DSN", "")

if SENTRY_DSN:
    sentry_sdk.init(
        SENTRY_DSN,
        traces_sample_rate=1.0,
        environment="production"
    )

# ==================== 2. RATE LIMITING ====================
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)

# ==================== 3. FASTAPI APP ====================
app = FastAPI(
    title="Car Damage Detection API",
    version="1.0.0",
    description="Detect car parts and damage from images"
)

# Привязываем limiter к app
app.state.limiter = limiter

# ==================== 4. CORS ====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Или конкретные домены: ["https://yourdomain.by", "https://app.yourdomain.by"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 5. EXCEPTION HANDLER для Rate Limiting ====================
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests. Please try again later."}
    )

# ==================== 6. ЗАГРУЗКА МОДЕЛИ ====================
print("📦 Загружаю модель...")
MODEL_PATH = "runs/detect/car_damage/v2_merged/weights/best.pt"

if not Path(MODEL_PATH).exists():
    print(f"❌ Модель не найдена: {MODEL_PATH}")
    model = None
else:
    model = YOLO(MODEL_PATH)
    print("✅ Модель загружена!")

# ==================== 7. API КЛЮЧИ ====================
VALID_KEYS = {
    "test-free-key": {"plan": "free", "limit": 10, "used": 0},
    "test-starter": {"plan": "starter", "limit": 1000, "used": 0},
}

def verify_api_key(api_key: str = Header(..., alias="X-API-Key")):
    """Проверяет API ключ"""
    if api_key not in VALID_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    user = VALID_KEYS[api_key]
    if user["used"] >= user["limit"]:
        raise HTTPException(status_code=429, detail="Limit exceeded. Upgrade your plan.")
    return api_key

# ==================== ENDPOINTS ====================

@app.get("/")
@limiter.limit("1000/minute")
async def root(request: Request):
    """Главная страница"""
    return FileResponse("index.html", media_type="text/html")

@app.get("/health")
@limiter.limit("1000/minute")
async def health(request: Request):
    """Health check"""
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/api/detect")
@limiter.limit("100/minute")  # ← макс 100 запросов в минуту
async def detect_damage(
    request: Request,
    file: UploadFile = File(...),
    api_key: str = Header(..., alias="X-API-Key"),
    confidence: float = 0.13
):
    """
    Загруженное фото → детекция → JSON ответ
    
    Headers:
        X-API-Key: твой API ключ
    
    Query params:
        confidence: пороговое значение уверенности (0.0-1.0)
    
    Returns:
        detections: список найденных частей
        count: кол-во найденных объектов
        requests_remaining: осталось запросов
    """
    
    # Проверяем ключ
    if api_key not in VALID_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    user = VALID_KEYS[api_key]
    if user["used"] >= user["limit"]:
        raise HTTPException(status_code=429, detail=f"Limit exceeded ({user['limit']} requests/month)")
    
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Читаем загруженное фото
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Инференс модели
        results = model(img, conf=confidence)
        
        detections = []
        
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                class_name = r.names[class_id]
                conf_score = float(box.conf[0])
                
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                detection = {
                    "class": class_name,
                    "confidence": round(conf_score, 3),
                    "bbox": {
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2),
                        "width": int(x2 - x1),
                        "height": int(y2 - y1)
                    }
                }
                detections.append(detection)
        
        # Считаем использование
        user["used"] += 1
        
        return {
            "status": "ok",
            "detections": detections,
            "count": len(detections),
            "image_size": {
                "width": img.shape[1],
                "height": img.shape[0]
            },
            "requests_used": user["used"],
            "requests_remaining": user["limit"] - user["used"],
            "plan": user["plan"]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/api/detect-and-save")
@limiter.limit("50/minute")
async def detect_and_save(
    request: Request,
    file: UploadFile = File(...),
    api_key: str = Header(..., alias="X-API-Key")
):
    """
    Детекция + сохранение фото с боксами
    (для дебага)
    """
    
    if api_key not in VALID_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Инференс
        results = model(img, conf=0.5)
        
        # Сохраняем фото с боксами
        output_path = "result.jpg"
        results[0].save(output_path)
        
        return FileResponse(output_path, media_type="image/jpeg")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models/info")
@limiter.limit("1000/minute")
async def model_info(request: Request):
    """Информация о модели"""
    if not model:
        return {"loaded": False}
    
    return {
        "loaded": True,
        "classes": model.names,
        "num_classes": len(model.names),
        "model_path": str(MODEL_PATH)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000))
    )
