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

# ==================== 4. CORS ====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 5. SLOWAPI MIDDLEWARE ====================
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda request, exc: JSONResponse(
    status_code=429,
    content={"detail": "Too many requests. Please try again later."}
))

# ==================== 6. ЗАГРУЗКА МОДЕЛИ ====================
print("📦 Загружаю модель...")

# Получаем рабочую директорию
BASE_DIR = Path(__file__).resolve().parent

# Возможные пути к модели (Railway, local, Docker)
possible_paths = [
    BASE_DIR / "runs/detect/car_damage/v2_merged/weights/best.pt",
    BASE_DIR / "../runs/detect/car_damage/v2_merged/weights/best.pt",
    Path("/app/runs/detect/car_damage/v2_merged/weights/best.pt"),
    Path("./runs/detect/car_damage/v2_merged/weights/best.pt"),
]

MODEL_PATH = None
model = None

for path in possible_paths:
    if path.exists():
        MODEL_PATH = path
        print(f"✅ Модель найдена: {path}")
        try:
            model = YOLO(str(MODEL_PATH))
            print("✅ Модель загружена успешно!")
        except Exception as e:
            print(f"❌ Ошибка при загрузке модели: {e}")
        break

if MODEL_PATH is None:
    print(f"❌ Модель не найдена. Проверенные пути:")
    for path in possible_paths:
        print(f"   - {path}")
    print("⚠️  API будет работать, но /api/detect вернёт ошибку 503")

# ==================== 7. API КЛЮЧИ ====================
VALID_KEYS = {
    "test-free-key": {"plan": "free", "limit": 10, "used": 0},
    "test-starter": {"plan": "starter", "limit": 1000, "used": 0},
}

# ==================== 8. МАППИНГ КЛАССОВ ====================
ALLOWED_DUPLICATES = {
    "wheel_left_front": 1,
    "wheel_right_front": 1,
    "wheel_left_rear": 1,
    "wheel_right_rear": 1,
    "wheel_left": 1,
    "wheel_right": 1,
    "headlight_left": 1,
    "headlight_right": 1,
    "taillight_left": 1,
    "taillight_right": 1,
    "door_front_left": 1,
    "door_front_right": 1,
    "door_rear_left": 1,
    "door_rear_right": 1,
    "door_left": 1,
    "door_right": 1,
    "window_left_front": 1,
    "window_right_front": 1,
    "window_left_rear": 1,
    "window_right_rear": 1,
    "window_left": 1,
    "window_right": 1,
    "mirror_left": 1,
    "mirror_right": 1,
    "fender_left": 1,
    "fender_right": 1,
}

# Маппинг на упрощённые английские названия
CLASS_NAMES_EN = {
    "wheel": "wheel",
    "wheel_left": "wheel",
    "wheel_right": "wheel",
    "door": "door",
    "door_left": "door",
    "door_right": "door",
    "window": "window",
    "window_left": "window",
    "window_right": "window",
    "mirror": "mirror",
    "mirror_left": "mirror",
    "mirror_right": "mirror",
    "fender": "fender",
    "fender_left": "fender",
    "fender_right": "fender",
    "bumper": "bumper",
    "bumper_left": "bumper",
    "bumper_right": "bumper",
    "headlight": "lamp",
    "headlight_left": "lamp",
    "headlight_right": "lamp",
    "taillight": "taillight",
    "taillight_left": "taillight",
    "taillight_right": "taillight",
    "bumper_front": "bumper",
    "bumper_rear": "bumper",
    "door_front_left": "door",
    "door_front_right": "door",
    "door_rear_left": "door",
    "door_rear_right": "door",
    "fender_left": "fender",
    "fender_right": "fender",
    "headlight_left": "lamp",
    "headlight_right": "lamp",
    "hood": "hood",
    "mirror_left": "mirror",
    "mirror_right": "mirror",
    "roof": "roof",
    "taillight_left": "taillight",
    "taillight_right": "taillight",
    "trunk": "trunk",
    "undamaged": "undamaged",
    "wheel_left_front": "wheel",
    "wheel_right_front": "wheel",
    "wheel_left_rear": "wheel",
    "wheel_right_rear": "wheel",
    "window_left_front": "window",
    "window_right_front": "window",
    "window_left_rear": "window",
    "window_right_rear": "window",
    "windscreen": "windscreen",
    "windscreen_front": "windscreen",
    "windscreen_rear": "windscreen",
    "window_rear": "windscreen",
    "rear_windscreen": "windscreen",
}

def map_class_name(original_class: str) -> str:
    """Маппит оригинальный класс на упрощённый английский"""
    return CLASS_NAMES_EN.get(original_class, original_class)

def correct_class_by_bbox(detections: list, img_width: int) -> list:
    """Исправляет left/right классы на основе X координат bbox"""
    center_x = img_width / 2
    
    swap_map = {
        "headlight_left": "headlight_right",
        "headlight_right": "headlight_left",
        "taillight_left": "taillight_right",
        "taillight_right": "taillight_left",
        "mirror_left": "mirror_right",
        "mirror_right": "mirror_left",
        "fender_left": "fender_right",
        "fender_right": "fender_left",
        "door_left": "door_right",
        "door_right": "door_left",
        "door_front_left": "door_front_right",
        "door_front_right": "door_front_left",
        "door_rear_left": "door_rear_right",
        "door_rear_right": "door_rear_left",
        "window_left": "window_right",
        "window_right": "window_left",
        "window_left_front": "window_right_front",
        "window_right_front": "window_left_front",
        "window_left_rear": "window_right_rear",
        "window_right_rear": "window_left_rear",
        "wheel_left": "wheel_right",
        "wheel_right": "wheel_left",
        "wheel_left_front": "wheel_right_front",
        "wheel_right_front": "wheel_left_front",
        "wheel_left_rear": "wheel_right_rear",
        "wheel_right_rear": "wheel_left_rear",
    }
    
    for det in detections:
        class_name = det["class_en"]
        center_bbox_x = (det["bbox"]["x1"] + det["bbox"]["x2"]) / 2
        
        if class_name in swap_map:
            if "_left" in class_name and center_bbox_x < center_x:
                det["class_en"] = swap_map[class_name]
            elif "_right" in class_name and center_bbox_x >= center_x:
                det["class_en"] = swap_map[class_name]
    
    return detections

def remove_conflicting_detections(detections: list) -> list:
    """Удаляет конфликтующие детекции"""
    conflict_pairs = [
        ("bumper_front", "bumper_rear"),
        ("windscreen_front", "windscreen_rear"),
        ("window_rear", "windscreen_rear"),
    ]
    
    for class1, class2 in conflict_pairs:
        det1 = next((d for d in detections if d["class_en"] == class1), None)
        det2 = next((d for d in detections if d["class_en"] == class2), None)
        
        if det1 and det2:
            if det1["confidence"] >= det2["confidence"]:
                detections = [d for d in detections if d["class_en"] != class2]
            else:
                detections = [d for d in detections if d["class_en"] != class1]
    
    return detections

def filter_detections(detections: list) -> list:
    detections = remove_conflicting_detections(detections)
    class_count = {}
    filtered = []
    
    for det in detections:
        class_en = det["class_en"]
        
        if class_en in ALLOWED_DUPLICATES:
            max_count = ALLOWED_DUPLICATES[class_en]
            current_count = class_count.get(class_en, 0)
            if current_count < max_count:
                filtered.append(det)
                class_count[class_en] = current_count + 1
        else:
            if class_en not in class_count:
                filtered.append(det)
                class_count[class_en] = 1
    
    class_en_names = [det["class_en"] for det in filtered]
    is_front_view = any(name in class_en_names for name in ["hood", "windscreen_front", "windscreen", "headlight_left", "headlight_right"])
    
    if is_front_view:
        rear_indicators = {"bumper_rear", "windscreen_rear", "rear_windscreen", "window_rear", "window_left_rear", "window_right_rear"}
        filtered = [det for det in filtered if det["class_en"] not in rear_indicators]
    
    for det in filtered:
        det["class"] = map_class_name(det["class_en"])
        del det["class_en"]
    
    return filtered

# Цвета для визуализации (BGR формат, БЕЗ белого и чёрного)
COLORS = {
    "door": (255, 100, 0),
    "lamp": (0, 255, 0),
    "taillight": (0, 0, 255),
    "wheel": (255, 255, 0),
    "window": (255, 0, 255),
    "mirror": (0, 255, 255),
    "fender": (255, 50, 100),
    "bumper": (50, 200, 255),
    "hood": (100, 255, 200),
    "roof": (200, 100, 255),
    "trunk": (100, 150, 255),
    "windscreen": (255, 150, 0),
    "undamaged": (50, 255, 100),
}

def get_color(class_name: str):
    for key, color in COLORS.items():
        if key in class_name.lower():
            return color
    return (50, 255, 100)

# ==================== МАРШРУТЫ ====================

@app.get("/")
@limiter.limit("1000/minute")
async def root(request: Request):
    static_dir = BASE_DIR / "static"
    index_file = static_dir / "index.html"
    
    if not index_file.exists():
        return JSONResponse(
            status_code=404,
            content={"error": "index.html not found"}
        )
    
    return FileResponse(index_file, media_type="text/html")

@app.get("/health")
@limiter.limit("1000/minute")
async def health(request: Request):
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH) if MODEL_PATH else None
    }

@app.post("/api/detect")
async def detect_damage(
    request: Request,
    file: UploadFile = File(...),
    api_key: str = Header(..., alias="X-API-Key"),
    confidence: float = 0.13
):
    if api_key not in VALID_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    user = VALID_KEYS[api_key]
    if user["used"] >= user["limit"]:
        raise HTTPException(status_code=429, detail=f"Limit exceeded ({user['limit']} requests/month)")
    
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        img_height, img_width = img.shape[:2]
        
        results = model(img, conf=confidence)
        
        detections = []
        
        print("\n=== YOLO INFERENCE ===")
        total_count = 0
        for r in results:
            for box in r.boxes:
                total_count += 1
                class_id = int(box.cls[0])
                original_class_name = r.names[class_id]
                conf_score = float(box.conf[0])
                
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                print(f"  Found: {original_class_name} (conf={conf_score:.3f}) bbox=[{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")
                
                detection = {
                    "class_en": original_class_name,
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
        print(f"Total raw detections: {total_count}")
        
        detections = correct_class_by_bbox(detections, img_width)
        detections = filter_detections(detections)
        
        user["used"] += 1
        
        return {
            "status": "ok",
            "detections": detections,
            "count": len(detections),
            "image_size": {
                "width": img_width,
                "height": img_height
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
        
        img_height, img_width = img.shape[:2]
        img_copy = img.copy()
        
        results = model(img, conf=0.5)
        
        detections = []
        
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                original_class_name = r.names[class_id]
                conf_score = float(box.conf[0])
                
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                detection = {
                    "class_en": original_class_name,
                    "confidence": round(conf_score, 3),
                    "bbox": {
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2),
                    }
                }
                detections.append(detection)
        
        detections = correct_class_by_bbox(detections, img_width)
        detections = filter_detections(detections)
        
        for det in detections:
            x1 = det["bbox"]["x1"]
            y1 = det["bbox"]["y1"]
            x2 = det["bbox"]["x2"]
            y2 = det["bbox"]["y2"]
            
            color = get_color(det["class"])
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            
            label = det["class"]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            text_x = x1
            text_y = max(y1 - 5, 20)
            
            cv2.rectangle(img_copy, (text_x, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y), color, -1)
            cv2.putText(img_copy, label, (text_x + 2, text_y - 2), font, font_scale, (255, 255, 255), thickness)
        
        output_path = BASE_DIR / "result.jpg"
        cv2.imwrite(str(output_path), img_copy)
        
        return FileResponse(output_path, media_type="image/jpeg")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models/info")
@limiter.limit("1000/minute")
async def model_info(request: Request):
    if not model:
        return {"loaded": False}
    
    return {
        "loaded": True,
        "classes": model.names,
        "num_classes": len(model.names),
        "model_path": str(MODEL_PATH) if MODEL_PATH else None
    }

# ==================== МОНТИРОВАНИЕ СТАТИЧЕСКИХ ФАЙЛОВ ====================
static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    print(f"✅ Статические файлы смонтированы: {static_dir}")
else:
    print(f"⚠️  Папка {static_dir} не найдена")

# ==================== ЗАПУСК ====================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 Запуск сервера на порту {port}")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )