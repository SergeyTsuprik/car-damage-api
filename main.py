from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends, Request, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import hashlib
import secrets

# ==================== DATABASE ====================
from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# Database Setup
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    logger_init = logging.getLogger("DATABASE")
    logger_init.error("DATABASE_URL environment variable not set!")
    raise ValueError("DATABASE_URL environment variable is required")

logger_init = logging.getLogger("DATABASE")
logger_init.info(f"Connecting to database...")

engine = create_engine(DATABASE_URL, pool_pre_ping=True, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ==================== USER MODEL ====================
class User(Base):
    """Модель пользователя с поддержкой подписок"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, index=True)
    password_hash = Column(String)
    session_token = Column(String, unique=True, index=True, nullable=True)
    api_key = Column(String, unique=True, index=True, nullable=True)
    
    # Subscription data
    plan = Column(String, default="free")  # free, starter, pro
    limit = Column(Integer, default=10)  # 10 для free, 1000 для starter, 10000 для pro
    used = Column(Integer, default=0)  # счётчик в текущем месяце (только для starter/pro)
    
    # Balance для всех планов (обновляется каждый месяц)
    balance = Column(Float, default=1.50)  # $1.50 для free = 10 запросов
    
    # Overage для Starter/Pro
    overage_charges = Column(Float, default=0)  # переплата сверх лимита
    
    # Subscription management
    subscription_active = Column(Boolean, default=False)
    subscription_started_at = Column(DateTime, nullable=True)
    subscription_ended_at = Column(DateTime, nullable=True)
    cancelled_at = Column(DateTime, nullable=True)
    
    # Reset info
    reset_date = Column(DateTime)  # когда сбросить used → 0 и обновить balance
    created_at = Column(DateTime, default=datetime.utcnow)

# ==================== FASTAPI SETUP ====================
app = FastAPI(title="KastikCars API", version="1.0.0")

# Sentry integration
sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN", ""),
    integrations=[FastApiIntegration()],
    traces_sample_rate=1.0
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# Static files
STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== CLASS MAPPING & FILTERING ====================

# Маппинг на упрощённые английские названия
CLASS_NAMES_EN = {
    "bumper_front": "bumper",
    "bumper_rear": "bumper",
    "bumper_left": "bumper",
    "bumper_right": "bumper",
    "door_front_left": "door",
    "door_front_right": "door",
    "door_rear_left": "door",
    "door_rear_right": "door",
    "door_left": "door",
    "door_right": "door",
    "headlight_left": "lamp",
    "headlight_right": "lamp",
    "taillight_left": "taillight",
    "taillight_right": "taillight",
    "mirror_left": "mirror",
    "mirror_right": "mirror",
    "wheel_left_front": "wheel",
    "wheel_right_front": "wheel",
    "wheel_left_rear": "wheel",
    "wheel_right_rear": "wheel",
    "wheel_left": "wheel",
    "wheel_right": "wheel",
    "window_left_front": "window",
    "window_right_front": "window",
    "window_left_rear": "window",
    "window_right_rear": "window",
    "window_left": "window",
    "window_right": "window",
    "fender_left": "fender",
    "fender_right": "fender",
    "hood": "hood",
    "roof": "roof",
    "trunk": "trunk",
    "windscreen": "windscreen",
    "windscreen_front": "windscreen",
    "windscreen_rear": "windscreen",
}

ALLOWED_DUPLICATES = {
    "wheel": 4,
    "door": 4,
    "window": 4,
    "lamp": 2,
    "taillight": 2,
    "mirror": 2,
}

def map_class_name(original_class: str) -> str:
    """Маппит оригинальный класс на упрощённый"""
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
        "door_left": "door_right",
        "door_right": "door_left",
        "door_front_left": "door_front_right",
        "door_front_right": "door_front_left",
        "door_rear_left": "door_rear_right",
        "door_rear_right": "door_rear_left",
        "window_left_front": "window_right_front",
        "window_right_front": "window_left_front",
        "window_left_rear": "window_right_rear",
        "window_right_rear": "window_left_rear",
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

def remove_nested_boxes(detections: list) -> list:
    """Удаляет вложенные боксы одного класса, оставляет только внешний (больший)"""
    filtered = []
    
    for i, det_i in enumerate(detections):
        is_nested = False
        
        for j, det_j in enumerate(detections):
            if i == j or det_i["class_en"] != det_j["class_en"]:
                continue
            
            # Проверяем, полностью ли det_i внутри det_j
            if (det_j["bbox"]["x1"] <= det_i["bbox"]["x1"] and
                det_j["bbox"]["y1"] <= det_i["bbox"]["y1"] and
                det_j["bbox"]["x2"] >= det_i["bbox"]["x2"] and
                det_j["bbox"]["y2"] >= det_i["bbox"]["y2"]):
                
                # Проверяем, что det_j заметно больше
                area_i = (det_i["bbox"]["x2"] - det_i["bbox"]["x1"]) * (det_i["bbox"]["y2"] - det_i["bbox"]["y1"])
                area_j = (det_j["bbox"]["x2"] - det_j["bbox"]["x1"]) * (det_j["bbox"]["y2"] - det_j["bbox"]["y1"])
                
                if area_j > area_i * 1.1:  # det_j на 10% больше
                    is_nested = True
                    break
        
        if not is_nested:
            filtered.append(det_i)
    
    return filtered

def remove_conflicting_detections(detections: list) -> list:
    """Удаляет конфликтующие детекции (bumper_front vs bumper_rear)"""
    conflict_pairs = [
        ("bumper_front", "bumper_rear"),
        ("windscreen_front", "windscreen_rear"),
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
    """Удаляет дубликаты и применяет маппинг классов"""
    detections = remove_conflicting_detections(detections)
    detections = remove_nested_boxes(detections)  # Новое: удаляем вложенные боксы
    
    class_count = {}
    filtered = []
    
    for det in detections:
        mapped_class = map_class_name(det["class_en"])
        
        if mapped_class in ALLOWED_DUPLICATES:
            max_count = ALLOWED_DUPLICATES[mapped_class]
            current_count = class_count.get(mapped_class, 0)
            if current_count < max_count:
                filtered.append(det)
                class_count[mapped_class] = current_count + 1
        else:
            if mapped_class not in class_count:
                filtered.append(det)
                class_count[mapped_class] = 1
    
    # Заменяем class_en на mapped names
    for det in filtered:
        det["class_en"] = map_class_name(det["class_en"])
    
    return filtered

# ==================== PLAN CONFIGURATION ====================
PLAN_PRICES = {
    "free": {
        "monthly_cost": 0,
        "limit": 10,
        "balance": 1.50,
        "cost_per_request": 0.15
    },
    "starter": {
        "monthly_cost": 29,
        "limit": 1000,
        "cost_per_request": 0.05
    },
    "pro": {
        "monthly_cost": 99,
        "limit": 10000,
        "cost_per_request": 0.02
    }
}

# ==================== MODEL LOADING ====================
BASE_DIR = Path(__file__).resolve().parent

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
        logger.info(f"✅ Модель найдена: {path}")
        try:
            model = YOLO(str(MODEL_PATH))
            logger.info("✅ Модель загружена успешно!")
        except Exception as e:
            logger.error(f"❌ Ошибка при загрузке модели: {e}")
        break

if MODEL_PATH is None:
    logger.warning(f"❌ Модель не найдена. API будет работать, но /api/detect вернёт ошибку 503")

# ==================== HELPER FUNCTIONS ====================

def get_db():
    """Dependency для получения сессии БД"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def hash_password(password: str) -> str:
    """Хэширует пароль с SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Проверяет пароль"""
    return hash_password(plain_password) == hashed_password

def generate_session_token() -> str:
    """Генерирует уникальный session token"""
    return secrets.token_urlsafe(32)

def generate_api_key() -> str:
    """Генерирует уникальный API ключ (длиннее для безопасности)"""
    return f"sk_{secrets.token_urlsafe(48)}"

def check_reset_date(user: User, db: Session):
    """Проверяет нужно ли сбросить счётчик и обновить balance"""
    if user.reset_date is None or datetime.utcnow() >= user.reset_date:
        user.used = 0
        
        # Обновляем balance для free плана
        if user.plan == "free":
            user.balance = PLAN_PRICES["free"]["balance"]  # $1.50
        
        user.reset_date = datetime.utcnow() + timedelta(days=30)
        db.commit()

def get_user_by_session_token(session_token: str, db: Session):
    """Получает юзера по session token"""
    user = db.query(User).filter(User.session_token == session_token).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user

def get_user_by_api_key(api_key: str, db: Session):
    """Получает юзера по API key"""
    user = db.query(User).filter(User.api_key == api_key).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return user

# ==================== STARTUP EVENT ====================

@app.on_event("startup")
async def startup_event():
    """Инициализация БД при старте приложения"""
    try:
        logger.info("🔄 Инициализирую БД...")
        Base.metadata.create_all(bind=engine)
        
        # Миграция: добавляем колонок api_key если его нет (для Railway PostgreSQL)
        db = SessionLocal()
        try:
            from sqlalchemy import text
            # Проверяем существует ли колонок
            result = db.execute(text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name='users' AND column_name='api_key'"
            ))
            if not result.fetchone():
                # Колонок не существует, добавляем
                db.execute(text(
                    "ALTER TABLE users ADD COLUMN api_key VARCHAR UNIQUE"
                ))
                db.execute(text(
                    "CREATE INDEX ix_users_api_key ON users (api_key)"
                ))
                db.commit()
                logger.info("✅ Добавлен колонок api_key")
            else:
                logger.info("✅ Колонок api_key уже существует")
        except Exception as migration_error:
            logger.warning(f"⚠️ Миграция api_key: {migration_error}")
            db.rollback()
        finally:
            db.close()
        
        logger.info("✅ БД инициализирована успешно!")
    except Exception as e:
        logger.error(f"❌ Ошибка при инициализации БД: {e}")
        raise

# ==================== ROOT & HEALTH ====================

@app.get("/")
async def root():
    """Serve index.html from static folder"""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path, media_type="text/html")
    return {"error": "index.html not found in static/"}

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "ok",
        "service": "KastikCars API",
        "model_loaded": model is not None
    }

@app.get("/api/models/info")
async def models_info():
    """Информация о модели"""
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model": "YOLOv8",
        "task": "object-detection",
        "classes": 15,
        "avg_response_time": "1.2s"
    }

# ==================== AUTH ENDPOINTS ====================

@app.post("/api/register")
async def register(email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    """Регистрирует нового юзера с plan=free"""
    
    # Проверяем дубликат
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"User with email {email} already registered"
        )
    
    # Генерируем session token
    session_token = generate_session_token()
    
    # Создаём юзера с plan=free
    user = User(
        email=email,
        password_hash=hash_password(password),
        session_token=session_token,
        plan="free",
        limit=PLAN_PRICES["free"]["limit"],  # 10
        balance=PLAN_PRICES["free"]["balance"],  # $1.50
        subscription_active=False,
        reset_date=datetime.utcnow() + timedelta(days=30),
        created_at=datetime.utcnow()
    )
    
    db.add(user)
    db.commit()
    db.refresh(user)
    
    logger.info(f"✅ User registered: {email}")
    
    return {
        "status": "ok",
        "message": "Registration successful",
        "email": email,
        "session_token": session_token,
        "plan": "free"
    }

@app.post("/api/login")
async def login(email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    """Вход по email и пароль"""
    
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.password_hash):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )
    
    session_token = generate_session_token()
    user.session_token = session_token
    db.commit()
    db.refresh(user)
    
    logger.info(f"✅ User logged in: {email}")
    
    return {
        "status": "ok",
        "message": "Login successful",
        "email": email,
        "session_token": session_token,
        "plan": user.plan
    }

@app.post("/api/generate-api-key")
async def generate_api_key_endpoint(session_token: str = Header(..., alias="X-Session-Token"), db: Session = Depends(get_db)):
    """Генерирует API ключ для пользователя (один на юзера, вечный)"""
    
    user = get_user_by_session_token(session_token, db)
    
    # Если ключ уже существует — возвращаем старый
    if user.api_key:
        logger.info(f"ℹ️ API key already exists for {user.email}, returning existing")
        return {
            "status": "ok",
            "message": "API key already exists",
            "api_key": user.api_key,
            "warning": "Keep your API key safe. It provides paid access to the API."
        }
    
    # Генерируем новый ключ только если его ещё нет
    user.api_key = generate_api_key()
    db.commit()
    db.refresh(user)
    
    logger.info(f"✅ API key generated for {user.email}")
    
    return {
        "status": "ok",
        "message": "API key generated",
        "api_key": user.api_key,
        "warning": "Keep your API key safe. It provides paid access to the API."
    }

@app.get("/api/me")
async def get_me(session_token: str = Header(..., alias="X-Session-Token"), db: Session = Depends(get_db)):
    """Получает информацию о текущем юзере"""
    
    user = get_user_by_session_token(session_token, db)
    check_reset_date(user, db)
    
    # Для free плана: requests_remaining считаем через balance
    if user.plan == "free":
        requests_remaining = int(user.balance / PLAN_PRICES["free"]["cost_per_request"])
        requests_used = PLAN_PRICES["free"]["limit"] - requests_remaining
    else:
        requests_used = user.used
        requests_remaining = max(0, user.limit - user.used)
    
    return {
        "status": "ok",
        "email": user.email,
        "plan": user.plan,
        "requests_used": requests_used,
        "requests_limit": user.limit,
        "requests_remaining": requests_remaining,
        "balance": round(user.balance, 2),
        "subscription_active": user.subscription_active,
        "created_at": user.created_at.isoformat()
    }

# ==================== DETECTION ENDPOINT ====================

@app.post("/api/detect")
@limiter.limit("100/minute")
async def detect_damage(
    request: Request,
    file: UploadFile = File(...),
    session_token: str = Header(None, alias="X-Session-Token"),
    api_key: str = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db)
):
    """Detects car damage from image
    
    Supports two auth methods:
    - X-Session-Token: Web UI (limit - used counter)
    - X-API-Key: API access ($0.15 per request)
    """
    
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # === GET USER & DETERMINE AUTH TYPE ===
    auth_type = None
    user = None
    
    if session_token:
        user = get_user_by_session_token(session_token, db)
        auth_type = "web"
    elif api_key:
        user = get_user_by_api_key(api_key, db)
        auth_type = "api"
    else:
        raise HTTPException(status_code=401, detail="X-Session-Token or X-API-Key required")
    
    # === CHECK RESET DATE ===
    check_reset_date(user, db)
    
    # === BILLING LOGIC ===
    plan_config = PLAN_PRICES[user.plan]
    
    if auth_type == "api":
        # API KEY: always paid model ($0.15 per request)
        cost_per_request = 0.15
        
        if user.balance < cost_per_request:
            raise HTTPException(
                status_code=402,
                detail=f"Insufficient balance. Need ${cost_per_request:.2f}, have ${user.balance:.2f}"
            )
        
        user.balance -= cost_per_request
        user.used += 1  # Считаем для счётчика
        logger.info(f"API request from {user.email}, balance: ${user.balance:.2f}")
    
    else:  # auth_type == "web"
        # WEB UI: use limit - used counter
        if user.plan == "free":
            if user.used >= user.limit:
                raise HTTPException(
                    status_code=402,
                    detail=f"Monthly limit exceeded. {user.limit} analyses per month."
                )
            user.used += 1
        
        elif user.plan in ["starter", "pro"]:
            if not user.subscription_active:
                raise HTTPException(
                    status_code=402,
                    detail="Subscription not active. Please renew your subscription."
                )
            
            if user.used >= user.limit:
                overage_cost = plan_config["cost_per_request"]
                user.overage_charges += overage_cost
                logger.warning(f"Overage for {user.email}: ${overage_cost:.2f}")
            
            user.used += 1  # Считаем все запросы
    
    db.commit()
    
    # === INFERENCE ===
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_height, img_width = img.shape[:2]
        
        # Model inference with confidence threshold
        results = model(img, conf=0.13)
        detections = []
        
        for result in results:
            for box in result.boxes:
                detections.append({
                    "class_en": result.names[int(box.cls)],
                    "confidence": round(float(box.conf), 3),
                    "bbox": {
                        "x1": int(float(box.xyxy[0][0])),
                        "y1": int(float(box.xyxy[0][1])),
                        "x2": int(float(box.xyxy[0][2])),
                        "y2": int(float(box.xyxy[0][3]))
                    }
                })
        
        logger.info(f"Raw detections: {len(detections)}")
        # Применяем фильтрацию и маппинг
        detections = correct_class_by_bbox(detections, img_width)
        detections = filter_detections(detections)
        logger.info(f"After filtering: {len(detections)} final detections")
        
        # Переименовываем class_en в class для ответа
        for det in detections:
            det["class"] = det.pop("class_en")
    
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail="Detection failed")
    
    # === RESPONSE ===
    response = {
        "status": "ok",
        "detections": detections,
        "count": len(detections),
        "plan": user.plan,
        "auth_type": auth_type
    }
    
    # Add billing info based on auth type
    if auth_type == "api":
        response["balance_remaining"] = round(user.balance, 2)
        response["cost_per_request"] = 0.15
    else:  # web
        response["requests_used"] = user.used
        response["requests_limit"] = user.limit
        response["requests_remaining"] = max(0, user.limit - user.used)
        response["overage_charges"] = round(user.overage_charges, 2) if user.plan != "free" else None
    
    return response

# ==================== SUBSCRIPTION MANAGEMENT ====================

@app.post("/api/upgrade/{session_token}")
async def upgrade_plan(
    session_token: str,
    new_plan: str,
    db: Session = Depends(get_db)
):
    """Upgrade to a higher plan"""
    
    user = get_user_by_session_token(session_token, db)
    
    PLAN_ORDER = {"free": 0, "starter": 1, "pro": 2}
    
    if PLAN_ORDER.get(new_plan, 0) <= PLAN_ORDER.get(user.plan, 0):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot upgrade from {user.plan} to {new_plan}"
        )
    
    plan_config = PLAN_PRICES[new_plan]
    
    user.plan = new_plan
    user.limit = plan_config["limit"]
    user.used = 0
    user.overage_charges = 0
    user.subscription_active = True
    user.subscription_started_at = datetime.utcnow()
    user.subscription_ended_at = datetime.utcnow() + timedelta(days=30)
    user.reset_date = datetime.utcnow() + timedelta(days=30)
    
    db.commit()
    db.refresh(user)
    
    logger.info(f"✅ User {user.email} upgraded to {new_plan}")
    
    return {
        "status": "ok",
        "message": f"Plan upgraded to {new_plan}",
        "plan": new_plan,
        "limit": plan_config["limit"],
        "monthly_cost": plan_config["monthly_cost"]
    }

# ==================== ERROR HANDLERS ====================

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request, exc):
    return {
        "status": "error",
        "detail": "Rate limit exceeded",
        "retry_after": 60
    }

if __name__ == "__main__":
    import uvicorn
    PORT = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=PORT)
