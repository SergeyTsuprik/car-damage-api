from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends, Request
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

# ==================== DATABASE ====================
from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker, Session
import secrets

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
    api_key = Column(String, unique=True, index=True)
    
    # Subscription data
    plan = Column(String, default="free")  # free, starter, pro
    limit = Column(Integer, default=0)  # 0 (pay-per-use), 1000, 10000
    used = Column(Integer, default=0)  # счётчик в текущем месяце
    
    # Pay-per-use для Free юзеров
    balance = Column(Float, default=0)  # баланс в $
    
    # Overage для Starter/Pro
    overage_charges = Column(Float, default=0)  # переплата сверх лимита
    
    # Subscription management (NEW)
    subscription_active = Column(Boolean, default=False)
    subscription_started_at = Column(DateTime, nullable=True)
    subscription_ended_at = Column(DateTime, nullable=True)
    cancelled_at = Column(DateTime, nullable=True)
    
    # Reset info
    reset_date = Column(DateTime)  # когда сбросить used → 0
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

# ==================== PLAN CONFIGURATION ====================
PLAN_PRICES = {
    "free": {
        "monthly_cost": 0,
        "limit": 0,  # unlimited (pay-per-use)
        "cost_per_request": 0.15
    },
    "starter": {
        "monthly_cost": 29,
        "limit": 1000,
        "cost_per_request": 0.05  # overage цена
    },
    "pro": {
        "monthly_cost": 99,
        "limit": 10000,
        "cost_per_request": 0.02  # overage цена
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

def generate_api_key():
    """Генерирует уникальный API key"""
    return f"sk_{secrets.token_urlsafe(24)}"

def check_reset_date(user: User, db: Session):
    """Проверяет нужно ли сбросить счётчик"""
    if user.reset_date is None or datetime.utcnow() >= user.reset_date:
        user.used = 0
        user.reset_date = datetime.utcnow() + timedelta(days=30)
        db.commit()

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

# ==================== DETECTION ENDPOINT ====================

@app.post("/api/detect")
@limiter.limit("100/minute")
async def detect_damage(
    request: Request,
    file: UploadFile = File(...),
    api_key: str = Header(..., alias="X-API-Key"),
    db: Session = Depends(get_db)
):
    """
    Detects car damage from image.
    
    Billing logic:
    - Free: pay-per-use ($0.15 per request)
    - Starter: subscription ($29/month = 1000 requests) + overage ($0.05 each)
    - Pro: subscription ($99/month = 10000 requests) + overage ($0.02 each)
    """
    
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # === GET USER ===
    user = get_user_by_api_key(api_key, db)
    
    # === CHECK RESET DATE ===
    check_reset_date(user, db)
    
    # === BILLING LOGIC ===
    plan_config = PLAN_PRICES[user.plan]
    
    if user.plan == "free":
        # Free: pay-per-use
        cost = plan_config["cost_per_request"]  # $0.15
        
        if user.balance < cost:
            raise HTTPException(
                status_code=402,
                detail=f"Insufficient balance. Need ${cost:.2f}, have ${user.balance:.2f}"
            )
        
        user.balance -= cost
    
    elif user.plan in ["starter", "pro"]:
        # Subscription + overage
        if user.used >= user.limit:
            # Лимит кончился → платит за избыток
            overage_cost = plan_config["cost_per_request"]
            user.overage_charges += overage_cost
        
        user.used += 1
    
    db.commit()
    
    # === INFERENCE ===
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Model inference with confidence threshold
        results = model(img, conf=0.13)
        detections = []
        
        for result in results:
            for box in result.boxes:
                detections.append({
                    "class": result.names[int(box.cls)],
                    "class_en": result.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": {
                        "x1": float(box.xyxy[0][0]),
                        "y1": float(box.xyxy[0][1]),
                        "x2": float(box.xyxy[0][2]),
                        "y2": float(box.xyxy[0][3])
                    }
                })
    
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail="Detection failed")
    
    # === RESPONSE ===
    return {
        "status": "ok",
        "detections": detections,
        "count": len(detections),
        "plan": user.plan,
        
        # For free users
        "balance_remaining": round(user.balance, 2) if user.plan == "free" else None,
        
        # For starter/pro
        "requests_used": user.used if user.plan != "free" else None,
        "requests_remaining": (user.limit - user.used) if user.plan != "free" and user.used < user.limit else 0,
        "overage_charges": round(user.overage_charges, 2) if user.plan != "free" else None
    }

# ==================== REGISTRATION ====================

@app.post("/api/register")
async def register(email: str, db: Session = Depends(get_db)):
    """
    Регистрирует нового юзера с plan=free
    Генерирует API key и отправляет на email (позже)
    """
    
    # Проверяем дубликат
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"User with email {email} already registered"
        )
    
    # Генерируем API key
    api_key = generate_api_key()
    
    # Создаём юзера с plan=free
    user = User(
        email=email,
        api_key=api_key,
        plan="free",
        limit=0,  # pay-per-use
        balance=1.00,  # стартовый баланс $1.00 для тестирования (~6 анализов)
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
        "api_key": api_key,
        "plan": "free",
        "note": "API key has been sent to your email (TODO: implement email)"
    }

# ==================== SUBSCRIPTION MANAGEMENT ====================

@app.post("/api/upgrade/{api_key}")
async def upgrade_plan(
    api_key: str,
    new_plan: str,
    db: Session = Depends(get_db)
):
    """
    Upgrade to a higher plan (Free → Starter → Pro)
    Usually called after successful payment
    """
    
    user = get_user_by_api_key(api_key, db)
    
    # Plan hierarchy
    PLAN_ORDER = {"free": 0, "starter": 1, "pro": 2}
    
    # Проверяем что это upgrade (не downgrade)
    if PLAN_ORDER.get(new_plan, 0) <= PLAN_ORDER.get(user.plan, 0):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot upgrade from {user.plan} to {new_plan}. Use downgrade endpoint instead."
        )
    
    plan_config = PLAN_PRICES[new_plan]
    
    # Update user
    user.plan = new_plan
    user.limit = plan_config["limit"]
    user.used = 0  # ← RESET счётчика
    user.overage_charges = 0  # ← ОЧИСТКА overage
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
        "monthly_cost": plan_config["monthly_cost"],
        "next_reset": user.reset_date.isoformat()
    }

@app.post("/api/downgrade/{api_key}")
async def downgrade_plan(
    api_key: str,
    new_plan: str,
    db: Session = Depends(get_db)
):
    """
    Downgrade to a lower plan (Pro → Starter → Free)
    Downgrade is effective immediately, no refund
    """
    
    user = get_user_by_api_key(api_key, db)
    
    # Plan hierarchy
    PLAN_ORDER = {"free": 0, "starter": 1, "pro": 2}
    
    # Проверяем что это downgrade (не upgrade)
    if PLAN_ORDER.get(new_plan, 0) >= PLAN_ORDER.get(user.plan, 0):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot downgrade from {user.plan} to {new_plan}. Use upgrade endpoint instead."
        )
    
    plan_config = PLAN_PRICES[new_plan]
    
    # Update user
    user.plan = new_plan
    user.limit = plan_config["limit"]
    user.used = 0  # ← RESET счётчика
    user.overage_charges = 0  # ← ОЧИСТКА overage (НЕ рефундим)
    user.subscription_active = (new_plan != "free")
    user.subscription_started_at = datetime.utcnow()
    user.subscription_ended_at = datetime.utcnow() + timedelta(days=30) if new_plan != "free" else None
    user.reset_date = datetime.utcnow() + timedelta(days=30)
    
    db.commit()
    db.refresh(user)
    
    logger.info(f"⬇️  User {user.email} downgraded to {new_plan}")
    
    return {
        "status": "ok",
        "message": f"Plan downgraded to {new_plan} (no refund)",
        "plan": new_plan,
        "limit": plan_config["limit"],
        "monthly_cost": plan_config["monthly_cost"],
        "next_reset": user.reset_date.isoformat()
    }

@app.post("/api/cancel-subscription/{api_key}")
async def cancel_subscription(
    api_key: str,
    db: Session = Depends(get_db)
):
    """
    Cancel subscription and downgrade to Free plan.
    Access is blocked immediately.
    No refund for current billing period.
    """
    
    user = get_user_by_api_key(api_key, db)
    
    if user.plan == "free":
        raise HTTPException(
            status_code=400,
            detail="User is already on Free plan"
        )
    
    # Downgrade to free
    user.plan = "free"
    user.limit = 0  # pay-per-use
    user.used = 0
    user.balance = 0  # очищаем баланс (или можно дать refund)
    user.overage_charges = 0
    user.subscription_active = False
    user.cancelled_at = datetime.utcnow()
    user.subscription_ended_at = datetime.utcnow()  # Действует сразу же
    user.reset_date = datetime.utcnow() + timedelta(days=30)
    
    db.commit()
    db.refresh(user)
    
    logger.info(f"❌ User {user.email} cancelled subscription")
    
    return {
        "status": "ok",
        "message": "Subscription cancelled. Plan downgraded to Free.",
        "plan": "free",
        "cancelled_at": user.cancelled_at.isoformat(),
        "note": "No refund for current billing period"
    }

# ==================== STATUS & USAGE ====================

@app.get("/api/usage/{api_key}")
async def get_usage(
    api_key: str,
    db: Session = Depends(get_db)
):
    """
    Показывает текущее использование и charges
    """
    
    user = get_user_by_api_key(api_key, db)
    
    # Проверяем нужно ли сбросить счётчик
    check_reset_date(user, db)
    
    plan_config = PLAN_PRICES[user.plan]
    
    if user.plan == "free":
        return {
            "plan": "free",
            "balance": round(user.balance, 2),
            "cost_per_request": plan_config["cost_per_request"],
            "created_at": user.created_at.isoformat()
        }
    else:
        monthly_charge = plan_config["monthly_cost"]
        overage_charge = user.overage_charges
        total = monthly_charge + overage_charge
        
        return {
            "plan": user.plan,
            "requests_used": user.used,
            "requests_limit": user.limit,
            "requests_remaining": max(0, user.limit - user.used),
            "monthly_charge": monthly_charge,
            "overage_charge": round(overage_charge, 2),
            "total_charge_this_month": round(total, 2),
            "subscription_active": user.subscription_active,
            "subscription_started_at": user.subscription_started_at.isoformat() if user.subscription_started_at else None,
            "subscription_ended_at": user.subscription_ended_at.isoformat() if user.subscription_ended_at else None,
            "reset_date": user.reset_date.isoformat()
        }

@app.get("/api/subscription/{api_key}")
async def get_subscription(
    api_key: str,
    db: Session = Depends(get_db)
):
    """
    Показывает статус подписки
    """
    
    user = get_user_by_api_key(api_key, db)
    plan_config = PLAN_PRICES[user.plan]
    
    return {
        "email": user.email,
        "plan": user.plan,
        "subscription_active": user.subscription_active,
        "subscription_started_at": user.subscription_started_at.isoformat() if user.subscription_started_at else None,
        "subscription_ended_at": user.subscription_ended_at.isoformat() if user.subscription_ended_at else None,
        "cancelled_at": user.cancelled_at.isoformat() if user.cancelled_at else None,
        "monthly_cost": plan_config["monthly_cost"],
        "limit": user.limit,
        "created_at": user.created_at.isoformat()
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