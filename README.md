# Car Damage Detection API  

REST API для автоматической детекции повреждений автомобилей.  

## 🚀 Что это?  

Анализирует фотографии повреждённых автомобилей и определяет:  
- Тип повреждения (царапина, вмятина, трещина, разбитое стекло и т.д.)  
- Местоположение (координаты на фото)  

**Среднее время ответа:** 1.2 секунды  

## 📊 Метрики модели  

- **mAP50:** 0.87  
- **mAP50-95:** 0.62  
- **Precision:** 0.89  
- **Recall:** 0.85  

## 🎯 Примеры использования  

### Python  

```python  
import requests  

API_URL = "https://car-damage-api-production.up.railway.app/api/detect"  
API_KEY = "test-free-key"  

with open("car.jpg", "rb") as f:  
    files = {"file": f}  
    headers = {"X-API-Key": API_KEY}  
    response = requests.post(API_URL, files=files, headers=headers)  

data = response.json()  
print(f"Найдено повреждений: {data['count']}")  
for detection in data['detections']:  
    print(f"  - {detection['class']}: {detection['confidence']:.0%}")  
```  

### JavaScript  

```javascript  
const form = new FormData();  
form.append("file", fileInput.files[0]);  

fetch("https://car-damage-api-production.up.railway.app/api/detect", {  
  method: "POST",  
  headers: {  
    "X-API-Key": "test-free-key"  
  },  
  body: form  
})  
.then(r => r.json())  
.then(data => {  
  console.log(`Найдено: ${data.count} повреждений`);  
  data.detections.forEach(d => {  
    console.log(`- ${d.class}: ${(d.confidence * 100).toFixed(0)}%`);  
  });  
});  
```  

### cURL  

```bash  
curl -X POST "https://car-damage-api-production.up.railway.app/api/detect" \
  -H "X-API-Key: test-free-key" \
  -F "file=@car.jpg"  
```  

## 📝 Формат ответа  

```json  
{  
  "status": "ok",  
  "detections": [  
    {  
      "class": "bumper_front",  
      "confidence": 0.87,  
      "bbox": {  
        "x1": 150,  
        "y1": 200,  
        "x2": 350,  
        "y2": 450  
      }  
    }  
  ],  
  "count": 1,  
  "requests_remaining": 9,  
  "plan": "free"  
}  
```  

## 💳 Ценообразование  

| План | Цена | Лимит |   
|------|------|-------|  
| Free | $0/месяц | 10 анализов/месяц |  
| Starter | $29/месяц | 1,000 анализов |  
| Pro | $99/месяц | 10,000 анализов |  
| Enterprise | Custom | Безлимит + SLA |  

## 🔑 API Key  

Для использования API нужен API ключ. Получи его на [landing page](https://car-damage-api-production.up.railway.app/).  

## 📚 Документация  

Интерактивная документация (Swagger UI):  
```  
https://car-damage-api-production.up.railway.app/docs  
```  

## 🛠️ Установка  

```bash  
# Клонировать репозиторий  
git clone https://github.com/YOUR_USERNAME/car-damage-api.git  
cd car-damage-api  

# Установить зависимости  
pip install -r requirements.txt  

# Запустить локально  
python main.py  
```  

Откроется на `http://localhost:8000`  

## 📦 Требования  

- Python 3.9+  
- FastAPI  
- YOLOv8  
- OpenCV  

Смотри `requirements.txt`  

## 🚀 Deploy  

API развёрнут на [Railway](https://railway.app/):  
```  
https://car-damage-api-production.up.railway.app/  
```  

## 📧 Контакты  

- Email: natalya.tsuprik@yandex.by  