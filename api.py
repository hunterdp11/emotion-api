from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import pickle
import io
import os
import math
from datetime import datetime, timezone
import sentiment_analyzer
import __main__

# ─── Firebase Admin ────────────────────────────────────────────────────────────
try:
    import firebase_admin
    from firebase_admin import credentials, firestore as fs_admin
    _cred_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "firebase_credentials.json")
    if not firebase_admin._apps and os.path.exists(_cred_path):
        firebase_admin.initialize_app(credentials.Certificate(_cred_path))
        _db = fs_admin.client()
        print("[Firebase] Firestore connected.")
    elif firebase_admin._apps:
        _db = fs_admin.client()
    else:
        _db = None
        print("[Firebase] firebase_credentials.json not found — logging disabled.")
except Exception as _e:
    _db = None
    print(f"[Firebase] Init failed: {_e}")

def _log_to_firestore(collection: str, data: dict):
    """Fire-and-forget Firestore write. Never crashes the API."""
    if _db is None:
        return
    try:
        data["timestamp"] = fs_admin.SERVER_TIMESTAMP
        _db.collection(collection).add(data)
    except Exception as e:
        print(f"[Firebase] Write error: {e}")

# ─── Pickle compatibility ──────────────────────────────────────────────────────
__main__.NaiveBayes = sentiment_analyzer.NaiveBayes
__main__.LogisticRegressionScratch = sentiment_analyzer.LogisticRegressionScratch
__main__.NeuralNetworkScratch = sentiment_analyzer.NeuralNetworkScratch

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == '__main__':
            return getattr(sentiment_analyzer, name)
        return super().find_class(module, name)

# ─── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(title="Multimodal Emotion & Sentiment API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Constants ────────────────────────────────────────────────────────────────
device = torch.device("cpu")
CLASS_NAMES = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
MODELS_DIR = "models"

# ─── Model Loaders (High Performance Mode) ───────────────────────────────────
_vision_cache = {}
_text_cache = {}

def load_vision_model(model_name: str):
    if model_name in _vision_cache:
        return _vision_cache[model_name]
    
    try:
        if model_name == 'resnet18':
            m = models.resnet18(weights=None)
            m.fc = nn.Sequential(nn.Linear(m.fc.in_features, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, len(CLASS_NAMES)))
        elif model_name == 'mobilenet_v2':
            m = models.mobilenet_v2(weights=None)
            m.classifier[1] = nn.Sequential(nn.Linear(m.classifier[1].in_features, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, len(CLASS_NAMES)))
        elif model_name == 'efficientnet_b0':
            m = models.efficientnet_b0(weights=None)
            m.classifier[1] = nn.Sequential(nn.Linear(m.classifier[1].in_features, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, len(CLASS_NAMES)))
        elif model_name == 'custom_cnn':
            import sys
            sys.path.insert(0, '.')
            from custom_model import CustomEmotionCNN
            m = CustomEmotionCNN(len(CLASS_NAMES))
        else:
            return None

        path = os.path.join(MODELS_DIR, f"{model_name}_emotion.pth")
        fallback = os.path.join(MODELS_DIR, "emotion_model.pth")
        
        state_dict = torch.load(path if os.path.exists(path) else fallback, map_location=device)
        m.load_state_dict(state_dict)
        
        m.eval()
        _vision_cache[model_name] = m
        return m
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return None

def load_text_models(requested_model: str = None):
    if requested_model and requested_model in _text_cache:
        return _text_cache

    try:
        # Load vocab first if not present
        if "vocab" not in _text_cache:
            pkl_path = os.path.join(MODELS_DIR, "text_sentiment_model.pkl")
            if os.path.exists(pkl_path):
                with open(pkl_path, "rb") as f:
                    data = CustomUnpickler(f).load()
                    if isinstance(data, tuple) and len(data) >= 3:
                        _text_cache["vocab"] = data[1]
                        _text_cache["num_to_label"] = data[2]
                        cls_name = type(data[0]).__name__
                        key = "Neural Network" if "Neural" in cls_name else ("Logistic Regression" if "Logistic" in cls_name else "Naive Bayes")
                        _text_cache[key] = data[0]

        # Load specific model if requested
        if requested_model and requested_model not in _text_cache:
            sep_files = {
                "Neural Network":      "text_nn_model.pkl",
                "Logistic Regression": "text_lr_model.pkl",
                "Naive Bayes":         "text_nb_model.pkl",
            }
            if requested_model in sep_files:
                pkl_path = os.path.join(MODELS_DIR, sep_files[requested_model])
                if os.path.exists(pkl_path):
                    with open(pkl_path, "rb") as f:
                        data = CustomUnpickler(f).load()
                        _text_cache[requested_model] = data[0] if isinstance(data, tuple) else data

    except Exception as e:
        print(f"Error loading text models: {e}")
    return _text_cache

# Use OpenCV built-in cascade path
_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
face_cascade = cv2.CascadeClassifier(_cascade_path)
if face_cascade.empty():
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ─── Prediction Endpoints ─────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    print("Preloading models for GCP high-performance...")
    load_text_models("Neural Network")
    load_vision_model("resnet18")
    print("All core models preloaded.")

@app.get("/logs")
async def get_logs(limit: int = 50):
    """Return latest emotion + sentiment logs from Firestore."""
    if _db is None:
        return {"error": "Firebase not configured on server."}
    try:
        emotion_docs = (
            _db.collection("emotion_logs")
            .order_by("timestamp", direction=fs_admin.Query.DESCENDING)
            .limit(limit)
            .stream()
        )
        sentiment_docs = (
            _db.collection("sentiment_logs")
            .order_by("timestamp", direction=fs_admin.Query.DESCENDING)
            .limit(limit)
            .stream()
        )
        def _serialize(doc):
            d = doc.to_dict()
            if d.get("timestamp"):
                try:
                    d["timestamp"] = d["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    d["timestamp"] = str(d["timestamp"])
            return d
        return {
            "emotion_logs":   [_serialize(d) for d in emotion_docs],
            "sentiment_logs": [_serialize(d) for d in sentiment_docs],
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def root():
    return {"status": "online", "version": "2.0-GCP", "endpoints": ["/predict/image", "/predict/text", "/stats/vision", "/stats/text"]}

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...), model: str = "resnet18"):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(image)

        if face_cascade.empty():
            return {"error": "Face detection module failed to load on server."}

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        # Ultra-strict parameters (alt2 cascade + minNeighbors=10) to completely prevent hands being detected as faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(80, 80))
        
        if len(faces) == 0:
            return {"error": "No human faces found in the image. Please try another."}
            
        x, y, w, h = faces[0]
        face_crop = img_np[y:y+h, x:x+w]
        faces_detected = len(faces)

        m = load_vision_model(model)
        if m is None:
            return {"error": f"Model '{model}' weights not found."}

        face_pil = Image.fromarray(face_crop)
        tensor = transform(face_pil).unsqueeze(0)

        with torch.no_grad():
            outputs = m(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            conf, idx = torch.max(probs, 0)

        all_probs = {CLASS_NAMES[i]: round(float(probs[i]) * 100, 1) for i in range(len(CLASS_NAMES))}

        result = {
            "emotion": CLASS_NAMES[idx.item()],
            "confidence": round(float(conf) * 100, 1),
            "all_probabilities": all_probs,
            "model_used": model,
            "faces_detected": faces_detected,
        }
        # Log to Firestore (fire-and-forget)
        _log_to_firestore("emotion_logs", {
            "emotion":        result["emotion"],
            "confidence":     result["confidence"],
            "model_used":     model,
            "faces_detected": faces_detected,
            "source":         "mobile_app",
        })
        return result
    except Exception as e:
        return {"error": f"Image analysis failed: {str(e)}"}

class TextRequest(BaseModel):
    text: str
    model: str = "Neural Network"

@app.post("/predict/text")
async def predict_text(req: TextRequest):
    try:
        cache = load_text_models(req.model)
        if not cache:
            return {"error": "Text models not found"}

        vocab = cache.get("vocab")
        num_to_label = cache.get("num_to_label")

        if vocab is None or num_to_label is None:
            return {"error": "Vocabulary not loaded"}

        clean = sentiment_analyzer.preprocess(req.text)
        vec = sentiment_analyzer.vectorize(clean, vocab)

        model_key = req.model
        mdl = cache.get(model_key) or cache.get("Neural Network")
        
        cls_name = type(mdl).__name__

        if "Neural" in cls_name:
            pred_num = mdl.predict([vec])[0]
            sentiment = num_to_label[pred_num]
            a1 = sentiment_analyzer.relu([sum(w*xi for w, xi in zip(ws, vec)) + b for ws, b in zip(mdl.W1, mdl.b1)])
            z2 = [sum(w*ai for w, ai in zip(ws, a1)) + b for ws, b in zip(mdl.W2, mdl.b2)]
            probs = sentiment_analyzer.softmax(z2)
            all_probs = {num_to_label[i]: round(probs[i] * 100, 1) for i in range(len(num_to_label))}
            conf = round(probs[pred_num] * 100, 1)
        elif "Logistic" in cls_name:
            pred_num = mdl.predict([vec])[0]
            sentiment = num_to_label[pred_num]
            scores = [sum(w*xi for w, xi in zip(mdl.W[c], vec)) + mdl.b[c] for c in range(len(num_to_label))]
            probs = sentiment_analyzer.softmax(scores)
            all_probs = {num_to_label[i]: round(probs[i] * 100, 1) for i in range(len(num_to_label))}
            conf = round(probs[pred_num] * 100, 1)
        else:
            sentiment = mdl.predict(vec)
            total_docs = sum(mdl.class_counts.values())
            scores = {}
            for label in mdl.class_counts:
                log_p = math.log(mdl.class_counts[label] / total_docs)
                total_words = sum(mdl.class_word_counts[label])
                for i, cnt in enumerate(vec):
                    prob = (mdl.class_word_counts[label][i] + 1) / (total_words + mdl.vocab_size)
                    log_p += cnt * math.log(prob)
                scores[label] = log_p
            min_score = min(scores.values())
            exp_scores = {k: math.exp(v - min_score) for k, v in scores.items()}
            total_exp = sum(exp_scores.values())
            all_probs = {k: round(v / total_exp * 100, 1) for k, v in exp_scores.items()}
            conf = all_probs.get(sentiment, 90.0)

        result = {
            "sentiment":        sentiment,
            "confidence":       conf,
            "all_probabilities": all_probs,
            "model_used":       model_key,
        }
        # Log to Firestore (fire-and-forget)
        _log_to_firestore("sentiment_logs", {
            "sentiment":   sentiment,
            "confidence":  conf,
            "model_used":  model_key,
            "text_length": len(req.text),
            "source":      "mobile_app",
        })
        return result
    except Exception as e:
        return {"error": str(e)}
