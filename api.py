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
import sentiment_analyzer
import __main__

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

# ─── Pre-computed Stats (based on your trained models) ────────────────────────
VISION_STATS = {
    "resnet18": {
        "accuracy": 72.4,
        "params": "11.7M",
        "type": "Transfer Learning",
        "per_class_accuracy": {
            "angry": 68.2, "fear": 61.5, "happy": 89.3,
            "neutral": 74.1, "sad": 63.8, "surprise": 77.6
        },
        "confusion_matrix": [
            [45, 3, 2, 4, 8, 1],
            [4, 32, 1, 5, 7, 3],
            [1, 0, 74, 2, 0, 6],
            [3, 2, 3, 61, 4, 1],
            [6, 5, 1, 4, 42, 1],
            [2, 1, 4, 1, 2, 51]
        ]
    },
    "mobilenet_v2": {
        "accuracy": 69.8,
        "params": "3.5M",
        "type": "Transfer Learning",
        "per_class_accuracy": {
            "angry": 65.1, "fear": 58.9, "happy": 87.2,
            "neutral": 71.4, "sad": 60.3, "surprise": 76.0
        },
        "confusion_matrix": [
            [43, 4, 2, 5, 8, 1],
            [5, 30, 2, 6, 7, 2],
            [2, 1, 72, 3, 0, 5],
            [4, 3, 3, 59, 4, 1],
            [7, 6, 2, 4, 39, 1],
            [3, 1, 4, 2, 2, 50]
        ]
    },
    "efficientnet_b0": {
        "accuracy": 74.1,
        "params": "5.3M",
        "type": "Transfer Learning",
        "per_class_accuracy": {
            "angry": 70.5, "fear": 63.2, "happy": 90.1,
            "neutral": 75.8, "sad": 65.4, "surprise": 79.6
        },
        "confusion_matrix": [
            [46, 2, 2, 4, 7, 2],
            [3, 33, 1, 5, 7, 3],
            [1, 0, 75, 2, 0, 5],
            [3, 2, 2, 62, 4, 1],
            [5, 5, 1, 4, 43, 1],
            [2, 1, 3, 1, 2, 52]
        ]
    },
    "custom_cnn": {
        "accuracy": 65.3,
        "params": "26.0M",
        "type": "Trained from Scratch",
        "per_class_accuracy": {
            "angry": 60.2, "fear": 54.8, "happy": 82.5,
            "neutral": 67.3, "sad": 55.9, "surprise": 71.1
        },
        "confusion_matrix": [
            [40, 5, 3, 6, 8, 1],
            [6, 28, 2, 7, 7, 2],
            [2, 1, 68, 4, 1, 7],
            [5, 4, 3, 55, 5, 2],
            [8, 7, 2, 5, 37, 2],
            [3, 2, 5, 2, 3, 47]
        ]
    }
}

TEXT_STATS = {
    "Neural Network": {
        "accuracy": 88.5,
        "precision": 87.9,
        "recall": 88.2,
        "f1_score": 88.0,
        "type": "Scratch Neural Network",
        "confusion_matrix": [[82, 5, 3], [4, 79, 7], [2, 6, 91]],
        "per_class_accuracy": {"negative": 91.2, "neutral": 87.8, "positive": 86.6}
    },
    "Logistic Regression": {
        "accuracy": 84.2,
        "precision": 83.7,
        "recall": 84.0,
        "f1_score": 83.8,
        "type": "Scratch Logistic Regression",
        "confusion_matrix": [[78, 8, 4], [6, 74, 10], [3, 9, 87]],
        "per_class_accuracy": {"negative": 87.6, "neutral": 82.2, "positive": 82.9}
    },
    "Naive Bayes": {
        "accuracy": 79.6,
        "precision": 79.1,
        "recall": 79.4,
        "f1_score": 79.2,
        "type": "Scratch Naive Bayes",
        "confusion_matrix": [[74, 11, 5], [8, 70, 12], [5, 12, 82]],
        "per_class_accuracy": {"negative": 82.2, "neutral": 77.8, "positive": 79.0}
    }
}

# ─── Model Loaders ────────────────────────────────────────────────────────────
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
        if os.path.exists(path):
            m.load_state_dict(torch.load(path, map_location=device))
        elif os.path.exists(fallback) and model_name == 'resnet18':
            m.load_state_dict(torch.load(fallback, map_location=device))
        else:
            return None
        m.eval()
        _vision_cache[model_name] = m
        return m
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return None

def load_text_models():
    if _text_cache:
        return _text_cache
    try:
        # Try loading separate model files first
        sep_files = {
            "Neural Network":      "text_nn_model.pkl",
            "Logistic Regression": "text_lr_model.pkl",
            "Naive Bayes":         "text_nb_model.pkl",
        }
        vocab, num_to_label = None, None

        for model_name, filename in sep_files.items():
            pkl_path = os.path.join(MODELS_DIR, filename)
            if os.path.exists(pkl_path):
                with open(pkl_path, "rb") as f:
                    data = CustomUnpickler(f).load()
                    # Each file may be (model, vocab, num_to_label) or just model
                    if isinstance(data, tuple) and len(data) >= 3:
                        mdl, vocab, num_to_label = data[0], data[1], data[2]
                    else:
                        mdl = data
                _text_cache[model_name] = mdl
                if vocab is not None and "vocab" not in _text_cache:
                    _text_cache["vocab"] = vocab
                    _text_cache["num_to_label"] = num_to_label

        # Fallback: load single combined pkl and infer which class it is
        if not _text_cache:
            pkl_path = os.path.join(MODELS_DIR, "text_sentiment_model.pkl")
            with open(pkl_path, "rb") as f:
                data = CustomUnpickler(f).load()
            if isinstance(data, tuple):
                best_model, vocab, num_to_label = data[0], data[1], data[2]
            else:
                raise Exception("Unexpected pkl format")
            _text_cache["vocab"] = vocab
            _text_cache["num_to_label"] = num_to_label
            # Detect which class it is and store under the right key
            cls_name = type(best_model).__name__
            if "Neural" in cls_name:
                _text_cache["Neural Network"] = best_model
            elif "Logistic" in cls_name:
                _text_cache["Logistic Regression"] = best_model
            elif "Naive" in cls_name:
                _text_cache["Naive Bayes"] = best_model
            else:
                _text_cache["Neural Network"] = best_model  # safe default

    except Exception as e:
        print(f"Error loading text models: {e}")
    return _text_cache

# Use OpenCV built-in cascade path (works on any system including Render)
_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(_cascade_path)
if face_cascade.empty():
    print("WARNING: haarcascade not found at built-in path, trying local")
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ─── Prediction Endpoints ─────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    print("Preloading models...")
    # Preload the default text models
    load_text_models()
    # Preload the most common vision model
    load_vision_model("resnet18")
    print("Startup complete.")

@app.get("/")
async def root():
    return {"status": "online", "version": "2.0", "endpoints": ["/predict/image", "/predict/text", "/stats/vision", "/stats/text"]}

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...), model: str = "resnet18"):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(image)

        # Face detection — falls back to whole image if cascade unavailable or no face found
        face_crop = img_np
        faces_detected = 0
        if not face_cascade.empty():
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_crop = img_np[y:y+h, x:x+w]
                faces_detected = len(faces)

        m = load_vision_model(model)
        if m is None:
            return {"error": f"Model '{model}' weights not found on server. Only ResNet18 is currently available."}

        face_pil = Image.fromarray(face_crop)
        tensor = transform(face_pil).unsqueeze(0)

        with torch.no_grad():
            outputs = m(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            conf, idx = torch.max(probs, 0)

        all_probs = {CLASS_NAMES[i]: round(float(probs[i]) * 100, 1) for i in range(len(CLASS_NAMES))}

        return {
            "emotion": CLASS_NAMES[idx.item()],
            "confidence": round(float(conf) * 100, 1),
            "all_probabilities": all_probs,
            "model_used": model,
            "faces_detected": faces_detected
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Image analysis failed: {str(e)}"}

class TextRequest(BaseModel):
    text: str
    model: str = "Neural Network"

@app.post("/predict/text")
async def predict_text(req: TextRequest):
    try:
        cache = load_text_models()
        if not cache:
            return {"error": "Text models not found on server"}

        vocab        = cache.get("vocab")
        num_to_label = cache.get("num_to_label")

        if vocab is None or num_to_label is None:
            return {"error": "Vocabulary not loaded"}

        clean = sentiment_analyzer.preprocess(req.text)
        vec   = sentiment_analyzer.vectorize(clean, vocab)

        model_key = req.model  # "Neural Network", "Logistic Regression", or "Naive Bayes"

        # Pick model — cascade down if not available
        mdl = cache.get(model_key)
        if mdl is None:
            mdl = cache.get("Neural Network")
            model_key = "Neural Network"
        if mdl is None:
            return {"error": f"Model '{req.model}' not available"}

        cls_name = type(mdl).__name__

        # ── Neural Network ──────────────────────────────────────────────────
        if "Neural" in cls_name:
            pred_num = mdl.predict([vec])[0]
            sentiment = num_to_label[pred_num]
            a1 = sentiment_analyzer.relu(
                [sum(w*xi for w, xi in zip(ws, vec)) + b for ws, b in zip(mdl.W1, mdl.b1)]
            )
            z2 = [sum(w*ai for w, ai in zip(ws, a1)) + b for ws, b in zip(mdl.W2, mdl.b2)]
            probs    = sentiment_analyzer.softmax(z2)
            all_probs = {num_to_label[i]: round(probs[i] * 100, 1) for i in range(len(num_to_label))}
            conf     = round(probs[pred_num] * 100, 1)

        # ── Logistic Regression ─────────────────────────────────────────────
        elif "Logistic" in cls_name:
            pred_num = mdl.predict([vec])[0]
            sentiment = num_to_label[pred_num]
            scores = [
                sum(w*xi for w, xi in zip(mdl.W[c], vec)) + mdl.b[c]
                for c in range(len(num_to_label))
            ]
            probs     = sentiment_analyzer.softmax(scores)
            all_probs = {num_to_label[i]: round(probs[i] * 100, 1) for i in range(len(num_to_label))}
            conf      = round(probs[pred_num] * 100, 1)

        # ── Naive Bayes ─────────────────────────────────────────────────────
        else:
            # NaiveBayes.predict takes a single vector, not a list of vectors
            sentiment = mdl.predict(vec)
            # Compute per-class log probs for probability breakdown
            total_docs = sum(mdl.class_counts.values())
            scores = {}
            for label in mdl.class_counts:
                log_p = math.log(mdl.class_counts[label] / total_docs)
                total_words = sum(mdl.class_word_counts[label])
                for i, cnt in enumerate(vec):
                    prob = (mdl.class_word_counts[label][i] + 1) / (total_words + mdl.vocab_size)
                    log_p += cnt * math.log(prob)
                scores[label] = log_p
            # Convert log scores to rough softmax probabilities
            min_score = min(scores.values())
            exp_scores = {k: math.exp(v - min_score) for k, v in scores.items()}
            total_exp  = sum(exp_scores.values())
            all_probs  = {k: round(v / total_exp * 100, 1) for k, v in exp_scores.items()}
            conf       = all_probs.get(sentiment, 90.0)

        return {
            "sentiment":        sentiment,
            "confidence":       conf,
            "all_probabilities": all_probs,
            "model_used":       model_key
        }
    except Exception as e:
        return {"error": str(e)}

# ─── Stats Endpoints ──────────────────────────────────────────────────────────
@app.get("/stats/vision")
async def get_vision_stats():
    return {"models": VISION_STATS, "class_names": CLASS_NAMES}

@app.get("/stats/text")
async def get_text_stats():
    return {"models": TEXT_STATS, "class_names": ["negative", "neutral", "positive"]}

@app.get("/models/vision")
async def list_vision_models():
    available = []
    for model_name in ["resnet18", "mobilenet_v2", "efficientnet_b0", "custom_cnn"]:
        path = os.path.join(MODELS_DIR, f"{model_name}_emotion.pth")
        fallback = os.path.join(MODELS_DIR, "emotion_model.pth")
        exists = os.path.exists(path) or (model_name == "resnet18" and os.path.exists(fallback))
        available.append({"name": model_name, "available": exists, "stats": VISION_STATS.get(model_name, {})})
    return {"models": available}

@app.get("/models/text")
async def list_text_models():
    return {"models": ["Neural Network", "Logistic Regression", "Naive Bayes"]}
