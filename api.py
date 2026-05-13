from fastapi import FastAPI, File, UploadFile, Form
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
import sentiment_analyzer
import __main__

# Bind classes for legacy pickle support
__main__.NaiveBayes = sentiment_analyzer.NaiveBayes
__main__.LogisticRegressionScratch = sentiment_analyzer.LogisticRegressionScratch
__main__.NeuralNetworkScratch = sentiment_analyzer.NeuralNetworkScratch

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == '__main__':
            return getattr(sentiment_analyzer, name)
        return super().find_class(module, name)

app = FastAPI(title="Multimodal Emotion API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cpu") # Cloud deployments usually use CPU for free tiers
class_names = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load Vision Model
vision_model = models.resnet18(weights=None)
vision_model.fc = nn.Sequential(nn.Linear(vision_model.fc.in_features, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, len(class_names)))
vision_model.load_state_dict(torch.load("models/resnet18_emotion.pth", map_location=device))
vision_model.eval()

# Load Text Model
with open("models/text_sentiment_model.pkl", "rb") as f:
    text_model, vocab, num_to_label, text_model_name = CustomUnpickler(f).load()

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(image)
    
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        return {"error": "No face detected"}
        
    x, y, w, h = faces[0]
    face_crop = img_np[y:y+h, x:x+w]
    face_pil = Image.fromarray(face_crop)
    tensor = transform(face_pil).unsqueeze(0)
    
    with torch.no_grad():
        outputs = vision_model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        conf, idx = torch.max(probs, 0)
        
    return {"emotion": class_names[idx.item()], "confidence": float(conf)}

class TextRequest(BaseModel):
    text: str

@app.post("/predict/text")
async def predict_text(req: TextRequest):
    clean = sentiment_analyzer.preprocess(req.text)
    vec = sentiment_analyzer.vectorize(clean, vocab)
    
    if text_model_name == "Neural Network":
        pred_num = text_model.predict([vec])[0]
        sentiment = num_to_label[pred_num]
        a1 = sentiment_analyzer.relu([sum(w*xi for w, xi in zip(ws, vec)) + b for ws, b in zip(text_model.W1, text_model.b1)])
        z2 = [sum(w*ai for w, ai in zip(ws, a1)) + b for ws, b in zip(text_model.W2, text_model.b2)]
        probs = sentiment_analyzer.softmax(z2)
        conf = probs[pred_num]
    elif text_model_name == "Logistic Regression":
        pred_num = text_model.predict([vec])[0]
        sentiment = num_to_label[pred_num]
        scores = [sum(w*xi for w, xi in zip(ws, vec)) + b for ws, b in zip(text_model.W, text_model.b)]
        probs = sentiment_analyzer.softmax(scores)
        conf = probs[pred_num]
    else:
        sentiment = text_model.predict(vec)
        conf = 1.0
        
    return {"sentiment": sentiment, "confidence": conf, "model": text_model_name}
