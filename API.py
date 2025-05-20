from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
from torchvision import transforms, models
import io
import numpy as np

# نفس الإعدادات من كودك الأصلي
class Config:
    DATA_PATH = "/kaggle/input/dataset/dataset"
    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 2
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_SAVE_PATH = "C:/Users/Dell/OneDrive/Desktop/AiModel/best_model.pth"

    #MODEL_SAVE_PATH = "C:\Users\Dell\OneDrive\Desktop\AiModel\best_model.pth"
    CLASS_NAMES = [
        "Dental benign tumors",
        "Dental Caries",
        "Dental Malignant tumors",
        "Gingivitis",
        "Hypodontia",
        "Mouth Ulcer",
        "Tooth Discoloration"
    ]
    NUM_CLASSES = len(CLASS_NAMES)
 
# نفس نموذجك
class DentalModel(torch.nn.Module):
    def __init__(self, num_classes=7, model_name="efficientnet_b2"):
        super(DentalModel, self).__init__()
        self.model = getattr(models, model_name)(weights="DEFAULT")
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.3, inplace=True),
            torch.nn.Linear(1408, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# تحميل النموذج
model = DentalModel(num_classes=Config.NUM_CLASSES)
model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE))
model.to(Config.DEVICE)
model.eval()

# التحويلات
transform = transforms.Compose([
    transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# التطبيق
app = FastAPI()

# للسماح بالطلبات من أي origin (اختياري لو هتجرب من فرونت منفصل)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # قراءة الصورة
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # تجهيز الصورة
        input_tensor = transform(image).unsqueeze(0).to(Config.DEVICE)

        # تنبؤ
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()

        result = {
            "prediction": Config.CLASS_NAMES[predicted_class],
            "confidence": round(confidence, 4),
            "probabilities": {
                Config.CLASS_NAMES[i]: round(float(probabilities[i]), 4)
                for i in range(len(Config.CLASS_NAMES))
            }
        }
        return JSONResponse(content=result)
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
############################################################################
import matplotlib.pyplot as plt
import uuid
import os
from fastapi.responses import FileResponse

# فولدر لحفظ الصور المؤقتة
CHARTS_DIR = "charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

@app.post("/predict-with-chart")
async def predict_with_chart(file: UploadFile = File(...)):
    try:
        # قراءة الصورة
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # تجهيز الصورة
        input_tensor = transform(image).unsqueeze(0).to(Config.DEVICE)

        # تنبؤ
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()

        # رسم الشارت
        labels = Config.CLASS_NAMES
        values = [float(probabilities[i]) for i in range(len(labels))]

        sorted_indices = np.argsort(values)[::-1]
        labels = [labels[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]

        plt.figure(figsize=(10, 6))
        bars = plt.barh(labels, values)
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                     f'{values[i]:.2%}', va='center')
        plt.xlabel("Probability")
        plt.title("Class Probabilities")
        plt.tight_layout()

        # حفظ الصورة
        chart_path = os.path.join(CHARTS_DIR, f"{uuid.uuid4()}.png")
        plt.savefig(chart_path)
        plt.close()

        return FileResponse(chart_path, media_type="image/png")

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
##############################################################################
from treatments import treatments
@app.get("/treatment/{condition}")
def get_treatment(condition: str):
    condition = condition.lower().strip()
    if condition in treatments:
        return {
            "condition": condition,
            "treatment": treatments[condition]
        }
    else:
        raise HTTPException(status_code=404, detail="Treatment not found")
    ##########################################################################
   