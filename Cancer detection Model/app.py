from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import AutoImageProcessor, ViTForImageClassification
import torch
from PIL import Image
import io

app = FastAPI(title="Skin Cancer Detection API")

# model path
MODEL_PATH = r"D:\Graduation Project\Cancer Model\vit_skin_cancer_model"

# load model and processor once at startup
print("Loading model...")
processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
model = ViTForImageClassification.from_pretrained(MODEL_PATH)
model.eval()
print("Model loaded!")

# label mapping
labels = {
    0: "akiec - Actinic Keratosis (Pre-malignant)",
    1: "bcc - Basal Cell Carcinoma (Malignant)",
    2: "bkl - Benign Keratosis (Benign)",
    3: "df - Dermatofibroma (Benign)",
    4: "mel - Melanoma (Malignant)",
    5: "nv - Melanocytic Nevi (Benign)",
    6: "vasc - Vascular Lesion (Benign)"
}

@app.get("/")
def root():
    return {"message": "Skin Cancer Detection API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[0]
        predicted_idx = probs.argmax().item()
        confidence = probs[predicted_idx].item()

    return JSONResponse({
        "prediction": labels[predicted_idx],
        "confidence": f"{confidence * 100:.2f}%",
        "all_probabilities": {
            labels[i]: f"{probs[i].item() * 100:.2f}%"
            for i in range(len(labels))
        }
    })