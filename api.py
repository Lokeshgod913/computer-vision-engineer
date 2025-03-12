from fastapi import FastAPI, UploadFile, File
from io import BytesIO
from PIL import Image
import uvicorn
import numpy as np
import tensorflow as tf

# Load Model
MODEL_PATH = "./model/mobilenetv2_saved_model"
IMG_SIZE = (224, 224)
model = tf.keras.models.load_model(MODEL_PATH)
class_names = {0: 'Class_A', 1: 'Class_B', 2: 'Class_C'}  # Modify based on dataset

# Initialize FastAPI app
app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    image = image.resize(IMG_SIZE)
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions)
    return {"class": class_names.get(predicted_class, "Unknown"), "confidence": float(np.max(predictions))}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
