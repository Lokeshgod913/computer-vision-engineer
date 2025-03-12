Image Classification API - README
Introduction
This repository contains an image classification model built with TensorFlow/Keras and deployed using FastAPI. The model classifies images into multiple categories and is served via a REST API.
Features
- Preprocesses images (resize, normalize, augment)
- Trains a MobileNetV2-based CNN model
- Deploys the model as a REST API using FastAPI
- Provides an endpoint for image classification
- Containerized using Docker
Setup
Ensure you have the following installed:
- Python 3.8+
- TensorFlow
- FastAPI
- Uvicorn
- NumPy
- Pillow

Install dependencies using:
```bash
pip install -r requirements.txt
```
Training the Model
Run the following command to train the model:
```bash
python train_model.py
```
This will preprocess the dataset, train the model, and save it.
Running the API
Start the FastAPI server by running:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```
Making Predictions
Use `curl` or Postman to send an image for classification:
```bash
curl -X 'POST' 'http://127.0.0.1:8000/predict' \
-H 'accept: application/json' \
-H 'Content-Type: multipart/form-data' \
-F 'file=@image.jpg'
```
Response Format
{
  "class": "Predicted Class Name",
  "confidence": 0.98
}
Docker Support
To build and run the Docker container:
```bash
docker build -t image-classification-api .
docker run -p 8000:8000 image-classification-api
```
Deployment
For cloud deployment, you can use AWS, GCP, or Azure to host the API.
License
This project is released under the MIT License.
