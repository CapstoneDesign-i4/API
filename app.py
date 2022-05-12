import io

import requests
from PIL import Image
import urllib
import torch
from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def hello_world():
    return "success!!"


@app.route('/predict', methods=["POST"])
def predict():
    if not request.method == "POST":
        return
    if request.files.get("image"):
        image_file = request.files["image"]
        image_bytes = image_file.read()

        img = Image.open(io.BytesIO(image_bytes))

        results = model(img, size=640)
        data = results.pandas().xyxy[0].to_json(orient="records")
        return data
    if request.files.get("url"):
        url_file = request.files["url"]
        url = url_file.read().decode("UTF-8")
        res = urllib.request.urlopen(url).read()

        img = Image.open(io.BytesIO(res))
        results = model(img, size=640)
        data = results.pandas().xyxy[0].to_json(orient="records")
        return data

if __name__ == "__main__":
    model = torch.hub.load(
         "ultralytics/yolov5", "custom", path="last.pt", force_reload=True)
    model.eval()
    app.run(host="0.0.0.0", port=5000)
