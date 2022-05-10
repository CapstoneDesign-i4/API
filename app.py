import io
from PIL import Image

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
        image_url = request.files["url"]
        image_bytes = image_url.read()
        print(image_url)
        img = Image.open(io.BytesIO(image_bytes))

        results = model(img, size=640)
        data = results.pandas().xyxy[0].to_json(orient="records")
        return data


if __name__ == "__main__":

    model = torch.hub.load(
         "ultralytics/yolov5", "custom", path="last.pt", force_reload=True)
    model.eval()
    app.run(host="0.0.0.0", port=5000)
    res = request.post("http://127.0.0.1:5000", files = {"url": "http://ec2-3-39-141-76.ap-northeast-2.compute.amazonaws.com/media/admin/%EC%BB%A8%ED%95%981734/22.jpg"}).json()
    print(res)