import requests

image_path = "test_xray.png"

with open(image_path, "rb") as f:
    response = requests.post(
        "http://127.0.0.1:8001/predict_gradcam",
        files={"file": f}
    )

print(response.status_code)

data = response.json()

print("Prediction:", data["prediction"])
print("Confidence:", data["confidence"])
print("Has Grad-CAM:", "overlay_base64" in data)