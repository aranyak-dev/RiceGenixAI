import requests
from PIL import Image
import io

API_URL = "https://your-huggingface-api-url"  # will update later

def load_model():
    return "api_model"

def predict(model, image):
    try:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")

        response = requests.post(
            API_URL,
            files={"file": buffered.getvalue()}
        )

        result = response.json()

        return result.get("prediction", "Unknown")

    except:
        return "AI Error"
