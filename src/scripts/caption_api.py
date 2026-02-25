from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

from generate_caption import generate_caption

app = FastAPI()

@app.post("/caption")
async def caption(image: UploadFile = File(...)):
    data = await image.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    text = caption(img, decode="beam")
    return {"caption": text}