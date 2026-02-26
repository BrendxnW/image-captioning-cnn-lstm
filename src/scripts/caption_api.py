import torch
import io
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from src.scripts.generate_caption import generate_caption, load_image, build_model
from src.utils.text import load_vocab
from pathlib import Path



app = FastAPI()

BASE_DIR = Path(__file__).resolve().parents[2]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PTH = BASE_DIR / "src" / "checkpoint" / "best_v7_retrain.pt"
VOCAB_PTH = BASE_DIR / "src" / "vocab" / "vocab_2.pkl"

_model = None
_vocab = None


def load_once():
    global _model, _vocab

    if _model is not None and _vocab is not None:
        return _model, _vocab

    vocab = load_vocab(VOCAB_PTH)
    pad_idx = vocab.word2idx["<PAD>"]

    model = build_model(len(vocab.word2idx), pad_idx).to(DEVICE)
    model.load_state_dict(torch.load(CKPT_PTH, map_location=DEVICE))
    model.eval()

    _model, _vocab = model, vocab
    return _model, _vocab

@app.post("/caption")
async def caption(image: UploadFile = File(...)):
    model, vocab = load_once()

    data = await image.read()
    pil_img = Image.open(io.BytesIO(data)).convert("RGB")

    img_tensor = load_image(pil_img).to(DEVICE)

    text = generate_caption(model=model, image=img_tensor, vocab=vocab, decode="beam")
    return {"caption": text}
