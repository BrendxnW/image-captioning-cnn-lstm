import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from src.model.photo_captioner import PhotoCaptioner
from src.utils.data_loader import get_dataloaders
from src.utils.text import load_vocab
from src.scripts.generate_caption import generate_caption

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None
_vocab = None

def _load_once(ckpt_path="src/checkpoint/best_v7_retrain.pt"):
    global _model, _vocab
    if _model is None:
        _vocab = load_vocab("vocab_2.pkl")

        vocab_size = len(_vocab.word2idx)
        pad_idx = _vocab.word2idx["<PAD>"]

        resnet_model = models.resnet50(weights="IMAGENET1K_V1")
        feat_extract = nn.Sequential(*list(resnet_model.children())[:-1])
        model = PhotoCaptioner(feat_extract, vocab_size, pad_idx).to(device)

        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        model.eval().to(device)
        _model = model

def caption_image(image_path, ckpt_path="src/checkpoint/best_v7_retrain.pt"):
    _load_once(ckpt_path)

    train_loader, test_loader, val_loader, _ = get_dataloaders()
    image = Image.open(image_path).convert("RGB")
    transform = val_loader()
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        caption = _model.generate_caption(_model, image, _vocab, decode=None, beam_size=3, max_len=30)

    return caption