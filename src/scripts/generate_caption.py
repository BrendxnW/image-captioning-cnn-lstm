import argparse
import torch
from PIL import Image
import torchvision.transforms as T
from src.model.photo_captioner import PhotoCaptioner
from src.utils.text import build_vocab_from_csv 
from torchvision import models
import torch.nn as nn
from src.utils.text import load_vocab
import torch.nn.functional as F



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 30
SOS = "<SOS>"
EOS = "<EOS>"


def load_image(image_path):
    """
    Docstring for load_image
    
    :param image_path: Description
    """
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)



def apply_repetition_penalty(logits, generated_ids, penalty=1.2):
    """
    Penalize tokens that have already appeared.
    logits: (vocab_size,)
    generated_ids: list[int]
    """
    if penalty is None or penalty <= 1.0 or len(generated_ids) == 0:
        return logits
    # Reduce logit for previously used tokens
    for tok in set(generated_ids):
        logits[tok] = logits[tok] / penalty
    return logits

def ban_repeat_ngrams(logits, generated_ids, no_repeat_ngram_size=3):
    """
    Prevent generating any n-gram that already appeared (like HF no_repeat_ngram).
    logits: (vocab_size,)
    """
    n = no_repeat_ngram_size
    if n is None or n <= 1 or len(generated_ids) < n - 1:
        return logits

    # Build map: prefix (n-1 tokens) -> set(next_tokens) that would repeat an ngram
    prefix_len = n - 1
    ngram_map = {}
    for i in range(len(generated_ids) - n + 1):
        prefix = tuple(generated_ids[i:i+prefix_len])
        nxt = generated_ids[i+prefix_len]
        ngram_map.setdefault(prefix, set()).add(nxt)

    current_prefix = tuple(generated_ids[-prefix_len:])
    banned = ngram_map.get(current_prefix, set())
    if banned:
        logits[list(banned)] = -1e9  # effectively impossible
    return logits

@torch.no_grad()
def generate_caption(model, image, vocab, decode="beam",beam_size=3, max_len=MAX_LEN):
    """
    
    """
    model.eval()
    device = next(model.parameters()).device
    image = image.to(device)

    if decode == "beam":
        return generate_caption_beam(
            model=model,
            image=image,
            vocab=vocab,
            beam_size=beam_size,
            max_len=max_len,
            length_penalty=0.7,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
        )
    else:
        return generate_caption_greedy(
            model=model,
            image=image,
            vocab=vocab,
            max_len=max_len,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )


@torch.no_grad()
def generate_caption_beam(
    model,
    image,
    vocab,
    beam_size=3,
    max_len=20,
    length_penalty=0.9,
    repetition_penalty=1.25,
    no_repeat_ngram_size=3,
):
    device = next(model.parameters()).device
    image = image.to(device)

    # Encode image once
    feature = model.encoder(image)
    feature = torch.flatten(feature, 1)          # [1, 2048]
    projected = model.projection(feature)        # [1, 512]
    projected_step = projected.unsqueeze(1)      # [1, 1, 512]

    sos = vocab.word2idx["<SOS>"]
    eos = vocab.word2idx["<EOS>"]
    pad = vocab.word2idx.get("<PAD>", None)
    unk = vocab.word2idx.get("<UNK>", None)

    projected_step = projected.unsqueeze(1)  # [1,1,512]
    _, initial_hidden = model.decoder(projected_step)

    beams = [([sos], initial_hidden, 0.0)]
    finished = []

    def score_with_lp(token_ids, logprob):
        L = max(1, len(token_ids))
        return logprob / (L ** length_penalty)

    for _ in range(max_len):
        candidates = []

        for tokens, hidden, score in beams:
            if tokens[-1] == eos:
                finished.append((tokens, score))
                continue

            # Emergency brake: if last 3 tokens same, stop expanding
            if len(tokens) >= 4 and tokens[-1] == tokens[-2] == tokens[-3]:
                continue

            # Step using last token (this updates hidden based on history)
            prev_token = torch.tensor([[tokens[-1]]], device=device)  # [1,1]
            emb = model.embed(prev_token)                             # [1,1,512]
            x = torch.cat([projected_step, emb], dim=1)               # [1,2,512]

            outputs, hidden_after = model.decoder(x, hidden)
            logits = model.fc_out(outputs[:, -1, :]).squeeze(0)       # [vocab]

            # Ban specials
            if pad is not None:
                logits[pad] = -1e9
            if unk is not None:
                logits[unk] = -1e9

            # Anti-repeat based on current partial sequence
            logits = apply_repetition_penalty(logits, tokens, penalty=repetition_penalty)
            logits = ban_repeat_ngrams(logits, tokens, no_repeat_ngram_size=no_repeat_ngram_size)

            log_probs = F.log_softmax(logits, dim=-1)
            topk = torch.topk(log_probs, k=beam_size)

            # For each token choice, create a NEW hidden by advancing one more step
            for i in range(beam_size):
                next_id = int(topk.indices[i].item())
                next_score = float(topk.values[i].item())

                new_tokens = tokens + [next_id]
                new_score = score + next_score

                # If EOS, we can keep hidden as-is; we won't expand it further
                if next_id == eos:
                    candidates.append((new_tokens, hidden_after, new_score))
                    continue

                # Advance hidden with the chosen next token (unique per branch)
                next_token_tensor = torch.tensor([[next_id]], device=device)  # [1,1]
                next_emb = model.embed(next_token_tensor)                     # [1,1,512]
                x2 = torch.cat([projected_step, next_emb], dim=1)             # [1,2,512]
                _, new_hidden = model.decoder(x2, hidden_after)

                candidates.append((new_tokens, new_hidden, new_score))

        if not candidates:
            break

        # Keep top beams
        candidates.sort(key=lambda item: score_with_lp(item[0], item[2]), reverse=True)
        beams = candidates[:beam_size]

        if len(finished) >= beam_size:
            break

    # Choose best completed if exists
    if finished:
        finished.sort(key=lambda x: score_with_lp(x[0], x[1]), reverse=True)
        best_tokens = finished[0][0]
    else:
        beams.sort(key=lambda x: score_with_lp(x[0], x[2]), reverse=True)
        best_tokens = beams[0][0]

    # Convert to words
    words = []
    for tid in best_tokens:
        w = vocab.idx2word[tid]
        if w in ("<SOS>", "<PAD>"):
            continue
        if w == "<EOS>":
            break
        words.append(w)

    return " ".join(words)


@torch.no_grad()
def generate_caption_greedy(
    model,
    image,
    vocab,
    max_len=20,
    repetition_penalty=1.25,
    no_repeat_ngram_size=3,
):
    device = next(model.parameters()).device

    # Encode image once
    feature = model.encoder(image)
    feature = torch.flatten(feature, 1)          # [1, 2048]
    projected = model.projection(feature)        # [1, 512]

    sos_idx = vocab.word2idx["<SOS>"]
    eos_idx = vocab.word2idx["<EOS>"]
    pad_idx = vocab.word2idx.get("<PAD>", None)
    unk_idx = vocab.word2idx.get("<UNK>", None)

    generated = [sos_idx]
    prev_token = torch.tensor([[sos_idx]], device=device)  # [1,1]
    projected_step = projected.unsqueeze(1)                # [1,1,512]
    _, hidden = model.decoder(projected_step)

    for _ in range(max_len):
        emb = model.embed(prev_token)                 # [1,1,512]

        outputs, hidden = model.decoder(emb, hidden)
        logits = model.fc_out(outputs[:, -1, :]).squeeze(0)  # [vocab]

        # Optional: ban special tokens
        if pad_idx is not None:
            logits[pad_idx] = -1e9
        if unk_idx is not None:
            logits[unk_idx] = -1e9

        # Anti-repeat
        logits = apply_repetition_penalty(logits, generated, penalty=repetition_penalty)
        logits = ban_repeat_ngrams(logits, generated, no_repeat_ngram_size=no_repeat_ngram_size)

        next_id = int(torch.argmax(logits).item())
        generated.append(next_id)

        if next_id == eos_idx:
            break

        prev_token = torch.tensor([[next_id]], device=device)

    # Convert ids to words (skip SOS/EOS/PAD)
    words = []
    for tid in generated:
        w = vocab.idx2word[tid]
        if w in ("<SOS>", "<PAD>"):
            continue
        if w == "<EOS>":
            break
        words.append(w)

    return " ".join(words)


def build_model(vocab_size, pad_idx):
    """
    Docstring for build_model
    
    :param vocab_size: Description
    """
    resnet = models.resnet50(weights="IMAGENET1K_V1")
    resnet.fc = nn.Identity()
    encoder = nn.Sequential(*list(resnet.children())[:-1])
    model = PhotoCaptioner(encoder, vocab_size=vocab_size, pad_idx=pad_idx)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--ckpt", type=str, default="best.pt", help="Path to model checkpoint")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--captions_csv", type=str, help="CSV used to build vocab (training-time)")
    group.add_argument("--vocab_pkl", type=str, help="Pickle file with saved vocab (recommended)")
    args = parser.parse_args()

    vocab = load_vocab("vocab_2.pkl")
    vocab_size = len(vocab.word2idx)

    pad_idx = vocab.word2idx["<PAD>"]
    model = build_model(vocab_size, pad_idx).to(DEVICE)
    model.load_state_dict(torch.load(args.ckpt, map_location=DEVICE))

    image = load_image(args.image)

    caption = generate_caption(model, image, vocab, decode=None, beam_size=3, max_len=MAX_LEN)
    print("Greedy:", generate_caption(model, image, vocab, decode="greedy"))
    print("Beam:", generate_caption(model, image, vocab, decode="beam", beam_size=3))
    print("\nImage:", args.image)
    print("Caption:", caption)


if __name__ == "__main__":
    main()