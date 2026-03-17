---
title: Photo Captioning Bot
colorFrom: purple
colorTo: pink
sdk: docker
sdk_version: 28.3.2
app_port: 7860
python_version: 3.13.7
pinned: false
---

<h1 align="center">Photo Captioning Bot</h1>
<p align="center">
  <strong>Offline CNN–LSTM image captioning in PyTorch</strong>
</p>
<p align="center">
  An offline image captioning model built with PyTorch that generates short, relevant captions for photos using a CNN–LSTM pipeline.  
  Designed as a local, privacy-friendly baseline for multimodal learning and experimentation.
</p>
<p align="center">
  <a href="https://www.python.org/downloads/">
      <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" />
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-2.10-orange?logo=pytorch&logoColor=white" />
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/Docker-containerized-blue?logo=Docker&logoColor=white" />
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green" />
  </a>
</p>

## Table of Contents
- [Security](#security)
- [Background](#background)
- [Demo Video](#demo-video)
- [Project Structure](#project-structure)
- [Install](#install)
- [Usage](#usage)
- [API](#api)
- [Model Details](#model-details)
- [Testing](#testing)
- [Deployment](#deployment)
- [License](#license)

## Security
This project is designed to run fully locally and doesnt uplaod images or captions to external servers to protect user data. All photo processing and caption generation happens on the user's machine to minimize privacy risks. Since the tool will operate on unencrypted local backups or personal photo directiores, users should ensure that sensitive data is handled securiy and the generated outputs are stored appropriately. This project is inteded for research and educational use.


## Background
Photo captioning sits in between computer vision and natural language processing where it enables machines to translate visual content into human readable text. While there are many solutions that exist, they rely on cloud APIs and closed systems, this project focuses on building a fully local CNN-LSTM captioning pipleine in PyTorch so users can keep their data private.

## Demo Video
[Filler]
[Link Filler]

## Project Structure
```text
photo-captioning/
|____src/
    |____models/
        |____photo_captioner.py
    |____scripts/
        |____caption_api.py
        |____generate_caption.py
    |____test/
        |____test_output.py
    |____training/
        |____train.py
    |____utils/
        |____convert_coco_csv.py
        |____data_loader.py
        |____split_data.py
        |____text.py
|____.dockerignore
|____.gitattributes
|____.gitignore
|____DockerFile
|____LICENSE
|____README.md
|____requirements.txt

```


## Install
```bash
git clone https://github.com/BrendxnW/photo-captioner.git  
cd photo-captioner

# Windows
python -m venv venv
source venv/Scripts/activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```
## Usage
**Training**
```bash
python -m src.training.train
```
**Inference**
```bash
python -m src.scripts.generate_caption \
  --image data/Images/[your_image.jpg] \
```
*NOTE:* Change '[your_image.jpg]' to the name of your .jpg file without the brackets.  


**Example**
```bash
python -m src.scripts.generate_caption \
  --image data/Images/10815824_2997e03d76.jpg \
```
**Output**  
```bash
Caption: a man in a red shirt is standing in front of a large crowd
```

## API
**POST /generate_caption**

Generate a caption for an uploaded image.

**Request**
- image: an image file

**Response**
```json
{
  "image": data/Images/10815824_2997e03d76.jpg
  "caption": "a man in a red shirt is standing in front of a large crowd",
}
```

## Model Details
**Architecture:**
- Encoder: ResNet50
- Decoder: LSTM

  
**Framework:**
- PyTorch


**Datasets:**
- flickr8k
- flickr30k
- COCO

  
**Vocabulary:**


A vocabulary was constructed from the training captions, including special tokens:
- '<SOS\>' - Start of sequence
- '<EOS\>' - End of sequence
- '<PAD\>' - Padding token
- '<UNK\>' - Unkown token

 
## Testing

Run tests with:
```bash

```

## Deployment

Backend: <a href="https://huggingface.co/spaces/BrendxnW/photo-captioner">HuggingFace</a>

## License
[MIT © Richard McRichface.](./LICENSE)
