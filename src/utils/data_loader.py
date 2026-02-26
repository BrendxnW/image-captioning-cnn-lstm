import torch
import os
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset
from torch.nn.utils.rnn import pad_sequence
from src.dataset.flickr_dataset import FlickrDataset
from src.utils.text import build_vocab_from_csv
from src.utils.text import save_vocab, load_vocab
from typing import Tuple, List
from jaxtyping import Float, Int
from torch import Tensor


if os.path.exists("src/vocab/vocab_2.pkl"):
    vocab = load_vocab("src/vocab/vocab_2.pkl")
else:
    train8 = pd.read_csv("data/flickr8k/Train/train.csv")
    train30 = pd.read_csv("data/flickr30k/Train/train.csv")
    traincoco = pd.read_csv("data/COCO/Train/coco_train.csv")
    train_all = pd.concat([train8, train30, traincoco], ignore_index=True)
    train_all.to_csv("data/train_combined.csv", index=False)

    vocab = build_vocab_from_csv("data/train_combined.csv", threshold=2)
    save_vocab(vocab, "src/vocab/vocab_2.pkl")


def caption_collate_fn(batch) -> Tuple[Int[Tensor, "t", Float[Tensor, "c h w"]]]:
    """
    Custom collate function for batching image-caption pairs.

    This function stacks image temsors into a single batch tensor and pads
    the variable-length caption temsors so that all captions in the batch 
    have the same sequence length.
    
    Args:
        batch (list of tuples): A list of image, caption pairs
            - image (Tensor): Image tensor
            - caption (Tensor): 1d tensor that consists tokenized caption

    Returns:
        Tuple[Int[Tensor, "t", Float[Tensor, "c h w"]]]: A tuple that contains the image tensor and caption tesnor
    """
    images, captions = zip(*batch)
    images = torch.stack(images)

    captions = pad_sequence(
        captions,
        batch_first=True,
        padding_value=0
    )

    return images, captions


def get_dataloaders(batch_size: int = 64, num_workers: int = 2) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Creates DataLoaders for the Flickr8k training and test datasets.

    This function defines an image preprocessing pipeline, initializes
    the Flickr8kDataset objects from their CSV files, and wraps them in
    PyTorch DataLoaders for batching and iteration during training/evaluation.

    Args:
        batch_size (int): Number of samples per batch. Defaults to 64.
        num_workers (int): Number of subprocesses used for data loading.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, List[str]]: A tuple that contains the data loader for training, 
                validation, and test and a list of the vocabulary tokens.
    """

    # Alters image and preprocesses for training and validation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
    ])

    # Data loaders for Flickr8k 
    training_data_8k = FlickrDataset(
        csv_file="data/flickr8k/Train/train.csv",
        root_dir="data/flickr8k/Images/Images/",
        vocab=vocab,
        transform=train_transform
    )

    test_data_8k = FlickrDataset(
        csv_file="data/flickr8k/Test/test.csv",
        root_dir="data/flickr8k/Images/Images/",
        vocab=vocab,
        transform=eval_transform
    )

    val_data_8k = FlickrDataset(
        csv_file="data/flickr8k/Validate/validate.csv",
        root_dir="data/flickr8k/Images/Images/",
        vocab=vocab,
        transform=eval_transform
    )

    # Data loaders for Flickr30k 
    training_data_30k = FlickrDataset(
        csv_file="data/flickr30k/Train/train.csv",
        root_dir="data/flickr30k/Images/",
        vocab=vocab,
        transform=train_transform
    )

    test_data_30k = FlickrDataset(
        csv_file="data/flickr30k/Test/test.csv",
        root_dir="data/flickr30k/Images/",
        vocab=vocab,
        transform=eval_transform
    )

    val_data_30k = FlickrDataset(
        csv_file="data/flickr30k/Validate/validate.csv",
        root_dir="data/flickr30k/Images/",
        vocab=vocab,
        transform=eval_transform
    )

    # Data loaders for COCO 
    training_data_coco = FlickrDataset(
        csv_file="data/COCO/Train/coco_train.csv",
        root_dir="data/COCO/Train/Images/train2014/",
        vocab=vocab,
        transform=train_transform
    )

    val_data_coco = FlickrDataset(
        csv_file="data/COCO/Validate/coco_val.csv",
        root_dir="data/COCO/Validate/Images/",
        vocab=vocab,
        transform=eval_transform
    )

    # Concatanates all three datasets
    train_ds = ConcatDataset([training_data_8k, training_data_30k, training_data_coco])
    val_ds = ConcatDataset([val_data_8k, val_data_30k, val_data_coco])
    test_ds = ConcatDataset([test_data_8k, test_data_30k])

    # Combined data loaders
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=caption_collate_fn
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=caption_collate_fn
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=caption_collate_fn
    )

    return train_loader, test_loader, val_loader, vocab