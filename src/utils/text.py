import re
import pandas as pd
import pickle
from collections import Counter
from typing import List


def tokenize(text: str) -> str:
    """
    Tokenizes and normalizes a caption string.

    This function lowercases the input text and extract word tokens using
    a simple regex-based tokenizer.

    Args:
        text (str): Captions or sentence to tokenize

    Returns:
        list: List of lowercase words tokens extracted from input text.
    """
    return re.findall(r"\b\w+\b", text.lower())


class Vocabulary:
    """
    Vocabulary class for mapping tokens to integer indices and back.

    This class builds a word-to-index and index-to-word mapping from a list
    of tokens, filtering out rare words below a frequency threshold.

    Special tokens:
        <PAD>: Padding token (index 0)
        <UNK>: Unknown token
        <SOS>: Start-of-sequence token
        <EOS>: End-of-sequence token

    Args:
        tokens (list): List of tokens used to build the vocabulary.
        threshold (int, optional): Minimum frequency required for a token to be
                                   included in the vocabulary. Defaults to 2.
    """
    def __init__(self, tokens, threshold=2):
        """
        Initializes the Vocabulary class
        """
        self.threshold = threshold

        self.word2idx = {"<PAD>":0, "<UNK>":1, "<SOS>":2, "<EOS>":3}
        self.idx2word = {i:w for w, i in self.word2idx.items()}
        self.idx = 4

        self.freq = Counter(tokens)

        for word, count in self.freq.items():
            if count >= self.threshold:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def token_to_id(self, token: str) -> int:
        """
        Assigns a token to an integer

        Args:
            token (str): a string that was tokenized from a list

        Returns:
            int: Returns an assigned integer to the string
        """
        return self.word2idx.get(token, self.word2idx["<UNK>"])

    def id_to_token(self, idx: int) -> str:
        """
        Turns an integer to the corresponding token that was previously assigned

        Args:
            idx (int): The id number

        Returns:
            str: Returns the corresponding token from given id integer
        """
        return self.idx2word.get(idx, "<UNK>")
    

def build_vocab_from_csv(csv_path: str, threshold: int = 2) :
    """
    Docstring for build_vocab_from_csv
    
    Args:
        csv_path: Description
        threshold: Description

    Returns:

    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["caption"])

    all_tokens = []
    for caption in df["caption"].astype(str).tolist():
        all_tokens.extend(tokenize(caption))

    return Vocabulary(all_tokens, threshold=threshold)


def save_vocab(vocab: List[str], path="src/vocab/vocab_2.pkl") -> pickle:
    """
    Saves a list of vocabulary into a .pkl file

    Args:
        vocab List[str]: A list of vocabluary strings
        path (str): the path of where to save the .pkl file

    Returns:
        pickle: Saved .pkl file
    """
    with open(path, "wb") as f:
        pickle.dump(vocab, f)


def load_vocab(path="src/vocab/vocab_2.pkl"):
    """
    Loads in the .pkl file

    Args:
        path (str): the path of where the saved .pkl file is

    Returns:
        pickle: loaded .pkl file
    """
    with open(path, "rb") as f:
        return pickle.load(f)