from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Union, List
import torch

from .simple_tokenizer import SimpleTokenizer

"""
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

def tokenize(texts: Union[str, List[str]], context_length: int = 77):
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            tokens = tokens[:context_length]
            # raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")

        result[i, :len(tokens)] = torch.tensor(tokens)

    return result
"""