import gzip
import html
import os
from functools import lru_cache

import ftfy
import regex as re
from typing import Union, List

import torch
import pdb
import copy

@lru_cache()
def default_bpe():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe()):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text

    def get_vocab_size(self):
        return 49408

    def get_eot_token(self):
        return self.encoder["<|endoftext|>"]

    def get_sot_token(self):
        return self.encoder["<|startoftext|>"]

    def check_added_tokens(self):
        return 0

    def get_tokenizer_obj(self):
        return None

    def tokenize(self, texts: Union[str, List[str]], context_length: int = 77):
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self.encoder["<|startoftext|>"]
        eot_token = self.encoder["<|endoftext|>"]
        all_tokens = [[sot_token] + self.encode(text) + [eot_token] for text in texts]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                tokens = tokens[:context_length]
                # raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")

            result[i, :len(tokens)] = torch.tensor(tokens)

        return result

    def encode_with_idx(self, text, input_idxs):
        bpe_tokens = []
        cleaned_text = whitespace_clean(basic_clean(text)).lower()

        parsed_text = re.findall(self.pat, cleaned_text)
        # TODO not sure about that
        if not len(cleaned_text.split(' ')) == len(text.split(' ')) == len(parsed_text):
            for ii, input_id in enumerate(input_idxs):
                refer_word = text.split(' ')[input_id]
                if len(refer_word.split('-')) != 1:
                    refer_word = refer_word.split('-')[-1]
                refer_exist = [parse_id for parse_id, parse_word in enumerate(parsed_text) if parse_word == refer_word]
                # assert len(refer_exist) == 1, 'after cleaning, size is different'
                if len(refer_exist) != 1:
                    refer_exist_distance = [(refer_id- input_id)**2 for refer_id in refer_exist]
                    input_idxs[ii] = refer_exist[refer_exist_distance.index(min(refer_exist_distance))]
                else:
                    input_idxs[ii] = refer_exist[0]
            # raise ('after cleaning, size is different')
        total_added_input_idx = [0] * len(input_idxs)
        for token_idx, token in enumerate(re.findall(self.pat, cleaned_text)):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens_list = self.bpe(token).split(' ')
            if len(bpe_tokens_list) > 1:
                added_length = len(bpe_tokens_list) - 1
                added_input_idx = [added_length if token_idx <= iii else 0 for iii in input_idxs]
                total_added_input_idx = [added_input_idx[iii] + jjj for iii, jjj in enumerate(total_added_input_idx)]
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in bpe_tokens_list)

        input_idxs = [total_added_input_idx[iii] + jjj for iii, jjj in enumerate(input_idxs)]
        return bpe_tokens, input_idxs

    def tokenize_with_idx(self, texts: Union[str, List[str]], context_length: int = 77, input_idxs=None):
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self.encoder["<|startoftext|>"]
        eot_token = self.encoder["<|endoftext|>"]
        assert len(texts) == 1
        encoded_text, output_idxs = self.encode_with_idx(texts[0], input_idxs)
        all_tokens = [[sot_token] + encoded_text + [eot_token]]
        output_idxs = [iii+1 for iii in output_idxs]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                tokens = tokens[:context_length]
                # raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")

            result[i, :len(tokens)] = torch.tensor(tokens)

        return result, output_idxs


    def __call__(self, texts: Union[str, List[str]], context_length: int = 77):
        return self.tokenize(texts, context_length)
