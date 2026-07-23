from typing import Iterable, Iterator
import regex as re

from cs336_basics.train_bpe import merge_bp

def split_pre_tokens_linear(
    corpus: str, 
    special_tokens: list[str] = None,
) -> list[str]:
    res: list[str] = []

    if special_tokens: 
        special_pat = "|".join(
            re.escape(token)
            for token in sorted(special_tokens, key=len, reverse=True)
        )
        segments = re.split(f"({special_pat})", corpus)
    else:
        segments = [corpus]
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for segment  in segments:
        if segment in special_tokens:
            res.append(segment)
        words = re.findall(PAT, segment)
        for word in words:
            res.append(word)
    return res

class tokenizer:
    def __init__(
        self, 
        vocab: dict[int, bytes], 
        merges: list[tuple[bytes, bytes]], 
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.re_vocab = {}
        for i, b in vocab.items():
            self.re_vocab[b] = i
        self.merges = merges
        self.special_tokens = special_tokens or []

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        pass

    def encode(self, text: str) -> list[int]:
        from cs336_basics.train_bpe import split_pre_tokens
        pre_tokens = split_pre_tokens(text, self.special_tokens)
        for pre_token, (bl, ct) in pre_tokens.items():
            for lb, rb in self.merges:
                if (lb, rb) in zip(bl[:-1], bl[1:]):
                    new_bl = merge_bp(bl, (lb, rb))
                    pre_tokens[pre_token] = (new_bl, ct)
        pre_tokens_linear = split_pre_tokens_linear(text, self.special_tokens)
        res = []
        for pre_token in pre_tokens_linear:
            for bts in pre_tokens[pre_token][0]:
                if bts in self.re_vocab:
                    res.append(self.re_vocab[bts])
                else:
                    raise ValueError(f"Pretoken not found in vocab: {pre_token}")
        return res

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass

    def decode(self, ids: list[int]) -> str:
        bl: list[bytes] = []
        for id in ids:
            if id not in self.vocab:
                bl.append('\ufffd'.encode('utf-8'))
            else:
                bl.append(self.vocab[id])
        str_bytes = b''.join(bl)
        return str_bytes.decode('utf-8', errors='replace')
        

        

if __name__ == "__main__":
    from cs336_basics.train_bpe import load_vocab, load_merges
    vocab = load_vocab("data/TinyStoriesV2-GPT4-train.txt.bpe.vocab.json")
    merges = load_merges("data/TinyStoriesV2-GPT4-train.txt.bpe.merges.json")
    tokenizer = tokenizer(vocab, merges)
    encoded = tokenizer.encode("Hello, world! 你好啦啦啦")
    print(encoded)
    print(tokenizer.decode(encoded))