from typing import Iterable, Iterator
import regex as re

from cs336_basics.train_bpe import merge_bp

class Tokenizer:
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
        self.merge_ranks : dict[tuple[bytes, bytes], int] = {
            pair: rank
            for rank, pair in enumerate(merges)
        }

        self.special_tokens = special_tokens or []
        next_id = max(self.vocab, default=-1) + 1
        for special_token in self.special_tokens:
            token_bytes = special_token.encode("utf-8")
            if token_bytes not in self.re_vocab:
                self.vocab[next_id] = token_bytes
                self.re_vocab[token_bytes] = next_id
                next_id += 1

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        from cs336_basics.train_bpe import load_merges, load_vocab
        return cls(load_vocab(vocab_filepath), load_merges(merges_filepath), special_tokens)

    def encode(self, text: str) -> list[int]:
        from cs336_basics.train_bpe import split_pre_tokens
        pre_tokens, pre_tokens_linear = split_pre_tokens(text, self.special_tokens, retain_linear_translation=True)
        for pre_token, (bl, ct) in pre_tokens.items():
            while len(bl) > 1:
                bps = zip(bl[:-1], bl[1:])
                hbp, hbpr = None, float('inf')
                for bp in bps:
                    if bp in self.merge_ranks.keys() and self.merge_ranks[bp] < hbpr:
                        hbp, hbpr = bp, self.merge_ranks[bp]
                if hbp is None:
                    break
                new_bl = merge_bp(bl, hbp)
                bl = new_bl
            pre_tokens[pre_token] = (bl, ct)
        res = []
        for pre_token in pre_tokens_linear:
            if pre_token in self.special_tokens:
                res.append(self.re_vocab[pre_token.encode("utf-8")])
                continue
            for bts in pre_tokens[pre_token][0]:
                if bts in self.re_vocab:
                    res.append(self.re_vocab[bts])
                else:
                    raise ValueError(f"Pretoken not found in vocab: {pre_token}")
        return res

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

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
    Tokenizer = Tokenizer(vocab, merges)
    encoded = Tokenizer.encode("Hello, world! 你好啦啦啦")
    print(encoded)
    print(Tokenizer.decode(encoded))
