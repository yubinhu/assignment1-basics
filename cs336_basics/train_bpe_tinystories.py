from time import time

from train_bpe import train_bpe, save_vocab


def main():
    start_time = time()
    vocab, _ = train_bpe("data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])
    end_time = time()
    print(f"Training BPE took {end_time - start_time:.2f} seconds")

    save_vocab("data/TinyStoriesV2-GPT4-train.txt.bpe.vocab.json", vocab)

    print("Vocabulary size:", len(vocab))
    print("First 10 vocabulary items:", list(vocab.items())[:10])


if __name__ == "__main__":
    main()
