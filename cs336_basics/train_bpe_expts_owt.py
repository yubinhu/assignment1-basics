from time import time

from cs336_basics.modal_utils import DATA_PATH, VOLUME_MOUNTS, app, build_image
from cs336_basics.train_bpe import save_merges, save_vocab, train_bpe


@app.function(
    image=build_image(),
    volumes=VOLUME_MOUNTS,
    cpu=8.0,
    memory= 100_000,
    timeout= 12 * 60 * 60,
)
def train_bpe_owt():
    start_time = time()
    vocab, merges = train_bpe(DATA_PATH / "owt_train.txt", 32000, ["<|endoftext|>"])
    end_time = time()
    print(f"Training BPE took {end_time - start_time:.2f} seconds")

    save_vocab(DATA_PATH / "owt_train.txt.bpe.vocab.json", vocab)
    save_merges(DATA_PATH / "owt_train.txt.bpe.merges.json", merges)

    print("Vocabulary size:", len(vocab))
    print("First 10 vocabulary items:", list(vocab.items())[:10])

    print(
        "Longest 10 vocabulary items:",
        sorted(vocab.items(), key=lambda item: len(item[1]), reverse=True)[:10],
    )

if __name__ == "__main__":
    train_bpe_owt.local()
