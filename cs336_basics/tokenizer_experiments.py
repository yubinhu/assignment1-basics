# NOTE: this script is mostly logistics and wholy written by grok4.5 atm

# (a) Sample 10 documents from TinyStories and OpenWebText. Using your previously-trained 
# TinyStories and OpenWebText tokenizers (10K and 32K vocabulary size, respectively), 
# encode these sampled documents into integer IDs. What is each tokenizer’s compression ratio 
# (bytes/token)?
# Deliverable: A one-to-two sentence response.
# (b) What happens if you tokenize your OpenWebText sample with the TinyStories tokenizer? 
# Compare the compression ratio and/or qualitatively describe what happens.
# Deliverable: A one-to-two sentence response.
# (c) Estimate the throughput of your tokenizer (e.g., in bytes/second). How long would it take to 
# tokenize the Pile dataset (825GB of text)?
# Deliverable: A one-to-two sentence response.
# (d) Using your TinyStories and OpenWebText tokenizers, encode the respective training and 
# development datasets into a sequence of integer token IDs. We’ll use this later to train our 
# 12
# language model. We recommend serializing the token IDs as a NumPy array of datatype 
# uint16. Why is uint16 an appropriate choice?
# Deliverable: A one-to-two sentence response

"""Tokenizer experiments (a)–(d) for CS336 Assignment 1."""

from __future__ import annotations

import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train_bpe import find_chunk_boundaries

DATA = Path("data")
SPECIAL = ["<|endoftext|>"]
SPECIAL_BYTES = b"<|endoftext|>"

# Worker-local tokenizer (loaded once per process via initializer).
_WORKER_TOK: Tokenizer | None = None

# Trained tokenizer artifacts (from train_bpe_tinystories / train_bpe_expts_owt)
TS_VOCAB = DATA / "TinyStoriesV2-GPT4-train.txt.bpe.vocab.json"
TS_MERGES = DATA / "TinyStoriesV2-GPT4-train.txt.bpe.merges.json"
OWT_VOCAB = DATA / "owt_train.txt.bpe.vocab.json"
OWT_MERGES = DATA / "owt_train.txt.bpe.merges.json"

# Raw corpora
TS_TRAIN = DATA / "TinyStoriesV2-GPT4-train.txt"
TS_VALID = DATA / "TinyStoriesV2-GPT4-valid.txt"
OWT_TRAIN = DATA / "owt_train.txt"
OWT_VALID = DATA / "owt_valid.txt"


def load_tokenizer(vocab_path: Path, merges_path: Path) -> Tokenizer:
    return Tokenizer.from_files(vocab_path, merges_path, special_tokens=SPECIAL)


def sample_documents(path: Path, n: int = 10) -> list[str]:
    """Sample first n documents separated by <|endoftext|>."""
    docs: list[str] = []
    buf: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if "<|endoftext|>" in line:
                before, _, after = line.partition("<|endoftext|>")
                buf.append(before)
                docs.append("".join(buf).rstrip("\n"))
                buf = [after] if after.strip() else []
                if len(docs) >= n:
                    break
            else:
                buf.append(line)
    if len(docs) < n and buf:
        docs.append("".join(buf))
    assert len(docs) >= n, f"Only found {len(docs)} docs in {path}"
    return docs[:n]


def compression_ratio(text: str, token_ids: list[int]) -> float:
    """bytes / token (UTF-8). Higher ⇒ more compression."""
    n_bytes = len(text.encode("utf-8"))
    n_tokens = len(token_ids)
    return n_bytes / n_tokens if n_tokens else float("nan")


def part_a() -> None:
    print("=== (a) Compression ratios on 10 sampled docs ===")
    ts_tok = load_tokenizer(TS_VOCAB, TS_MERGES)
    owt_tok = load_tokenizer(OWT_VOCAB, OWT_MERGES)

    ts_docs = sample_documents(TS_TRAIN, 10)
    owt_docs = sample_documents(OWT_TRAIN, 10)

    ts_text = "<|endoftext|>".join(ts_docs)
    owt_text = "<|endoftext|>".join(owt_docs)

    ts_ids = ts_tok.encode(ts_text)
    owt_ids = owt_tok.encode(owt_text)

    print(f"TinyStories tokenizer on TinyStories sample: {compression_ratio(ts_text, ts_ids):.3f} bytes/token")
    print(f"OpenWebText tokenizer on OpenWebText sample: {compression_ratio(owt_text, owt_ids):.3f} bytes/token")
    print(f"  (#bytes ts={len(ts_text.encode())}, #toks={len(ts_ids)}; "
          f"#bytes owt={len(owt_text.encode())}, #toks={len(owt_ids)})")


def part_b() -> None:
    print("\n=== (b) Cross-tokenize: OWT sample with TinyStories tokenizer ===")
    ts_tok = load_tokenizer(TS_VOCAB, TS_MERGES)
    owt_tok = load_tokenizer(OWT_VOCAB, OWT_MERGES)

    owt_docs = sample_documents(OWT_TRAIN, 10)
    owt_text = "<|endoftext|>".join(owt_docs)

    ids_matched = owt_tok.encode(owt_text)
    ids_cross = ts_tok.encode(owt_text)

    r_matched = compression_ratio(owt_text, ids_matched)
    r_cross = compression_ratio(owt_text, ids_cross)
    print(f"OWT tok on OWT: {r_matched:.3f} bytes/token ({len(ids_matched)} toks)")
    print(f"TS  tok on OWT: {r_cross:.3f} bytes/token ({len(ids_cross)} toks)")
    print("Expect worse (higher bytes/token) with TS: less domain match ⇒ shorter merges / more tokens.")
    # Optional qualitative peek
    print("Sample decode prefix (TS tok):", repr(ts_tok.decode(ids_cross[:40])))


def part_c(num_bytes: int = 10_000_000) -> None:
    """Throughput on a ~num_bytes prefix; extrapolate to The Pile (825GB)."""
    print("\n=== (c) Throughput estimate ===")
    tok = load_tokenizer(TS_VOCAB, TS_MERGES)  # or OWT; pick one / report both

    with TS_TRAIN.open("r", encoding="utf-8") as f:
        text = f.read(num_bytes)
    # trim to valid UTF-8 / complete chars already handled by text mode
    raw_bytes = len(text.encode("utf-8"))

    t0 = time.perf_counter()
    ids = tok.encode(text)
    elapsed = time.perf_counter() - t0

    throughput = raw_bytes / elapsed  # bytes/sec
    pile_bytes = 825 * (1024**3)
    pile_seconds = pile_bytes / throughput

    print(f"Encoded {raw_bytes:,} bytes → {len(ids):,} tokens in {elapsed:.2f}s")
    print(f"Throughput: {throughput:,.0f} bytes/s ({throughput / 1e6:.2f} MB/s)")
    print(f"Time for The Pile (825GB): {pile_seconds / 3600:.1f} hours "
          f"({pile_seconds / 86400:.1f} days)")


def _init_encode_worker(vocab_path: str, merges_path: str) -> None:
    global _WORKER_TOK
    _WORKER_TOK = Tokenizer.from_files(vocab_path, merges_path, special_tokens=SPECIAL)


def _encode_chunk(args: tuple[str, int, int, str]) -> tuple[str, int, int]:
    """Encode [start, end) of path; stream uint16 ids to a raw shard file."""
    path, start, end, shard_path = args
    assert _WORKER_TOK is not None

    with open(path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")

    n_tokens = 0
    max_id = 0
    # Encode document-by-document so peak RAM stays proportional to one doc,
    # not the whole chunk.
    docs = text.split("<|endoftext|>")
    with open(shard_path, "wb") as out:
        for i, doc in enumerate(docs):
            piece = doc if i == len(docs) - 1 else doc + "<|endoftext|>"
            if not piece:
                continue
            ids = _WORKER_TOK.encode(piece)
            if not ids:
                continue
            arr = np.asarray(ids, dtype=np.uint16)
            out.write(arr.tobytes())
            n_tokens += arr.size
            max_id = max(max_id, int(arr.max()))
    return shard_path, n_tokens, max_id


def encode_file_to_uint16(
    path: Path,
    vocab_path: Path,
    merges_path: Path,
    out_path: Path,
    num_workers: int | None = None,
) -> None:
    """
    Parallel stream-encode a corpus to a uint16 .npy array.

    Splits on <|endoftext|> boundaries, encodes chunks in a process pool,
    writes per-chunk raw shards, then concatenates into a memmap .npy.
    """
    num_workers = num_workers or max(20, (os.cpu_count() or 4))
    # More chunks than workers for load balance across uneven doc sizes.
    num_chunks = max(num_workers * 4, num_workers)

    print(f"Encoding {path} → {out_path}  ({num_workers} workers, ~{num_chunks} chunks)")
    t0 = time.perf_counter()

    with path.open("rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, SPECIAL_BYTES)
    ranges = list(zip(boundaries[:-1], boundaries[1:]))
    print(f"  split into {len(ranges)} chunks")

    shard_dir = out_path.with_suffix(".shards.tmp")
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True)

    tasks = [
        (str(path), start, end, str(shard_dir / f"shard_{i:05d}.bin"))
        for i, (start, end) in enumerate(ranges)
    ]

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_encode_worker,
        initargs=(str(vocab_path), str(merges_path)),
    ) as pool:
        results = list(pool.map(_encode_chunk, tasks))

    total = sum(n for _, n, _ in results)
    max_id = max((m for _, _, m in results), default=0)
    assert max_id < 2**16, f"token id {max_id} does not fit in uint16"

    # Stream shards into a memmapped .npy without holding all tokens in RAM.
    arr = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.uint16, shape=(total,)
    )
    offset = 0
    for shard_path, n, _ in results:
        if n:
            shard = np.fromfile(shard_path, dtype=np.uint16)
            assert shard.size == n
            arr[offset : offset + n] = shard
            offset += n
        Path(shard_path).unlink(missing_ok=True)
    arr.flush()
    shutil.rmtree(shard_dir, ignore_errors=True)

    elapsed = time.perf_counter() - t0
    bytes_in = path.stat().st_size
    print(
        f"  wrote {total:,} tokens, max_id={max_id} in {elapsed:.1f}s "
        f"({bytes_in / elapsed / 1e6:.1f} MB/s)"
    )


def part_d() -> None:
    print("\n=== (d) Serialize train/valid as uint16 ===")

    # Why uint16? vocab sizes 10_000 and 32_000 are both < 65_536, so every
    # token id fits in 2 bytes; uint8 (256) is too small, int32 wastes space.
    print(
        "uint16 is appropriate because max token id < 65536 for both vocabs "
        "(10k / 32k), so 2 bytes/token is enough without wasting space like int32."
    )

    encode_file_to_uint16(
        TS_TRAIN, TS_VOCAB, TS_MERGES, DATA / "tinystories_train_ids.npy"
    )
    encode_file_to_uint16(
        TS_VALID, TS_VOCAB, TS_MERGES, DATA / "tinystories_valid_ids.npy"
    )
    encode_file_to_uint16(
        OWT_TRAIN, OWT_VOCAB, OWT_MERGES, DATA / "owt_train_ids.npy"
    )
    encode_file_to_uint16(
        OWT_VALID, OWT_VOCAB, OWT_MERGES, DATA / "owt_valid_ids.npy"
    )


if __name__ == "__main__":
    # part_a()
    # part_b()
    # part_c()
    part_d()  # slow / large — uncomment when ready