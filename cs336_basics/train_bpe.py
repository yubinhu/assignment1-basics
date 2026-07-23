import os 
import regex as re
from typing import BinaryIO
import multiprocessing as mp
import json
from pathlib import Path

def save_vocab(path, vocab):
    serialized = [
        {
            "index": token_id,
            "hex": vocab[token_id].hex(),
            "repr": repr(vocab[token_id]),
        }
        for token_id in range(len(vocab))
    ]
    Path(path).write_text(json.dumps(serialized, indent=2) + "\n", encoding="utf-8")


def save_merges(path, merges):
    serialized = [[left.hex(), right.hex()] for left, right in merges]
    Path(path).write_text(json.dumps(serialized, indent=2) + "\n", encoding="utf-8")


def load_merges(path):
    serialized = json.loads(Path(path).read_text(encoding="utf-8"))
    return [(bytes.fromhex(left), bytes.fromhex(right)) for left, right in serialized]


def load_vocab(path):
    serialized = json.loads(Path(path).read_text(encoding="utf-8"))

    # Continue to support vocabularies written by the original compact format.
    if serialized and isinstance(serialized[0], str):
        return {
            token_id: bytes.fromhex(token)
            for token_id, token in enumerate(serialized)
        }

    return {
        token["index"]: bytes.fromhex(token["hex"])
        for token in serialized
    }

def find_chunk_boundaries(
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
):
    split_chunks = 32
    num_processes = 8
    
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, split_chunks, b"<|endoftext|>")
    with mp.Pool(processes=num_processes) as pool:
        tasks = [(input_path, special_tokens, boundary) for boundary in zip(boundaries[:-1], boundaries[1:])]
        chunk_pre_tokens = pool.starmap(
            split_pre_tokens_from_file, tasks
        )

    # Merge the pre-token counts from all chunks
    merged_pre_tokens : dict[str, tuple[list[bytes], int]] = {}
    for pre_tokens in chunk_pre_tokens:
        for pre_token, (bl, ct) in pre_tokens.items():
            if pre_token in merged_pre_tokens:
                merged_bl, merged_ct = merged_pre_tokens[pre_token]
                merged_pre_tokens[pre_token] = (bl, merged_ct + ct)
            else:
                merged_pre_tokens[pre_token] = (bl, ct)
    
    return bpe_from_pre_tokens(merged_pre_tokens, special_tokens, vocab_size)

def split_pre_tokens(
    corpus: str, 
    special_tokens: list[str],
    retain_linear_translation: bool = False,
) -> dict[str, tuple[list[bytes], int]]:
    pre_tokens : dict[str, tuple[list[bytes], int]] = {}
    linear_translation : list[str] = []
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
        if segment in (special_tokens or []):
            if retain_linear_translation:
                linear_translation.append(segment)
            continue

        words = re.findall(PAT, segment)
        for word in words:
            if retain_linear_translation:
                linear_translation.append(word)
            if word in pre_tokens:
                word_bl, ct = pre_tokens[word]
                pre_tokens[word] = (word_bl, ct + 1)
            else:
                word_bytes = bytes(word, encoding="utf-8")
                word_bl : list[bytes] = [word_bytes[i:i+1] for i in range(len(word_bytes))]
                pre_tokens[word] = (word_bl, 1)

    return pre_tokens, linear_translation

def split_pre_tokens_from_file(
    input_path: str | os.PathLike, 
    special_tokens: list[str],
    boundary: tuple[int, int] | None = None,
) -> dict[str, tuple[list[bytes], int]]:
    with open(input_path, "rb") as f:
        if boundary:
            f.seek(boundary[0])
            corpus = f.read(boundary[1] - boundary[0]).decode("utf-8", errors="ignore")
        else:
            corpus = f.read().decode("utf-8", errors="ignore")
    return split_pre_tokens(corpus, special_tokens)[0]

def bpe_from_pre_tokens(
    pre_tokens,
    special_tokens,
    vocab_size,
):
    from collections import defaultdict
    vocab : dict[int, bytes] = {}
    vocab_ctr = 0
    for i in range(256):
        vocab[vocab_ctr] = bytes([i])
        vocab_ctr += 1
    for token in special_tokens:
        vocab[vocab_ctr] = bytes(token, encoding='utf-8')
        vocab_ctr += 1
    merges = []
    bp_counter : dict[tuple[bytes, bytes], int] = defaultdict(int)
    bp_to_pretoken : dict[tuple[bytes, bytes], set[str]] = defaultdict(set)
    for pre_token, (bl, ct) in pre_tokens.items():
        for bp in zip(bl[:-1], bl[1:]):
            bp_counter[bp] += ct
            bp_to_pretoken[bp].add(pre_token)

    while len(vocab) < vocab_size:
        # Find max bp
        mbp:tuple = max(bp_counter.keys(), key=lambda bp: (bp_counter[bp], bp))        
        
        # Merge
        merges.append(mbp)
        vocab[vocab_ctr] = mbp[0] + mbp[1]
        vocab_ctr += 1
        
        for pre_token in bp_to_pretoken[mbp]:
            bl, ct = pre_tokens.pop(pre_token)
            new_bl = merge_bp(bl, mbp)
            pre_tokens[pre_token] = (new_bl, ct)
            # Update counts
            for bp in zip(bl[:-1], bl[1:]):
                bp_counter[bp] -= ct
            for bp in zip(new_bl[:-1], new_bl[1:]):
                bp_counter[bp] += ct
                bp_to_pretoken[bp].add(pre_token)

        bp_counter.pop(mbp) # This is not a bp anymore
        bp_to_pretoken.pop(mbp)

    return vocab, merges

def merge_bp(bl: list[bytes], bp: tuple[bytes, bytes]) -> list[bytes]:
    new_bl = []
    i = 0
    while i < len(bl):
        if i == len(bl) - 1:
            new_bl.append(bl[i])
            break
        if (bl[i], bl[i+1]) == bp:
            new_bl.append(bl[i] + bl[i+1])
            i += 1
        else:
            new_bl.append(bl[i])
        i += 1
    return new_bl