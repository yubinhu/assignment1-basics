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
    num_processes = 8
    
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    with mp.Pool(processes=num_processes) as pool:
        tasks = [(input_path, special_tokens, boundary) for boundary in zip(boundaries[:-1], boundaries[1:])]
        chunk_pre_tokens = pool.starmap(
            split_pre_tokens, tasks
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
    pre_tokens : dict[str, tuple[list[bytes], int]] = {}
    special_pat = "|".join(
        re.escape(token)
        for token in sorted(special_tokens, key=len, reverse=True)
    )

    segments = re.split(special_pat, corpus)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for segment  in segments:
        words = re.findall(PAT, segment)
        for word in words:
            if word in pre_tokens:
                word_bl, ct = pre_tokens[word]
                pre_tokens[word] = (word_bl, ct + 1)
            else:
                word_bytes = bytes(word, encoding="utf-8")
                word_bl : list[bytes] = [word_bytes[i:i+1] for i in range(len(word_bytes))]
                pre_tokens[word] = (word_bl, 1)

    return pre_tokens

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
            i = 0
            new_bl = []
            while i < len(bl):
                if i == len(bl) - 1:
                    new_bl.append(bl[i])
                    break
                if (bl[i], bl[i+1]) == mbp:
                    # update bytes list
                    new_bytes = bl[i] + bl[i+1]
                    new_bl.append(new_bytes)
                    i += 1 # skip the next bytes
                else:
                    new_bl.append(bl[i])
                i += 1
            # print("new_pre_token", new_pre_token)
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
