import os 
import heapq
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
    checkpoint: bool = False,
):
    split_chunks = 32
    num_processes = 4
    
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
    
    return bpe_from_pre_tokens(merged_pre_tokens, special_tokens, vocab_size, input_path if checkpoint else None)

def split_pre_tokens(
    corpus: str, 
    special_tokens: list[str],
    retain_linear_translation: bool = False,
) -> tuple[
    dict[str, tuple[list[bytes], int]],
    list[str],
]:
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

class _MergeCandidate:
    __slots__ = ("count", "pair")

    def __init__(self, count, pair):
        self.count, self.pair = count, pair

    def __lt__(self, other):
        return (self.count, self.pair) > (other.count, other.pair)

def bpe_from_pre_tokens(
    pre_tokens,
    special_tokens,
    vocab_size,
    checkpoint_path=None,
):
    from collections import defaultdict
    checkpoint_paths = [Path(f"{checkpoint_path}.bpe.{kind}.json") for kind in ("vocab", "merges")] if checkpoint_path else []
    vocab : dict[int, bytes] = {}
    vocab_ctr = 0
    for i in range(256):
        vocab[vocab_ctr] = bytes([i])
        vocab_ctr += 1
    for token in special_tokens:
        vocab[vocab_ctr] = bytes(token, encoding='utf-8')
        vocab_ctr += 1
    merges = []
    if checkpoint_paths and all(path.exists() for path in checkpoint_paths):
        vocab, merges = load_vocab(checkpoint_paths[0]), load_merges(checkpoint_paths[1])
        vocab_ctr = len(vocab)
        print(f"Resuming from {len(merges)} merges")
        if len(vocab) >= vocab_size:
            return vocab, merges
        merge_ranks = {bp: rank for rank, bp in enumerate(merges)}
        for pre_token, (bl, ct) in pre_tokens.items():
            while len(bl) > 1:
                mbp = min(zip(bl[:-1], bl[1:]), key=lambda bp: merge_ranks.get(bp, len(merges)))
                if mbp not in merge_ranks:
                    break
                bl = merge_bp(bl, mbp)
            pre_tokens[pre_token] = (bl, ct)

    def save_checkpoint():
        if checkpoint_paths:
            save_vocab(checkpoint_paths[0], vocab)
            save_merges(checkpoint_paths[1], merges)

    bp_counter : dict[tuple[bytes, bytes], int] = defaultdict(int)
    bp_to_pretoken : dict[tuple[bytes, bytes], set[str]] = defaultdict(set)
    for pre_token, (bl, ct) in pre_tokens.items():
        for bp in zip(bl[:-1], bl[1:]):
            bp_counter[bp] += ct
            bp_to_pretoken[bp].add(pre_token)
    bp_heap = [_MergeCandidate(ct, bp) for bp, ct in bp_counter.items()]
    heapq.heapify(bp_heap)

    while len(vocab) < vocab_size:
        while bp_heap:
            candidate = heapq.heappop(bp_heap)
            if bp_counter.get(candidate.pair) == candidate.count:
                mbp = candidate.pair
                break
        else:
            break
        
        # Merge
        merges.append(mbp)
        vocab[vocab_ctr] = mbp[0] + mbp[1]
        vocab_ctr += 1

        changed_bps = set()
        affected_pre_tokens = bp_to_pretoken.pop(mbp)
        for pre_token in affected_pre_tokens:
            bl, ct = pre_tokens.pop(pre_token)
            new_bl = merge_bp(bl, mbp)
            pre_tokens[pre_token] = (new_bl, ct)

            old_bps = list(zip(bl[:-1], bl[1:]))
            new_bps = list(zip(new_bl[:-1], new_bl[1:]))
            # Update counts
            for bp in old_bps:
                bp_counter[bp] -= ct
            for bp in new_bps:
                bp_counter[bp] += ct
            old_bp_set, new_bp_set = set(old_bps), set(new_bps)
            changed_bps.update(old_bp_set | new_bp_set)

            for bp in old_bp_set - new_bp_set:
                bp_to_pretoken[bp].discard(pre_token)
            for bp in new_bp_set - old_bp_set:
                bp_to_pretoken[bp].add(pre_token)

        bp_counter.pop(mbp)
        for bp in changed_bps:
            count = bp_counter.get(bp, 0)
            if count > 0:
                heapq.heappush(bp_heap, _MergeCandidate(count, bp))
            else:
                bp_counter.pop(bp, None)
                bp_to_pretoken.pop(bp, None)
        if len(bp_heap) > max(1024, 2 * len(bp_counter)):
            bp_heap = [_MergeCandidate(ct, bp) for bp, ct in bp_counter.items()]
            heapq.heapify(bp_heap)
        if len(merges) % 1000 == 0:
            save_checkpoint()

    save_checkpoint()
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
