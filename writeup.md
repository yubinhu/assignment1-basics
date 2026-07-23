# Assigment 1 Write Up

### train_bpe_expts_owt
- (a) ÃÂ × 16. Makes sense but looks like OWT is a lower quality dataset than tiny stories. 
- (b) The longest tokens in tinystories are [(7160, b' accomplishment'), (9143, b' disappointment'), (9379, b' responsibility'), (3228, b' uncomfortable'), (3515, b' compassionate'), (5319, b' understanding'), (6386, b' neighbourhood'), (6497, b' Unfortunately'), (6874, b' determination'), (7756, b' encouragement')]
Top tokens in OWT: 
Token ID	Bytes	Decoded content
25822	64	ÃÂ × 16
25836	64	- × 64
31274	48	— × 16
10900	32	- × 32
15947	32	_ × 32
16885	32	ÃÂ × 8
25146	32	= × 32
28585	32	. × 32
31162	32	* × 32
15279	24	— × 8

The longest ones in tiny stories are more emotionally coded and the ones in OWT come from artifacts

### tokenizer_experiments

=== (a) Compression ratios on 10 sampled docs ===
TinyStories tokenizer on TinyStories sample: 4.187 bytes/token
OpenWebText tokenizer on OpenWebText sample: 4.702 bytes/token
  (#bytes ts=7533, #toks=1799; #bytes owt=31604, #toks=6721)

=== (b) Cross-tokenize: OWT sample with TinyStories tokenizer ===
OWT tok on OWT: 4.702 bytes/token (6721 toks)
TS  tok on OWT: 3.198 bytes/token (9882 toks)
Expect worse (higher bytes/token) with TS: less domain match ⇒ shorter merges / more tokens.
Sample decode prefix (TS tok): "What wouldn't you do to save someone you love?\n\nWhen They Come Calling is a modern ghost story, a suspenseful weaving of urban battles,"

=== (c) Throughput estimate ===
Encoded 10,003,969 bytes → 2,430,529 tokens in 0.94s
Throughput: 10,612,098 bytes/s (10.61 MB/s)
Time for The Pile (825GB): 23.2 hours (1.0 days)

=== (d) Serialize train/valid as uint16 ===
uint16 is appropriate because max token id < 65536 for both vocabs (10k / 32k), so 2 bytes/token is enough without wasting space like int32.
Encoding data/TinyStoriesV2-GPT4-train.txt → data/tinystories_train_ids.npy  (32 workers, ~128 chunks)
  split into 128 chunks
  wrote 541,229,347 tokens, max_id=9999 in 84.0s (26.5 MB/s)
Encoding data/TinyStoriesV2-GPT4-valid.txt → data/tinystories_valid_ids.npy  (32 workers, ~128 chunks)
  split into 128 chunks
  wrote 5,465,883 tokens, max_id=9999 in 0.9s (24.3 MB/s)
Encoding data/owt_train.txt → data/owt_train_ids.npy  (32 workers, ~128 chunks)
  split into 128 chunks
  wrote 2,727,120,452 tokens, max_id=31999 in 508.3s (23.5 MB/s)
Encoding data/owt_valid.txt → data/owt_valid_ids.npy  (32 workers, ~128 chunks)
  split into 128 chunks
  wrote 66,401,098 tokens, max_id=31999 in 13.6s (21.3 MB/s)

