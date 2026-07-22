from train_bpe import train_bpe

_, vocab = train_bpe("data/TinyStoriesV2-GPT4-train.txt", 1000, ["<|endoftext|>"])

print(len(vocab), vocab[:10])