from tokenizer import train_bpe

if __name__ == '__main__':
  vocab, merges = train_bpe('../data/TinyStoriesV2-GPT4-train.txt', 10000, ['<|endoftext|>'])
  print (vocab, merges)
