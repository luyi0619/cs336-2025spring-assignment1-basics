from tokenizer import train_bpe

if __name__ == '__main__':
  vocab, merges = train_bpe('../data/test.txt', 260, ['<|endoftext|>'])
  print (vocab, merges)
