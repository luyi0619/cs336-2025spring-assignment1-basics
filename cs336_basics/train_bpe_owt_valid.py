from tokenizer import train_bpe

if __name__ == '__main__':
  vocab, merges = train_bpe('../data/owt_valid.txt', 10000, ['<|endoftext|>'])
  print (vocab, merges)
