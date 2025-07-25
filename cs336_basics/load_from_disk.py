from tokenizer import load_from_disk

if __name__ == '__main__':
  vocab, merges = load_from_disk()
  print (vocab, merges)
