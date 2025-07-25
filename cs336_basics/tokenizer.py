from collections import defaultdict
from collections.abc import Iterable
import functools
import multiprocessing
import pickle
import time
from typing import Dict, Iterator, List, Optional, Tuple
from pretokenization_example import find_chunk_boundaries
import regex as re
from sortedcontainers import SortedSet
from tqdm import tqdm

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Token:
  """Represents a pre-tokenized sequence of bytes and its frequency.

  Attributes:
      str (tuple[bytes]): The sequence of bytes representing the token.
      cnt (int): The frequency count of this token.
  """

  def __init__(self, str: Tuple[bytes, ...], cnt: int):
    self.str = str
    self.cnt = cnt


def _merge_pair_for_new_token(
    token: Tuple[bytes, ...], pair: Tuple[bytes, bytes]
) -> Tuple[bytes, ...]:
  """Merges an occurrence of a `pair` within a `token` to form a new, combined token.

  Args:
      token: The original token represented as a tuple of bytes.
      pair: The pair of bytes to be merged.

  Returns:
      The new token with the merged pair.
  """
  result: List[bytes] = []
  i = 0
  while i < len(token):
    if i + 1 < len(token) and (token[i], token[i + 1]) == pair:
      result.append(b"".join(pair))
      i += 2
    else:
      result.append(token[i])
      i += 1
  return tuple(result)


def _split_by_sepcial_tokens(
    text: str, special_tokens: list[str] | None
) -> Tuple[List[str], List[str]]:
  """Splits the input text by a list of special tokens.

  Args:
      text: The input text to be split.
      special_tokens: A list of special tokens to split by. If None, the text is
        returned as a single element list.

  Returns:
      A tuple containing two lists:
      - The first list contains the text segments after splitting.
      - The second list contains the special tokens that were used for
      splitting.
  """
  split_by_which_token: List[str] = []
  if special_tokens is None:
    return [text], split_by_which_token
  # Escape special tokens to use them safely in a regex pattern
  pattern = "|".join([re.escape(s) for s in special_tokens])
  text_splits = re.split(pattern, text)
  # Find how the text is split
  cur = 0
  for i in range(len(text_splits) - 1):
    cur += len(text_splits[i])
    m = re.match(pattern, text[cur:])
    assert (
        m is not None
    ), f"Mismatch when splitting text. Expected a special token at {cur}"
    split_by_which_token.append(m.group())
    cur += len(m.group())
  assert (
      len(text_splits) == len(split_by_which_token) + 1
  ), "Number of splits and split tokens mismatch."
  return text_splits, split_by_which_token


def _pretokenize(split: str) -> List[Tuple[bytes, ...]]:
  """Pre-tokenizes a given text split into a list of tuples of bytes based on a regex pattern.

  Args:
      split: The text segment to pre-tokenize.

  Returns:
      A list where each element is a tuple of single-byte strings representing a
      pre-tokenized unit.
  """
  result: List[Tuple[bytes, ...]] = []
  for m in re.finditer(PAT, split):
    s = m.group().encode("utf-8")
    str_tuple = tuple(s[i : i + 1] for i in range(len(s)))
    result.append(str_tuple)
  return result


def _pretokenize_and_build_freq(
    text: str, special_tokens: List[str]
) -> List[Token]:
  """Pre-tokenizes the input text, handling special tokens, and builds frequency counts

  for the resulting byte sequence tokens.

  Args:
      text: The input text to process.
      special_tokens: A list of special tokens to consider during
        pre-tokenization.

  Returns:
      A list of Token objects, each containing a unique byte sequence token and
      its total frequency.
  """
  text_splits, _ = _split_by_sepcial_tokens(text, special_tokens)
  freq: Dict[Tuple[bytes, ...], int] = defaultdict(int)
  for s in text_splits:
    for token in _pretokenize(s):
      freq[token] += 1
  return [Token(str_val, freq[str_val]) for str_val in freq]


def _add_to_map(
    pair: Tuple[bytes, bytes],
    cnt: int,
    pair_freq: Dict[Tuple[bytes, bytes], int],
    sorted_set: SortedSet[Tuple[int, Tuple[bytes, bytes]]],
):
  """Adds to the frequency of a given pair and updates the sorted set.

  Args:
      pair: The pair of bytes.
      cnt: The count to add to the pair's frequency.
      pair_freq: Dictionary mapping pairs to their frequencies.
      sorted_set: Sorted set of (frequency, pair) tuples.
  """
  # if pair in pair_freq:
  #  assert (pair_freq[pair], pair) in sorted_set
  #  sorted_set.remove((pair_freq[pair], pair))

  pair_freq[pair] += cnt
  # sorted_set.add( (pair_freq[pair], pair) )


def _del_from_map(
    pair: Tuple[bytes, bytes],
    cnt: int,
    pair_freq: Dict[Tuple[bytes, bytes], int],
    sorted_set: SortedSet[Tuple[int, Tuple[bytes, bytes]]],
):
  """Decreases the frequency of a given pair and updates the sorted set.

  Removes the pair from `pair_freq` and `sorted_set` if its frequency becomes
  zero.

  Args:
      pair: The pair of bytes.
      cnt: The count to subtract from the pair's frequency.
      pair_freq: Dictionary mapping pairs to their frequencies.
      sorted_set: Sorted set of (frequency, pair) tuples.
  """
  assert pair in pair_freq, f"Attempted to delete non-existent pair: {pair}"
  # assert (pair_freq[pair], pair) in sorted_set
  # sorted_set.remove((pair_freq[pair], pair))
  pair_freq[pair] -= cnt
  assert pair_freq[pair] >= 0
  if pair_freq[pair] == 0:
    del pair_freq[pair]
  # else:
  #  sorted_set.add((pair_freq[pair], pair))


def _add_to_location(
    pair: Tuple[bytes, bytes],
    loc: int,
    location: Dict[Tuple[bytes, bytes], Dict[int, int]],
):
  """Adds an occurrence of a pair at a specific token index (location).

  Args:
      pair: The pair of bytes.
      loc: The index of the token where the pair occurs.
      location: Dictionary mapping pairs to a dictionary of token indices and
        counts.
  """
  location[pair][loc] += 1


def _del_from_location(
    pair: Tuple[bytes, bytes],
    loc: int,
    location: Dict[Tuple[bytes, bytes], Dict[int, int]],
):
  """Decreases the count of a pair at a specific token index (location).

  Removes entries if their counts become zero.

  Args:
      pair (tuple[bytes, bytes]): The pair of bytes.
      loc (int): The index of the token where the pair occurs.
      location (dict[tuple[bytes, bytes], dict[int, int]]): Dictionary mapping
        pairs to a dictionary of token indices and counts.
  """
  assert (
      pair in location
  ), f"Attempted to delete location for non-existent pair: {pair}"
  assert (
      loc in location[pair]
  ), f"Attempted to delete location {loc} for pair {pair} which does not exist."
  location[pair][loc] -= 1
  if location[pair][loc] == 0:
    del location[pair][loc]
  if len(location[pair]) == 0:
    del location[pair]


def _preprocess_special_tokens(
    special_tokens: list[str] | None,
) -> List[str] | None:
  """Preprocesses a list of special tokens by sorting them by length in descending order.

  Args:
      special_tokens: A list of special tokens.

  Returns:
      The sorted list of special tokens, or None if the input was None.
  """
  if special_tokens:
    # Sort by length in descending order to ensure longer matches are found first
    special_tokens = sorted(special_tokens, key=lambda x: -len(x))
  return special_tokens


def write_vocab_to_disk(vocab: list[bytes], vocab_filepath: str):
  """Writes the vocabulary dictionary to a file using pickle.

  Args:
      vocab: The vocabulary dictionary where keys are IDs and values are byte
        strings.
      vocab_filepath: The path to the file where the vocabulary will be saved.
  """
  with open(vocab_filepath, "wb") as file:
    pickle.dump(vocab, file)


def write_vocab_to_disk(merges: list[tuple[bytes]], merges_filepath: str):
  """Writes the list of merges to a file using pickle.

  Args:
      merges: The list of merge operations.
      merges_filepath: The path to the file where the merges will be saved.
  """
  with open(merges_filepath, "wb") as file:
    pickle.dump(merges, file)


def load_from_disk(
    filepath: str,
) -> Dict[int, bytes] | List[Tuple[bytes, bytes]]:
  """Loads data (vocabulary or merges) from a pickle file.

  Args:
      filepath: The path to the file to load.

  Returns:
      The loaded data, which can be a vocabulary dictionary or a list of merge
      operations.
  """
  with open(filepath, "rb") as file:
    return pickle.load(file)


def _merge_step(
    tokens: List[Token],
    vocab: Dict[int, bytes],
    pair_freq: Dict[Tuple[bytes, bytes], int],
    sorted_set: SortedSet[Tuple[int, Tuple[bytes, bytes]]],
    location: Dict[Tuple[bytes, bytes], Dict[int, int]],
) -> Tuple[bytes, bytes]:
  """Performs a single BPE merge step: finds the most frequent pair, merges it,

  updates frequencies, and adjusts the token list.

  Args:
      tokens: The list of current tokens to be merged.
      vocab: The current vocabulary mapping IDs to byte strings.
      pair_freq: Frequencies of all current byte pairs.
      sorted_set: A sorted set of (frequency, pair) for efficient retrieval.
      location: Maps pairs to the indices of tokens where they appear.

  Returns:
      The most frequent pair that was merged in this step.
  """

  # most_freq_cnt, most_freq_pair = sorted_set.pop()

  # Get and remove the highest freq pair
  most_freq_pair, most_freq_cnt = max(
      pair_freq.items(), key=lambda x: (x[1], x[0])
  )
  # Add the new merged token to the vocabulary
  vocab[len(vocab)] = b"".join(most_freq_pair)

  # Get the list of token indices where this most frequent pair occurs
  # We make a list copy because `location[most_freq_pair]` will be modified during the loop.
  token_locations = list(location[most_freq_pair].keys())

  for k in token_locations:
    token = tokens[k]
    old_token = token.str
    new_token = _merge_pair_for_new_token(token.str, most_freq_pair)

    # If no actual merge happened in this specific token, continue to the next location
    if old_token == new_token:
      continue

    # Update frequencies and locations for pairs affected by the merge in this specific token
    # Decrement counts for old pairs
    for i in range(len(old_token) - 1):
      current_pair = (old_token[i], old_token[i + 1])
      # We decrement the global pair_freq by token.cnt (the frequency of the entire pre-token)
      # This is because this one `Token` instance contributes `token.cnt` occurrences of the `current_pair`
      # to the global pair_freq if it contains `current_pair`.
      _del_from_location(current_pair, k, location)
      _del_from_map(current_pair, token.cnt, pair_freq, sorted_set)

    # Increment counts for new pairs formed after the merge
    for i in range(len(new_token) - 1):
      current_pair = (new_token[i], new_token[i + 1])
      _add_to_map(current_pair, token.cnt, pair_freq, sorted_set)
      _add_to_location(current_pair, k, location)

    token.str = new_token

  return most_freq_pair


def _merge(
    tokens: List[Token], steps: int
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
  """Performs the BPE merging process for a specified number of steps.

  Args:
      tokens: The initial list of pre-tokens.
      steps: The number of merge steps to perform.

  Returns:
      tuple: A tuple containing:
          - The final vocabulary mapping integer IDs to merged byte strings.
          - A list of the merged byte pairs in the order they were performed.
  """
  # Initialize pair frequencies
  pair_freq: Dict[Tuple[bytes, bytes], int] = defaultdict(int)
  for token in tokens:
    for i in range(0, len(token.str) - 1):
      pair_freq[(token.str[i], token.str[i + 1])] += token.cnt

  # Initialize a sorted set to keep track of pair frequencies
  # SortedSet stores (frequency, pair) for efficient retrieval of the max
  sorted_set: SortedSet[Tuple[int, Tuple[bytes, bytes]]] = SortedSet()
  for pair, freq in pair_freq.items():
    sorted_set.add((freq, pair))

  # Initialize a location dictionary to map pairs to the tokens they appear in
  location = defaultdict(lambda: defaultdict(int))
  for k in range(len(tokens)):
    token = tokens[k]
    for i in range(0, len(token.str) - 1):
      location[(token.str[i], token.str[i + 1])][k] += 1

  # Initialize vocabulary with base tokens (endoftext and single bytes)
  vocab: Dict[int, bytes] = {}
  vocab[0] = "<|endoftext|>".encode("utf-8")
  for i in range(256):
    vocab[i + 1] = bytes([i])

  merges: List[Tuple[bytes, bytes]] = []
  for i in tqdm(range(steps), desc="BPE Merging"):
    most_freq_pair = _merge_step(tokens, vocab, pair_freq, sorted_set, location)
    merges.append(most_freq_pair)

  return vocab, merges


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: Optional[List[str]]
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
  """Trains a Byte Pair Encoding (BPE) tokenizer on a text file.

  Args:
      input_path: The path to the input text file.
      vocab_size: The target size of the vocabulary.
      special_tokens: A list of special tokens to be included in the vocabulary.

  Returns:
      A tuple containing:
          - The final vocabulary mapping integer IDs to merged byte strings.
          - A list of the merged byte pairs in the order they were performed.
  """
  special_tokens = _preprocess_special_tokens(special_tokens)
  num_processes = 20  # Number of processes to use for parallel pre-tokenization
  all_text: List[str] = []

  # Find chunk boundaries to read the file in parallel
  with open(input_path, "rb") as f:
    boundaries = find_chunk_boundaries(
        f, num_processes, "<|endoftext|>".encode("utf-8")
    )

    for i in range(1, len(boundaries)):
      f.seek(boundaries[i - 1])
      text = f.read(boundaries[i] - boundaries[i - 1]).decode(
          "utf-8", errors="ignore"
      )
      all_text.append(text)

  all_encoded_tokens: List[Token] = []

  # Pre-tokenize and build initial frequencies in parallel
  with multiprocessing.Pool(processes=num_processes) as pool:
    # functools.partial is used to pass fixed arguments to the mapped function
    all_encoded_tokens = pool.map(
        functools.partial(
            _pretokenize_and_build_freq, special_tokens=special_tokens
        ),
        all_text,
    )
  # Flatten the list of lists of Tokens into a single list
  all_encoded_tokens = sum(all_encoded_tokens, [])

  # Initialize vocabulary with base tokens
  vocab: Dict[int, bytes] = {}
  vocab[0] = "<|endoftext|>".encode("utf-8")
  for i in range(256):
    vocab[i + 1] = bytes([i])

  # Calculate the number of merge steps needed
  num_steps = vocab_size - len(vocab)
  assert num_steps > 0

  vocab, merges = _merge(all_encoded_tokens, num_steps)
  return vocab, merges


def _get_reverse_vocab(vocab: Dict[int, bytes]) -> Dict[bytes, int]:
  """Creates a reverse mapping from byte strings to their integer IDs in the vocabulary.

  Args:
      vocab: The vocabulary mapping IDs to byte strings.

  Returns:
      The reverse vocabulary mapping byte strings to their IDs.
  """
  return {v: k for k, v in vocab.items()}


def _preprocess_vocab(
    vocab: Dict[int, bytes],
    vocab_reversed: Dict[bytes, int],
    special_tokens: Optional[List[str]] = None,
) -> Dict[int, bytes]:
  """Adds any missing special tokens to the vocabulary and its reverse mapping.

  Args:
      vocab: The vocabulary mapping IDs to byte strings.
      vocab_reversed: The reverse vocabulary mapping byte strings to IDs.
      special_tokens: A list of special tokens to ensure are in the vocab.

  Returns:
      The updated vocabulary.
  """
  if special_tokens is None:
    return vocab
  for special_token in special_tokens:
    encoded_special_token = special_token.encode("utf-8")
    if encoded_special_token not in vocab_reversed:
      id = len(vocab)
      vocab[id] = encoded_special_token
      vocab_reversed[encoded_special_token] = id
  return vocab


def _encode(
    token: Tuple[bytes, ...],
    reverse_vocab: Dict[bytes, int],
    merges: List[Tuple[bytes, bytes]],
) -> List[int]:
  """Encodes a pre-tokenized sequence of bytes into a list of vocabulary IDs using the BPE merges.

  Args:
      token: The pre-tokenized sequence of bytes (e.g., from _pretokenize).
      reverse_vocab: The reverse vocabulary mapping byte strings to IDs.
      merges: The ordered list of BPE merge operations.

  Returns:
      A list of integer IDs representing the encoded token.
  """

  current_token_parts = list(token)  # Convert to list to allow modification

  for merge_pair in merges:
    i = 0
    while i < len(current_token_parts) - 1:
      if (current_token_parts[i], current_token_parts[i + 1]) == merge_pair:
        # Merge the pair
        current_token_parts[i : i + 2] = [
            current_token_parts[i] + current_token_parts[i + 1]
        ]
      else:
        i += 1

  result: List[int] = []
  for part in current_token_parts:
    assert part in reverse_vocab
    result.append(reverse_vocab[part])
  return result


class Tokenizer:
  """A BPE (Byte Pair Encoding) tokenizer for encoding and decoding text."""

  def __init__(
      self,
      vocab: Dict[int, bytes],
      merges: List[Tuple[bytes, bytes]],
      special_tokens: List[str] | None = None,
  ):
    """Initializes the BPE tokenizer.

    Args:
        vocab: The vocabulary mapping IDs to byte strings.
        merges: The ordered list of BPE merge operations.
        special_tokens: An optional list of special tokens.
    """
    self.special_tokens = _preprocess_special_tokens(special_tokens)
    self.reverse_vocab = _get_reverse_vocab(vocab)
    self.vocab = _preprocess_vocab(vocab, self.reverse_vocab, special_tokens)
    self.merges = merges

  @classmethod
  def from_files(
      cls,
      vocab_filepath: str,
      merges_filepath: str,
      special_tokens: list[str] | None = None,
  ) -> "Tokenizer":
    """Creates a Tokenizer instance by loading vocabulary and merges from disk.

    Args:
        vocab_filepath: Path to the pickled vocabulary file.
        merges_filepath: Path to the pickled merges file.
        special_tokens: Optional list of special tokens.

    Returns:
        Tokenizer: A new Tokenizer instance.
    """
    vocab = load_from_disk(vocab_filepath)
    merges = load_from_disk(merges_filepath)
    return cls(vocab, merges, special_tokens)

  def encode(self, text: str) -> List[int]:
    """Encodes a given text string into a list of integer token IDs.

    Args:
        text: The input text string.

    Returns:
         A list of integer token IDs.
    """
    text_splits, split_by = _split_by_sepcial_tokens(text, self.special_tokens)
    result: List[int] = []
    for i in range(len(text_splits)):
      # Pre-tokenize the regular text segment
      pre_tokenized_parts = _pretokenize(text_splits[i])
      for pre_token in pre_tokenized_parts:
        result += _encode(pre_token, self.reverse_vocab, self.merges)

      if i < len(text_splits) - 1:
        special_token_bytes = split_by[i].encode("utf-8")
        assert (
            special_token_bytes in self.reverse_vocab
        ), f"Special token '{split_by[i]}' not in vocabulary."
        result.append(self.reverse_vocab[special_token_bytes])
    return result

  def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
    """Encodes an iterable of text strings into an iterator of integer token IDs.

    Args:
        iterable: An iterable yielding text strings.

    Yields:
        An iterator yielding integer token IDs.
    """
    for s in iterable:
      encoded_ids = self.encode(s)
      for r in encoded_ids:
        yield r

  def decode(self, ids: List[int]) -> str:
    """Decodes a list of integer token IDs back into a text string.

    Args:
        ids: A list of integer token IDs.

    Returns:
        The decoded text string.
    """
    # Join the byte strings corresponding to the IDs
    concat_string = b"".join([self.vocab[id] for id in ids])
    # Decode the combined byte string to a UTF-8 string, replacing invalid sequences
    return concat_string.decode("utf-8", errors="replace")
