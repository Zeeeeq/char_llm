import torch

def get_batch(text_input, B, T):
  """Create a random batch of data from text_input (train_data or test_data).

    Args:
      text_int: 1D tensor of token ids.
      B: batch size (number of sequences).
      T: sequence length (number of tokens per sequence).

    Returns:
      x: (B, T) int array input tokens.
      y: (B, T) int array target tokens.
  """
  # Generate random index
  ix = torch.randint(len(text_input) - T, (B,))
  x = torch.stack([text_input[i:i+T] for i in ix])
  y = torch.stack([text_input[i+1:i+1+T] for i in ix])
  return x, y

def encode(text, ctoi):
    """Encode text to a list of integers."""
    return [ctoi[ch] for ch in text]

def decode(indices, itoc):
    """Decode a list of integers back to text."""
    return ''.join([itoc[i] for i in indices])