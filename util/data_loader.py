import numpy as np
import jax.numpy as jnp
import pickle

def get_batch(text_int, B, T):
    """Create a random batch of data from text_int.

    Args:
      text_int: 1D array of token ids.
      B: batch size (number of sequences).
      T: sequence length (number of tokens per sequence).

    Returns:
      x: (B, T) int array input tokens.
      y: (B, T) int array target tokens.
    """
    # choose random starting indices for each sequence in the batch
    ix = np.random.randint(0, len(text_int) - T, size=B)
    # inputs are text from i to i+T
    x = np.stack([text_int[i:i+T] for i in ix])
    # targets are text from i+1 to i+T+1
    y = np.stack([text_int[i+1:i+T+1] for i in ix])
    return jnp.array(x, dtype=jnp.int32), jnp.array(y, dtype=jnp.int32)

def encode(text, ctoi):
    """Encode text to a list of integers."""
    id = [ctoi[ch] for ch in text]
    return np.array(id, dtype=np.uint8)

def decode(indices, itoc):
    """Decode a list of integers back to text."""
    return ''.join([itoc[i] for i in indices])

def load_data(encoded_path='data/encoded.pkl', device='cpu'):
    with open(encoded_path, 'rb') as f:
        data = pickle.load(f)
    train = data['train_data']
    test = data['test_data']
    return train, test, data['ctoi'], data['itoc'], data['vocab_size']