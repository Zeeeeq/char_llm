import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCharBigram(nn.Module):
  def __init__(self, vocab_size, n_embd=64) -> None:
     super().__init__()
     self.vocab_size = vocab_size
     self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

  def forward(self, idx, targets=None):
    logits = self.token_embedding_table(idx)
    B,T,C = logits.shape

    if targets is None:
      loss = None
    else:
      # Reshape to allow use of pytorch cross_entropy function
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, new_tokens):
    for _ in range(new_tokens):
      logits, _ = self(idx)
      logits = logits[:, -1, :] # Only interested in the last character for predicting next character
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)
    return idx