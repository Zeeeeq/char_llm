import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
  """
  Single self-attention head
  
  head_size: size of the head
  n_embd: embedding size
  block_size: context length (L)
  """
  def __init__(self, head_size, n_embd, block_size, dropout=0.1):
    super().__init__()
    self.key   = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B, T, C = x.shape
    k = self.key(x)      # (B,T,head_size)
    q = self.query(x)    # (B,T,head_size)
    wei = q @ k.transpose(-2, -1) * (C ** -0.5)  # scaled dot-product attention
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # 
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei)
    v = self.value(x)
    out = wei @ v        # (B,T,head_size)
    return out

class FeedForward(nn.Module):
  """
  Simple 2-layer feedforward network
  
  n_embd: embedding size
  """
  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(n_embd, 4 * n_embd), # AAYN paper has inner dimensionality as 4x outer dimensionality
        nn.ReLU(),
        nn.Linear(4 * n_embd, n_embd),
        nn.Dropout(0.1)
    )

  def forward(self, x):
      return self.net(x)

class Block(nn.Module):
  """Transformer block: communication (attention) then computation (feedforward)"""
  def __init__(self, n_embd, block_size):
    super().__init__()
    head_size = n_embd  # single-head version
    self.sa = Head(head_size, n_embd, block_size)
    self.ffwd = FeedForward(n_embd)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)

  def forward(self, x):
    # Use pre-norm for stability
    x = x + self.sa(self.ln1(x))  # residual connection
    x = x + self.ffwd(self.ln2(x))
    return x

class CharTransformerSmall(nn.Module):
  """
  Mini Transformer for character-level modeling
  
  vocab_size: number of unique characters from text8
  n_embd: embedding size; default 64
  block_size: context length (L); default 32
  """
  def __init__(self, vocab_size, n_embd=64, block_size=32):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.block_size = block_size
    self.transformer_block = Block(n_embd, block_size)
    self.ln_f = nn.LayerNorm(n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def forward(self, idx, targets=None):
    B, T = idx.shape
    tok_emb = self.token_embedding_table(idx)              # (B,T,n_embd)
    pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T,n_embd)
    x = tok_emb + pos_emb                                  # combine token + position
    x = self.transformer_block(x)
    x = self.ln_f(x)
    logits = self.lm_head(x)                               # (B,T,vocab_size)

    if targets is None:
        loss = None
    else:
        logits = logits.view(B*T, -1)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)

    return logits, loss

  @torch.no_grad()
  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -self.block_size:]
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :]  # last time step
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx