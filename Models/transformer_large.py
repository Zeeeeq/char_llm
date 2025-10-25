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
    wei = q @ k.transpose(-2, -1) * (C ** -0.5)  # scaled dot-product attention; (B, T, head_size) @ (B, head_size, T) --> (B, T, T)
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) #
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei)
    v = self.value(x)    # (B,T,head_size)
    out = wei @ v        # (B, T, T) @ (B, T, head_size) --> (B,T,head_size)
    return out

# --- Multi-Head Self-Attention ---
class MultiHeadAttention(nn.Module):
  def __init__(self, n_head, head_size, n_embd, block_size, dropout=0.1):
    super().__init__()
    self.heads = nn.ModuleList(Head(head_size, n_embd, block_size, dropout) for _ in range(n_head))
    self.proj = nn.Linear(n_head * head_size, n_embd)

  def forward(self, x):
    concat_att = torch.cat([h(x) for h in self.heads], dim=-1) # Concatenate on last dimension --> (B, T, n_heads * head_size)
    out = self.proj(concat_att)
    return out

# --- Feed Forward Network ---
class FeedForward(nn.Module):
  def __init__(self, n_embd, dropout=0.1):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(n_embd, 4 * n_embd),
        nn.ReLU(),
        nn.Linear(4 * n_embd, n_embd),
        nn.Dropout(dropout),
    )

  def forward(self, x):
    return self.net(x)


# --- Transformer Block ---
class Block(nn.Module):
  def __init__(self, n_embd, n_head, block_size, dropout=0.1):
    super().__init__()
    head_size = n_embd // n_head
    self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
    self.ffwd = FeedForward(n_embd, dropout)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)

  def forward(self, x):
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x


# --- Large Transformer Model ---
class CharTransformerLarge(nn.Module):
  """
  Large Transformer model for character-level language modeling.

  vocab_size: number of unique characters from text8
  n_embd: embedding size; default 128
  block_size: context length (L); default 64
  """

  def __init__(self, vocab_size, n_embd=128, block_size=64, n_layer=4, n_head=4, dropout=0.1):
    super().__init__()
    self.block_size = block_size
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.blocks = nn.Sequential(
        *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
    )
    self.ln_f = nn.LayerNorm(n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    # Weight tying (saves params & improves generalization)
    self.lm_head.weight = self.token_embedding_table.weight

  def forward(self, idx, targets=None):
    B, T = idx.shape
    tok_emb = self.token_embedding_table(idx)
    pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # Provide positional context of characters
    x = tok_emb + pos_emb
    x = self.blocks(x)
    x = self.ln_f(x)
    logits = self.lm_head(x)

    loss = None
    if targets is not None:
        logits = logits.view(B * T, -1)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)

    return logits, loss

  @torch.no_grad()
  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -self.block_size:]
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx