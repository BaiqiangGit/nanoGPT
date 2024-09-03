
## replicated from karpathy's bigram.py

import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
batch_size = 64 # how many independent sequence will be processed in prarallel 
block_size = 256 # maximum context length
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4 # self attention favors lower learning rate
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

#----------
# curl https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt --output input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(set(text))
vocab_size = len(chars)
# create a mapping between characters and integers
itos = {i:ch for i,ch in enumerate(chars)}
stoi = {ch:i for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # encode string into list of ints
decode = lambda l: ''.join([itos[i] for i in l]) # decode list of ints into string

# create train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(data)*0.9) # first 90% as train set
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small abtch of data of input x and targes y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-block_size, (batch_size, )) # sample a batch indices
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval() # put model in eval mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters) # hold losses
        for k in range(eval_iters):
            X, Y = get_batch(split) # sample with replacement
            logits, loss = model(X, Y) 
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B,T,C=x.shape
        q = self.query(x) # [B,T,head_size]
        k = self.key(x) # [B,T,head_size]
        v = self.value(x) # [B,T,head_size]
        # compute attention scores "affinities"
        wei = q@k.transpose(-2,-1) * q.shape[-1]**-0.5 # [B,T,head_size] @ [B, head_size, T] => [B,T,T]
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # [B,T,T]
        wei = F.softmax(wei, dim=-1) # [B,T,T]
        wei = self.dropout(wei)
        out = wei @ v # [B, T, T] @ [B, T, C] => [B,T,C]
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x): # [B, T, C]
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out
    
class FeedForward(nn.Module):
    """ a simple token level linear layer followed by non-linearity """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), 
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), 
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer's block: communication followed by computation """
    
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimenssion
        # n_head: number of heads 
        super().__init__()
        head_size = n_embd//n_head
        assert head_size * n_head == n_embd
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = self.ln1(x) # pre norm formulation
        x = self.sa(x)
        x = self.ln2(x)
        x = self.ffwd(x)
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        #self.head = Head(n_embd)
        #self.sa_heads = MultiHeadAttention(4, n_embd//4) # 4 heads, 8 channels
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=4) for _ in range(3)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd) # project channels
        self.lm_head = nn.Linear(n_embd, vocab_size) # get logits

        
    def forward(self, idx, targets = None):
        B, T = idx.shape
        
        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # [B, T, C=n_embd]
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # [T, C=n_embd]
        x = tok_emb + pos_emb # [B, T, C=n_embd]
        # x = self.head(x) # apply one head self-attention
        # x = self.sa_heads(x) # apply multi-head sa
        x = x + self.blocks(x) # residual multihead attention
        x =self.ln_f(x)
        # x = x + self.ffwd(x) # residual feed forward
        logits = self.lm_head(x) # [B, T, C=vocab_size]

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) 
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_token):
        # idx is (B, T) tensor of indices in current context
        for _ in range(max_new_token):
            # crop idx to the last block_size tokens (avoid context overflow)
            idx_cond = idx[:,-block_size:]
            # get the predictions
            logits, loss = self(idx_cond) # equivalent of self.forward(idx) # logits [B, T, C]
            # focus only to the last time step
            logits = logits[:, -1, :] # [B, C]
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim = -1) # [B, C]
            # sample from the distribution: https://pytorch.org/docs/stable/generated/torch.multinomial.html
            idx_next = torch.multinomial(probs, num_samples=1) # [B, 1]
            # append sampled index to the running sentence
            idx = torch.cat([idx, idx_next], dim=1) # [B, T+1]
        return idx
    
model = BigramLanguageModel()
m = model.to(device)
print(sum([p.numel() for p in m.parameters()])/1e6, 'M parameters')
# create a pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    
    # evaluate periodically
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f'step {iter:5d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}')

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((2,1), dtype=torch.long, device=device)
print('----------')
print(decode(m.generate(context, max_new_token=500)[0].tolist())) # [0] as we have only one 
