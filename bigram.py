
## replicated from karpathy's bigram.py

import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
batch_size = 32 # how many independent sequence will be processed in prarallel 
block_size = 8 # maximum context length
max_iters = 10000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

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

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets = None):
        
        # idx and targes are both (B, T) tensor of integers
        logits = self.token_embedding_table(idx) # [B, T, C]

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
            # get the predictions
            logits, loss = self(idx) # equivalent of self.forward(idx) # logits [B, T, C]
            # focus only to the last time step
            logits = logits[:, -1, :] # [B, C]
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim = -1) # [B, C]
            # sample from the distribution: https://pytorch.org/docs/stable/generated/torch.multinomial.html
            idx_next = torch.multinomial(probs, num_samples=1) # [B, 1]
            # append sampled index to the running sentence
            idx = torch.cat([idx, idx_next], dim=1) # [B, T+1]
        return idx
    
model = BigramLanguageModel(vocab_size)
m = model.to(device)

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
