import torch
import numpy as np

def getPositionEncoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k / denominator)
            P[k, 2*i+1] = np.cos(k / denominator)
    return torch.tensor(P, dtype=torch.float32)  # Convert to PyTorch tensor


class Decoder(torch.nn.Module):
    def __init__(self, Wemb_dim, Pemb_dim, new_dim, num_heads, hidden_dim_ff, voc_size):
        super().__init__()
        self.num_heads = num_heads
        self.Whead_dim = new_dim // num_heads  # Embedding dimension for words per head
        self.Phead_dim = new_dim // num_heads  # Embedding dimension for images per head

        assert Wemb_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"
        assert Pemb_dim % num_heads == 0, "Embedding dimension for images must be divisible by the number of heads"

        self.embeddings = torch.nn.Embedding(num_embeddings=voc_size, embedding_dim=Wemb_dim)

        # Linear layers for query, key, and value transformations
        self.linear_q = torch.nn.Linear(Wemb_dim, new_dim)
        self.linear_k = torch.nn.Linear(Pemb_dim, new_dim)
        self.linear_v = torch.nn.Linear(Pemb_dim, new_dim)
        
        # Feedforward layer (two linear layers with ReLU in between)
        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(Wemb_dim, hidden_dim_ff),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim_ff, Wemb_dim)
        )

    def forward(self, wemb, pemb):
        # Embedding layer for word embeddings (Wemb)
        Wemb = self.embeddings(wemb)
        Pemb = pemb
        print('The Wemb after embeddings:', Wemb.shape)

        # Positional encoding for word embeddings
        Wseq_len, Wd = Wemb.size(0), Wemb.size(1)
        Wsin_emb = getPositionEncoding(Wseq_len, Wd).to(Wemb.device)
        print('The Wemb after adding positional encoding:', Wsin_emb.shape)
        Wemb = Wemb + Wsin_emb

        # No positional encoding needed for image embeddings (Pemb)
        Pseq_len = Pemb.size(0)  # Image sequence length is just the first dimension
        print("The Pemb shape:", Pemb.shape) 

        # Transform embeddings for query, key, and value
        query = self.linear_q(Wemb).view(Wseq_len, self.num_heads, self.Whead_dim).transpose(0, 1)
        key = self.linear_k(Pemb).view(Pseq_len, self.num_heads, self.Phead_dim).transpose(0, 1)
        value = self.linear_v(Pemb).view(Pseq_len, self.num_heads, self.Phead_dim).transpose(0, 1)

        print("Query shape after linear transformation:", query.shape)
        print("Key shape after linear transformation:", key.shape)
        print("Value shape after linear transformation:", value.shape)

        # Attention computation: query * key^T
        scaling_factor = self.Whead_dim ** 0.5  # or use self.Phead_dim if necessary
        attention = torch.matmul(query, key.transpose(-2, -1)) / scaling_factor

        # Apply upper triangular mask (if required for causality)
        mask = torch.triu(torch.ones_like(attention), diagonal=1) * -1e9
        attention = attention + mask

        # Apply softmax to get attention weights
        soft_matrix = torch.softmax(attention, dim=-1)

        # Attention output
        sim_mat = torch.matmul(soft_matrix, value)
        sim_mat = sim_mat.transpose(0, 1).contiguous()
        sim_mat = sim_mat.view(Wseq_len, -1)  # Reshape to (Wseq_len, num_heads * Whead_dim)

        # Apply the feedforward layer
        output = self.feedforward(sim_mat)

        return output



#%%

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Testing
args = (64, 64,128, 4, 32,6000)  # (vocab size, embedding size, num heads, FFN hidden dim)
# self, Wemb_dim, Pemb_dim, new_dim, num_heads, hidden_dim_ff, voc_size)
model = Decoder(*args)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# Random input tensor with integer values within embedding vocab range
pemb = torch.rand((600,64)).to(device)  # Integer values for embedding lookup
wemb = torch.randint(10, 6000, (10,)).to(device)


print(pemb.shape)
print(wemb.shape)
output = model(wemb,pemb)
print("Output shape:", output.shape)