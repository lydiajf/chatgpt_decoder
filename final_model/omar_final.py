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

class Encoder(torch.nn.Module):
    def __init__(self, pxl_size, emb_dim, num_heads, hidden_dim_ff):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads  # Dimension per head
        print(self.head_dim)
        assert emb_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"
        #linear layer 
        self.linear_q = torch.nn.Linear(emb_dim, emb_dim)
        self.linear_k = torch.nn.Linear(emb_dim, emb_dim)
        self.linear_v = torch.nn.Linear(emb_dim, emb_dim)
        
        # Learnable bias for attention
        self.attn_embedding_bias = torch.nn.Parameter(torch.zeros(emb_dim))
        
        # Feedforward layer (two linear layers with ReLU in between)
        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, hidden_dim_ff),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim_ff, emb_dim)
        )

    def forward(self, emb):
        # postion encoding 
        seq_len, d = emb.size(0), emb.size(1)
        sin_emb = getPositionEncoding(seq_len, d).to(emb.device)
        emb = emb + sin_emb
        
        # Transform embeddings for query, key, and value, then reshape for multi-head attention
        query = self.linear_q(emb).view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        print("Query shape after linear transformation:", query.shape) 
        key = self.linear_k(emb).view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        value = self.linear_v(emb).view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)

        # Calculate attention scores and apply softmax
        scaling_factor = self.head_dim ** 0.5
        similarity_matrix = torch.matmul(query, key.transpose(-2, -1)) / scaling_factor

        # Apply upper triangular mask (if required for causality)
        mask = torch.triu(torch.ones_like(similarity_matrix), diagonal=1) * -1e9
        similarity_matrix = similarity_matrix + mask

        # Apply softmax to get attention weights
        soft_matrix = torch.softmax(similarity_matrix, dim=-1)
    
        # Apply attention weights to values and reshape back
        attention = torch.matmul(soft_matrix, value)
        attention = attention.transpose(0, 1).contiguous()
        attention = attention.view(seq_len, -1)  # Reshape
        

        # Apply feedforward layer
        output = self.feedforward(attention)
        
        return output

class Decoder(torch.nn.Module):
    def __init__(self, Wemb_dim, Pemb_dim, new_dim, num_heads, hidden_dim_ff, voc_size):
        super().__init__()
        self.num_heads = num_heads
        self.Whead_dim = new_dim  // num_heads  # Embedding dimension for words per head
        self.Phead_dim = new_dim // num_heads  # Embedding dimension for images per head

        assert Wemb_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"
        assert Pemb_dim % num_heads == 0, "Embedding dimension for images must be divisible by the number of heads"


        # Linear layers for query, key, and value transformations
        self.linear_q = torch.nn.Linear(Wemb_dim, new_dim)
        self.linear_k = torch.nn.Linear(Pemb_dim, new_dim)
        self.linear_v = torch.nn.Linear(Pemb_dim, new_dim)
        self.linear_w = torch.nn.Linear(new_dim,Wemb_dim)
        
        # Feedforward layer (two linear layers with ReLU in between)
        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(Wemb_dim, hidden_dim_ff),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim_ff, Wemb_dim)
        )

    def forward(self, wemb, pemb):
        # Embedding layer for word embeddings (Wemb)
        Wemb = wemb
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

        og = self.linear_w(sim_mat)

        print('size of og',og.shape)

        # Apply the feedforward layer
        output = self.feedforward(og)

        return output

import torch
import torch.nn as nn
import numpy as np


def getPositionEncoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d / 2)):
            denominator = np.power(n, 2 * i / d)
            P[k, 2 * i] = np.sin(k / denominator)
            P[k, 2 * i + 1] = np.cos(k / denominator)
    return torch.tensor(P, dtype=torch.float32)  # Convert to PyTorch tensor


class EncoderLayer(nn.Module):
    def __init__(self, pxl_size, emb_dim, num_heads, hidden_dim_ff):
        super().__init__()
        self.encoder = Encoder(pxl_size, emb_dim, num_heads, hidden_dim_ff)

    def forward(self, x):
        return self.encoder(x)


class DecoderLayer(nn.Module):
    def __init__(self, Wemb_dim, Pemb_dim, new_dim, num_heads, hidden_dim_ff, voc_size):
        super().__init__()
        self.decoder = Decoder(Wemb_dim, Pemb_dim, new_dim, num_heads, hidden_dim_ff, voc_size)

    def forward(self, wemb, pemb):
        return self.decoder(wemb, pemb)


class Transformer(nn.Module):
    def __init__(
        self,
        pxl_size,
        emb_dim,
        num_heads,
        hidden_dim_ff,
        Wemb_dim,
        Pemb_dim,
        new_dim,
        voc_size,
        num_encoder_layers=6,
        num_decoder_layers=6,
    ):
        super().__init__()
        # Stacking encoder layers
        self.encoders = nn.ModuleList(
            [EncoderLayer(pxl_size, emb_dim, num_heads, hidden_dim_ff) for _ in range(num_encoder_layers)]
        )

        # Stacking decoder layers
        self.decoders = nn.ModuleList(
            [DecoderLayer(Wemb_dim, Pemb_dim, new_dim, num_heads, hidden_dim_ff, voc_size) for _ in range(num_decoder_layers)]
        )
    
    

    def forward(self, pxl_input, wemb):
        # Pass through all encoder layers
        encoder_output = pxl_input
        self.linear = nn.Linear(pxl_size,emb_dim)
        encoder_output = self.linear(encoder_output)
        for i, encoder in enumerate(self.encoders):
            encoder_output = encoder(encoder_output)
            print(f"Encoder Layer {i + 1} output shape:", encoder_output.shape)

        # Pass through all decoder layers
        self.embeddings = torch.nn.Embedding(num_embeddings=voc_size, embedding_dim=Wemb_dim)
        wemb = self.embeddings(wemb)
        print('the output of wemb after embedding',wemb.shape)
        decoder_output = (wemb, encoder_output)
        wemb, encoder_output = decoder_output
        for i, decoder in enumerate(self.decoders):
            decoder_output = decoder(wemb, encoder_output)
            print(f"Decoder Layer {i + 1} output shape:", decoder_output.shape)

        return decoder_output


# Example instantiation and usage
if __name__ == "__main__":
    # Define dimensions
    # if emb_dim chnaged you need to change Pemb_dim aswell
    pxl_size = 784
    emb_dim = 256
    num_heads = 4
    hidden_dim_ff = 512
    Wemb_dim = 128
    Pemb_dim = 256
    new_dim = 256
    voc_size = 1000
    num_encoder_layers = 20
    num_decoder_layers = 20

    # Create model
    model = Transformer(
        pxl_size,
        emb_dim,
        num_heads,
        hidden_dim_ff,
        Wemb_dim,
        Pemb_dim,
        new_dim,
        voc_size,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
    )

    # Dummy input data
    pxl_input = torch.rand((600, 784))  # Batch of 32 pixel inputs
    print(pxl_input.shape)
    wemb = torch.randint(0, voc_size, (32,))  # Batch of 32 word indices

    # Forward pass
    output = model(pxl_input, wemb)
    print("Final Transformer output shape:", output)
