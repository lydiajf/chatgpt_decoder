from datasets import load_dataset
import torch
import numpy as np

loaded_ds = torch.load('Tranformer_project/flickr30k_patches.pt')

# encoder no masking . decoder cross attention no masking, separate class. only masking on the text - before cross attention, another class. no need for additional classes for encoder and decoder

def getPositionEncoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k / denominator)
            P[k, 2*i+1] = np.cos(k / denominator)
    return torch.tensor(P, dtype=torch.float32)

class encoder(torch.nn.Module):
    def __init__(self, pxl_size, emb_dim, num_heads, hidden_dim_ff):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads  # Dimension per head
        print(self.head_dim)
        assert emb_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"

        self.embeddings = torch.nn.Linear(pxl_size,emb_dim)
        
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
        emb = self.embeddings(emb)
        print('the emb',emb.shape)
        # Fix: Get dimensions correctly using size()
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



# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

embedding_dim = 64 # Set the embedding dimension
# embedding_layer = torch.nn.Embedding(64, embedding_dim).to(device)

args = (768, 64, 4, 4)  # Example arguments (vocab size, embedding size, num heads)
model = encoder(*args)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()


for idx in range(len(loaded_ds)):
    patches = loaded_ds[idx]['patches']  # Get the patches for the current image
    unrolled_patches = []

    
    # Unroll each patch into a 1D vector
    for patch in patches:
        if isinstance(patch, list):
            print(patch)
            patch = np.array(patch)  # Convert to NumPy array

        # Flatten the patch and convert to a tensor
        unrolled_patch = torch.tensor(patch.flatten(), dtype=torch.float32).to(device)
        unrolled_patches.append(unrolled_patch)

    # Stack the unrolled patches into a single tensor
    unrolled_patches_tensor = torch.stack(unrolled_patches) 
    print(unrolled_patches_tensor.shape) # Shape: [num_patches, patch_size*patch_size*channels]

    output = model(unrolled_patches_tensor)  # Model output, expected shape [seq_len, vocab_size]
        
    print(f"Output shape for image {idx}: {output.shape}")

# Save the model's state dict if needed
torch.save(model.state_dict(), 'model_overfit.pth')


