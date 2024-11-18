import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import random
import sentencepiece as spm
from transformer import Transformer

# Custom Dataset Class
class Flickr30kPatchesDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.tokenizer = spm.SentencePieceProcessor(model_file="final_model/tokenizer.model")
        self.vocab_size = self.tokenizer.get_piece_size()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the patches for the image at index idx
        patches = torch.tensor(self.data[idx]['patches'], dtype=torch.float32)
        num_patches, h, w, c = patches.shape
        captions = self.data[idx]['caption']
        cap_idx = random.randint(0, len(captions)-1)
        caption = captions[cap_idx]
        
        # tokenize caption (TODO)

        # You may want to apply any transformations here
        # For example, converting to tensor
        return patches.view(num_patches, -1), self.encode_label(caption)
    
    def encode_label(self, label):
        return torch.LongTensor(self.tokenizer.encode("<s>"+label+"</s>"))

# Load the dataset
dataset_path = './final_model/flickr30k_patches.pt'
loaded_ds = torch.load(dataset_path)

# Create an instance of the dataset
flickr_dataset = Flickr30kPatchesDataset(loaded_ds)

# Create a DataLoader for batching
batch_size = 1  # Adjust batch size as needed
data_loader = DataLoader(flickr_dataset, batch_size=batch_size, shuffle=True)

single_data_row = loaded_ds[0]
args = (768, 256, 4, 400, 128, 256, 256, flickr_dataset.vocab_size)
model = Transformer(*args)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()

learning_rate = 0.0001

for img, cap in data_loader:
    wrd_inpt = cap[0, :-1]
    targ = cap[0, 1:]
    model.train()  # Set model to training mode

    # Zero gradients
    optimizer.zero_grad()
    
    # Forward pass
    out = model(img.squeeze(), wrd_inpt.squeeze())  # Model output, expected shape [seq_len, vocab_size]
    
    # Calculate the loss
    loss = criterion(out, targ)
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    
    # Log the loss
    print(f"loss: {loss.item()}")
    # wandb.log({'loss': loss.item(), 'learning_rate': learning_rate})

    # print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

