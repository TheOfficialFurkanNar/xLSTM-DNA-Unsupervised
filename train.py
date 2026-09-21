# train.py
import os
os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'hf_home')
os.environ['HF_DATASETS_CACHE'] = os.path.join(os.getcwd(), 'hf_home', 'datasets')
os.environ['HF_HUB_CACHE'] = os.path.join(os.getcwd(), 'hf_home', 'hub')

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import save_file
from model import sLSTM, mLSTM
import time

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


class xLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=2, dropout=0.3):
        super(xLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Stack of LSTM layers (alternating sLSTM and mLSTM)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i % 2 == 0:
                self.layers.append(sLSTM(embedding_dim if i == 0 else hidden_size, hidden_size))
            else:
                self.layers.append(mLSTM(embedding_dim if i == 0 else hidden_size, hidden_size))
        
        self.output_proj = nn.Linear(hidden_size, vocab_size)  # Output vocab_size for next-token prediction
    
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        embedded = self.dropout(embedded)
        
        # Process through LSTM layers
        batch_size, seq_len, _ = embedded.shape
        states = [None] * len(self.layers)
        
        # Output logits at each timestep for next-token prediction
        logits_list = []
        
        for t in range(seq_len):
            x_t = embedded[:, t, :]  # (batch_size, embedding_dim)
            x_t = self.dropout(x_t)
            for i, layer in enumerate(self.layers):
                if isinstance(layer, sLSTM):
                    x_t, cell, stab, norm = layer(x_t, states[i])
                    states[i] = (x_t, cell, stab, norm)
                else:  # mLSTM
                    x_t, cell, norm, stab = layer(x_t, states[i])
                    states[i] = (cell, norm, stab)
            logits = self.output_proj(x_t)  # (batch_size, vocab_size)
            logits_list.append(logits)
        
        # Stack logits: (batch_size, seq_len, vocab_size)
        logits = torch.stack(logits_list, dim=1)
        return logits


class DNADataset(Dataset):
    def __init__(self, sequences, seq_length=200):
        self.sequences = sequences
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # Encode DNA sequence (A=0, C=1, G=2, T=3, N=4)
        encoding = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        encoded = [encoding.get(base.upper(), 4) for base in seq]
        
        # Pad or truncate to seq_length
        if len(encoded) > self.seq_length:
            encoded = encoded[:self.seq_length]
        else:
            encoded = encoded + [4] * (self.seq_length - len(encoded))
        
        return torch.tensor(encoded, dtype=torch.long)


def train_model(model, train_loader, test_loader, device, epochs=20, lr=5e-3, checkpoint_epochs=[5, 10, 15, 20]):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    warmup_epochs = 5
    warmup_scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, 
        lambda epoch: min(1.0, (epoch + 1) / warmup_epochs)
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    criterion = nn.CrossEntropyLoss()
    
    # Track metrics at checkpoint epochs
    metrics = {
        'train_loss': {},
        'test_loss': {},
        'train_perplexity': {},
        'test_perplexity': {}
    }

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for X_batch in pbar:
            X_batch = X_batch.to(device)
            
            # Shift inputs for next-token prediction
            # Input: X_batch[:, :-1], Target: X_batch[:, 1:]
            input_seq = X_batch[:, :-1]
            target_seq = X_batch[:, 1:]
            
            optimizer.zero_grad()
            logits = model(input_seq)  # (batch_size, seq_len-1, vocab_size)
            
            # Reshape for loss calculation
            logits = logits.reshape(-1, logits.size(-1))  # (batch_size * (seq_len-1), vocab_size)
            target_seq = target_seq.reshape(-1)  # (batch_size * (seq_len-1),)
            
            loss = criterion(logits, target_seq)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * target_seq.size(0)
            total_tokens += target_seq.size(0)
            
            perplexity = torch.exp(torch.tensor(loss.item()))
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'ppl': f"{perplexity:.2f}"})
            
            # Small delay to reduce GPU heat
            time.sleep(0.01)

        # Use warmup scheduler for first warmup_epochs, then cosine annealing
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            if epoch == warmup_epochs:
                # Switch to cosine annealing after warmup
                for _ in range(warmup_epochs):
                    scheduler.step()
            scheduler.step()

        avg_loss = total_loss / total_tokens
        train_perplexity = torch.exp(torch.tensor(avg_loss))

        # Evaluate at checkpoint epochs
        if epoch in checkpoint_epochs:
            model.eval()
            test_loss = 0.0
            test_tokens = 0
            
            with torch.no_grad():
                for X_batch in tqdm(test_loader, desc=f"Test Epoch {epoch}"):
                    X_batch = X_batch.to(device)
                    
                    # Shift inputs for next-token prediction
                    input_seq = X_batch[:, :-1]
                    target_seq = X_batch[:, 1:]
                    
                    logits = model(input_seq)
                    logits = logits.reshape(-1, logits.size(-1))
                    target_seq = target_seq.reshape(-1)
                    
                    loss = criterion(logits, target_seq)
                    test_loss += loss.item() * target_seq.size(0)
                    test_tokens += target_seq.size(0)
            
            test_avg_loss = test_loss / test_tokens
            test_perplexity = torch.exp(torch.tensor(test_avg_loss))
            
            # Store metrics
            metrics['train_loss'][epoch] = avg_loss
            metrics['test_loss'][epoch] = test_avg_loss
            metrics['train_perplexity'][epoch] = train_perplexity.item()
            metrics['test_perplexity'][epoch] = test_perplexity.item()
            
            print(f"Epoch {epoch:2d} | Train Loss: {avg_loss:.4f} | Test Loss: {test_avg_loss:.4f} | Train PPL: {train_perplexity:.2f} | Test PPL: {test_perplexity:.2f}")
            model.train()
        else:
            print(f"Epoch {epoch:2d}/{epochs} | Loss: {avg_loss:.4f} | Perplexity: {train_perplexity:.2f}")

    return metrics


def plot_metrics(metrics, checkpoint_epochs, output_dir='.'):
    """Plot loss and perplexity metrics and save as JPG files"""
    epochs = sorted(metrics['train_loss'].keys())
    
    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [metrics['train_loss'][e] for e in epochs], 'o-', label='Train Loss')
    plt.plot(epochs, [metrics['test_loss'][e] for e in epochs], 's-', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs Test Loss')
    plt.legend()
    plt.grid(True)
    plt.xticks(epochs)
    plt.savefig(f'{output_dir}/loss.jpg', format='jpg', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Perplexity plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [metrics['train_perplexity'][e] for e in epochs], 'o-', label='Train Perplexity')
    plt.plot(epochs, [metrics['test_perplexity'][e] for e in epochs], 's-', label='Test Perplexity')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Train vs Test Perplexity')
    plt.legend()
    plt.grid(True)
    plt.xticks(epochs)
    plt.savefig(f'{output_dir}/perplexity.jpg', format='jpg', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {output_dir}/")


if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Constrain GPU memory usage to allow multitasking
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.7)  # Use 70% of GPU memory
        print("GPU memory constrained to 70% for multitasking")
        # Enable CUDA graphs for better power efficiency
        torch.backends.cudnn.benchmark = True
        print("CUDA benchmark enabled for power efficiency")
    
    # Load dataset
    print("Loading dataset from Hugging Face...")
    dataset = load_dataset("dnagpt/human_genome_GCF_009914755.1")
    
    # Use official train/test split - take first N samples for demo
    train_samples = 50000
    test_samples = 10000
    
    train_seqs = []
    for i, item in enumerate(dataset['train']):
        if i >= train_samples:
            break
        train_seqs.append(item['text'])
    
    test_seqs = []
    for i, item in enumerate(dataset['test']):
        if i >= test_samples:
            break
        test_seqs.append(item['text'])
    
    print(f"Train samples: {len(train_seqs)}, Test samples: {len(test_seqs)}")
    
    # Check for data overlap
    overlap = set(train_seqs) & set(test_seqs)
    print(f"Data overlap between train and test: {len(overlap)} sequences")
    
    # Create datasets and dataloaders
    seq_length = 200
    batch_size = 128
    
    train_dataset = DNADataset(train_seqs, seq_length=seq_length)
    test_dataset = DNADataset(test_seqs, seq_length=seq_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create xLSTM model for next-token prediction
    vocab_size = 5  # A, C, G, T, N
    embedding_dim = 64
    hidden_size = 128
    num_layers = 2
    dropout = 0.3
    
    model = xLSTM(vocab_size, embedding_dim, hidden_size, num_layers, dropout).to(device)
    
    param_count = count_parameters(model)
    print(f"Model parameters: {param_count:,}")
    
    # Train model
    checkpoint_epochs = [5, 10, 15, 20]
    metrics = train_model(model, train_loader, test_loader, device, epochs=20, lr=1e-3, checkpoint_epochs=checkpoint_epochs)
    
    # Save model weights in safetensors format
    print("Saving model weights in safetensors format...")
    state_dict = model.state_dict()
    metadata = {
        'vocab_size': str(vocab_size),
        'embedding_dim': str(embedding_dim),
        'hidden_size': str(hidden_size),
        'num_layers': str(num_layers),
        'total_parameters': str(param_count),
        'seq_length': str(seq_length),
        'train_samples': str(len(train_seqs)),
        'test_samples': str(len(test_seqs))
    }
    save_file(state_dict, "xlstm_weights.safetensors", metadata=metadata)
    print("Model weights and metadata saved to model_weights.safetensors")
    
    # Plot metrics
    plot_metrics(metrics, checkpoint_epochs)