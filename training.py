import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

# ==========================================
# 1) Hyperparameters
# ==========================================
hyperparams = {
    'block_size': 128,             # Sequence length for context
    'batch_size': 32,              # Batch size
    'embed_dim': 256,              # Transformer embedding dimension
    'n_heads': 8,                  # Number of attention heads
    'n_layers': 16,                # Number of Transformer blocks
    'num_epochs': 20,              # Number of epochs
    'steps_per_epoch': 1000,       # Steps per epoch
    'eval_interval': 200,          # Steps between loss evaluations
    'eval_iters': 100,             # Iterations to average validation loss
    'generate_num_tokens': 200,    # Number of tokens to generate after each epoch
    'start_prompt': "This is not love",  # Start text for generation
    'checkpoint_path': "contrastive_residual2.pt"  # File for saving/loading checkpoint
}

# Select device
device = "mps" if torch.backends.mps.is_available() else \
         ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 1.5) Read Data & Build Vocabulary
# ==========================================
# Read the input text
with open("books.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Build vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}  # string to integer
itos = {i: ch for i, ch in enumerate(chars)}  # integer to string

# Create training data
data = [stoi[ch] for ch in text]
n = int(0.9 * len(data))  # 90% for training
train_data = data[:n]
val_data = data[n:]

print(f"Vocabulary size: {vocab_size}")
print(f"Training data size: {len(train_data)}")
print(f"Validation data size: {len(val_data)}")

# ==========================================
# 2) Improved Emergent Threshold Layer
# ==========================================
class ImprovedEmergentThresholdLayer(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.norm = nn.LayerNorm(feature_dim)
        
        # Initialize adaptive threshold parameters
        self.register_buffer('running_mean', torch.zeros(feature_dim))
        self.register_buffer('running_var', torch.ones(feature_dim))
        self.adaptive_threshold = nn.Parameter(torch.ones(1) * 0.5)
        self.momentum = 0.01  # Slower momentum for stability
        
    def forward(self, x):
        """
        Forward pass with normalized thresholding and stable gradients
        """
        # Apply layer normalization first
        x_norm = self.norm(x)
        
        if self.training:
            # Update statistics conservatively
            with torch.no_grad():
                batch_mean = x_norm.mean(dim=(0, 1))
                batch_var = x_norm.var(dim=(0, 1), unbiased=False)
                
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
        
        # Compute dynamic threshold based on normalized statistics
        threshold = torch.sigmoid(self.adaptive_threshold) * torch.sqrt(self.running_var + 1e-5)
        
        # Soft gating with controlled gradient flow
        gate = torch.sigmoid((torch.abs(x_norm) - threshold.view(1, 1, -1)) / 0.1)
        
        # Mix gated and residual paths with learned ratio
        alpha = torch.sigmoid(self.adaptive_threshold)
        return alpha * (gate * x) + (1 - alpha) * x

# ==========================================
# ADDITIONAL THRESHOLDING METHODS
# (1) Multi-Scale
# (2) Frequency-Domain
# (3) Memory-Augmented
# (4) Hierarchical Feature Selection
# (5) Enhanced Contrastive (with extra loss)
# (6) Uncertainty-Aware
# ==========================================

class MultiScaleThresholdLayer(nn.Module):
    """
    (1) Multiple parallel threshold layers, combined by learned weights.
    """
    def __init__(self, feature_dim, num_scales=3, base_layer_cls=None):
        super().__init__()
        if base_layer_cls is None:
            base_layer_cls = ImprovedEmergentThresholdLayer

        # Create multiple sub-layers
        self.thresholds = nn.ModuleList([
            base_layer_cls(feature_dim) for _ in range(num_scales)
        ])
        # Learned combination weights
        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)

    def forward(self, x):
        # Apply each threshold layer
        scale_outputs = [thr(x) for thr in self.thresholds]
        # Weighted sum of results
        weights = F.softmax(self.scale_weights, dim=0)
        out = sum(w * s for w, s in zip(weights, scale_outputs))
        return out


class FrequencyAwareThreshold(nn.Module):
    """
    (2) Thresholding with FFT-based gating via single-head attention in frequency domain.
    """
    def __init__(self, feature_dim, base_layer_cls=None):
        super().__init__()
        if base_layer_cls is None:
            base_layer_cls = ImprovedEmergentThresholdLayer

        # The original threshold layer for gating
        self.threshold = base_layer_cls(feature_dim)
        # Single-head attention
        self.freq_attention = nn.MultiheadAttention(feature_dim, num_heads=1, batch_first=True)

    def forward(self, x):
        # FFT on input
        freq_repr = torch.fft.fft2(x.float()).abs()  # shape (B, T, E)
        # Single-head attention in freq space
        freq_attn, _ = self.freq_attention(freq_repr, freq_repr, freq_repr)

        # Original threshold gating
        threshold_out = self.threshold(x)
        # Frequency gating
        freq_gate = torch.sigmoid(freq_attn)
        return threshold_out * freq_gate


class MemoryThresholdLayer(nn.Module):
    """
    (3) Memory-Augmented Thresholding: 
    a memory bank is attended to, then added to the thresholded output.
    """
    def __init__(self, feature_dim, memory_size=128, base_layer_cls=None):
        super().__init__()
        if base_layer_cls is None:
            base_layer_cls = ImprovedEmergentThresholdLayer

        self.threshold = base_layer_cls(feature_dim)
        self.memory = nn.Parameter(torch.randn(memory_size, feature_dim))
        self.memory_query = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        # Query memory
        query = self.memory_query(x)  # (B, T, E)
        # Attn weights: (B, T, M)
        memory_attn = torch.matmul(query, self.memory.transpose(0, 1))
        attn_weights = F.softmax(memory_attn, dim=-1)  # (B, T, M)
        # Weighted sum of memory: (B, T, E)
        memory_context = torch.matmul(attn_weights, self.memory)

        threshold_out = self.threshold(x)
        out = threshold_out + memory_context
        return out


class HierarchicalThreshold(nn.Module):
    """
    (4) Hierarchical Feature Selection:
    separate global vs. local threshold, then mix.
    """
    def __init__(self, feature_dim, base_layer_cls=None):
        super().__init__()
        if base_layer_cls is None:
            base_layer_cls = ImprovedEmergentThresholdLayer

        self.global_threshold = base_layer_cls(feature_dim)
        self.local_threshold = base_layer_cls(feature_dim)
        self.mixer = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        B, T, E = x.size()
        # Global average
        global_avg = x.mean(dim=1, keepdim=True)  # (B, 1, E)

        global_features = self.global_threshold(global_avg)      # (B, 1, E)
        local_features = self.local_threshold(x)                 # (B, T, E)

        # Expand global to match shape (B, T, E)
        global_expanded = global_features.expand(B, T, E)

        mix = torch.sigmoid(self.mixer)
        out = mix * global_expanded + (1 - mix) * local_features
        return out


class ContrastiveThreshold(nn.Module):
    """
    (5) Enhanced Contrastive Learning:
    Returns (thresholded_x, contrastive_loss). 
    Incorporate contrastive_loss in your training loop.
    """
    def __init__(self, feature_dim, queue_size=1024, base_layer_cls=None):
        super().__init__()
        if base_layer_cls is None:
            base_layer_cls = ImprovedEmergentThresholdLayer

        self.threshold = base_layer_cls(feature_dim)
        self.register_buffer('feature_queue', torch.randn(queue_size, feature_dim))
        self.temperature = nn.Parameter(torch.tensor(0.07))

        self.queue_size = queue_size
        self.feature_dim = feature_dim

    def forward(self, x):
        # Threshold features
        thresh_features = self.threshold(x)

        # Flatten (B, T, E) => (N, E)
        N = x.size(0) * x.size(1)
        flattened_thresh = thresh_features.view(N, self.feature_dim)

        # Similarities to the queue
        sim_matrix = torch.matmul(
            F.normalize(flattened_thresh, dim=-1),
            F.normalize(self.feature_queue, dim=-1).transpose(0, 1)
        )  # (N, queue_size)

        # Contrastive-style loss (simple negative log-softmax of queue)
        contrastive_loss = -torch.log_softmax(sim_matrix / self.temperature, dim=-1).mean()

        # Update queue with the average of the new features
        with torch.no_grad():
            self.feature_queue = torch.roll(self.feature_queue, shifts=-1, dims=0)
            self.feature_queue[-1] = flattened_thresh.mean(dim=0)

        # Return both the thresholded output and the contrastive loss
        return thresh_features, contrastive_loss


class UncertaintyThreshold(nn.Module):
    """
    (6) Uncertainty-Aware Thresholding:
    Weighted by (1 - uncertainty), where uncertainty is learned.
    """
    def __init__(self, feature_dim, base_layer_cls=None):
        super().__init__()
        if base_layer_cls is None:
            base_layer_cls = ImprovedEmergentThresholdLayer

        self.threshold = base_layer_cls(feature_dim)
        self.uncertainty_predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1)
        )

    def forward(self, x):
        # Original threshold
        threshold_out = self.threshold(x)
        # Predict uncertainty
        uncertainty = torch.sigmoid(self.uncertainty_predictor(x))  # (B, T, 1)
        # Scale by (1 - uncertainty)
        out = threshold_out * (1.0 - uncertainty)
        return out

# ==========================================
# 3) Improved Transformer Block
# ==========================================
class ImprovedTransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        
        # Two-stage feed-forward with intermediate activation
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            ImprovedEmergentThresholdLayer(4 * embed_dim),  # Replace here if desired
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        
        # Threshold layers with smooth transitions (replace as needed)
        self.threshold1 = ImprovedEmergentThresholdLayer(embed_dim)
        self.threshold2 = ImprovedEmergentThresholdLayer(embed_dim)

    def forward(self, x):
        # x shape: (B, T, E)
        B, T, E = x.size()
        
        # Create causal mask of shape (T, T)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        
        # Self-attention with causal mask
        attn_out, _ = self.attention(
            x,      # query
            x,      # key
            x,      # value
            attn_mask=causal_mask
        )
        
        # Thresholded residual connection
        x = x + self.threshold1(attn_out)
        
        # Feed-forward with thresholded residual connection
        ff_out = self.feed_forward(x)
        x = x + self.threshold2(ff_out)
        
        return x

# ==========================================
# 4) Improved Character Transformer
# ==========================================
class ImprovedCharTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, n_heads=4, n_layers=4):
        super().__init__()
        self.block_size = hyperparams['block_size']
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(self.block_size, embed_dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            ImprovedTransformerBlock(embed_dim, n_heads)
            for _ in range(n_layers)
        ])
        
        # Final layer with improved threshold
        self.final_threshold = ImprovedEmergentThresholdLayer(embed_dim)
        self.ln_f = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        B, T = x.size()
        
        # Token + position embeddings
        token_emb = self.token_embedding(x)
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)
        x = token_emb + pos_emb
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final threshold and projection
        x = self.final_threshold(x)
        logits = self.ln_f(x)
        
        return logits

# ==========================================
# 5) Training Functions
# ==========================================
def get_batch(split='train'):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - hyperparams['block_size'], 
                       (hyperparams['batch_size'],))
    x = torch.stack([torch.tensor(
        data_split[i:i+hyperparams['block_size']], 
        dtype=torch.long
    ) for i in ix])
    y = torch.stack([torch.tensor(
        data_split[i+1:i+hyperparams['block_size']+1], 
        dtype=torch.long
    ) for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(hyperparams['eval_iters'])
        for k in range(hyperparams['eval_iters']):
            X, Y = get_batch(split)
            X, Y = X.to(device), Y.to(device)
            logits = model(X)
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), Y.view(B*T))
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def generate_text(model, start_str="", max_new_tokens=200):
    model.eval()
    context = torch.tensor([stoi[ch] for ch in start_str], 
                           dtype=torch.long, 
                           device=device).unsqueeze(0)
    
    generated = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(context[:, -hyperparams['block_size']:])
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat([context, next_token], dim=1)
            generated.append(next_token.item())
    
    return start_str + ''.join(itos[i] for i in generated)


# ==========================================
# NEW: Utility to compute model size (N)
# ==========================================
def compute_model_size(model):
    """
    Returns total number of trainable parameters in `model`.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==========================================
# 6) Training Loop (Modified to log scaling law metrics)
# ==========================================

# Initialize model and optimizer
model = ImprovedCharTransformer(
    vocab_size=vocab_size,
    embed_dim=hyperparams['embed_dim'],
    n_heads=hyperparams['n_heads'],
    n_layers=hyperparams['n_layers']
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1
)

# Calculate N (Model Size) once
N = compute_model_size(model)
print(f"Model Size (N): {N}")

# We'll define dataset size (D) based on train_data
D = len(train_data)
print(f"Dataset Size (D): {D}")

grad_clip = 1.0

def get_lr(step, warmup_steps=2000, base_lr=3e-4, min_lr=1e-4):
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    decay_steps = hyperparams['num_epochs'] * hyperparams['steps_per_epoch'] - warmup_steps
    step = step - warmup_steps
    cosine_decay = 0.5 * (1 + math.cos(math.pi * step / decay_steps))
    return min_lr + (base_lr - min_lr) * cosine_decay

# Load checkpoint if exists
checkpoint_path = hyperparams['checkpoint_path']
start_epoch = 0
start_step = 0

if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    # start_epoch = checkpoint.get('epoch', 0)
    # start_step = checkpoint.get('step', 0)
    print("Checkpoint loaded. Resuming training.")
else:
    print("No checkpoint found. Starting training from scratch.")

print("Starting training...")
early_stop = False
total_steps = hyperparams['num_epochs'] * hyperparams['steps_per_epoch']
current_step = start_epoch * hyperparams['steps_per_epoch'] + start_step

# We'll store each epoch's (N, D, C, L) in a list
scaling_log = []

for epoch in range(start_epoch, hyperparams['num_epochs']):
    if early_stop:
        print("\nEarly stopping triggered. Training completed.")
        break
        
    print(f"\n--- Epoch {epoch+1}/{hyperparams['num_epochs']} ---")
    
    for step in range(start_step, hyperparams['steps_per_epoch']):
        if step % hyperparams['eval_interval'] == 0:
            losses = estimate_loss()
            print(f"Step {step}, train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")

        xb, yb = get_batch('train')
        xb, yb = xb.to(device), yb.to(device)

        # Forward
        logits = model(xb)
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B*T, C), yb.view(B*T))

        # Exploding loss check
        if loss.item() > 100:
            print("\nLoss exploded. Triggering early stop.")
            early_stop = True
            break

        # Update learning rate
        lr = get_lr(current_step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        current_step += 1
    
    if not early_stop:
        # Evaluate at the end of the epoch
        losses = estimate_loss()
        val_loss = losses["val"].item()  # Performance (L)
        
        # Generate sample text after each epoch
        sample = generate_text(
            model,
            start_str=hyperparams['start_prompt'],
            max_new_tokens=hyperparams['generate_num_tokens']
        )
        print(f"\n[Sample generated text]\n{sample}\n")
        
        # Save checkpoint
        torch.save({
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch + 1,
            'step': 0  # reset step for the next epoch
        }, hyperparams['checkpoint_path'])
        print(f"Checkpoint saved at epoch {epoch+1}.")

        # ==========================================
        # LOG SCALING LAW METRICS
        # ==========================================
        # Model Size N (constant), Dataset Size D (constant)
        # Training Cost C => total tokens processed so far
        C = current_step * hyperparams['batch_size'] * hyperparams['block_size']
        L = val_loss  # performance

        epoch_scaling_info = {
            'epoch': epoch + 1,
            'N': N,
            'D': D,
            'C': C,
            'val_loss': L
        }
        scaling_log.append(epoch_scaling_info)

        print(f"Epoch {epoch+1} Scaling Info => "
              f"N: {N}, D: {D}, C: {C}, Performance (L): {L:.4f}")

    # Reset step to 0 at the end of the epoch
    start_step = 0

print("Training complete!")

# Print out the scaling log at the end
print("\n--- Final Scaling Law Log ---")
for record in scaling_log:
    print(record)
