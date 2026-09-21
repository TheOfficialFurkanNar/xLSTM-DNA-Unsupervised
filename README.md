# xLSTM Architecture Technical Documentation

## Model Architecture

### Overall Structure
```
xLSTM(
  embedding_dim: 64
  hidden_size: 128
  num_layers: 2 (alternating sLSTM and mLSTM)
  dropout: 0.3
  parameters: ~250K
)
```

The xLSTM architecture alternates between two LSTM variants to leverage their complementary strengths:
- **Layer 1**: sLSTM (Scalar LSTM)
- **Layer 2**: mLSTM (Matrix LSTM)

### sLSTM (Scalar LSTM)

The sLSTM uses exponential gates for numerical stability and includes a normalizer state:

**Key Components:**
- **Input Gate**: Controls information flow into the cell state
- **Forget Gate**: Controls information retention from previous cell state
- **Output Gate**: Controls information flow to the hidden state
- **Candidate Cell State**: Proposed new cell state content
- **Stabilizer Gate**: Prevents numerical instability in exponential gates
- **Normalizer State**: Tracks cumulative gate values for stable normalization

**Mathematical Operations:**
- Exponential gates with log-space stabilization: $gate = \exp(\log_{gate} - stab_{gate})$
- Cell state update: $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$
- Normalizer update: $n_t = f_t \odot n_{t-1} + i_t$
- Hidden state: $h_t = o_t \odot (C_t / (n_t + \epsilon))$

**Initialization:**
- Forget gate bias initialized to 1.0 for stable initial state (encourages remembering)

### mLSTM (Matrix LSTM)

The mLSTM uses matrix memory and attention-like mechanisms for enhanced information processing:

**Key Components:**
- **Query Projection**: Projects input to query space (eq. 22)
- **Key Projection**: Projects input to key space with 1/√d scaling (eq. 23)
- **Value Projection**: Projects input to value space (eq. 24)
- **Input Gate**: Scalar gate controlling new information (eq. 25)
- **Forget Gate**: Scalar gate controlling memory retention (eq. 26)
- **Output Gate**: Controls output generation (eq. 27)
- **Matrix Cell State**: Stores information as outer products of keys and values

**Mathematical Operations:**
- Outer product memory update: $C_t = f_t \odot C_{t-1} + i_t \odot (v_t \otimes k_t^T)$
- Normalizer update: $n_t = f_t \odot n_{t-1} + i_t \odot k_t$
- Query-key interaction: $\tilde{h}_t = (C_t q_t) / \max(|n_t^T q_t|, 1)$
- Final output: $h_t = o_t \odot \tilde{h}_t$

**Key Features:**
- Matrix memory enables richer representation than scalar memory
- Attention-like query-key-value mechanism
- Stabilizer gates prevent numerical overflow in exponential operations
- Denominator clamping ensures numerical stability

### Architecture Rationale

The alternating sLSTM-mLSTM design combines:
- **sLSTM**: Efficient scalar operations with exponential gating for stability
- **mLSTM**: Matrix memory for capturing complex sequence dependencies

This hybrid architecture is particularly effective for genomic sequences where:
- Local patterns (codons, motifs) are captured by sLSTM
- Long-range dependencies and structural patterns are captured by mLSTM

### DNA Encoding
- A = 0, C = 1, G = 2, T = 3, N = 4
- Sequences are converted to uppercase before encoding to handle soft-masking

## Training Configuration

### Hyperparameters
- **Epochs**: 20
- **Learning Rate**: 0.001
- **Weight Decay**: 0.0001 (L2 regularization)
- **Batch Size**: 128
- **Gradient Clipping**: max_norm=1.0
- **Optimizer**: Adam with cosine annealing scheduler
- **Loss**: Cross-Entropy Loss

### Regularization Techniques
- Dropout (0.3) on embeddings, timesteps, and pooled representation
- L2 regularization via weight decay
- Gradient clipping to prevent explosion
