# Slot Attention for Object-Centric Learning

A PyTorch implementation of [Slot Attention](https://arxiv.org/abs/2006.15055) for unsupervised object discovery and scene decomposition.

## Overview

Slot Attention is an attention-based module that learns to decompose scenes into object-centric representations (slots) without supervision. Each slot learns to represent a different object in the scene through competitive attention.

## Architecture

The model consists of three main components:

1. **Encoder**: CNN that extracts visual features from input images
2. **Slot Attention**: Iterative attention mechanism that groups features into object slots
3. **Decoder**: Spatial broadcast decoder that reconstructs the image from slots

### Key Features

- Multi-head attention with competitive softmax across slots
- GRU-based slot updates for temporal coherence
- Learnable slot initialization with reparameterization trick
- Mask-based composition for final reconstruction

## Implementation Details

### Slot Attention Module

The core innovation is the attention mechanism where:
- Slots compete for input features via `softmax(dim=-2)` over slots
- Each input feature is assigned to slots probabilistically
- Slots are updated iteratively using a GRU
- An MLP with residual connection refines slots after each iteration

### Reparameterization Trick

Slots are initialized stochastically to break symmetry:
```python
slots = slot_mu + slot_sigma * torch.randn(...)
```
This allows the model to learn a good initialization distribution while maintaining randomness for slot specialization.

## Training

The model is trained with Multi-dSprites dataset using MSE reconstruction loss:

```bash
python train.py
```

### Hyperparameters

- **feature_dim**: 128 (encoder output channels)
- **attn_dim**: 64 (slot attention dimension)
- **n_slots**: 5 (number of object slots)
- **n_heads**: 4 (attention heads)
- **t_iters**: 3 (slot attention iterations)
- **batch_size**: 32
- **learning_rate**: 1e-4
- **epochs**: 50

## Visualization

The training script generates two types of visualizations:

1. **Reconstructions** (`recon_*.png`): Original images vs full reconstructions
2. **Slot Decomposition** (`slot_vis_*.png`): Individual slot contributions showing what each slot learned to represent

## Requirements

```
torch
torchvision
matplotlib
multi_object_datasets_torch
```

## Dataset

Uses the Multi-dSprites dataset with colored sprites on colored backgrounds. The dataset is automatically downloaded on first run.

## Model Architecture Details

### Encoder
- 4 convolutional layers (3→32→64→128→128 channels)
- Stride-2 convolutions for spatial downsampling
- Output: (B, num_patches, 128) feature vectors

### Slot Attention
- Learnable slot initialization distribution (mu, sigma)
- Multi-head attention with competitive assignment
- GRU for integrating information over iterations
- LayerNorm + MLP with residual for slot refinement

### Decoder
- Spatial broadcast architecture
- Concatenates slot vectors with position encodings
- CNN tower predicts RGB + alpha mask per slot
- Softmax over masks for final composition

## Results

After training, slots learn to specialize to different objects in the scene without any object-level supervision. Each slot typically captures one object, including its appearance and spatial extent.

## References

- [Object-Centric Learning with Slot Attention (Locatello et al., 2020)](https://arxiv.org/abs/2006.15055)

## License

MIT