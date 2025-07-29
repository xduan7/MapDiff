# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MapDiff (Mask-prior-guided denoising Diffusion) is a deep learning framework for inverse protein folding - predicting amino acid sequences from 3D protein backbone structures. It uses a two-stage training approach: mask-prior IPA pre-training followed by denoising diffusion network training.

## Key Commands

### Environment Setup
```bash
# Create and activate conda environment
conda create --name mapdiff python=3.8
conda activate mapdiff

# Install dependencies
conda install pytorch==1.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg==2.4.0 -c pyg
conda install -c ostrokach dssp
pip install --no-index torch-cluster torch-scatter -f https://pytorch-geometric.com/whl/torch-1.13.0+cu117.html
pip install rdkit==2023.3.3
pip install hydra-core==1.3.2
pip install biopython==1.81
pip install einops==0.7.0
pip install prettytable
pip install comet-ml  # Optional for experiment tracking
```

### Data Preparation
```bash
# Download CATH dataset (version can be 4.2 or 4.3)
python data/download_cath.py --cath_version=4.2

# Process data and generate graphs
python data/generate_graph_cath.py --download_dir=${download_dir}
```

### Training
```bash
# Stage 1: Mask-prior IPA pre-training
python mask_ipa_pretrain.py --config-name=mask_pretrain \
    dataset.train_dir=${train_data} \
    dataset.val_dir=${val_data}

# Stage 2: Diffusion network training
python main.py --config-name=diff_config \
    prior_model.path=${ipa_model} \
    dataset.train_dir=${train_data} \
    dataset.val_dir=${val_data} \
    dataset.test_dir=${test_data}
```

### Inference
See `model_inference.ipynb` for detailed inference pipeline. Pre-trained weights available at: https://github.com/peizhenbai/MapDiff/releases/download/v1.0.1/mapdiff_weight.pt

## Architecture Overview

### Two-Stage Training Pipeline

1. **Mask-Prior IPA Pre-training** (`mask_ipa_pretrain.py`)
   - Pre-trains an Invariant Point Attention (IPA) network
   - Uses masked language modeling on protein structures
   - Model: `IPANetPredictor` with structure-aware attention

2. **Diffusion Training** (`main.py`)
   - Trains denoising diffusion model with IPA guidance
   - Model: `Prior_Diff` combining EGNN and pre-trained IPA
   - Supports multiple noise schedules (uniform, marginal, BLOSUM)

### Key Model Components

- **EGNN** (`model/egnn_pytorch/`): Equivariant Graph Neural Network
  - Maintains 3D equivariance for structure-aware processing
  - Updates both node features and 3D coordinates
  
- **IPA** (`model/ipa/`): Invariant Point Attention
  - Based on AlphaFold2's IPA mechanism
  - Rotation/translation invariant features
  
- **Diffusion** (`model/diffusion.py`, `model/prior_diff.py`)
  - Discrete diffusion on amino acid types
  - Entropy-based mask-prior guidance

### Data Flow

```
PDB Structure → Graph Construction → Residue Features + 3D Coordinates → Model → AA Sequence
```

- **Input**: Protein backbone structures (N, CA, C, O atoms)
- **Features**: Amino acid type, SASA, secondary structure, pairwise distances
- **Output**: Predicted amino acid sequences

### Configuration System

Uses Hydra configuration management with configs in `conf/`:
- `diff_config.yaml`: Main diffusion training config
- `mask_pretrain.yaml`: IPA pre-training config
- Modular configs for dataset, model, training parameters

### Important Design Decisions

1. **Mask-Prior Guidance**: High-entropy predictions receive guidance from pre-trained IPA
2. **Discrete Diffusion**: Works directly on discrete amino acid types
3. **Equivariance**: EGNN maintains 3D structure awareness
4. **Ensemble Sampling**: Multiple forward passes for robust predictions

## Development Notes

- No explicit test suite found - validation happens during training
- Experiment tracking via optional Comet ML integration
- Evaluation metrics: Recovery rate, perplexity, BLOSUM-based NSSR scores
- Sampling uses DDIM for faster inference