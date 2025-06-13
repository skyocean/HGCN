# Outcome-Oriented Predictive Business Process Monitoring with GCN/GraphConv Models

This repository contains implementations of Graph Convolutional Network (GCN) and Graph Convolution (GraphConv) models for outcome-oriented predictive business process monitoring using event sequence data from event logs.

## Repository Structure

### Python Model Files
- **Core Models**:
  - `OneLevelGCN.py`, `OneLevelGraphConv.py` - 1-level models (balanced datasets)
  - `OneLevelGCNIm.py`, `OneLevelGraphConvIm.py` - 1-level models for imbalanced datasets
  - `TwoLevelGCN.py`, `TwoLevelGraphConv.py` - 2-level hierarchy models (balanced)
  - `TwoLevelGCNIm.py`, `TwoLevelGraphConvIm.py` - 2-level hierarchy models (imbalanced)
  - `TwoLevelDurationGCN.py`, `TwoLevelDurationGraphConv.py` - 2-level with duration embedding (balanced)
  - `TwoLevelDurationGCNIm.py`, `TwoLevelDurationGraphConvIm.py` - 2-level with duration embedding (imbalanced)
  - `TwoLevelEmbeddingGCN.py`, `TwoLevelEmbeddingGraphConv.py` - 2-level with NLP-style activity embedding (balanced)
  - `TwoLevelEmbeddingGCNIm.py`, `TwoLevelEmbeddingGraphConvIm.py` - 2-level with NLP-style activity embedding (imbalanced)

### Support Files
- `DataEncoder.py` - Data encoding functions
- `DurationEmbedding.py` - Duration pseudo-embedding matrix generation
- `utils.py` - Utility functions

### Demonstration Notebooks
Jupyter notebooks demonstrating each model type:
- `*_call.ipynb` files correspond to each model implementation
- `DurationBin_call.ipynb` demonstrates duration embedding generation

## Model Architectures

### 1-Level Models
- Take event-level and sequence-level (broadcasted) attributes as a single input
- Suffixes:
  - No suffix: Balanced dataset version
  - `Im`: Imbalanced dataset version

### 2-Level Models
- Take event-level and sequence-level attributes separately as hierarchical inputs
- Variants:
  - Basic: Standard hierarchical input
  - Duration: Adds duration pseudo-embedding matrix at event level
  - Embedding: Separates activity/event label as extra NLP-style embedded input

## Datasets
- Located in `output/` directory
- Naming convention:
  - `BPI12*.csv`: Balanced datasets
  - Other filenames: Imbalanced datasets

## Usage
1. Encode your event log data using functions from `DataEncoder.py`
2. For duration-aware models, generate duration embeddings using `DurationEmbedding.py`
3. Choose the appropriate model architecture for your needs (1-level vs 2-level, duration/embedding variants)
4. Run the corresponding demonstration notebook or import the model directly

