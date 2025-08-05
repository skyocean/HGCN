# HGCN (O): A Self-Tuning GCN HyperModel Toolkit for Outcome Prediction in Event-Sequence Data

***Graph-based predictive process monitoring library** featuring self-tuning GCN architectures (using GCNConv and GraphConv layers) for outcome prediction. Designed for real-world event-sequence analytics, the framework supports dynamic model selection, hierarchical inputs, and pseudo-embedding strategies.*

**Authors**: Fang Wang (Florence Wong), Paolo Ceravolo, Ernesto Damiani  
**Repository**: Code and Demonstrations for the associated research article.

---

## 📖 Overview  
**[Download Full Paper](https://arxiv.org/abs/2507.22524)**  
This repository provides implementations of **Graph Convolutional Network (GCN) HyperModels** for outcome-oriented event-sequence prediction. The framework offers:

- Support for **both balanced and imbalanced datasets**  
- **Hierarchical input structures** for event- and sequence-level attributes  
- **Pseudo-embedding matrix (eg. Duration-aware)** and **embedding-based** model variants  
- Flexible architecture to support **dynamic self-tuning workflows**  

The toolkit is optimized for robust performance across diverse process mining and predictive monitoring scenarios.

---

## 🧩 Repository Structure  

### 🔧 **Model Architectures**

| Model Type | File(s) | Description |
|------------|---------|-------------|
| **1-Level GCN** | `OneLevelGCN.py`, `OneLevelGraphConv.py` | Single input: event-level + broadcasted sequence-level attributes (balanced). |
| **1-Level GCN (Imbalanced)** | `OneLevelGCNIm.py`, `OneLevelGraphConvIm.py` | As above, tuned for class imbalance. |
| **2-Level GCN** | `TwoLevelGCN.py`, `TwoLevelGraphConv.py` | Separate inputs for event-level and sequence-level attributes (balanced). |
| **2-Level GCN (Imbalanced)** | `TwoLevelGCNIm.py`, `TwoLevelGraphConvIm.py` | Same as above, tuned for imbalanced classes. |
| **Duration-Aware GCN** | `TwoLevelDurationGCN.py`, `TwoLevelDurationGraphConv.py` | Adds pseudo-embedding (duration-aware) at event level (balanced). |
| **Duration-Aware GCN (Imbalanced)** | `TwoLevelDurationGCNIm.py`, `TwoLevelDurationGraphConvIm.py` | Same as above, with imbalance handling. |
| **Embedding GCN** | `TwoLevelEmbeddingGCN.py`, `TwoLevelEmbeddingGraphConv.py` | Uses NLP-style embedding of activity labels (balanced). |
| **Embedding GCN (Imbalanced)** | `TwoLevelEmbeddingGCNIm.py`, `TwoLevelEmbeddingGraphConvIm.py` | Embedding variant for imbalanced datasets. |

### 📎 **Support Modules**

| File | Purpose |
|------|---------|
| `DurationEmbedding.py` | Constructs the duration pseudo-embedding matrix. |
| `utils.py` | Shared helper functions for preprocessing and training. |

### 📓 **Demonstration Notebooks**

| Notebook | Purpose |
|----------|---------|
| `*_call.ipynb` | Run and visualize the behavior of each model variant. |
| `DurationBin_call.ipynb` | Demonstrates how to generate and apply duration embeddings. |

---

## ⚙️ Model Selection Guide

| Scenario | Recommended Model |
|----------|-------------------|
| Simple, balanced and imbalanced dataset | `OneLevelGCN(GraphConv).py` / `OneLevelGCN(GraphConv)Im.py` 
| Hierarchical input (sequence + event) | `TwoLevelGCN(GraphConv).py` / `TwoLevelGCN(GraphConv)Im.py` |
| Pseudo Embeding Matrix | `TwoLevelDurationGCN(GraphConv).py`/ `TwoLevelDurationGCN(GraphConv)Im.py` |
| Activity label semantics | `TwoLevelEmbeddingGCN(GraphConv).py`/`TwoLevelEmbeddingGCN(GraphConv)Im.py` |

**Tip**: Duration-aware models require running `DurationEmbedding.py` or `DurationBin_call.ipynb` to prepare pseudo-embeddings.

---

## 📁 Datasets

- Preprocessed datasets are stored under `output/` and `DataProcessed/`.
- Naming convention:
  - `BPI12*.csv` → Balanced datasets
  - Other filenames → Imbalanced datasets

You may extend the toolkit to other event log datasets by formatting them according to the provided examples.

---

## 🧠 Research Context

This work contributes to the field of **Predictive Business Process Monitoring (PBPM)** by offering:

- A **modular graph-based framework** supporting outcome prediction on event sequences  
- Integration of **pseudo-embeddings** for duration/context encoding  
- **Self-tuning HyperModel structure** to adapt to varied dataset characteristics  

The proposed GCN variants outperform static architectures across multiple real-world process logs.

---

## 🔄 Layer Support & Roadmap

This release includes **GCNConv** and **GraphConv** layers as the core GNN backbones. These were selected for their simplicity and strong performance on graph-structured process data.

🚧 **Coming soon**:  
We are actively extending support for additional GNN architectures, including:  
- 🧠 **GAT** — for attention-based message passing  
- 🧬 **GIN** — for powerful discriminative graph representations  
- 🌐 **GraphSAGE** — for inductive, scalable neighborhood aggregation  

These additions will enable broader experimentation and plug-and-play flexibility within the HyperModel framework.

💡 **Prefer LSTM-based models?**  
Check out our complementary repository:  
**[Comprehensive Attribute Encoding and Dynamic LSTM HyperModels for Predictive Business Process Monitoring](https://github.com/skyocean/HyperLSTM-PBPM)** — focused on sequence-based predictive process analytics with dynamic embeddings and hyperparameter tuning.
📄 [Read the preprint](https://arxiv.org/abs/2506.03696)


## 📜 Citation  
If you use this code or model, please cite the associated paper:  

```bibtex
@article{wang2025hgcn,
  title={HGCN (O): A Self-Tuning GCN HyperModel Toolkit for Outcome Prediction in Event-Sequence Data},
  author={Wang, Fang and Ceravolo, Paolo and Damiani, Ernesto},
  journal={arXiv preprint arXiv:2507.22524},
  year={2025}
}
```

## 🔗 **About the Author**
This repository is maintained by Florence Wong, Ph.D. in Business Analytics and Applied Machine Learning.
For collaboration, contact via http://www.linkedin.com/in/florence-wong-fw
