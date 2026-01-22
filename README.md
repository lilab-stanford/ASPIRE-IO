# ASPIRE-IO

**Abstract:** 
- placeholder for now

## Dependencies

**Hardware:**
- NVIDIA GPU (Tested on NVIDIA L40S) with CUDA 11.8+ (Ubuntu 22.04)

**Software:**
- Python (3.10), PyTorch (2.0+)

**Additional Python Libraries:**
- numpy, pandas, scipy, scikit-learn
- Pillow, h5py, tqdm
- timm, transformers, huggingface_hub
- [MUSK](https://github.com/lilab-stanford/MUSK)

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Demo Notebooks

- `demo.ipynb`: Full pipeline (pre-training → fine-tuning → inference)
- `demo_IO_response_prediction.ipynb`: Patient-level IO response prediction

**Before running**, update the following:
```python
TOKENIZER_PATH = "<path-to-MUSK>/musk/models/tokenizer.spm"
HF_TOKEN = "<your-huggingface-token>"
```

### Pipeline Overview

| Step | Script | Description |
|------|--------|-------------|
| 1 | `pretrain.py` | Pre-train MUSK backbone with organ-specific heads |
| 2 | `finetune_mlp.py` | Fine-tune organ-specific MLP for gene prediction |
| 3 | `train_moe.py` | Train organ classifier for routing |
| 4 | `spatial_gene_signature_prediction.py` | Generate spatial predictions |
| 5 | `train_IO_response_prediction_mlp.py` | Train patient-level response predictor |

### Data Format

**Sample Data** (`sample_data/`):
- `sample_train.csv`, `sample_val.csv`: Metadata used for training
- `patches/`: H&E patch images (PNG)
- `gene_data/`: Gene expression CSVs
- `text_descriptions/`: Organ descriptions

**IO Response Data** (`pseudo_data_IO/`):
- `train_outcomes.csv`, `val_outcomes.csv`: Patient outcomes
- `embeddings/{patient_id}/`: Patch-level MUSK embeddings (`.pt`)
- `gene_predictions/`: Patch-level predictions (`.csv`)

## Acknowledgments

This project builds upon the following open-source repositories:

- [MUSK](https://github.com/lilab-stanford/MUSK) - Multimodal foundation model for pathology
- [CLAM](https://github.com/mahmoodlab/CLAM) - Weakly supervised whole slide image analysis
- [timm](https://github.com/huggingface/pytorch-image-models) - PyTorch Image Models

## License

This repository is licensed under the CC-BY-NC-ND 4.0 license.
