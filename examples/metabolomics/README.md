# BMFM Metabolomics Training

Training BMFM on metabolomics data (HMDB + MetaboLights).

## Quick Start

### Prerequisites
- Python 3.10 or 3.11
- CUDA GPU
- ~50 GB disk space

### Setup
```bash
# 1. Create environment
conda create -n bmfm python=3.11 -y
conda activate bmfm

# 2. Install BMFM-NVP
cd ../../  # to bmfm-nvp root
pip install -e .

# 3. Download and build vocabulary
cd examples/metabolomics
python build_vocab.py
./setup_vocabulary.sh

# 4. Download data (2-4 hours)
python hmdb_downloader.py
python metabolights_batch_processor.py

# 5. Preprocess data
python preprocess_h5ad.py \
    h5ad_data_batch/metabolomics_training_large.h5ad \
    data/metabolomics_training_processed.h5ad

# 6. Train
cd ../../  # back to root
bmfm-targets-run -cn metabolomics_pretrain max_epochs=50
```

---

## What Gets Created

- `bmfm_metabolomics_vocab/` - Raw vocabulary
- `all_genes_vocab/` - BMFM-compatible vocab  
- `hmdb_cache/` - HMDB database
- `metabolights_cache/` - MetaboLights downloads
- `h5ad_data_batch/` - Individual study files
- `data/` - Preprocessed training data
- `outputs/` - Training logs
- `checkpoints/` - Model checkpoints

**All data directories are git-ignored.**

---

## Files to Commit

- `build_vocab.py`, `hmdb_downloader.py`, `metabolights_batch_processor.py`
- `setup_vocabulary.sh` - Creates BMFM vocab structure
- `preprocess_h5ad.py` - Preprocesses h5ad files
- `README.md` - This file
- `distributed/*.sh`, `distributed/*.md` - Distributed training

---

## Troubleshooting

**"Command not found: bmfm-targets-run"**
```bash
pip install -e ../../
```

**"Python 3.13 not supported"**
```bash
conda create -n bmfm python=3.11 -y
```

**"Vocabulary not found"**
```bash
./setup_vocabulary.sh
```

---

## Configuration

After setup, edit the config file to use your absolute paths:
```bash
cd ../../run
nano metabolomics_pretrain.yaml

# Update these lines with your actual paths:
#   identifier: /your/path/to/examples/metabolomics/all_genes_vocab
#   data_dir: /your/path/to/examples/metabolomics/data
#   working_dir: /your/path/to/examples/metabolomics/outputs
```

Or use environment variables:
```bash
export METABOLOMICS_DIR=/your/path/to/examples/metabolomics
# Then edit config to use $METABOLOMICS_DIR
```

---

## Distributed Training (Advanced)

For multi-GPU training, see `distributed/` directory. 

**Note:** Distributed training scripts require manual path configuration for your cluster.

Documentation available:
- `distributed/DISTRIBUTED_TRAINING_GUIDE.md`
- `distributed/DISTRIBUTED_TRAINING_PATTERNS.md`
- `distributed/CROSS_MODAL_APPLICATION_GUIDE.md`
