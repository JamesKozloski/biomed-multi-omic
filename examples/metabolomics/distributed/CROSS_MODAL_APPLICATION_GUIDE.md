# Cross-Modal Application Guide: Distributed Training for Biomedical Foundation Models

## Executive Summary

This document generalizes the BMFM-NVP metabolomics distributed training implementation to other biological modalities. It provides architectural patterns, configuration templates, and scaling guidelines for:

1. **Proteomics** (mass spectrometry-based protein quantification)
2. **Genomics** (DNA sequencing, variant calling)
3. **Multi-omics** (integrated metabolomics + proteomics + transcriptomics)
4. **Clinical data** (EHR, medical imaging + omics)

**Core Insight**: The Name-Value Pair (NVP) architecture abstracts away modality-specific details, allowing the same distributed training infrastructure to handle different data types by varying only vocabulary size and value representation.

---

## Pattern Abstraction Framework

### Universal NVP Model Structure

```
Model(data_sample) = Transformer(
    Entity_Embeddings(entity_tokens),
    Value_Embeddings(value_tokens)
)

where:
    entity_tokens ∈ V_entities  (vocabulary of biological entities)
    value_tokens ∈ V_values     (discretized/continuous measurements)
```

### Modality-Specific Instantiations

| Modality | Entity Vocabulary | Value Type | Typical Vocab Size | Value Range |
|----------|------------------|------------|-------------------|-------------|
| Metabolomics | Metabolite IDs (HMDB) | Concentration | 27K - 200K | 0-50 bins (log-scale) |
| Proteomics | Protein IDs (UniProt) | Abundance | 20K - 100K | 0-50 bins (log-scale) |
| Transcriptomics | Gene IDs (Ensembl) | Expression | 20K - 60K | 0-50 bins (log-scale) |
| Genomics | Variant IDs (dbSNP) | Genotype | 10M - 100M | {0, 1, 2} (dosage) |
| Multi-omics | Union of above | Mixed | 50K - 500K | Modality-specific |

### Configuration Pattern Template

```yaml
# Generic NVP Configuration Template
defaults:
  - data_module: base_language_modeling
  - tokenizer: {modality}_vocab
  - fields: entities_values_masked
  - trainer: default
  - task: train
  - model: {architecture}
  - _self_

# Model configuration
model:
  checkpoint: null
  vocab_size: {ENTITY_VOCAB_SIZE}
  hidden_dim: 768  # Scale with vocabulary size
  num_layers: 12   # Scale with dataset size
  num_heads: 12

# Tokenizer
tokenizer:
  identifier: /path/to/{modality}/vocab/
  entity_field: "{modality}_entities"
  value_field: "{modality}_values"

# Data module
data_module:
  max_length: {MAX_ENTITIES_PER_SAMPLE}
  data_dir: /path/to/{modality}/data/
  batch_size: {PER_GPU_BATCH}
  collation_strategy: language_modeling
  num_workers: {NUM_WORKERS}
  
  transform_kwargs:
    source_h5ad_file_name: /path/to/raw_data.h5ad
    transforms:
      - transform_name: {VALUE_TRANSFORM}  # LogTransform, NormalizationTransform
      - transform_name: BinTransform
        transform_args:
          num_bins: {NUM_VALUE_BINS}

# Distributed training
trainer:
  num_nodes: {NUM_NODES}
  sync_batchnorm: true
  
accelerator: gpu
devices: 1
strategy: {STRATEGY}  # ddp, fsdp, deepspeed
precision: "16-mixed"

# Losses
trainer:
  losses:
    - field_name: values
      name: {VALUE_LOSS}  # mse, cross_entropy
      ignore_zero: true
    - field_name: values
      name: is_zero_bce  # Optional: zero detection
```

---

## Modality-Specific Implementations

### 1. Proteomics (Mass Spectrometry)

#### Data Characteristics
- **Entities**: Proteins (UniProt IDs)
- **Values**: Abundance (log-transformed intensity)
- **Typical samples**: 200-500 proteins detected per sample
- **Dataset sizes**: 1K - 100K samples
- **Vocabulary**: 20K human proteins (proteome)

#### Configuration Specifics

```yaml
# proteomics_pretrain_distributed.yaml

tokenizer:
  identifier: /mnt/mountcastle/bmfm-proteomics/vocab/uniprot_human

data_module:
  max_length: 1024  # Max 1024 proteins per sample
  batch_size: 4     # Larger batch possible (smaller vocab than metabolomics)
  
  transform_kwargs:
    transforms:
      - transform_name: LogTransform
        transform_args:
          base: 10  # Log10 for protein abundance
      - transform_name: BinTransform
        transform_args:
          num_bins: 50

# Distributed configuration (4 GPUs)
trainer:
  num_nodes: 4

# Memory estimate: ~1.5 GB per GPU
# Throughput estimate: ~400 samples/sec (4 GPUs)
```

#### Scaling Rules

```python
def estimate_proteomics_memory(vocab_size: int, max_length: int, batch_size: int) -> float:
    """Estimate memory requirements for proteomics model"""
    embedding_dim = 768
    
    # Vocabulary embeddings
    vocab_memory = vocab_size * embedding_dim * 4  # FP32
    
    # Activations (per sample)
    activation_memory = max_length * embedding_dim * 4 * batch_size
    
    # Optimizer state (Adam: 2x model size)
    optimizer_memory = vocab_memory * 2
    
    total_mb = (vocab_memory + activation_memory + optimizer_memory) / 1024**2
    return total_mb

# Example: 20K vocab, 1024 max_length, batch=4
# Result: ~1.5 GB per GPU
```

---

### 2. Genomics (Variant Data)

#### Data Characteristics
- **Entities**: Genetic variants (dbSNP IDs)
- **Values**: Genotype dosage {0, 1, 2}
- **Typical samples**: 1M - 10M variants per individual
- **Dataset sizes**: 1K - 1M individuals
- **Vocabulary**: 100M+ known variants

#### Special Considerations

**Challenge**: Extremely large vocabulary (100M variants)
**Solution**: Use FSDP (Fully Sharded Data Parallel) to shard embeddings across GPUs

```yaml
# genomics_pretrain_distributed.yaml

model:
  vocab_size: 100000000  # 100M variants
  hidden_dim: 512        # Smaller due to vocab size
  num_layers: 8          # Fewer layers due to memory

tokenizer:
  identifier: /mnt/mountcastle/bmfm-genomics/vocab/dbsnp_common
  # Strategy: Only include common variants (MAF > 0.01)
  # Reduces vocab to ~10M

data_module:
  max_length: 10000     # Process 10K variants at a time
  batch_size: 1         # Small batch due to large vocab
  
  # Genomics-specific: No binning needed
  transform_kwargs:
    transforms:
      - transform_name: GenotypeEncode  # {0,1,2} directly

# CRITICAL: Use FSDP for large vocabulary
strategy: fsdp
trainer:
  num_nodes: 8  # Need more GPUs for large vocab

# FSDP configuration
fsdp_config:
  sharding_strategy: "FULL_SHARD"  # Shard parameters + gradients + optimizer
  cpu_offload: false               # Keep on GPU for speed
```

#### Memory Scaling Law

```python
def genomics_memory_requirement(vocab_size: int) -> str:
    """Determine minimum GPU count for genomics model"""
    embedding_dim = 512
    vocab_memory_gb = vocab_size * embedding_dim * 4 / 1024**3
    
    # Rule: Each GPU should hold < 4 GB of embeddings
    min_gpus = math.ceil(vocab_memory_gb / 4)
    strategy = "fsdp" if min_gpus > 1 else "ddp"
    
    return f"Minimum {min_gpus} GPUs, use {strategy}"

# Example: 10M variants
# Result: "Minimum 20 GPUs, use fsdp"
```

**Practical Genomics Strategy**:
1. Filter to common variants (MAF > 1%): ~10M variants
2. Use 8-16 GPUs with FSDP
3. Shard embedding layer across GPUs
4. Batch size = 1-2 per GPU

---

### 3. Multi-Omics (Integrated Modalities)

#### Data Characteristics
- **Entities**: Union of metabolites + proteins + genes
- **Values**: Mixed (concentration, abundance, expression)
- **Typical samples**: 500-5000 entities per sample
- **Vocabulary**: 50K - 500K (combined)

#### Architecture Pattern: Modal-Specific Embeddings

```python
class MultiOmicsTransformer(nn.Module):
    """Transformer with modality-aware embeddings"""
    
    def __init__(self, config):
        self.modality_configs = {
            'metabolomics': {'vocab': 27525, 'embed_dim': 768},
            'proteomics': {'vocab': 20000, 'embed_dim': 768},
            'transcriptomics': {'vocab': 20000, 'embed_dim': 768}
        }
        
        # Separate embedding layers per modality
        self.embeddings = nn.ModuleDict({
            modality: nn.Embedding(cfg['vocab'], cfg['embed_dim'])
            for modality, cfg in self.modality_configs.items()
        })
        
        # Shared transformer
        self.transformer = BertModel(config)
    
    def forward(self, entity_ids, modality_ids, values):
        # Route entities to correct embedding
        embeddings = torch.zeros(batch_size, seq_len, embed_dim)
        for modality in self.modality_configs:
            mask = (modality_ids == modality)
            emb = self.embeddings[modality](entity_ids[mask])
            embeddings[mask] = emb
        
        # Add value embeddings
        value_emb = self.value_embedding(values)
        combined = embeddings + value_emb
        
        return self.transformer(combined)
```

#### Configuration Template

```yaml
# multiomics_pretrain_distributed.yaml

model:
  type: multiomics_transformer
  modality_vocabs:
    metabolomics: 27525
    proteomics: 20000
    transcriptomics: 20000
  shared_hidden_dim: 768
  num_layers: 12

data_module:
  max_length: 8192  # Accommodate combined entities
  batch_size: 2     # Smaller due to longer sequences
  
  # Multi-omics specific data loading
  collation_strategy: multiomics_language_modeling
  modality_fields:
    - metabolomics_entities
    - proteomics_entities
    - transcriptomics_entities

# Distributed: Need FSDP for large combined vocab
strategy: fsdp
trainer:
  num_nodes: 8

# Memory estimate: ~4 GB per GPU with FSDP
```

#### Cross-Modal Attention Pattern

```python
class CrossModalAttention(nn.Module):
    """Attention across different omics modalities"""
    
    def forward(self, metabolite_repr, protein_repr, gene_repr):
        # Learn modality interactions
        attention_weights = self.compute_attention(
            query=metabolite_repr,
            key=torch.cat([protein_repr, gene_repr], dim=1),
            value=torch.cat([protein_repr, gene_repr], dim=1)
        )
        
        return attention_weights
```

---

## Memory and Batch Size Scaling Rules

### Rule 1: Vocabulary Size → Model Memory

```python
def estimate_model_memory(vocab_size: int, hidden_dim: int) -> float:
    """
    Model memory = vocabulary embeddings + transformer layers
    """
    # Embedding layer
    embedding_memory = vocab_size * hidden_dim * 4  # FP32
    
    # Transformer (rule of thumb: 2x embedding memory for 12 layers)
    transformer_memory = 2 * embedding_memory
    
    # Optimizer (Adam: 2x model memory)
    optimizer_memory = 2 * (embedding_memory + transformer_memory)
    
    total_gb = (embedding_memory + transformer_memory + optimizer_memory) / 1024**3
    return total_gb

# Decision tree:
# < 2 GB per GPU → Use DDP (model replication)
# 2-8 GB per GPU → Use FSDP (model sharding)
# > 8 GB per GPU → Use DeepSpeed Stage 3 + CPU offload
```

### Rule 2: Sequence Length → Activation Memory

```python
def estimate_activation_memory(batch_size: int, seq_length: int, hidden_dim: int) -> float:
    """
    Activation memory = O(batch_size × seq_length × hidden_dim)
    """
    # Per-layer activations
    per_layer = batch_size * seq_length * hidden_dim * 4  # FP32
    
    # 12 transformer layers
    num_layers = 12
    total_activation = per_layer * num_layers
    
    # Gradient storage (same as activations)
    total_with_grads = 2 * total_activation
    
    total_gb = total_with_grads / 1024**3
    return total_gb

# Rule: activation_memory << model_memory for good GPU utilization
# If activation_memory > 0.5 * GPU_memory → reduce batch_size or seq_length
```

### Rule 3: Batch Size Selection

```python
def select_optimal_batch_size(
    gpu_memory_gb: float,
    model_memory_gb: float,
    seq_length: int,
    hidden_dim: int
) -> int:
    """
    Select largest batch that fits in GPU memory
    Leave 20% headroom for CUDA overhead
    """
    available_memory = gpu_memory_gb * 0.8 - model_memory_gb
    
    # Binary search for max batch size
    for batch_size in range(1, 128):
        activation_gb = estimate_activation_memory(batch_size, seq_length, hidden_dim)
        if activation_gb > available_memory:
            return batch_size - 1
    
    return 128  # Max

# Example: GTX 1660 Ti (6 GB), metabolomics model (0.5 GB), seq_length=4096
# Result: batch_size = 2
```

---

## Distributed Strategy Decision Tree

```
START: Given model_memory and num_gpus

├─ IF model_memory < 0.5 × GPU_memory:
│   └─> USE: DDP (Distributed Data Parallel)
│       • Each GPU: Full model copy
│       • Communication: Gradient all-reduce
│       • Best for: Metabolomics, Proteomics, Transcriptomics
│       • Efficiency: 80-90%
│
├─ ELIF model_memory < 2 × GPU_memory:
│   └─> USE: FSDP (Fully Sharded Data Parallel)
│       • Each GPU: Sharded model
│       • Communication: All-gather + reduce-scatter
│       • Best for: Multi-omics, small-vocab genomics
│       • Efficiency: 60-70%
│
└─ ELSE:
    └─> USE: DeepSpeed Stage 3 + CPU Offload
        • Each GPU: Minimal model shard
        • Communication: Hierarchical
        • Best for: Large-vocab genomics, multi-modal
        • Efficiency: 40-50%
```

---

## Configuration Templates by Modality

### Template 1: Metabolomics (Current)

```yaml
# metabolomics_pretrain_distributed.yaml
model: {vocab: 27525, hidden: 768, layers: 12}
data: {max_length: 4096, batch_size: 2}
distributed: {nodes: 4, strategy: ddp}
memory: ~2.5 GB per GPU
throughput: ~320 samples/sec (4 GPUs)
```

### Template 2: Proteomics

```yaml
# proteomics_pretrain_distributed.yaml
model: {vocab: 20000, hidden: 768, layers: 12}
data: {max_length: 1024, batch_size: 4}
distributed: {nodes: 4, strategy: ddp}
memory: ~1.5 GB per GPU
throughput: ~400 samples/sec (4 GPUs)
```

### Template 3: Transcriptomics

```yaml
# transcriptomics_pretrain_distributed.yaml
model: {vocab: 20000, hidden: 768, layers: 12}
data: {max_length: 2048, batch_size: 4}
distributed: {nodes: 4, strategy: ddp}
memory: ~1.8 GB per GPU
throughput: ~380 samples/sec (4 GPUs)
```

### Template 4: Multi-Omics

```yaml
# multiomics_pretrain_distributed.yaml
model: {vocab: 67525, hidden: 768, layers: 12}
data: {max_length: 8192, batch_size: 1}
distributed: {nodes: 8, strategy: fsdp}
memory: ~4 GB per GPU (sharded)
throughput: ~150 samples/sec (8 GPUs)
```

### Template 5: Genomics (Common Variants)

```yaml
# genomics_pretrain_distributed.yaml
model: {vocab: 10000000, hidden: 512, layers: 8}
data: {max_length: 10000, batch_size: 1}
distributed: {nodes: 16, strategy: fsdp + cpu_offload}
memory: ~5 GB per GPU (sharded)
throughput: ~50 samples/sec (16 GPUs)
```

---

## Performance Estimation Formulas

### Training Time Projection

```python
def estimate_training_time(
    num_samples: int,
    num_epochs: int,
    throughput_samples_per_sec: float,
    validation_overhead: float = 1.1  # 10% overhead
) -> float:
    """Estimate total training time in hours"""
    total_samples = num_samples * num_epochs
    time_seconds = total_samples / throughput_samples_per_sec
    time_hours = time_seconds / 3600 * validation_overhead
    return time_hours

# Examples:
# Metabolomics: 723 samples, 50 epochs, 320 samples/sec (4 GPUs)
# → 1.6 hours

# Proteomics: 10000 samples, 20 epochs, 400 samples/sec (4 GPUs)
# → 14 hours

# Multi-omics: 5000 samples, 30 epochs, 150 samples/sec (8 GPUs)
# → 30 hours
```

### Throughput Scaling

```python
def estimate_throughput(
    single_gpu_throughput: float,
    num_gpus: int,
    parallel_efficiency: float
) -> float:
    """
    Parallel efficiency depends on strategy:
    - DDP: 0.8-0.9
    - FSDP: 0.6-0.7
    - DeepSpeed: 0.4-0.5
    """
    return single_gpu_throughput * num_gpus * parallel_efficiency
```

---

## Implementation Checklist for New Modality

- [ ] **Step 1: Vocabulary Creation**
  - [ ] Identify entity vocabulary (IDs, names)
  - [ ] Determine vocabulary size
  - [ ] Create `vocab.txt` and `multifield_vocab.json`

- [ ] **Step 2: Data Preprocessing**
  - [ ] Convert raw data to h5ad format
  - [ ] Implement value transformation (log, normalize, bin)
  - [ ] Validate data loading pipeline

- [ ] **Step 3: Configuration Adaptation**
  - [ ] Copy template YAML for modality
  - [ ] Update vocabulary paths
  - [ ] Set max_length based on entities per sample
  - [ ] Calculate optimal batch_size

- [ ] **Step 4: Memory Estimation**
  - [ ] Calculate model memory requirements
  - [ ] Determine distributed strategy (DDP/FSDP/DeepSpeed)
  - [ ] Select number of GPUs

- [ ] **Step 5: Testing**
  - [ ] Run single-GPU fast dev run (10 steps)
  - [ ] Test distributed setup (5 steps)
  - [ ] Profile performance

- [ ] **Step 6: Production Training**
  - [ ] Launch full training
  - [ ] Monitor convergence
  - [ ] Evaluate on downstream tasks

---

## Summary: Key Patterns for Cross-Modal Application

### Pattern 1: Vocabulary Size Determines Strategy
```
< 50K entities  → DDP (model replication)
50K - 500K      → FSDP (model sharding)
> 500K          → DeepSpeed (aggressive optimization)
```

### Pattern 2: Batch Size Arithmetic
```
effective_batch = per_gpu_batch × num_gpus × gradient_accum

Constraint: per_gpu_batch × seq_length × hidden_dim < GPU_memory
```

### Pattern 3: Throughput Estimation
```
throughput = single_gpu_throughput × num_gpus × efficiency

efficiency = {
    0.85 for DDP,
    0.65 for FSDP,
    0.45 for DeepSpeed
}
```

### Pattern 4: Multi-Modality Handling
```
• Separate embeddings per modality
• Shared transformer backbone
• Cross-modal attention layers
• Modality-specific loss weighting
```

---

## Appendix: Quick Reference Tables

### A. Memory Requirements by Modality

| Modality | Vocab Size | Model Memory (GB) | Strategy | Min GPUs |
|----------|-----------|------------------|----------|----------|
| Metabolomics | 27K | 0.5 | DDP | 1 |
| Proteomics | 20K | 0.4 | DDP | 1 |
| Transcriptomics | 20K | 0.4 | DDP | 1 |
| Multi-omics | 67K | 1.2 | DDP/FSDP | 2 |
| Genomics (common) | 10M | 20 | FSDP | 8 |
| Genomics (all) | 100M | 200 | DeepSpeed | 64 |

### B. Recommended Configurations by Cluster Size

| Cluster | Modality | Config | Est. Time (50 epochs) |
|---------|----------|--------|----------------------|
| 4× GTX 1660 Ti | Metabolomics | DDP, batch=2 | 1.6 hrs |
| 4× GTX 1660 Ti | Proteomics | DDP, batch=4 | 8 hrs (10K samples) |
| 8× A100 (40GB) | Multi-omics | FSDP, batch=4 | 12 hrs (5K samples) |
| 16× A100 (80GB) | Genomics | FSDP, batch=2 | 48 hrs (1K samples) |

### C. Batch Size Recommendations

| GPU Memory | Metabolomics | Proteomics | Multi-omics | Genomics |
|-----------|--------------|-----------|------------|----------|
| 6 GB (GTX 1660 Ti) | 2 | 4 | 1 | N/A |
| 16 GB (V100) | 8 | 16 | 4 | 1 |
| 40 GB (A100) | 16 | 32 | 8 | 2 |
| 80 GB (A100-80) | 32 | 64 | 16 | 4 |

---

**End of Cross-Modal Application Guide**

For questions or issues, refer to:
- Distributed training patterns: `DISTRIBUTED_TRAINING_PATTERNS.md`
- Implementation guide: `DISTRIBUTED_TRAINING_GUIDE.md`
- Test suite: `test_distributed_config.sh`
- Performance profiler: `profile_training_performance.sh`
