# BMFM-NVP Distributed Training Implementation Guide
# PyTorch Lightning Native Strategy on Mountcastle Cluster

## Architecture Overview

### Physical Topology
```
Mountcastle Cluster - Homogeneous GPU Pool
├── neuro02 (Master)
│   └── GTX 1660 Ti (6GB VRAM)
├── neuro03 (Worker)
│   └── GTX 1660 Ti (6GB VRAM)
├── neuro08 (Worker)
│   └── GTX 1660 Ti (6GB VRAM)
└── neuro09 (Worker)
    └── GTX 1660 Ti (6GB VRAM)

Total: 4 nodes × 1 GPU = 4 workers
Aggregate VRAM: 24 GB
```

### Logical Topology (DDP Strategy)
```
Distributed Data Parallel Pattern:
┌─────────────────────────────────────────────┐
│ Global Batch (size=8)                       │
│ Distributed across 4 workers               │
└─────────────────────────────────────────────┘
         │
         ├──────┬──────┬──────┬──────┐
         ▼      ▼      ▼      ▼      ▼
    Worker0 Worker1 Worker2 Worker3
    (batch=2)(batch=2)(batch=2)(batch=2)
         │      │      │      │
         └──────┴──────┴──────┘
                 │
         [All-Reduce Gradients]
                 │
         [Synchronized Update]
```

### Numerical Properties
- **Per-GPU Batch Size**: 2
- **Global Batch Size**: 2 × 4 = 8
- **Gradient Sync Frequency**: Every training step
- **Communication Overhead**: O(model_parameters)
- **Expected Speedup**: ~3.2x (80% parallel efficiency)

## File Structure

```
/mnt/mountcastle/bmfm-metabolomics/
├── configs/
│   └── metabolomics_pretrain_distributed.yaml  # Training configuration
├── data/
│   └── merged/
│       ├── metabolomics_training.h5ad
│       └── metabolomics_training_processed.h5ad
├── vocab/
│   └── all_genes_vocab/
│       ├── multifield_vocab.json
│       └── vocab.txt
├── checkpoints/                                 # Auto-created
├── outputs/                                     # Auto-created
└── scripts/
    ├── launch_pytorch_lightning.sh             # Main launcher
    ├── validate_distributed_setup.sh           # Pre-flight checks
    └── monitor_training.sh                     # Real-time monitoring
```

## Implementation Steps

### Phase 1: Setup and Validation

1. **Copy files to cluster**
   ```bash
   # On your local machine
   scp metabolomics_pretrain_distributed.yaml neuro02:/mnt/mountcastle/bmfm-metabolomics/configs/
   scp launch_pytorch_lightning.sh neuro02:/mnt/mountcastle/bmfm-metabolomics/scripts/
   scp validate_distributed_setup.sh neuro02:/mnt/mountcastle/bmfm-metabolomics/scripts/
   scp monitor_training.sh neuro02:/mnt/mountcastle/bmfm-metabolomics/scripts/
   ```

2. **Make scripts executable**
   ```bash
   ssh neuro02
   cd /mnt/mountcastle/bmfm-metabolomics/scripts
   chmod +x *.sh
   ```

3. **Run validation**
   ```bash
   ./validate_distributed_setup.sh
   ```
   
   This checks:
   - ✓ GPU availability on all 4 nodes
   - ✓ Network connectivity
   - ✓ CephFS mounts
   - ✓ Data and vocabulary files
   - ✓ PyTorch distributed capabilities
   - ✓ Configuration file syntax
   - ✓ Single-node fast dev run (3 steps)

### Phase 2: Test Distributed Training

4. **Fast dev run (distributed)**
   ```bash
   cd /mnt/mountcastle/bmfm-metabolomics
   
   bmfm-targets-run \
       -cn metabolomics_pretrain_distributed \
       trainer.fast_dev_run=10 \
       trainer.num_nodes=4
   ```
   
   This runs 10 training steps to verify:
   - Multi-node coordination
   - Gradient synchronization
   - Loss computation
   - Checkpoint saving

### Phase 3: Full Training

5. **Launch full training**
   ```bash
   # Option A: Using launch script (recommended)
   cd /mnt/mountcastle/bmfm-metabolomics/scripts
   ./launch_pytorch_lightning.sh
   
   # Option B: Direct command
   cd /mnt/mountcastle/bmfm-metabolomics
   bmfm-targets-run -cn metabolomics_pretrain_distributed
   ```

6. **Monitor training (in separate terminal)**
   ```bash
   ssh neuro02
   cd /mnt/mountcastle/bmfm-metabolomics/scripts
   ./monitor_training.sh
   ```

## Configuration Parameters

### Key Settings in metabolomics_pretrain_distributed.yaml

```yaml
# Physical topology
trainer:
  num_nodes: 4              # Use all 4 GTX 1660 Ti nodes
  sync_batchnorm: true      # Sync batch norm across GPUs

# Strategy
strategy: ddp               # Distributed Data Parallel
accelerator: gpu
devices: 1                  # 1 GPU per node
precision: "16-mixed"       # Mixed precision (FP16 + FP32)

# Batch configuration
data_module:
  batch_size: 2             # Per-GPU (global = 2×4 = 8)
  num_workers: 4            # Data loading workers per GPU

# Checkpointing
checkpoints_every_n_train_steps: 500
default_root_dir: /mnt/mountcastle/bmfm-metabolomics/checkpoints
```

### Tunable Parameters for Performance

```yaml
# Increase global batch size (adjust per-GPU batch)
data_module.batch_size: 4              # Global = 4×4 = 16

# Gradient accumulation (simulate larger batch)
accumulate_grad_batches: 2             # Effective global = 8×2 = 16

# Mixed precision control
precision: "16-mixed"                  # FP16 (fastest)
precision: "32"                        # FP32 (more stable)

# Checkpoint frequency
checkpoints_every_n_train_steps: 1000 # Less frequent
checkpoints_every_n_train_steps: 250  # More frequent
```

## Troubleshooting

### Issue: "NCCL initialization failed"
**Cause**: Network communication problem between GPUs
**Solution**:
```bash
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eno1
```

### Issue: "Out of memory"
**Cause**: Batch size too large for 6GB VRAM
**Solution**: Reduce batch size or enable gradient accumulation
```yaml
data_module.batch_size: 1  # Smaller per-GPU batch
accumulate_grad_batches: 4  # Simulate larger batch
```

### Issue: "Worker timeout"
**Cause**: Slow data loading or node failure
**Solution**: Check data workers and node status
```bash
# Reduce data workers
data_module.num_workers: 2

# Check node status
ssh neuro03 nvidia-smi
```

### Issue: "Gradient synchronization slow"
**Cause**: Large model or slow network
**Solution**: Enable gradient compression (if available)
```yaml
strategy:
  ddp:
    find_unused_parameters: false
    gradient_as_bucket_view: true
```

## Performance Expectations

### Training Speed Estimates

**Single GTX 1660 Ti baseline:**
- ~100 samples/second
- 50 epochs × 723 samples = 36,150 samples
- Estimated time: ~6 minutes per epoch
- Total: ~5 hours

**4x GTX 1660 Ti distributed (80% efficiency):**
- ~320 samples/second (3.2x speedup)
- Estimated time: ~1.9 minutes per epoch
- Total: ~1.6 hours

### Memory Usage per GPU
- Model parameters: ~100-200 MB
- Activations (batch=2): ~1-2 GB
- Optimizer state: ~200-400 MB
- CUDA overhead: ~500 MB
- **Total: ~2-3 GB per GPU** (well within 6GB limit)

### Communication Overhead
- Gradient all-reduce per step: ~106M parameters × 4 bytes = ~400 MB
- Network bandwidth needed: ~400 MB / step time
- With 1 Gbps Ethernet: ~8 seconds per all-reduce (acceptable)

## Checkpointing and Resume

### Automatic Checkpointing
- Checkpoints saved every 500 steps to CephFS
- Location: `/mnt/mountcastle/bmfm-metabolomics/checkpoints/`
- Format: `epoch=X-step=Y.ckpt`

### Resume Training
```bash
bmfm-targets-run \
    -cn metabolomics_pretrain_distributed \
    model.checkpoint=/path/to/checkpoint.ckpt
```

### Best Model Selection
```yaml
# In configuration
callbacks:
  - class_path: ModelCheckpoint
    init_args:
      monitor: val_loss
      mode: min
      save_top_k: 3
```

## Monitoring and Logging

### Real-time Monitoring
```bash
# Terminal 1: Training
./launch_pytorch_lightning.sh

# Terminal 2: Monitoring
./monitor_training.sh

# Terminal 3: Per-node GPU stats
watch -n 1 'for node in neuro02 neuro03 neuro08 neuro09; do \
    echo "=== $node ==="; \
    ssh $node nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; \
done'
```

### Log Files
- Training logs: `/mnt/mountcastle/bmfm-metabolomics/outputs/*.log`
- Checkpoint metadata: `/mnt/mountcastle/bmfm-metabolomics/checkpoints/*.yaml`

## Next Steps After Training

1. **Evaluate model**
   ```bash
   bmfm-targets-run \
       -cn metabolomics_eval \
       model.checkpoint=/path/to/best.ckpt
   ```

2. **Extract embeddings**
   ```bash
   bmfm-targets-run \
       -cn metabolomics_predict \
       model.checkpoint=/path/to/best.ckpt \
       task.output_embeddings=true
   ```

3. **Fine-tune on downstream task**
   ```bash
   bmfm-targets-run \
       -cn metabolomics_finetune \
       model.checkpoint=/path/to/best.ckpt \
       label_column_name=disease_state
   ```

## Summary Checklist

- [ ] Files copied to cluster
- [ ] Scripts made executable
- [ ] Validation script passes
- [ ] Fast dev run (10 steps) successful
- [ ] Full training launched
- [ ] Monitoring active
- [ ] Checkpoints being saved
- [ ] Training metrics improving

## Contact and Support

For issues or questions about this implementation:
1. Check logs in `/mnt/mountcastle/bmfm-metabolomics/outputs/`
2. Review validation output
3. Verify all nodes are responsive
4. Check GPU memory with `nvidia-smi`
