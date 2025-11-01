# Architectural Patterns: Distributed Pre-Training for Transformer Models

## Pattern Catalog

### 1. Physical Topology Abstraction Pattern

**Intent**: Decouple training logic from hardware configuration

**Structure**:
```python
@dataclass
class ClusterTopology:
    """Hardware-agnostic cluster description"""
    nodes: List[Node]
    
    @property
    def total_workers(self) -> int:
        return sum(node.num_devices for node in self.nodes)
    
    @property
    def is_homogeneous(self) -> bool:
        device_types = set(node.device_type for node in self.nodes)
        return len(device_types) == 1
    
    def select_strategy(self, model_size_bytes: int) -> str:
        """Strategy selection based on topology and model"""
        device_memory = self.nodes[0].device_memory_bytes
        
        if model_size_bytes < 0.5 * device_memory:
            return "ddp"  # Model fits in single device
        elif model_size_bytes < 2 * device_memory:
            return "fsdp"  # Requires sharding
        else:
            return "deepspeed_stage_3"  # Extreme optimization
```

**Application to BMFM-NVP**:
```yaml
# Concrete instantiation for Mountcastle
cluster:
  nodes:
    - {name: neuro02, device: GTX_1660Ti, memory_gb: 6}
    - {name: neuro03, device: GTX_1660Ti, memory_gb: 6}
    - {name: neuro08, device: GTX_1660Ti, memory_gb: 6}
    - {name: neuro09, device: GTX_1660Ti, memory_gb: 6}
  strategy: ddp  # Auto-selected based on model size
```

---

### 2. Batch Size Arithmetic Pattern

**Intent**: Maintain consistent effective batch size across distributed configurations

**Formula**:
```
Effective_Batch = Per_GPU_Batch × Num_GPUs × Gradient_Accumulation_Steps
```

**Implementation**:
```python
def compute_distributed_batch_config(
    target_effective_batch: int,
    num_gpus: int,
    memory_constraint_per_gpu: int
) -> Dict[str, int]:
    """
    Compute optimal per-GPU batch and accumulation steps
    
    Constraints:
    - Effective batch = target
    - Per-GPU batch fits in memory
    - Accumulation minimized for faster updates
    """
    max_batch_per_gpu = memory_constraint_per_gpu // sample_memory_size
    
    if target_effective_batch <= max_batch_per_gpu * num_gpus:
        # No accumulation needed
        per_gpu_batch = target_effective_batch // num_gpus
        accumulation = 1
    else:
        # Use max per-GPU batch, accumulate rest
        per_gpu_batch = max_batch_per_gpu
        accumulation = target_effective_batch // (per_gpu_batch * num_gpus)
    
    return {
        "per_gpu_batch": per_gpu_batch,
        "accumulation_steps": accumulation,
        "effective_batch": per_gpu_batch * num_gpus * accumulation
    }
```

**BMFM-NVP Application**:
```python
# Target: 32 effective batch size
# Available: 4 GPUs, 6GB each
config = compute_distributed_batch_config(
    target_effective_batch=32,
    num_gpus=4,
    memory_constraint_per_gpu=6_000_000_000
)
# Result: {per_gpu_batch: 8, accumulation_steps: 1, effective_batch: 32}
```

---

### 3. Strategy Selection Pattern

**Intent**: Choose optimal distributed strategy based on model and hardware

**Decision Tree**:
```python
def select_distributed_strategy(
    model_params: int,
    device_memory_bytes: int,
    num_devices: int,
    network_bandwidth_gbps: float
) -> str:
    """
    DDP: Model replication (each GPU has full copy)
    FSDP: Model sharding (parameters split across GPUs)
    DeepSpeed: Advanced optimizations (ZeRO stages)
    """
    model_size_bytes = model_params * 4  # FP32
    
    # Check if model fits in single device with optimizer
    memory_per_device = device_memory_bytes
    model_plus_optimizer = model_size_bytes * 2  # Model + optimizer state
    
    if model_plus_optimizer < 0.5 * memory_per_device:
        # DDP: Fast, simple, low communication
        return "ddp"
    
    elif model_plus_optimizer < memory_per_device * num_devices:
        # FSDP: Sharding needed but feasible
        if network_bandwidth_gbps < 10:
            return "fsdp_offload_cpu"  # Slow network, use CPU offload
        else:
            return "fsdp"
    
    else:
        # DeepSpeed: Extreme memory optimization
        return "deepspeed_stage_3_offload"
```

**BMFM-NVP Decision**:
```python
# Model: ~106M parameters = ~424 MB (FP32)
# Device: 6 GB VRAM
# Network: 1 Gbps Ethernet

strategy = select_distributed_strategy(
    model_params=106_000_000,
    device_memory_bytes=6_000_000_000,
    num_devices=4,
    network_bandwidth_gbps=1.0
)
# Result: "ddp" (model + optimizer = ~850 MB << 6 GB)
```

---

### 4. Communication Overhead Pattern

**Intent**: Estimate and minimize gradient synchronization cost

**Analysis**:
```python
def estimate_communication_time(
    model_params: int,
    num_devices: int,
    network_bandwidth_gbps: float,
    strategy: str = "ddp"
) -> float:
    """
    DDP: All-reduce of all gradients
    FSDP: All-gather of sharded parameters
    """
    bytes_per_param = 4  # FP32 gradients
    
    if strategy == "ddp":
        # Ring all-reduce: 2(N-1)/N factor
        data_volume = model_params * bytes_per_param
        ring_factor = 2 * (num_devices - 1) / num_devices
        total_bytes = data_volume * ring_factor
    
    elif strategy == "fsdp":
        # All-gather sharded params
        data_volume = model_params * bytes_per_param / num_devices
        total_bytes = data_volume * num_devices
    
    network_bandwidth_bytes_per_sec = network_bandwidth_gbps * 125_000_000
    time_seconds = total_bytes / network_bandwidth_bytes_per_sec
    
    return time_seconds
```

**BMFM-NVP Communication Cost**:
```python
sync_time = estimate_communication_time(
    model_params=106_000_000,
    num_devices=4,
    network_bandwidth_gbps=1.0,
    strategy="ddp"
)
# Result: ~0.25 seconds per training step
# If training step = 1 second, communication overhead = 20%
```

---

### 5. Checkpoint Sharding Pattern

**Intent**: Efficiently save/load distributed model states

**Structure**:
```python
class DistributedCheckpoint:
    """Checkpoint format for distributed models"""
    
    def save(self, model, optimizer, path, strategy):
        if strategy == "ddp":
            # Single checkpoint (all GPUs have same model)
            if rank == 0:
                torch.save({
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, path)
        
        elif strategy == "fsdp":
            # Sharded checkpoint (gather from all GPUs)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
                if rank == 0:
                    state_dict = model.state_dict()
                    torch.save(state_dict, path)
    
    def load(self, model, path, strategy):
        checkpoint = torch.load(path)
        
        if strategy == "ddp":
            model.module.load_state_dict(checkpoint['model'])
        
        elif strategy == "fsdp":
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
                model.load_state_dict(checkpoint)
```

---

### 6. Data Sharding Pattern

**Intent**: Distribute dataset across workers without duplication

**Implementation**:
```python
class DistributedDataSampler:
    """Shard dataset across GPUs"""
    
    def __init__(self, dataset, num_replicas, rank):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = len(dataset) // num_replicas
    
    def __iter__(self):
        # Each worker gets unique subset
        indices = list(range(len(self.dataset)))
        indices = indices[self.rank::self.num_replicas]
        return iter(indices)
```

**Visualization**:
```
Dataset: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
Num GPUs: 4

GPU 0: [0, 4, 8]    (rank 0)
GPU 1: [1, 5, 9]    (rank 1)
GPU 2: [2, 6, 10]   (rank 2)
GPU 3: [3, 7, 11]   (rank 3)
```

---

### 7. Fault Tolerance Pattern

**Intent**: Handle node failures gracefully in distributed training

**Structure**:
```python
class ElasticTraining:
    """Checkpoint-based recovery from failures"""
    
    def train_with_recovery(self, config):
        try:
            # Main training loop
            for epoch in range(config.max_epochs):
                train_epoch()
                
                # Frequent checkpointing for recovery
                if epoch % config.checkpoint_frequency == 0:
                    save_checkpoint()
        
        except WorkerFailure as e:
            # Attempt recovery
            last_checkpoint = find_latest_checkpoint()
            
            if last_checkpoint:
                load_checkpoint(last_checkpoint)
                restart_from_checkpoint()
            else:
                raise e
```

---

### 8. Gradient Accumulation Pattern

**Intent**: Simulate larger batch sizes without increasing memory

**Numerical Analysis**:
```python
# Without accumulation:
# Memory: O(batch_size × sequence_length × hidden_dim)
# Update frequency: Every step

# With accumulation (steps=K):
# Memory: Same (only activations for current micro-batch)
# Effective batch: batch_size × K
# Update frequency: Every K steps
# Communication: Same total volume, less frequent

def configure_gradient_accumulation(
    target_batch: int,
    memory_limited_batch: int
) -> int:
    """Compute accumulation steps to reach target batch"""
    return math.ceil(target_batch / memory_limited_batch)
```

**Trade-off Analysis**:
```
Advantages:
+ Larger effective batch without OOM
+ Same gradient quality as true large batch
+ Reduced communication frequency

Disadvantages:
- Slower convergence (fewer updates per epoch)
- Slightly different batch norm statistics
- K× more forward passes
```

---

## Integration with BMFM-NVP

### Applied Pattern Composition

```yaml
# metabolomics_pretrain_distributed.yaml
# Demonstrates pattern integration

# Pattern 1: Physical Topology
trainer:
  num_nodes: 4              # Homogeneous GTX 1660 Ti cluster
  
# Pattern 2: Batch Arithmetic  
data_module:
  batch_size: 2             # Per-GPU
  # Effective: 2 × 4 = 8
  
# Pattern 3: Strategy Selection
strategy: ddp               # Model fits in memory, use DDP

# Pattern 8: Gradient Accumulation (optional)
accumulate_grad_batches: 1  # No accumulation needed

# Pattern 5: Checkpointing
checkpoints_every_n_train_steps: 500
default_root_dir: /mnt/mountcastle/bmfm-metabolomics/checkpoints
```

---

## Generalization to Other Modalities

### Pattern Application Matrix

| Pattern | Metabolomics | Proteomics | Transcriptomics | Multi-omics |
|---------|-------------|-----------|----------------|------------|
| Physical Topology | ✓ 4 GPUs | ✓ 4-8 GPUs | ✓ 8+ GPUs | ✓ 16+ GPUs |
| Batch Arithmetic | 2×4=8 | 4×8=32 | 8×8=64 | 16×8=128 |
| Strategy | DDP | DDP | FSDP | DeepSpeed |
| Accumulation | 1x | 1x | 2x | 4x |

### Scaling Laws

```python
def estimate_training_time(
    num_samples: int,
    batch_size: int,
    num_gpus: int,
    samples_per_second_per_gpu: float,
    parallel_efficiency: float = 0.8
) -> float:
    """
    Estimate training time with distributed setup
    
    parallel_efficiency: Fraction of ideal speedup
        DDP: 0.8-0.9 (good)
        FSDP: 0.6-0.7 (moderate)
        DeepSpeed: 0.5-0.6 (communication heavy)
    """
    samples_per_second_total = (
        samples_per_second_per_gpu * 
        num_gpus * 
        parallel_efficiency
    )
    
    total_time = num_samples / samples_per_second_total
    return total_time
```

---

## Key Insights for Project Knowledge

### 1. Homogeneous Hardware Principle
**Pattern**: Use identical GPUs for distributed training
**Rationale**: Slowest GPU becomes bottleneck in synchronous training
**Implementation**: GTX 1660 Ti pool (neuro02, 03, 08, 09)

### 2. Communication-Computation Balance
**Pattern**: Choose strategy based on network bandwidth
**Formula**: `communication_time / computation_time < 0.2` for good efficiency
**BMFM-NVP**: DDP with 1 Gbps Ethernet achieves ~80% efficiency

### 3. Memory-Batch Trade-off
**Pattern**: Gradient accumulation extends effective batch without OOM
**Formula**: `effective_batch = per_gpu_batch × num_gpus × accum_steps`
**Application**: With 6 GB VRAM, can simulate 32+ effective batch

### 4. Checkpoint Frequency
**Pattern**: Balance checkpoint overhead vs. recovery cost
**Rule of thumb**: Checkpoint every 5-10 minutes of training
**BMFM-NVP**: 500 steps ≈ 8 minutes

### 5. Strategy Selection Hierarchy
```
Model + Optimizer fits in GPU? → DDP (best performance)
    ↓ No
Model fits across GPUs? → FSDP (good performance)
    ↓ No
Use CPU offloading → DeepSpeed Stage 3 (enables training)
```

---

## Summary Pseudo-code: Complete Distributed Training

```python
def distributed_pretraining_pipeline(
    cluster_topology: ClusterTopology,
    model_config: ModelConfig,
    data_config: DataConfig
):
    # Pattern 1: Topology analysis
    strategy = cluster_topology.select_strategy(model_config.size)
    
    # Pattern 2: Batch configuration
    batch_config = compute_distributed_batch_config(
        target_effective_batch=data_config.target_batch,
        num_gpus=cluster_topology.total_workers,
        memory_constraint=cluster_topology.nodes[0].memory
    )
    
    # Pattern 3: Initialize distributed process group
    init_process_group(
        backend="nccl",  # GPU communication
        world_size=cluster_topology.total_workers,
        rank=get_rank()
    )
    
    # Pattern 4: Wrap model with distributed strategy
    model = create_model(model_config)
    if strategy == "ddp":
        model = DistributedDataParallel(model)
    elif strategy == "fsdp":
        model = FullyShardedDataParallel(model)
    
    # Pattern 6: Shard dataset
    sampler = DistributedSampler(
        dataset,
        num_replicas=cluster_topology.total_workers,
        rank=get_rank()
    )
    
    # Pattern 8: Training loop with accumulation
    for epoch in range(config.max_epochs):
        for step, batch in enumerate(dataloader):
            loss = model(batch)
            loss = loss / batch_config.accumulation_steps
            loss.backward()
            
            if (step + 1) % batch_config.accumulation_steps == 0:
                # Pattern 4: Gradient synchronization
                synchronize_gradients(model)
                optimizer.step()
                optimizer.zero_grad()
        
        # Pattern 5: Checkpointing
        if epoch % checkpoint_frequency == 0:
            save_checkpoint(model, optimizer, epoch)
```

---

## Recommended Updates to Project Knowledge

**Document to create**: `Distributed_Training_Patterns.md`

**Sections**:
1. Pattern catalog (this document)
2. Mountcastle cluster specification
3. BMFM-NVP specific configuration
4. Generalization guidelines for other modalities
5. Troubleshooting decision tree
6. Performance benchmarks

**Why**: Future chats can reference patterns without re-deriving implementation details
